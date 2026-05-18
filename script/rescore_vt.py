"""
Temp script: re-run velocity-reward (R_T) scoring on a pre-existing
eval_rollout.jsonl, without retraining.

Mirrors the post-training scoring block in script/run_game24_one.py, but
reads puzzles/solutions stateless-ly via enumerate_solutions so we don't
need to reconstruct the original train/eval/probe split.

The input file is read, every row that has a non-empty `completion` is
scored, and three new fields are written back in-place:
    R_T, R_per_token, cumR_resampled

Usage:
    python script/rescore_vt.py \
        --rollouts logs/game24_sweep_exp4/len512/Qwen__Qwen3-4B/eval_rollout.jsonl \
        --scorer-model Qwen/Qwen3-4B

By default the scorer is the same model used for training (passed via
--scorer-model). For a "trained-policy" R_T you would instead pass a
saved checkpoint directory. For a "base-model" R_T pass the HF hub id.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# cuDNN's SDPA backend can fail to build an execution plan for some
# Qwen3 / long-context bf16 shapes ("No valid execution plans built").
# Disable it so PyTorch falls back to FlashAttention or the math backend.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

# Repo root → sys.path so `src.*` imports resolve when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game24utils import enumerate_solutions, to_chat
from src.velocity import compute_vt_batched


def _canon(expr: str) -> str:
    """AST-canonicalize so '(5*5- (2 - 1))' == '(5*5)-(2-1)'."""
    try:
        return ast.dump(ast.parse(expr, mode="eval").body)
    except Exception:
        return expr


def _build_refs(row: dict) -> Tuple[List[str], int, int]:
    """Return (refs, n_canonical, own_idx).

    ``refs[:n_canonical]`` are the enumerated 24-solutions for the puzzle.
    If the row has a non-empty ``expr`` that is AST-distinct from all of
    them, it is appended as ``refs[-1]`` and ``own_idx`` points at it;
    otherwise ``own_idx`` is the matching canonical index (or -1 if the
    row has no expr at all).
    """
    sols = list(enumerate_solutions(tuple(row["numbers"])))
    n_can = len(sols)
    refs = list(sols)
    own_expr = row.get("expr") or ""
    if not own_expr:
        return refs, n_can, -1
    own_canon = _canon(own_expr)
    own_idx = next((i for i, s in enumerate(sols) if _canon(s) == own_canon), None)
    if own_idx is None:
        refs.append(own_expr)
        own_idx = len(refs) - 1
    return refs, n_can, own_idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", required=True, type=Path,
                   help="Path to eval_rollout.jsonl to re-score (modified in place).")
    p.add_argument("--scorer-model", required=True,
                   help="HF hub id or local path of the model to use as scorer. "
                        "Pass the base model for prompt-search R_T; pass a "
                        "trained checkpoint for information-gain R_T.")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer to use (defaults to --scorer-model). Override "
                        "if your scorer checkpoint doesn't ship its tokenizer.")
    p.add_argument("--micro-batch", type=int, default=8,
                   help="Forward-pass micro-batch for compute_vt_batched.")
    p.add_argument("--resample-pts", type=int, default=100,
                   help="Grid size for cumR_resampled.")
    p.add_argument("--device", default=None,
                   help="Device override, e.g. 'cuda:0'. Defaults to first CUDA "
                        "device or CPU.")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional output path. Default: overwrite --rollouts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load model + score but don't write the output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rollouts_path: Path = args.rollouts
    if not rollouts_path.exists():
        raise FileNotFoundError(f"rollouts file not found: {rollouts_path}")

    rows = [json.loads(l) for l in rollouts_path.read_text().splitlines()
            if l.strip()]
    print(f"[rescore] loaded {len(rows)} rows from {rollouts_path}")

    # --- Tokenizer + scorer load -------------------------------------------
    tok_src = args.tokenizer or args.scorer_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    device = (args.device
              or (f"cuda:{torch.cuda.current_device()}"
                  if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[rescore] loading scorer {args.scorer_model} → {device} ({dtype})")
    try:
        scorer = AutoModelForCausalLM.from_pretrained(
            args.scorer_model, attn_implementation="flash_attention_2",
            dtype=dtype,
        ).to(device).eval()
        print("[rescore] scorer using flash_attention_2")
    except (ImportError, ValueError) as e:
        print(f"[rescore] FA2 unavailable ({e.__class__.__name__}); using default attn")
        scorer = AutoModelForCausalLM.from_pretrained(
            args.scorer_model, dtype=dtype,
        ).to(device).eval()

    # --- Build prompt/completion/ref triples -------------------------------
    # For each scored row we score the rollout's CoT against EVERY canonical
    # 24-solution for the puzzle (plus the row's own answer if it's not in
    # that set). The canonical R_T is then max_s R_T(rollout → s) — the
    # "best decoding reward". We also expose R_T_own (decoding reward of the
    # rollout's own \boxed{} expression, possibly wrong) so downstream
    # analyses can study
    #     R_T_correct - R_T_own = log p(correct | CoT) - log p(own | CoT).
    prompts, completions, refs_flat = [], [], []
    valid: List[bool] = []
    row_n_can: List[int] = []   # # canonical refs scored for this row
    row_own_idx: List[int] = [] # own_idx (into row's ref slice), -1 if N/A
    skipped_no_completion = 0
    skipped_no_solution = 0
    for row in rows:
        if not row.get("completion"):
            valid.append(False); row_n_can.append(0); row_own_idx.append(-1)
            skipped_no_completion += 1
            continue
        refs, n_can, own_idx = _build_refs(row)
        if n_can == 0:
            valid.append(False); row_n_can.append(0); row_own_idx.append(-1)
            skipped_no_solution += 1
            continue
        puzzle = {"numbers": list(row["numbers"]), "solutions": refs}
        prompt = tokenizer.apply_chat_template(
            to_chat(puzzle)["prompt"], tokenize=False, add_generation_prompt=True)
        prompts.extend([prompt] * len(refs))
        completions.extend([row["completion"]] * len(refs))
        refs_flat.extend(refs)
        valid.append(True); row_n_can.append(n_can); row_own_idx.append(own_idx)

    n_score = sum(valid)
    print(f"[rescore] scoring {n_score}/{len(rows)} rollouts against "
          f"{len(refs_flat)} (rollout, ref) pairs "
          f"(skipped: {skipped_no_completion} empty completion, "
          f"{skipped_no_solution} no enumerable solution)")
    if n_score == 0:
        print("[rescore] nothing to score; exiting"); return

    # --- Score -------------------------------------------------------------
    t0 = time.time()
    scored = compute_vt_batched(
        prompts, completions, refs_flat, scorer, tokenizer,
        micro_batch_size=args.micro_batch,
    )
    print(f"[rescore] forward pass done in {time.time()-t0:.0f}s "
          f"({len(scored)} ref-scores)")

    # --- Augment rows ------------------------------------------------------
    grid = np.linspace(0.0, 1.0, args.resample_pts)
    cursor = 0
    for row, ok, n_can, own_idx in zip(rows, valid, row_n_can, row_own_idx):
        if not ok:
            row["R_T"] = None
            row["R_per_token"] = None
            row["cumR_resampled"] = None
            row["R_T_per_ref"] = None
            row["R_T_own"] = None
            row["best_ref_idx"] = None
            continue
        n_refs = n_can + (1 if own_idx >= n_can else 0)
        chunk = scored[cursor:cursor + n_refs]
        cursor += n_refs

        # canonical (puzzle-correct) R_Ts
        R_T_can = [float(s["R_T"]) if not np.isnan(s["R_T"]) else None
                   for s in chunk[:n_can]]
        valid_can = [(i, v) for i, v in enumerate(R_T_can) if v is not None]
        if valid_can:
            best_i, best_R = max(valid_can, key=lambda x: x[1])
            best = chunk[best_i]
            vt = np.asarray(best["vt"], dtype=float)
            row["R_T"] = best_R
            row["R_per_token"] = (float(best["R_per_token"])
                                  if not np.isnan(best["R_per_token"]) else None)
            if len(vt):
                R = np.cumsum(vt)
                x = np.linspace(0.0, 1.0, len(R))
                row["cumR_resampled"] = np.interp(grid, x, R).tolist()
            else:
                row["cumR_resampled"] = None
            row["best_ref_idx"] = best_i
        else:
            row["R_T"] = None
            row["R_per_token"] = None
            row["cumR_resampled"] = None
            row["best_ref_idx"] = None

        row["R_T_per_ref"] = R_T_can
        if 0 <= own_idx < n_can:
            # own expression matched a canonical solution
            row["R_T_own"] = R_T_can[own_idx]
        elif own_idx >= n_can:
            own_sc = chunk[own_idx]
            row["R_T_own"] = (float(own_sc["R_T"])
                              if not np.isnan(own_sc["R_T"]) else None)
        else:
            row["R_T_own"] = None

    # --- Write -------------------------------------------------------------
    R_Ts = np.array([r["R_T"] for r in rows if r["R_T"] is not None])
    if R_Ts.size:
        print(f"[rescore] R_T  mean={R_Ts.mean():+.4f}  "
              f"median={np.median(R_Ts):+.4f}  "
              f"std={R_Ts.std():.4f}  n={R_Ts.size}")

    if args.dry_run:
        print("[rescore] --dry-run set, not writing output")
        return

    out_path = args.output or rollouts_path
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(out_path)
    print(f"[rescore] wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
