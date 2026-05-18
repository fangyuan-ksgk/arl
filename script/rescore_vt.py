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
import json
import sys
import time
from pathlib import Path

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
    # For each row we need:
    #   prompt      — chat-formatted question
    #   completion  — the rollout's CoT (already in the row)
    #   ref         — any canonical solution string (enumerate_solutions)
    # Rows without a usable completion (or with no enumerable solution) are
    # marked invalid and assigned None R_T fields, matching run_game24_one.py.
    prompts, completions, refs, valid = [], [], [], []
    skipped_no_completion = 0
    skipped_no_solution = 0
    for row in rows:
        if not row.get("completion"):
            valid.append(False)
            skipped_no_completion += 1
            continue
        numbers = tuple(row["numbers"])
        sols = enumerate_solutions(numbers)
        if not sols:
            valid.append(False)
            skipped_no_solution += 1
            continue
        puzzle = {"numbers": list(numbers), "solutions": sols}
        prompts.append(tokenizer.apply_chat_template(
            to_chat(puzzle)["prompt"], tokenize=False, add_generation_prompt=True))
        completions.append(row["completion"])
        refs.append(sols[0]) # -> Issue 1. Only the first correct answer is used for velocity reward computation (wrong)
                             #    A dumb fix : take the correct answer with max decoding reward
                             #    A request: when mode is not producing the correct answer, record p(current answer | cot + query) for reference
                             #    I suspect the delta between p(correct answer | cot + query) - p(current answer | cot + query) is meaningful
                             #    
        valid.append(True)

    n_score = sum(valid)
    print(f"[rescore] scoring {n_score}/{len(rows)} rollouts "
          f"(skipped: {skipped_no_completion} empty completion, "
          f"{skipped_no_solution} no enumerable solution)")
    if n_score == 0:
        print("[rescore] nothing to score; exiting"); return

    # --- Score -------------------------------------------------------------
    t0 = time.time()
    scored = compute_vt_batched(
        prompts, completions, refs, scorer, tokenizer,
        micro_batch_size=args.micro_batch,
    )
    print(f"[rescore] forward pass done in {time.time()-t0:.0f}s")

    # --- Augment rows ------------------------------------------------------
    grid = np.linspace(0.0, 1.0, args.resample_pts)
    scored_iter = iter(scored)
    for row, ok in zip(rows, valid):
        if not ok:
            row["R_T"] = None
            row["R_per_token"] = None
            row["cumR_resampled"] = None
            continue
        sc = next(scored_iter)
        vt = np.asarray(sc["vt"], dtype=float)
        row["R_T"] = (float(sc["R_T"])
                     if not np.isnan(sc["R_T"]) else None)
        row["R_per_token"] = (float(sc["R_per_token"])
                              if not np.isnan(sc["R_per_token"]) else None)
        if len(vt):
            R = np.cumsum(vt)
            x = np.linspace(0.0, 1.0, len(R))
            row["cumR_resampled"] = np.interp(grid, x, R).tolist()
        else:
            row["cumR_resampled"] = None

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
