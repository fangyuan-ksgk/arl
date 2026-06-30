"""Mass-produce per-correct-answer velocity-reward analysis for a run dir.

Given a run directory that contains an ``eval_rollout.jsonl`` (e.g.
``logs/game24_sweep_exp4/len512/Qwen__Qwen3-0.6B``), this script:

1. For every eval rollout at the requested global step(s), enumerates all
   canonical 24-solutions for the puzzle and scores the rollout's CoT
   against each of them with :func:`src.velocity.compute_vt_batched`. The
   per-reference ``R_T`` / ``R_per_token`` / ``cumR_resampled`` arrays are
   written to a sidecar ``eval_rollout_vt.jsonl`` (one row per input
   rollout, with new ``refs``, ``R_T_per_ref``, ``R_per_token_per_ref``,
   ``cumR_resampled_per_ref``, ``own_ref_idx`` fields). The original
   ``eval_rollout.jsonl`` is **not** modified.
2. For a subset of those rollouts (default: only ``correct`` ones, capped
   at ``--max-gifs``), renders the streaming animation from
   :func:`src.velo_viz.make_animation` and saves it as
   ``<run_dir>/vt_gifs/step{step}_idx{idx}.gif``.

Usage::

    python script/visualize_vt.py \
        --run-dir logs/game24_sweep_exp4/len512/Qwen__Qwen3-0.6B \
        --scorer-model Qwen/Qwen3-0.6B \
        --step 1200 \
        --max-gifs 16
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game24utils import enumerate_solutions, to_chat  # noqa: E402
from src.velocity import compute_vt_batched  # noqa: E402
from src.velo_viz import prepare, make_animation  # noqa: E402


# ───────────────────────────── helpers ──────────────────────────────
def _canon(expr: str) -> str:
    """AST-canonicalize an expression so '(5*5- (2 - 1))' == '(5*5)-(2-1)'."""
    try:
        return ast.dump(ast.parse(expr, mode="eval").body)
    except Exception:
        return expr


def _build_refs(row: Dict[str, Any]) -> Tuple[List[str], int]:
    """Return (refs, own_idx). own_idx is -1 if the row has no `expr`.

    `refs` is the canonical solutions, with the row's own expression
    appended iff it is AST-distinct from all of them (mirrors the
    notebook's logic).
    """
    sols = enumerate_solutions(tuple(row["numbers"]))
    refs = list(sols)
    own_expr = row.get("expr") or ""
    if not own_expr:
        return refs, -1
    own_canon = _canon(own_expr)
    own_idx = next((i for i, s in enumerate(sols) if _canon(s) == own_canon), None)
    if own_idx is None:
        refs.append(own_expr)
        own_idx = len(refs) - 1
    return refs, own_idx


def _resample(vt: np.ndarray, n: int) -> List[float]:
    if len(vt) == 0:
        return []
    R = np.cumsum(vt)
    x = np.linspace(0.0, 1.0, len(R))
    return np.interp(np.linspace(0.0, 1.0, n), x, R).tolist()


# ───────────────────────────── main ─────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Directory containing eval_rollout.jsonl.")
    p.add_argument("--scorer-model", required=True,
                   help="HF hub id or local path of scorer model.")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer to use (defaults to --scorer-model).")
    p.add_argument("--step", type=int, nargs="+", default=[1200],
                   help="global_step(s) to score (default: 1200).")
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--resample-pts", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--output-jsonl", type=Path, default=None,
                   help="Where to write per-ref R_T sidecar. "
                        "Default: <run_dir>/eval_rollout_vt.jsonl.")
    p.add_argument("--gif-dir", type=Path, default=None,
                   help="Where to save GIFs. Default: <run_dir>/vt_gifs/.")
    p.add_argument("--gif-rows", choices=["correct", "all", "none"],
                   default="correct",
                   help="Which rows to render GIFs for (default: correct).")
    p.add_argument("--max-gifs", type=int, default=16,
                   help="Cap on the number of GIFs produced (default: 16).")
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--skip-vt", action="store_true",
                   help="Don't write the per-ref jsonl; only render GIFs "
                        "(still requires the forward passes).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_path = args.run_dir / "eval_rollout.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"missing {eval_path}")

    rows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    sel_rows = [r for r in rows if r.get("global_step") in set(args.step)]
    print(f"[viz_vt] {len(rows)} total rows · {len(sel_rows)} matching "
          f"global_step ∈ {args.step}")
    if not sel_rows:
        return

    # --- load scorer ---------------------------------------------------------
    tok_src = args.tokenizer or args.scorer_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    device = (args.device or
              (f"cuda:{torch.cuda.current_device()}"
               if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[viz_vt] loading scorer {args.scorer_model} → {device} ({dtype})")
    try:
        scorer = AutoModelForCausalLM.from_pretrained(
            args.scorer_model, attn_implementation="flash_attention_2",
            dtype=dtype,
        ).to(device).eval()
    except (ImportError, ValueError):
        scorer = AutoModelForCausalLM.from_pretrained(
            args.scorer_model, dtype=dtype,
        ).to(device).eval()

    # --- per-row scoring -----------------------------------------------------
    out_path = args.output_jsonl or (args.run_dir / "eval_rollout_vt.jsonl")
    gif_dir = args.gif_dir or (args.run_dir / "vt_gifs")
    gif_dir.mkdir(parents=True, exist_ok=True)

    n_gif = 0
    t_start = time.time()
    out_records: List[Dict[str, Any]] = []
    for r_i, row in enumerate(sel_rows):
        if not row.get("completion"):
            continue
        refs, own_idx = _build_refs(row)
        if not refs:
            continue

        puzzle = {"numbers": list(row["numbers"]), "solutions": refs}
        prompt = tokenizer.apply_chat_template(
            to_chat(puzzle)["prompt"], tokenize=False, add_generation_prompt=True)
        prompts     = [prompt] * len(refs)
        completions = [row["completion"]] * len(refs)

        t0 = time.time()
        scored = compute_vt_batched(
            prompts, completions, refs, scorer, tokenizer,
            micro_batch_size=args.micro_batch,
        )
        dt = time.time() - t0

        R_T = [float(s["R_T"]) if not np.isnan(s["R_T"]) else None for s in scored]
        R_pt = [float(s["R_per_token"]) if not np.isnan(s["R_per_token"]) else None
                for s in scored]
        cumR = [_resample(np.asarray(s["vt"], dtype=float), args.resample_pts)
                for s in scored]
        valid_R = [v for v in R_T if v is not None]
        best_idx = int(np.argmax([v if v is not None else -1e18 for v in R_T])) \
            if valid_R else -1

        out_records.append({
            "step": row.get("step"),
            "global_step": row.get("global_step"),
            "idx": row.get("idx"),
            "numbers": row.get("numbers"),
            "expr": row.get("expr"),
            "correct": row.get("correct"),
            "refs": refs,
            "own_ref_idx": own_idx,
            "best_ref_idx": best_idx,
            "R_T_per_ref": R_T,
            "R_per_token_per_ref": R_pt,
            "cumR_resampled_per_ref": cumR,
        })

        gif_ok = (
            (args.gif_rows == "all") or
            (args.gif_rows == "correct" and bool(row.get("correct")))
        )
        if gif_ok and n_gif < args.max_gifs and own_idx >= 0:
            try:
                ctx = prepare(row, scored, refs, own_idx, tokenizer)
                gif_path = (gif_dir /
                            f"step{row['global_step']}_idx{row['idx']}.gif")
                make_animation(ctx, save_path=str(gif_path), fps=args.gif_fps)
                n_gif += 1
                print(f"[viz_vt] {r_i+1}/{len(sel_rows)} step={row['global_step']} "
                      f"idx={row['idx']} refs={len(refs)} "
                      f"score={dt:.1f}s gif→{gif_path.name}")
            except Exception as e:
                print(f"[viz_vt] gif failed for idx={row['idx']}: "
                      f"{e.__class__.__name__}: {e}")
        else:
            print(f"[viz_vt] {r_i+1}/{len(sel_rows)} step={row['global_step']} "
                  f"idx={row['idx']} refs={len(refs)} score={dt:.1f}s")

    print(f"[viz_vt] total {time.time()-t_start:.0f}s · "
          f"scored {len(out_records)} rows · {n_gif} gifs")

    if not args.skip_vt:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w") as f:
            for rec in out_records:
                f.write(json.dumps(rec) + "\n")
        tmp.replace(out_path)
        print(f"[viz_vt] wrote {out_path} ({len(out_records)} rows)")


if __name__ == "__main__":
    main()
