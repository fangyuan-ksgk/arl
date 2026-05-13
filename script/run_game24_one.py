#!/usr/bin/env python3
"""
Run the Phase-I GRPO + diagnostics pipeline on a SINGLE base model.

Pipeline
--------
1. Train vanilla GRPO on Game-of-24 with rollout logging.
2. Compute D1/diversity/D3 from the rollout log (cheap).
3. Compute D2/D4 v_t-based probes against the *base* model (slow).
4. Save figures + metrics.json under ``--output-root/<model_slug>/``.

This script runs ONE model end-to-end, then exits — so the OS reclaims all
GPU memory (vLLM + training graph). Use ``script/run_game24_sweep.sh`` to
loop over a list of models cleanly.

Example
-------
::

    python script/run_game24_one.py --model Qwen/Qwen3-0.6B
    python script/run_game24_one.py --model Qwen/Qwen3-0.6B --max-steps 50 --skip-vt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def vllm_mem_for(model_name: str) -> float:
    """Heuristic vLLM colocate memory share. Tune for your GPU."""
    name = model_name.lower()
    if "4b" in name or "3b" in name:
        return 0.55
    if "1.7b" in name or "1b" in name:
        return 0.45
    return 0.35  # 0.6B and smaller


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_one(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.game24utils import (
        build_puzzle_pool, bucket_by_difficulty, make_splits,
        build_datasets, to_chat,
        correctness_reward, format_reward, _text, extract_expr,
        verify_24,
    )
    from src.game24diagnostics import (
        load_rollouts,
        d1_length_diversity, coverage_probe, d3_pass_rate_by_bucket,
        make_vt_scorer, d4_vt_on_failed,
        score_rollout_sample, d2_pair_figures, decoding_reward_stats,
    )

    model_name = args.model
    out_dir = Path(args.output_root) / slug(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_log = out_dir / "rollouts.jsonl"
    rollout_log.write_text("")

    print(f"\n[run_one] model={model_name}  output={out_dir}", flush=True)

    # Reproducible split
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    puzzles = build_puzzle_pool(max_n=9)
    easy, medium, hard = bucket_by_difficulty(puzzles, easy_min=8, hard_max=2)
    train_puzzles, eval_puzzles, hard_probe = make_splits(
        easy, medium, hard, eval_frac=0.10, probe_frac=0.40,
    )
    train_ds, _, _ = build_datasets(train_puzzles, eval_puzzles, hard_probe)
    print(f"  train={len(train_puzzles)} eval={len(eval_puzzles)} probe={len(hard_probe)}",
          flush=True)

    # Tokenizer (Llama needs an explicit pad token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  set pad_token = eos_token ({tokenizer.eos_token!r})", flush=True)

    # Rollout logger — piggybacks on a reward callback to capture every rollout
    class RolloutLogger:
        __name__ = "rollout_logger"

        def __init__(self, path: Path):
            self.path = path
            self.step = 0

        def __call__(self, completions, numbers, **_):
            with self.path.open("a") as f:
                for i, (c, nums) in enumerate(zip(completions, numbers)):
                    text = _text(c)
                    expr = extract_expr(text)
                    correct = verify_24(list(nums), expr)
                    n_tok = len(tokenizer.encode(text, add_special_tokens=False))
                    f.write(json.dumps({
                        "step": self.step, "idx": i, "numbers": list(nums),
                        "completion": text, "expr": expr,
                        "correct": bool(correct), "n_tokens": int(n_tok),
                    }) + "\n")
            self.step += 1
            return [0.0] * len(completions)

    rollout_logger = RolloutLogger(rollout_log)

    # ------- Train --------------------------------------------------------
    config = GRPOConfig(
        output_dir=str(out_dir / "grpo"),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        save_strategy="no",
        report_to="none",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=vllm_mem_for(model_name),
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward, rollout_logger],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    train_time = time.time() - t0
    print(f"  training done in {train_time:.0f}s", flush=True)

    # ------- Cheap diagnostics from rollout log ---------------------------
    df = load_rollouts(rollout_log)
    metrics: Dict[str, Any] = {
        "model": model_name,
        "n_rollouts": int(len(df)),
        "train_seconds": train_time,
    }

    for name, fn in [
        ("d1",        lambda: d1_length_diversity(df)),
        ("coverage",  lambda: coverage_probe(df, train_puzzles)),
        ("d3",        lambda: d3_pass_rate_by_bucket(df, train_puzzles)),
    ]:
        m, fig = fn()
        metrics.update(m)
        if fig is not None:
            fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")
            fig.clear()

    # ------- v_t probes against the BASE model ----------------------------
    # We score a balanced sample of correct + incorrect rollouts ONCE (slow:
    # T+1 forward passes per rollout), then derive both the paired figures
    # and the global decoding-reward statistics from the cached results.
    if not args.skip_vt:
        # Free vLLM/trainer memory before loading the v_t scorer.
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_puzzles = train_puzzles + eval_puzzles + hard_probe
        vt_scorer = make_vt_scorer(model_name, tokenizer)

        print(f"\n[scoring] computing v_t for up to "
              f"{args.n_per_class} correct + {args.n_per_class} incorrect rollouts",
              flush=True)
        scored = score_rollout_sample(
            df, all_puzzles, vt_scorer, to_chat,
            n_per_class=args.n_per_class, seed=args.seed,
        )
        # Persist the scored sample (drop the heavy per-token arrays for the CSV).
        scored.drop(columns=["toks", "vt"]).to_csv(out_dir / "scored_rollouts.csv", index=False)
        print(f"[scoring] scored {len(scored)} rollouts → scored_rollouts.csv", flush=True)

        # 50 paired (correct, incorrect) figures into <out_dir>/d2_pairs/
        n_written = d2_pair_figures(
            scored, out_dir / "d2_pairs",
            n_pairs=args.n_pairs, seed=args.seed,
        )
        metrics["d2_n_pair_figures"] = int(n_written)
        print(f"[d2] wrote {n_written} paired figures → {out_dir / 'd2_pairs'}", flush=True)

        # Global decoding-reward statistics
        dr_metrics, dr_fig = decoding_reward_stats(scored)
        metrics.update(dr_metrics)
        if dr_fig is not None:
            dr_fig.savefig(out_dir / "decoding_reward_stats.png", dpi=120, bbox_inches="tight")
            dr_fig.clear()
        print(f"[decoding-reward] AUC={dr_metrics.get('R_T_auc', float('nan')):.3f}  "
              f"gap={dr_metrics.get('R_T_gap', float('nan')):+.2f}  "
              f"d={dr_metrics.get('R_T_cohens_d', float('nan')):+.2f}",
              flush=True)

        # D4 (productive tokens in failed rollouts) — reuse cached incorrect rollouts.
        d4_metrics, d4_fig = d4_vt_on_failed(
            df, all_puzzles, vt_scorer, to_chat, n_sample=args.d4_sample,
        )
        metrics.update(d4_metrics)
        if d4_fig is not None:
            d4_fig.savefig(out_dir / "d4.png", dpi=120, bbox_inches="tight")
            d4_fig.clear()

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  metrics → {out_dir / 'metrics.json'}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HuggingFace model id, e.g. Qwen/Qwen3-0.6B")
    p.add_argument("--output-root", default="output/game24_sweep",
                   help="Per-model artefacts go to <output-root>/<model-slug>/")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--n-per-class", type=int, default=50,
                   help="Number of correct + incorrect rollouts to v_t-score (each).")
    p.add_argument("--n-pairs", type=int, default=50,
                   help="Number of (correct, incorrect) pair figures to write.")
    p.add_argument("--d4-sample", type=int, default=16,
                   help="Number of failed rollouts to score for D4 histogram.")
    p.add_argument("--skip-vt", action="store_true",
                   help="Skip D2/D4 v_t probes (faster; only D1/diversity/D3).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    run_one(args)


if __name__ == "__main__":
    main()
