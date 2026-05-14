#!/usr/bin/env python3
"""
Run the Phase-I GRPO + diagnostics pipeline on a SINGLE base model.

Pipeline
--------
1. Train vanilla GRPO on Game-of-24 with rollout logging (train + eval JSONL).
2. Compute D1 / coverage / D3 from the eval rollout log (cheap, no GPU).
3. Score every eval rollout's decoding-velocity reward via
   ``src.velocity.compute_vt_batched`` against the *base* model and augment
   ``eval_rollout.jsonl`` with R_T, R_per_token, cumR_resampled.
4. Render R_T figures: ``rt_progress.png`` (mean R_T per eval cycle, split
   by correctness) and one ``rt_steps/rt_step{N}.png`` per eval cycle
   (2-panel: one (correct, incorrect) pair + global mean ±1σ).
5. Save figures + metrics.json under ``--output-root/<model_slug>/``.

This script runs ONE model end-to-end, then exits — so the OS reclaims all
GPU memory (vLLM + training graph). Use ``script/run_game24_sweep.sh`` to
loop over a list of models cleanly.

Example
-------
::

    python script/run_game24_one.py --model Qwen/Qwen3-0.6B
    python script/run_game24_one.py --model Qwen/Qwen3-0.6B --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from src.game24utils import (
        build_puzzle_pool, bucket_by_difficulty, make_splits,
        build_datasets, to_chat,
        correctness_reward, format_reward,
        RolloutLogger,
    )
    from src.game24diagnostics import (
        load_rollouts,
        d1_length_diversity, coverage_probe, d3_pass_rate_by_bucket,
        rt_dynamics, rt_progress,
    )

    model_name = args.model
    out_dir = Path(args.output_root) / slug(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_log = out_dir / "rollouts.jsonl"
    eval_rollout_log = out_dir / "eval_rollout.jsonl"
    rollout_log.write_text("")
    eval_rollout_log.write_text("")

    print(f"\n[run_one] model={model_name}  output={out_dir}", flush=True)

    # Reproducible split
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    puzzles = build_puzzle_pool(max_n=9)
    easy, medium, hard = bucket_by_difficulty(puzzles, easy_min=8, hard_max=2)
    train_puzzles, eval_puzzles, hard_probe = make_splits(
        easy, medium, hard, eval_frac=0.20, probe_frac=0.40,
    )
    train_ds, eval_ds, _probe_ds = build_datasets(train_puzzles, eval_puzzles, hard_probe)
    print(f"  train={len(train_puzzles)} eval={len(eval_puzzles)} probe={len(hard_probe)}",
          flush=True)

    # Tokenizer (Llama needs an explicit pad token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  set pad_token = eos_token ({tokenizer.eos_token!r})", flush=True)

    rollout_logger = RolloutLogger(rollout_log, eval_rollout_log, tokenizer)

    class EvalFlagCallback(TrainerCallback):
        """Toggles `rollout_logger.in_eval` around HF Trainer's eval loop.

        `on_prediction_step` fires per eval batch → sets the flag True.
        `on_step_begin` fires at the start of each training step → clears it.
        That ordering is enough to route every reward-fn call to the right log.
        """
        def __init__(self, logger): self.logger = logger
        def on_prediction_step(self, args, state, control, **kw):
            self.logger.in_eval = True
            # Stamp the training step that triggered this eval. Multiple
            # prediction batches happen per eval; they all share global_step.
            self.logger.global_step = state.global_step
        def on_step_begin(self, args, state, control, **kw):
            self.logger.in_eval = False
            self.logger.global_step = state.global_step
        def on_evaluate(self, args, state, control, **kw):
            self.logger.in_eval = False

    # ------- Train --------------------------------------------------------
    config_kwargs = dict(
        output_dir=str(out_dir / "grpo"),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        save_strategy="no",
        report_to="none",
        use_vllm=True,
        vllm_mode=args.vllm_mode,
    )
    if args.vllm_mode == "colocate":
        config_kwargs["vllm_gpu_memory_utilization"] = vllm_mem_for(model_name)
    else:  # server
        config_kwargs["vllm_server_host"] = args.vllm_server_host
        config_kwargs["vllm_server_port"] = args.vllm_server_port

    if args.eval_steps > 0:
        config_kwargs["eval_strategy"] = "steps"
        config_kwargs["eval_steps"] = args.eval_steps
        # Pre-training baseline eval at global_step=0.
        config_kwargs["eval_on_start"] = True

    config = GRPOConfig(**config_kwargs)

    t0 = time.time()
    # In server mode, pin training to --train-device so it never lands on the
    # GPU(s) running the vLLM server.
    model_load_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if args.vllm_mode == "server" and torch.cuda.is_available():
        model_load_kwargs["device_map"] = {"": f"cuda:{args.train_device}"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward, rollout_logger],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if args.eval_steps > 0 else None,
        processing_class=tokenizer,
        callbacks=[EvalFlagCallback(rollout_logger)],
    )
    trainer.train()
    train_time = time.time() - t0
    print(f"  training done in {train_time:.0f}s", flush=True)

    # ------- Cheap diagnostics — preferentially from EVAL rollouts --------
    # Eval re-runs the same prompts every --eval-steps cycle, so length /
    # pass-rate trajectories over `global_step` are not confounded by changes
    # in the train sampler. Each eval cycle shares one global_step value.
    # We rebind that to `step` so the existing d1/coverage/d3 functions
    # (which groupby("step")) plot one point per cycle.
    diag_path = eval_rollout_log if args.eval_steps > 0 else rollout_log
    diag_puzzles = eval_puzzles if args.eval_steps > 0 else train_puzzles
    df = load_rollouts(diag_path)
    if len(df) and "global_step" in df.columns:
        df["step"] = df["global_step"].astype(int)
    print(f"  diagnostics on {len(df)} rollouts from "
          f"{'eval' if args.eval_steps > 0 else 'train'} log "
          f"(path={diag_path.name})", flush=True)

    metrics: Dict[str, Any] = {
        "model": model_name,
        "n_rollouts": int(len(df)),
        "train_seconds": train_time,
        "diag_source": "eval" if args.eval_steps > 0 else "train",
    }

    for name, fn in [
        ("d1",        lambda: d1_length_diversity(df)),
        ("coverage",  lambda: coverage_probe(df, diag_puzzles)),
        ("d3",        lambda: d3_pass_rate_by_bucket(df, diag_puzzles)),
    ]:
        m, fig = fn()
        metrics.update(m)
        if fig is not None:
            fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")
            fig.clear()

    # ------- Decoding-velocity scoring → augment eval_rollout.jsonl ------
    # Every eval rollout gets:
    #   - R_T              total decoding reward log p(a* | q, o) - log p(a* | q)
    #   - R_per_token      R_T / T
    #   - cumR_resampled   cumulative R_t resampled to a fixed grid so the
    #                      notebook can plot mean-trajectory bands without
    #                      handling variable-length CoTs.
    # Reference answer for both correct and incorrect rollouts is the
    # puzzle's canonical solutions[0] — consistent across classes so the
    # left-vs-right contrast in the dynamics plot is meaningful.
    if args.eval_steps > 0 and args.score_vt and eval_rollout_log.exists():
        from src.velocity import compute_vt_batched

        print(f"\n[vt-score] augmenting {eval_rollout_log.name} ...", flush=True)
        # Free training graph + (colocated) vLLM before loading the scorer.
        del trainer
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scorer_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        scorer_device = (f"cuda:{args.train_device}"
                         if torch.cuda.is_available() and args.vllm_mode == "server"
                         else ("cuda" if torch.cuda.is_available() else "cpu"))
        scorer_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=scorer_dtype,
        ).to(scorer_device).eval()

        puzzle_index = {tuple(sorted(p["numbers"])): p
                        for p in (train_puzzles + eval_puzzles + hard_probe)}

        eval_rows = [json.loads(l) for l in eval_rollout_log.read_text().splitlines()
                     if l.strip()]
        prompts, completions, refs, valid = [], [], [], []
        for row in eval_rows:
            key = tuple(sorted(row["numbers"]))
            puzzle = puzzle_index.get(key)
            if puzzle is None or not row.get("completion"):
                valid.append(False); continue
            prompts.append(tokenizer.apply_chat_template(
                to_chat(puzzle)["prompt"], tokenize=False, add_generation_prompt=True))
            completions.append(row["completion"])
            refs.append(puzzle["solutions"][0])
            valid.append(True)

        print(f"[vt-score] scoring {sum(valid)}/{len(eval_rows)} rollouts "
              f"(micro_batch={args.vt_micro_batch}) ...", flush=True)
        t_vt = time.time()
        scored = compute_vt_batched(
            prompts, completions, refs, scorer_model, tokenizer,
            micro_batch_size=args.vt_micro_batch,
        )

        import numpy as np
        N_PTS = args.vt_resample_pts
        grid = np.linspace(0.0, 1.0, N_PTS)
        scored_iter = iter(scored)
        for row, ok in zip(eval_rows, valid):
            if not ok:
                row["R_T"] = None
                row["R_per_token"] = None
                row["cumR_resampled"] = None
                continue
            sc = next(scored_iter)
            vt = np.asarray(sc["vt"], dtype=float)
            R_T = float(sc["R_T"]) if not np.isnan(sc["R_T"]) else None
            row["R_T"] = R_T
            row["R_per_token"] = (float(sc["R_per_token"])
                                  if not np.isnan(sc["R_per_token"]) else None)
            if len(vt):
                R = np.cumsum(vt)
                x = np.linspace(0.0, 1.0, len(R))
                row["cumR_resampled"] = np.interp(grid, x, R).tolist()
            else:
                row["cumR_resampled"] = None

        tmp = eval_rollout_log.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for row in eval_rows:
                f.write(json.dumps(row) + "\n")
        tmp.replace(eval_rollout_log)
        print(f"[vt-score] done in {time.time()-t_vt:.0f}s "
              f"→ R_T, R_per_token, cumR_resampled added to {eval_rollout_log.name}",
              flush=True)

        # Aggregate scalar R_T into metrics for the sweep summary.
        R_Ts = np.array([r["R_T"] for r in eval_rows if r["R_T"] is not None])
        if R_Ts.size:
            metrics["vt_R_T_mean"]   = float(R_Ts.mean())
            metrics["vt_R_T_median"] = float(np.median(R_Ts))

        # ------- R_T figures (consume the augmented eval JSONL) ----------
        # 1. rt_progress.png  — one summary curve across all eval cycles.
        # 2. rt_step{N}.png   — one 2-panel figure per eval cycle.
        df_rt = load_rollouts(eval_rollout_log)
        if len(df_rt):
            df_rt["step"] = df_rt["global_step"].astype(int)

            m, fig = rt_progress(df_rt)
            metrics.update(m)
            if fig is not None:
                fig.savefig(out_dir / "rt_progress.png", dpi=120, bbox_inches="tight")
                fig.clear()

            rt_dir = out_dir / "rt_steps"; rt_dir.mkdir(exist_ok=True)
            n_written = 0
            for step in sorted(df_rt.global_step.dropna().unique()):
                m, fig = rt_dynamics(df_rt, int(step), pair_seed=args.rt_pair_seed)
                metrics.update(m)
                if fig is not None:
                    fig.savefig(rt_dir / f"rt_step{int(step)}.png",
                                dpi=120, bbox_inches="tight")
                    fig.clear()
                    n_written += 1
            print(f"[rt-figs] wrote rt_progress.png + {n_written} per-step "
                  f"figures → {rt_dir}", flush=True)

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
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    # Periodic evaluation (rollouts → eval_rollout.jsonl)
    p.add_argument("--eval-steps", type=int, default=200,
                   help="Run eval on the full eval_ds every N steps (0 to disable). "
                        "Eval rollouts land in eval_rollout.jsonl alongside rollouts.jsonl.")
    # vLLM mode
    p.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate",
                   help="'colocate' shares one GPU with training (single-GPU). "
                        "'server' connects to a separately launched vLLM server "
                        "(use a different GPU; see --train-device).")
    p.add_argument("--vllm-server-host", default="0.0.0.0",
                   help="(server only) vLLM server host.")
    p.add_argument("--vllm-server-port", type=int, default=8000,
                   help="(server only) vLLM server port.")
    p.add_argument("--train-device", type=int, default=0,
                   help="(server only) CUDA index for training; must not overlap "
                        "with the GPU(s) running the vLLM server.")
    # Decoding-velocity scoring (post-training pass; augments eval_rollout.jsonl)
    p.add_argument("--score-vt", action="store_true", default=True,
                   help="After training, score every eval rollout's R_T via "
                        "src.velocity.compute_vt_batched and append the fields "
                        "(R_T, R_per_token, cumR_resampled) to eval_rollout.jsonl.")
    p.add_argument("--no-score-vt", dest="score_vt", action="store_false",
                   help="Disable the post-training v_t scoring pass.")
    p.add_argument("--vt-micro-batch", type=int, default=8,
                   help="Forward-pass batch size for compute_vt_batched. "
                        "Lower if you hit OOM on long CoTs.")
    p.add_argument("--vt-resample-pts", type=int, default=100,
                   help="Grid size for the resampled cumulative R_t curve "
                        "stored per rollout.")
    p.add_argument("--rt-pair-seed", type=int, default=0,
                   help="random_state for picking the (correct, incorrect) "
                        "pair shown in each rt_step{N}.png figure.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    run_one(args)


if __name__ == "__main__":
    main()
