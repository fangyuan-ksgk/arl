#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 variant of ``run_game24_one.py``.

Differences from the single-GPU script
--------------------------------------
1. Must be launched under ``accelerate launch --config_file configs/zero3.yaml``.
   We refuse to run if no distributed launcher env vars are present.
2. ``RolloutLogger`` writes **rank-suffixed** JSONL files
   (``rollouts.jsonl.r{R}`` / ``eval_rollout.jsonl.r{R}``) to avoid the
   concurrent-append race that a single shared path would cause under DDP.
   After training, rank 0 concatenates them into canonical
   ``rollouts.jsonl`` / ``eval_rollout.jsonl``.
3. R_T scoring is **sharded**: each rank scores ``eval_rows[rank::world]``
   independently using its own scorer-model copy, writes a rank-local
   partial JSONL, then rank 0 merges into the final augmented file.
4. All non-scoring post-train work (diagnostics, figures, metrics.json) runs
   on rank 0 only; other ranks block on ``dist.barrier()`` until rank 0
   finishes so the launcher exits cleanly.

Launch
------
::

    # GPU 0 → vLLM server
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve \\
      --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --enforce-eager &

    # GPUs 1,2 → ZeRO-3 training
    CUDA_VISIBLE_DEVICES=1,2 accelerate launch \\
      --config_file configs/zero3.yaml \\
      script/run_game24_deepspeed.py \\
      --model Qwen/Qwen3-0.6B \\
      --output-root output/zero3_smoketest \\
      --max-steps 30 --eval-steps 30 \\
      --num-generations 8 --max-completion-length 1024 \\
      --per-device-batch-size 2 --grad-accum 4 \\
      --learning-rate 5e-6 \\
      --vllm-mode server --vllm-server-port 8000
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def _canon_expr(expr: str) -> str:
    """AST-canonicalize so '(5*5- (2 - 1))' == '(5*5)-(2-1)'."""
    try:
        return ast.dump(ast.parse(expr, mode="eval").body)
    except Exception:
        return expr


def _build_refs(puzzle: Dict[str, Any], own_expr: str) -> Tuple[List[str], int, int]:
    """Return (refs, n_canonical, own_idx).

    ``refs[:n_canonical]`` are the puzzle's enumerated 24-solutions. If
    ``own_expr`` is non-empty and AST-distinct from all of them it is
    appended as ``refs[-1]`` and ``own_idx`` points at it; otherwise
    ``own_idx`` is the matching canonical index (or -1 if no own_expr).
    """
    sols = list(puzzle["solutions"])
    n_can = len(sols)
    refs = list(sols)
    if not own_expr:
        return refs, n_can, -1
    own_canon = _canon_expr(own_expr)
    own_idx = next((i for i, s in enumerate(sols) if _canon_expr(s) == own_canon), None)
    if own_idx is None:
        refs.append(own_expr)
        own_idx = len(refs) - 1
    return refs, n_can, own_idx


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _rank_world() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


def _under_distributed_launcher() -> bool:
    return (
        "ACCELERATE_USE_DEEPSPEED" in os.environ
        or "ACCELERATE_USE_FSDP" in os.environ
        or "LOCAL_RANK" in os.environ
    )


def _barrier() -> None:
    """Best-effort barrier; works under torch.distributed and is a no-op otherwise."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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

    if not _under_distributed_launcher():
        raise RuntimeError(
            "run_game24_deepspeed.py must be launched via `accelerate launch "
            "--config_file configs/zero3.yaml` (or another DDP launcher). "
            "For single-GPU runs, use run_game24_one.py instead."
        )

    rank, world = _rank_world()
    is_main = rank == 0

    model_name = args.model
    out_dir = Path(args.output_root) / slug(model_name)
    # All ranks mkdir; exist_ok=True makes this race-safe. We can't rely on
    # `_barrier()` here because torch.distributed isn't initialized yet —
    # accelerate spins up the process group lazily, around model/trainer
    # construction, not at script entry.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical (merged) paths — used by all post-train diagnostics.
    rollout_log = out_dir / "rollouts.jsonl"
    eval_rollout_log = out_dir / "eval_rollout.jsonl"
    # Per-rank shards written live during training. Each rank owns its own
    # path so the appends never collide.
    rollout_shard = out_dir / f"rollouts.jsonl.r{rank}"
    eval_shard    = out_dir / f"eval_rollout.jsonl.r{rank}"
    rollout_shard.write_text("")
    eval_shard.write_text("")
    if is_main:
        rollout_log.write_text("")
        eval_rollout_log.write_text("")

    if is_main:
        print(f"\n[run_one][rank{rank}/{world}] model={model_name}  "
              f"output={out_dir}", flush=True)

    # Reproducible split — same on every rank.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    puzzles = build_puzzle_pool(max_n=9)
    easy, medium, hard = bucket_by_difficulty(puzzles, easy_min=8, hard_max=2)
    train_puzzles, eval_puzzles, hard_probe = make_splits(
        easy, medium, hard, eval_frac=0.20, probe_frac=0.40,
    )
    train_ds, eval_ds, _probe_ds = build_datasets(train_puzzles, eval_puzzles, hard_probe)
    if is_main:
        print(f"  train={len(train_puzzles)} eval={len(eval_puzzles)} "
              f"probe={len(hard_probe)}", flush=True)

    # Train global batch divisibility check (must include num_processes here).
    train_global = args.per_device_batch_size * args.grad_accum * world
    if train_global % args.num_generations != 0:
        raise ValueError(
            f"train global batch (pdbs*grad_accum*world = "
            f"{args.per_device_batch_size}*{args.grad_accum}*{world} = "
            f"{train_global}) must be a multiple of "
            f"--num-generations ({args.num_generations})."
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if is_main:
            print(f"  set pad_token = eos_token ({tokenizer.eos_token!r})", flush=True)

    # Rank-scoped logger so the appends never collide across ranks.
    rollout_logger = RolloutLogger(rollout_shard, eval_shard, tokenizer)

    class EvalFlagCallback(TrainerCallback):
        def __init__(self, logger): self.logger = logger
        def on_prediction_step(self, args, state, control, **kw):
            self.logger.in_eval = True
            self.logger.global_step = state.global_step
        def on_step_begin(self, args, state, control, **kw):
            self.logger.in_eval = False
            self.logger.global_step = state.global_step
        def on_evaluate(self, args, state, control, **kw):
            self.logger.in_eval = False

    class StepTimerCallback(TrainerCallback):
        """Append one JSON line per training step to out_dir/step_times.jsonl.
        Rank-0 only. Useful for cross-config wall-time comparison without
        having to grep tqdm output."""
        def __init__(self, path: Path, rank: int):
            self.path = path
            self.rank = rank
            self.t0 = None
            self.fh = None
            if rank == 0:
                self.fh = path.open("w", buffering=1)  # line-buffered
        def on_step_begin(self, args, state, control, **kw):
            if self.fh is None: return
            self.t0 = time.time()
        def on_step_end(self, args, state, control, **kw):
            if self.fh is None or self.t0 is None: return
            dt = time.time() - self.t0
            self.fh.write(json.dumps({
                "step": int(state.global_step),
                "step_time_s": round(dt, 4),
                "epoch": float(state.epoch) if state.epoch is not None else None,
            }) + "\n")
        def on_train_end(self, args, state, control, **kw):
            if self.fh is not None:
                self.fh.close()

    # ------- Train --------------------------------------------------------
    config_kwargs = dict(
        output_dir=str(out_dir / "grpo"),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.num_generations,
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
    if args.vllm_mode == "server":
        config_kwargs["vllm_server_host"] = args.vllm_server_host
        config_kwargs["vllm_server_port"] = args.vllm_server_port
    else:
        raise ValueError(
            "deepspeed variant requires --vllm-mode server (colocate puts "
            "vLLM on the training GPUs, which conflicts with ZeRO-3 sharding)."
        )

    if args.eval_steps > 0:
        config_kwargs["eval_strategy"] = "steps"
        config_kwargs["eval_steps"] = args.eval_steps
        config_kwargs["eval_on_start"] = True

    config = GRPOConfig(**config_kwargs)

    t0 = time.time()
    # Under DeepSpeed, do NOT set device_map — let DS init the model on `meta`
    # and shard params via zero3_init.
    model_load_kwargs = dict(
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    print(f"[rank{rank}] loading base model {model_name} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
    print(f"[rank{rank}] base model loaded in {time.time()-t0:.1f}s; "
          f"building GRPOTrainer (will NCCL-handshake vLLM server) ...", flush=True)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward, rollout_logger],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if args.eval_steps > 0 else None,
        processing_class=tokenizer,
        callbacks=[
            EvalFlagCallback(rollout_logger),
            StepTimerCallback(out_dir / "step_times.jsonl", rank),
        ],
    )
    print(f"[rank{rank}] GRPOTrainer ready; starting trainer.train() ...", flush=True)
    trainer.train()
    train_time = time.time() - t0
    if is_main:
        print(f"  training done in {train_time:.0f}s", flush=True)

    # ------- Save trained policy for R_T scoring -------------------------
    # R_T here is computed against the *trained* policy (information-gain
    # interpretation), not the base model. The policy is sharded under
    # ZeRO-3, so we can't use it directly for raw forward passes.
    #
    # Note: we deliberately avoid `trainer.save_model()` here because the
    # accelerate yaml flag `zero3_save_16bit_model: true` does not reliably
    # propagate into the DeepSpeed engine's runtime config across versions
    # — when it doesn't, save_model() leaves a pile of zero_pp_rank_*.pt
    # files instead of a model.safetensors, and reloading fails. Instead
    # we use the documented HF pattern: `accelerator.get_state_dict()`
    # explicitly gathers the full state dict across ranks, and we save it
    # via `unwrap_model().save_pretrained(state_dict=...)`.
    trained_ckpt_dir = out_dir / "trained_for_vt"
    if args.eval_steps > 0 and args.score_vt:
        if is_main:
            print(f"  saving trained policy → {trained_ckpt_dir} "
                  "(gathering ZeRO-3 shards, used as R_T scorer)", flush=True)
        # Collective op — must be called on every rank.
        full_sd = trainer.accelerator.get_state_dict(trainer.model_wrapped)
        if is_main:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model_wrapped)
            unwrapped.save_pretrained(
                trained_ckpt_dir,
                state_dict=full_sd,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(trained_ckpt_dir)
            print(f"  saved {sum(t.numel() for t in full_sd.values())/1e6:.1f}M "
                  f"parameters as safetensors", flush=True)
        del full_sd

    # ------- Merge per-rank shards into canonical files (rank 0) ---------
    _barrier()
    if is_main:
        def _merge_shards(canonical: Path, pattern: str) -> int:
            n = 0
            with canonical.open("w") as out:
                for r in range(world):
                    shard = out_dir / pattern.format(r=r)
                    if not shard.exists():
                        continue
                    with shard.open("r") as f_in:
                        for line in f_in:
                            if line.strip():
                                out.write(line.strip() + "\n")
                                n += 1
            return n

        n_train = _merge_shards(rollout_log,      "rollouts.jsonl.r{r}")
        n_eval  = _merge_shards(eval_rollout_log, "eval_rollout.jsonl.r{r}")
        print(f"  merged shards → rollouts.jsonl ({n_train} rows), "
              f"eval_rollout.jsonl ({n_eval} rows)", flush=True)
    _barrier()

    # ------- Cheap diagnostics (rank 0 only) ------------------------------
    metrics: Dict[str, Any] = {
        "model": model_name,
        "train_seconds": train_time,
        "diag_source": "eval" if args.eval_steps > 0 else "train",
        "world_size": world,
    }
    if is_main:
        diag_path = eval_rollout_log if args.eval_steps > 0 else rollout_log
        diag_puzzles = eval_puzzles if args.eval_steps > 0 else train_puzzles
        df = load_rollouts(diag_path)
        if len(df) and "global_step" in df.columns:
            df["step"] = df["global_step"].astype(int)
        print(f"  diagnostics on {len(df)} rollouts from {diag_path.name}", flush=True)
        metrics["n_rollouts"] = int(len(df))

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

    _barrier()

    # ------- Sharded R_T scoring -----------------------------------------
    # Every rank still has the policy model + DeepSpeed engine loaded. Free
    # them BEFORE building the scorer model on each rank, otherwise we
    # double the per-GPU footprint.
    if args.eval_steps > 0 and args.score_vt:
        from src.velocity import compute_vt_batched

        del trainer
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # All ranks read the merged eval log (rank 0 wrote it; barrier above
        # guarantees it's flushed).
        eval_rows = []
        with eval_rollout_log.open("r") as f_in:
            for l in f_in:
                if l.strip():
                    eval_rows.append(json.loads(l))
        if is_main:
            print(f"\n[vt-score] sharding {len(eval_rows)} rollouts over "
                  f"{world} ranks ({(len(eval_rows)+world-1)//world} per rank)",
                  flush=True)

        # Each rank takes a strided slice — preserves a uniform distribution
        # over eval cycles instead of giving one rank "all the late steps".
        my_indices = list(range(rank, len(eval_rows), world))
        my_rows    = [eval_rows[i] for i in my_indices]

        puzzle_index = {tuple(sorted(p["numbers"])): p
                        for p in (train_puzzles + eval_puzzles + hard_probe)}
        # For each scored row we score against EVERY canonical 24-solution
        # for the puzzle (plus the rollout's own \boxed{} expr if it's not
        # in that set). Canonical R_T = max_s R_T(rollout → s); R_T_own
        # exposes the own-expr score for delta-vs-correct analysis.
        prompts, completions, refs_flat = [], [], []
        valid: List[bool] = []
        row_n_can: List[int] = []   # # canonical refs scored for this row
        row_own_idx: List[int] = [] # own_idx into row's ref slice, -1 if N/A
        for row in my_rows:
            key = tuple(sorted(row["numbers"]))
            puzzle = puzzle_index.get(key)
            if puzzle is None or not row.get("completion") or not puzzle["solutions"]:
                valid.append(False); row_n_can.append(0); row_own_idx.append(-1)
                continue
            refs, n_can, own_idx = _build_refs(puzzle, row.get("expr") or "")
            prompt = tokenizer.apply_chat_template(
                to_chat(puzzle)["prompt"], tokenize=False, add_generation_prompt=True)
            prompts.extend([prompt] * len(refs))
            completions.extend([row["completion"]] * len(refs))
            refs_flat.extend(refs)
            valid.append(True); row_n_can.append(n_can); row_own_idx.append(own_idx)

        try:
            from transformers.integrations.deepspeed import (
                unset_hf_deepspeed_config,
            )
            unset_hf_deepspeed_config()
        except ImportError:
            pass

        scorer_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # Logical device on this rank under CUDA_VISIBLE_DEVICES mask.
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        scorer_device = (f"cuda:{local_rank}"
                         if torch.cuda.is_available() else "cpu")
        # R_T is measured against the TRAINED policy (information-gain
        # interpretation), so we load from the consolidated checkpoint we
        # saved right after `trainer.train()`, not from the base HF hub id.
        scorer_src = str(trained_ckpt_dir)
        if is_main:
            print(f"[vt-score] loading trained policy from {scorer_src} "
                  "as R_T scorer", flush=True)
        scorer_load_kwargs = dict(dtype=scorer_dtype)
        try:
            scorer_model = AutoModelForCausalLM.from_pretrained(
                scorer_src, attn_implementation="flash_attention_2",
                **scorer_load_kwargs,
            ).to(scorer_device).eval()
            if is_main:
                print("[vt-score] scorer using flash_attention_2", flush=True)
        except (ImportError, ValueError) as e:
            if is_main:
                print(f"[vt-score] FA2 unavailable ({e.__class__.__name__}); "
                      "falling back to default attn", flush=True)
            scorer_model = AutoModelForCausalLM.from_pretrained(
                scorer_src, **scorer_load_kwargs,
            ).to(scorer_device).eval()

        if is_main:
            print(f"[vt-score] scoring {sum(valid)}/{len(my_rows)} rollouts on rank0 "
                  f"against {len(refs_flat)} (rollout, ref) pairs "
                  f"(micro_batch={args.vt_micro_batch}) ...", flush=True)
        t_vt = time.time()
        scored = compute_vt_batched(
            prompts, completions, refs_flat, scorer_model, tokenizer,
            micro_batch_size=args.vt_micro_batch,
        )

        N_PTS = args.vt_resample_pts
        grid = np.linspace(0.0, 1.0, N_PTS)
        cursor = 0
        out_records: List[Dict[str, Any]] = []
        for gidx, row, ok, n_can, own_idx in zip(
            my_indices, my_rows, valid, row_n_can, row_own_idx,
        ):
            if not ok:
                row["R_T"] = None
                row["R_per_token"] = None
                row["cumR_resampled"] = None
                row["R_T_per_ref"] = None
                row["R_T_own"] = None
                row["best_ref_idx"] = None
                out_records.append({"_gidx": gidx, "row": row})
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
                row["R_T_own"] = R_T_can[own_idx]
            elif own_idx >= n_can:
                own_sc = chunk[own_idx]
                row["R_T_own"] = (float(own_sc["R_T"])
                                  if not np.isnan(own_sc["R_T"]) else None)
            else:
                row["R_T_own"] = None
            out_records.append({"_gidx": gidx, "row": row})

        # Write this rank's scored slice as ``eval_rollout.jsonl.scored.r{R}``
        # with the original global index inlined so rank 0 can reassemble in
        # the right order.
        scored_shard = out_dir / f"eval_rollout.jsonl.scored.r{rank}"
        with scored_shard.open("w") as f:
            for rec in out_records:
                f.write(json.dumps(rec) + "\n")

        # Drop the scorer before the barrier so peak memory doesn't hold
        # while waiting for the slowest rank.
        del scorer_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _barrier()

        # Rank 0 merges per-rank scored shards back into eval_rollout.jsonl
        # preserving original order, then writes figures + metrics.
        if is_main:
            scored_by_idx: Dict[int, Dict[str, Any]] = {}
            for r in range(world):
                p = out_dir / f"eval_rollout.jsonl.scored.r{r}"
                if not p.exists():
                    continue
                for line in p.open("r"):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    scored_by_idx[int(rec["_gidx"])] = rec["row"]
            merged = [scored_by_idx[i] for i in range(len(eval_rows))
                      if i in scored_by_idx]

            tmp = eval_rollout_log.with_suffix(".jsonl.tmp")
            with tmp.open("w") as f:
                for row in merged:
                    f.write(json.dumps(row) + "\n")
            tmp.replace(eval_rollout_log)
            print(f"[vt-score] done in {time.time()-t_vt:.0f}s "
                  f"(merged {len(merged)}/{len(eval_rows)} rows from {world} ranks)",
                  flush=True)

            R_Ts = np.array([r["R_T"] for r in merged if r["R_T"] is not None])
            if R_Ts.size:
                metrics["vt_R_T_mean"]   = float(R_Ts.mean())
                metrics["vt_R_T_median"] = float(np.median(R_Ts))

            # ------- R_T figures ----------------------------------------
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

        # Drop the transient trained-policy checkpoint unless --keep-ckpt.
        # We've already extracted R_T into eval_rollout.jsonl; the weights
        # served only as the scorer and aren't needed downstream.
        if is_main and not args.keep_ckpt and trained_ckpt_dir.exists():
            import shutil
            shutil.rmtree(trained_ckpt_dir, ignore_errors=True)
            print(f"  removed transient checkpoint {trained_ckpt_dir.name} "
                  "(pass --keep-ckpt to retain)", flush=True)

    if is_main:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  metrics → {out_dir / 'metrics.json'}", flush=True)
    else:
        # Wait for rank 0 to finish all post-processing and writing metrics
        for _ in range(7200):
            if (out_dir / "metrics.json").exists():
                break
            time.sleep(5)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--output-root", default="output/game24_sweep")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--vllm-mode", choices=["server"], default="server",
                   help="Only 'server' is supported under DeepSpeed (colocate "
                        "would put vLLM on the training GPUs).")
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--score-vt", action="store_true", default=True)
    p.add_argument("--no-score-vt", dest="score_vt", action="store_false")
    p.add_argument("--keep-ckpt", action="store_true", default=False,
                   help="Retain the transient trained-policy checkpoint "
                        "(<out_dir>/trained_for_vt) used as the R_T scorer. "
                        "By default it is deleted after R_T scoring completes.")
    p.add_argument("--vt-micro-batch", type=int, default=8)
    p.add_argument("--vt-resample-pts", type=int, default=100)
    p.add_argument("--rt-pair-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    run_one(args)


if __name__ == "__main__":
    main()
