"""Head-to-head ablation runner for GRPO vs PerTokenAdvantageTrainer on Game-of-24.

Variants (one PTrainer design toggled at a time):
  grpo                : vanilla TRL GRPOTrainer (baseline).
  pt-placeholder      : PT subclass + ramp reward. Isolates trainer subclass
                        from the velocity reward.
  pt-velocity         : PT + VelocityRewardComputer (answer buffer seeded).
                        Isolates per-token reward from prefix injection.
  pt-velocity-prefix  : PT + velocity + PrefixTrieBuffer + PrefixInjector.

Buffer wiring (verified):
  grpo, pt-placeholder : no buffers.
  pt-velocity          : answer_buffer seeded from solutions; prefix_buffer=None.
  pt-velocity-prefix   : answer_buffer seeded; prefix_buffer = PrefixTrieBuffer
                         (PrefixInjector reads it; PT writes accepted CoTs back).

Each run drops everything under output/ablate_game24/<run_name>/:
  config.json     all CLI args
  rollouts.jsonl  per-rollout log
  buffers.json    final buffer stats

Usage:
  python script/ablate_game24.py --variant grpo               --steps 200
  python script/ablate_game24.py --variant pt-velocity        --adv_mode token
  python script/ablate_game24.py --variant pt-velocity-prefix --p_inject 0.5
"""
from __future__ import annotations

import argparse, json, os, random, re, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.game24utils import (
    bucket_by_difficulty, build_datasets, build_puzzle_pool,
    correctness_reward, extract_expr, format_reward,
    make_splits, verify_24, _text,
)
from src.online_buffer import OnlineBuffer
from src.pertoken_trainer import PerTokenAdvantageTrainer
from src.prefix_inject import PrefixInjector, PrefixTrieBuffer
from src.velocity import VelocityRewardComputer


# -------- task hooks --------

_NUM_RE = re.compile(r"Given numbers:\s*([\d,\s]+)")

def _nums_from_prompt(text: str):
    m = _NUM_RE.search(text)
    return [int(x) for x in m.group(1).split(",")] if m else []

def game24_query_key(prompt_str: str):
    return tuple(_nums_from_prompt(prompt_str))

def game24_is_correct(completion_str: str, prompt_str: str) -> bool:
    nums = _nums_from_prompt(prompt_str)
    return bool(nums) and bool(verify_24(nums, extract_expr(completion_str)))


def placeholder_per_token_reward(prompt_ids, completion_ids, completion_mask, *, tokenizer):
    """Trajectory-correctness × position-ramp. Sanity baseline only."""
    B, T = completion_ids.shape
    device = completion_ids.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    traj_r = torch.zeros(B, device=device)
    for i in range(B):
        text = tokenizer.decode(
            completion_ids[i][completion_mask[i].bool()], skip_special_tokens=True,
        )
        ptxt = tokenizer.decode(
            [int(x) for x in prompt_ids[i].tolist() if x != pad_id], skip_special_tokens=True,
        )
        nums = _nums_from_prompt(ptxt)
        traj_r[i] = float(verify_24(nums, extract_expr(text))) if nums else 0.0
    T_eff = completion_mask.sum(dim=1).clamp(min=1).float()
    pos = torch.arange(T, device=device, dtype=torch.float32)
    ramp = pos.unsqueeze(0) / T_eff.unsqueeze(1)
    return ramp * traj_r.unsqueeze(1) * completion_mask.float()


class RolloutLogger:
    """Zero-reward callback: dumps every rollout to JSONL."""
    __name__ = "rollout_logger"
    def __init__(self, path: Path, tok):
        self.path, self.tok, self.step = path, tok, 0
        path.write_text("")
    def __call__(self, completions, numbers, solutions=None, **kwargs):
        with self.path.open("a") as f:
            for i, (c, nums) in enumerate(zip(completions, numbers)):
                text = _text(c); expr = extract_expr(text)
                ok = verify_24(list(nums), expr)
                n_tok = len(self.tok.encode(text, add_special_tokens=False))
                f.write(json.dumps({
                    "step": self.step, "idx": i, "numbers": list(nums),
                    "completion": text, "expr": expr,
                    "correct": bool(ok), "n_tokens": int(n_tok),
                }) + "\n")
        self.step += 1
        return [0.0] * len(completions)


def build_run_name(a) -> str:
    parts = [a.variant]
    if a.variant.startswith("pt-"):
        parts.append(f"adv-{a.adv_mode}")
    if a.variant == "pt-velocity-prefix":
        parts.append(f"pinj-{a.p_inject}")
        parts.append(f"L-{a.prefix_max_layer}")
    parts.append(f"seed-{a.seed}")
    return "_".join(parts)


def make_grpo_config(a, output_dir: Path, with_prefix: bool) -> GRPOConfig:
    kw = dict(
        output_dir=str(output_dir),
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_length,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        learning_rate=a.learning_rate,
        max_steps=a.steps,
        logging_steps=a.logging_steps,
        bf16=True,
        save_strategy="no",
        report_to="none",
        use_vllm=True,
        vllm_mode=a.vllm_mode,
    )
    if a.vllm_mode == "colocate":
        kw["vllm_gpu_memory_utilization"] = a.vllm_mem
        # Generous budget: prompt + (optional) prefix + completion + slack.
        kw["vllm_max_model_length"] = max(2 * a.max_completion_length, 2048) \
            if with_prefix else max(a.max_completion_length + 512, 1024)
    else:  # server
        kw["vllm_server_host"] = a.vllm_server_host
        kw["vllm_server_port"] = a.vllm_server_port
    return GRPOConfig(**kw)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["grpo", "pt-placeholder", "pt-velocity", "pt-velocity-prefix"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # GRPO
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--vllm_mem", type=float, default=0.4)
    # vLLM mode
    ap.add_argument("--vllm_mode", choices=["colocate", "server"], default="colocate",
                    help="'colocate' = single-GPU shared with training. "
                         "'server' = connect to a separately launched `trl vllm-serve`.")
    ap.add_argument("--vllm_server_host", default="0.0.0.0")
    ap.add_argument("--vllm_server_port", type=int, default=8000)
    ap.add_argument("--train_device", type=int, default=0,
                    help="(server only) CUDA index for training; must not overlap with vLLM.")
    # PT
    ap.add_argument("--adv_mode", default="token", choices=["token", "position", "progress"])
    ap.add_argument("--adv_n_chunks", type=int, default=8)
    ap.add_argument("--adv_stride", type=int, default=5)
    ap.add_argument("--vel_chunk_size", type=int, default=16)
    ap.add_argument("--vel_chunk_strategy", default="uniform", choices=["uniform", "random"])
    ap.add_argument("--answer_capacity", type=int, default=64)
    # Prefix
    ap.add_argument("--p_inject", type=float, default=0.5)
    ap.add_argument("--share_within_group", type=int, default=1)
    ap.add_argument("--prefix_max_depth", type=int, default=100)
    ap.add_argument("--prefix_max_layer", type=int, default=3)
    ap.add_argument("--prefix_truncate", default="none", choices=["none", "uniform"])
    # Output
    ap.add_argument("--output_root", default="output/ablate_game24")
    return ap.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # PrefixInjector only supports colocate (see src/prefix_inject.py).
    if a.variant == "pt-velocity-prefix" and a.vllm_mode != "colocate":
        raise SystemExit(
            "pt-velocity-prefix requires --vllm_mode colocate "
            "(PrefixInjector is not implemented for vllm_mode='server'). "
            "Run prefix variants on a single GPU instead."
        )

    run_name = build_run_name(a)
    output_dir = Path(a.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(a), indent=2))
    print(f"[{run_name}] output → {output_dir.resolve()}")

    # data
    puzzles = build_puzzle_pool(max_n=9)
    easy, medium, hard = bucket_by_difficulty(puzzles, easy_min=8, hard_max=2)
    train_puzzles, eval_puzzles, hard_probe = make_splits(
        easy, medium, hard, eval_frac=0.10, probe_frac=0.40,
    )
    train_ds, eval_ds, probe_ds = build_datasets(train_puzzles, eval_puzzles, hard_probe)
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}  probe={len(probe_ds)}")

    # model
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model_kw = dict(torch_dtype=torch.bfloat16)
    _under_dist = any(k in os.environ for k in ("ACCELERATE_USE_FSDP", "LOCAL_RANK"))
    if a.vllm_mode == "server" and torch.cuda.is_available() and not _under_dist:
        # Pin training to --train_device so it never lands on the GPU(s) running vLLM.
        model_kw["device_map"] = {"": f"cuda:{a.train_device}"}
    model = AutoModelForCausalLM.from_pretrained(a.model, **model_kw)
    rollout_logger = RolloutLogger(output_dir / "rollouts.jsonl", tokenizer)
    rewards = [correctness_reward, format_reward, rollout_logger]

    # ---- variant dispatch ----
    answer_buffer = None
    prefix_buffer = None

    if a.variant == "grpo":
        cfg = make_grpo_config(a, output_dir, with_prefix=False)
        trainer = GRPOTrainer(
            model=model, reward_funcs=rewards, args=cfg, train_dataset=train_ds,
        )
    else:
        with_prefix = a.variant == "pt-velocity-prefix"
        cfg = make_grpo_config(a, output_dir, with_prefix=with_prefix)
        kw = dict(
            model=model, reward_funcs=rewards, args=cfg, train_dataset=train_ds,
            adv_mode=a.adv_mode, adv_n_chunks=a.adv_n_chunks, adv_stride=a.adv_stride,
        )

        if a.variant == "pt-placeholder":
            kw["per_token_reward_fn"] = placeholder_per_token_reward
        else:
            # velocity route — needs a seeded answer buffer
            answer_buffer = OnlineBuffer(capacity_per_query=a.answer_capacity)
            n_seed = 0
            for row in train_ds:
                qk = tuple(row["numbers"])
                for expr in (row.get("solutions") or []):
                    n_seed += int(answer_buffer.add(qk, expr))
            print(f"  answer_buffer seeded: {n_seed} entries / {answer_buffer.num_queries()} queries")

            kw.update(
                velocity_computer=VelocityRewardComputer(
                    answer_buffer,
                    chunk_strategy=a.vel_chunk_strategy,
                    chunk_size=a.vel_chunk_size,
                    normalize_by_chunk=True,
                ),
                is_correct=game24_is_correct,
                query_key_fn=game24_query_key,
            )

            if a.variant == "pt-velocity-prefix":
                prefix_buffer = PrefixTrieBuffer(
                    max_depth=a.prefix_max_depth, max_layer=a.prefix_max_layer,
                )
                kw["prefix_buffer"] = prefix_buffer
                kw["rollout_func"] = PrefixInjector(
                    prefix_buffer,
                    query_key_fn=game24_query_key,
                    p_inject=a.p_inject,
                    truncate=a.prefix_truncate,
                    share_within_group=bool(a.share_within_group),
                    rng=np.random.default_rng(a.seed),
                )

        trainer = PerTokenAdvantageTrainer(**kw)

    print(f"  trainer = {type(trainer).__name__}")
    trainer.train()

    # final buffer stats
    stats = {}
    if answer_buffer is not None:
        stats["answer_buffer"] = answer_buffer.stats()
    if prefix_buffer is not None:
        stats["prefix_buffer"] = prefix_buffer.stats()
    if stats:
        (output_dir / "buffers.json").write_text(json.dumps(stats, indent=2))
        print(f"  buffer stats → {output_dir / 'buffers.json'}")

    print(f"[{run_name}] done.")


if __name__ == "__main__":
    main()
