"""
GRPO on MATH (Hendrycks) with Qwen3-0.6B + vLLM

Single-GPU:
    python script/grpo_math.py                        # smoke test (20 steps)
    python script/grpo_math.py --max_steps -1         # full run (1 epoch)

Server mode (2-GPU):
    GPU 0: trl vllm-serve --model Qwen/Qwen3-0.6B --port 8000
    GPU 1: CUDA_VISIBLE_DEVICES=1 python script/grpo_math.py \
               --use_vllm --vllm_mode server --vllm_server_port 8000 --max_steps 200
"""

import argparse
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


class SaveAtStepsCallback(TrainerCallback):
    """Force checkpoint saves at arbitrary steps (HF save_steps only supports fixed intervals) ->
    checkpoint-<step>. run_seed_scaling_lora / local_sgd read the adapter from checkpoint-<steps>."""
    def __init__(self, steps):
        self.steps = set(int(s) for s in steps)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step in self.steps:
            control.should_save = True
        return control


# ---------------------------------------------------------------------------
# Answer extraction & comparison  (MATH uses \boxed{...})
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str:
    """Extract the last \\boxed{...} content from text, handling nested braces."""
    # Find all \boxed{ positions
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        # Walk forward matching braces
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            break
        continue
    return results[-1].strip() if results else ""


def normalize_answer(ans: str) -> str:
    """Normalize a MATH answer string for comparison."""
    ans = ans.strip()
    # Remove surrounding $ signs
    ans = ans.strip("$")
    # Remove \text{...} wrappers
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    # Remove \mathrm{...} wrappers
    ans = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", ans)
    # Normalize \frac{a}{b} → a/b for simple cases
    ans = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", ans)
    # Remove unnecessary spaces
    ans = re.sub(r"\s+", "", ans)
    # Remove trailing period
    ans = ans.rstrip(".")
    return ans


def answers_equal(predicted: str, gold: str) -> bool:
    """Compare two MATH answers, trying string match then numeric."""
    if not predicted or not gold:
        return False
    p = normalize_answer(predicted)
    g = normalize_answer(gold)
    # Direct string match
    if p == g:
        return True
    # Try numeric comparison
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    # Try evaluating simple fractions
    try:
        pv = eval(p, {"__builtins__": {}})  # noqa: S307
        gv = eval(g, {"__builtins__": {}})  # noqa: S307
        return abs(float(pv) - float(gv)) < 1e-6
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def extract_answer_from_completion(text: str) -> str:
    """Extract answer from model completion: try \\boxed{}, then #### fallback."""
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    # Fallback: #### format (model might learn GSM8K-style)
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    return ""


def correctness_reward(completions, gold_answer, **kwargs):
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        text = completion[0]["content"]
        predicted = extract_answer_from_completion(text)
        correct = answers_equal(predicted, gold)
        rewards.append(1.0 if correct else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Reward for producing a \\boxed{} answer."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        has_boxed = bool(extract_boxed(text))
        rewards.append(0.5 if has_boxed else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


def load_math_any():
    """Load Hendrycks MATH via a WORKING mirror ('hendrycks/competition_math' is gone from the Hub).
    DigitalLearningGmbH/MATH-lighteval = the official 7500 train / 5000 test (disjoint). eval_math
    imports this."""
    last = None
    for repo, cfg in [("DigitalLearningGmbH/MATH-lighteval", None),
                      ("EleutherAI/hendrycks_math", "default"),
                      ("nlile/hendrycks-MATH-benchmark", None)]:
        try:
            return load_dataset(repo) if cfg is None else load_dataset(repo, cfg)
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(f"Could not load any MATH mirror: {last}")


def load_math_dataset():
    dataset = load_math_any()

    def format_example(example):
        example["prompt"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        example["gold_answer"] = extract_boxed(example["solution"])
        return example

    train_dataset = dataset["train"].map(format_example)
    test_dataset = dataset["test"].map(format_example)
    print(f"MATH dataset — Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    # Show level/type distribution
    if "level" in train_dataset.column_names:
        from collections import Counter
        levels = Counter(train_dataset["level"])
        types = Counter(train_dataset["type"])
        print(f"  Levels: {dict(sorted(levels.items()))}")
        print(f"  Types:  {dict(sorted(types.items()))}")
    return train_dataset, test_dataset


class VirtualRolloutGRPOTrainer(GRPOTrainer):
    """GRPOTrainer + virtual-rollout advantage shaping (insert-max etc.) — the necessary complement to
    the MBE-velocity reward. No-op when virtual_rollout_mode is None. See src/arsenal.virtual_rollout_advantages."""
    def _calculate_rewards(self, *args, **kwargs):
        rpf = super()._calculate_rewards(*args, **kwargs)
        self._last_rewards_per_func = rpf
        return rpf

    def _local_rewards_per_func(self, out):
        rpf = getattr(self, "_last_rewards_per_func", None)
        adv = out.get("advantages")
        if rpf is None or adv is None:
            return None
        Bp = adv.shape[0]
        lo = self.accelerator.process_index * Bp
        return rpf[lo:lo + Bp]

    def _virtual_rollout_advantages(self, out, local):
        from src.arsenal import virtual_rollout_advantages
        adv = out.get("advantages")
        names = self.reward_func_names
        rewards = local.sum(dim=1)
        if "correctness_reward" in names:
            corrects = (local[:, names.index("correctness_reward")] == 1.0)
        else:
            corrects = torch.zeros_like(rewards, dtype=torch.bool)
        return virtual_rollout_advantages(
            rewards, corrects, self.num_generations,
            max_reward=getattr(self, "virtual_max_reward", 1.2),
            mode=self.virtual_rollout_mode).to(adv)

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if getattr(self, "virtual_rollout_mode", None) and self.model.training:
            local = self._local_rewards_per_func(out)
            if local is not None and out.get("advantages") is not None:
                out["advantages"] = self._virtual_rollout_advantages(out, local)
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO on MATH (Hendrycks)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="grpo_math_output")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=3072)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=20, help="-1 for full epoch")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--no_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["colocate", "server"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--train_device", type=int, default=0,
                        help="CUDA device for training (server mode)")
    # Eval
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Run eval every N steps (0 to disable)")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Subsample N test examples for eval (default: full test set)")
    # MBE reward
    parser.add_argument("--mbe_reward", action="store_true",
                        help="Add scaled MBE reward")
    parser.add_argument("--gated_mbe_reward", action="store_true",
                        help="Add correctness-gated MBE reward")
    parser.add_argument("--mbe_scale", type=float, default=5.0)
    parser.add_argument("--mbe_clip", type=float, default=2.0)
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=512)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Dr.GRPO recipe + scheduler + save-at-step + seed (consumed by run_seed_scaling_lora / local_sgd)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="dr_grpo",
                        choices=["grpo", "dapo", "bnpo", "dr_grpo"])
    parser.add_argument("--scale_rewards", type=str, default="none",
                        choices=["group", "batch", "none"])
    parser.add_argument("--save_steps_list", type=str, default=None,
                        help="comma-sep steps to force-save (e.g. the final step) -> checkpoint-<step>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_truncated_completions", action="store_true")
    # MBE velocity reward + insert-max virtual-rollout (the necessary anti-reward-hacking pair).
    parser.add_argument("--mbe_velocity_reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mbe_velocity_scale", type=float, default=5.0)
    parser.add_argument("--mbe_velocity_clip", type=float, default=1.0)
    parser.add_argument("--mbe_velocity_stride", type=int, default=8)
    parser.add_argument("--mbe_velocity_layers", type=str, default="-1")
    parser.add_argument("--mbe_velocity_mode", type=str, default="trajectory", choices=["trajectory", "rollercoaster"])
    parser.add_argument("--virtual_rollout", type=str, default="none",
                        choices=["none", "insert_max", "insert_min", "insert_max_min",
                                 "insert_max_all_incorrect", "insert_max_mixed"])
    parser.add_argument("--virtual_max_reward", type=float, default=1.2)
    parser.add_argument("--data_shard", default="",
                        help="i/N -> train on the disjoint shard i of N (MoLoRA v2: per-expert data split)")
    args = parser.parse_args()

    train_dataset, test_dataset = load_math_dataset()

    if args.data_shard:                                    # MoLoRA v2: disjoint per-expert data shard
        i, N = (int(x) for x in args.data_shard.split("/"))
        train_dataset = train_dataset.shuffle(seed=1234).shard(num_shards=N, index=i)
        print(f"[shard] training on disjoint shard {i}/{N}: {len(train_dataset)} examples", flush=True)

    config_kwargs = dict(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_strategy,
        report_to=args.report_to,
        # Dr.GRPO: unbiased length (loss_type) + no std/group reward scaling (scale_rewards none -> False)
        loss_type=args.loss_type,
        scale_rewards=(False if args.scale_rewards == "none" else args.scale_rewards),
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        beta=0.0,
        seed=args.seed,
        mask_truncated_completions=args.mask_truncated_completions,
    )
    if args.max_steps > 0:
        config_kwargs["max_steps"] = args.max_steps
    if not args.no_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "colocate":
            config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        elif args.vllm_mode == "server":
            config_kwargs["vllm_server_host"] = args.vllm_server_host
            config_kwargs["vllm_server_port"] = args.vllm_server_port

    if args.eval_steps > 0:
        config_kwargs["eval_strategy"] = "steps"
        config_kwargs["eval_steps"] = args.eval_steps

    config = GRPOConfig(**config_kwargs)

    # LoRA
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        from transformers import AutoConfig as _AC
        _mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        _g4 = "gemma4" in (getattr(_AC.from_pretrained(args.model), "model_type", "") or "")
        _tm = [f"{m}.linear" for m in _mods] if _g4 else _mods  # gemma4: inner nn.Linear of Gemma4ClippableLinear
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=_tm,
            task_type="CAUSAL_LM",
        )
        print(f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}")

    # Model loading
    if not args.no_vllm and args.vllm_mode == "server":
        num_processes = int(os.environ.get("WORLD_SIZE", "1"))
        if num_processes > 1:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{args.train_device}"},
            )
    else:
        model = args.model

    # Reward functions
    reward_funcs = [correctness_reward, format_reward]
    mbe_reward_obj = None

    if args.mbe_reward or args.gated_mbe_reward:
        from src.mbe_reward import MBEReward, CorrectnessGatedMBEReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.gated_mbe_reward:
            mbe_reward_obj = CorrectnessGatedMBEReward(
                tokenizer, scale=args.mbe_scale, clip=args.mbe_clip,
            )
        else:
            mbe_reward_obj = MBEReward(
                tokenizer, scale=args.mbe_scale, clip=args.mbe_clip,
            )
        reward_funcs.append(mbe_reward_obj)
        print(f"MBE reward: {'gated' if args.gated_mbe_reward else 'plain'}, "
              f"scale={args.mbe_scale}, clip={args.mbe_clip}")

    mbe_velo_reward_obj = None
    if args.mbe_velocity_reward:
        from src.mbe_reward import MBEVeloReward
        velo_layers = [int(x) for x in args.mbe_velocity_layers.split(",") if x.strip()]
        mbe_velo_reward_obj = MBEVeloReward(
            AutoTokenizer.from_pretrained(args.model),
            layers=velo_layers, stride=args.mbe_velocity_stride,
            scale=args.mbe_velocity_scale, clip=args.mbe_velocity_clip, mode=args.mbe_velocity_mode)
        reward_funcs.append(mbe_velo_reward_obj)
        print(f"MBE velocity reward enabled: scale={args.mbe_velocity_scale}, clip=±{args.mbe_velocity_clip}")

    # Eval dataset
    eval_dataset = None
    if args.eval_steps > 0:
        eval_dataset = test_dataset
        if args.eval_samples is not None:
            eval_dataset = test_dataset.select(range(min(args.eval_samples, len(test_dataset))))
        print(f"Eval enabled: {len(eval_dataset)} samples every {args.eval_steps} steps")

    callbacks = []
    if args.save_steps_list:
        steps_list = [int(s) for s in args.save_steps_list.split(",") if s.strip()]
        callbacks.append(SaveAtStepsCallback(steps_list))
        print(f"Forced checkpoint saves at steps {sorted(set(steps_list))} -> {args.output_dir}/checkpoint-<step>")

    trainer = VirtualRolloutGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.virtual_rollout_mode = None if args.virtual_rollout == "none" else args.virtual_rollout
    trainer.virtual_max_reward = args.virtual_max_reward
    if trainer.virtual_rollout_mode:
        print(f"Virtual-rollout advantage shaping: mode={trainer.virtual_rollout_mode}, "
              f"max_reward={trainer.virtual_max_reward}")

    if mbe_reward_obj is not None:
        mbe_reward_obj.set_model(trainer.model)
    if mbe_velo_reward_obj is not None:
        mbe_velo_reward_obj.set_model(trainer.model)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
