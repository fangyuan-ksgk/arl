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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


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


def load_math_dataset():
    dataset = load_dataset("hendrycks/competition_math")

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO on MATH (Hendrycks)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="grpo_math_output")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=1024)
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
    parser.add_argument("--mbe_scale", type=float, default=40.0)
    parser.add_argument("--mbe_clip", type=float, default=2.0)
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    train_dataset, test_dataset = load_math_dataset()

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
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
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

    # Eval dataset
    eval_dataset = None
    if args.eval_steps > 0:
        eval_dataset = test_dataset
        if args.eval_samples is not None:
            eval_dataset = test_dataset.select(range(min(args.eval_samples, len(test_dataset))))
        print(f"Eval enabled: {len(eval_dataset)} samples every {args.eval_steps} steps")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    if mbe_reward_obj is not None:
        mbe_reward_obj.set_model(trainer.model)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
