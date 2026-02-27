"""
GRPO on GSM8K with Qwen3-0.6B + vLLM

Single-GPU:
    python scripts/grpo_gsm8k.py                        # smoke test (20 steps)
    python scripts/grpo_gsm8k.py --max_steps -1         # full run (1 epoch)

Multi-GPU:
    accelerate launch --num_processes 4 scripts/grpo_gsm8k.py --no_vllm --max_steps -1
    accelerate launch --config_file scripts/configs/multi_gpu.yaml scripts/grpo_gsm8k.py --no_vllm

DeepSpeed ZeRO-2:
    accelerate launch --config_file scripts/configs/deepspeed_zero2.yaml scripts/grpo_gsm8k.py --no_vllm

Note: vLLM colocate mode is single-GPU only. For multi-GPU, use --no_vllm
      (TRL falls back to its native generation which supports distributed).
"""

import argparse
import os
import re

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def extract_answer_from_completion(text: str) -> str:
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def correctness_reward(completions, gold_answer, **kwargs):
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        text = completion[0]["content"]
        predicted = extract_answer_from_completion(text)
        try:
            correct = float(predicted) == float(gold)
        except (ValueError, TypeError):
            correct = False
        rewards.append(1.0 if correct else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        has_format = bool(re.search(r"####\s*[\d,\.\-]+", text))
        rewards.append(0.5 if has_format else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")

    def extract_gold_answer(answer_text: str) -> str:
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return ""

    def format_example(example):
        example["prompt"] = [{"role": "user", "content": example["question"]}]
        example["gold_answer"] = extract_gold_answer(example["answer"])
        return example

    train_dataset = dataset["train"].map(format_example)
    test_dataset = dataset["test"].map(format_example)
    return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="grpo_gsm8k_output")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=20, help="-1 for full epoch")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--no_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["colocate", "server"],
                        help="'colocate': vLLM shares GPU with training (single-GPU). "
                             "'server': vLLM runs as separate server on dedicated GPUs.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5,
                        help="(colocate only) Fraction of GPU VRAM for vLLM KV cache.")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--report_to", type=str, default="none")
    args = parser.parse_args()

    train_dataset, test_dataset = load_gsm8k()
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

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

    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
