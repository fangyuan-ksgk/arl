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

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0",
                        help="(server only) Host of the vLLM server.")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                        help="(server only) Port of the vLLM server.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--train_device", type=int, default=0,
                        help="CUDA device index for training (server mode only). "
                             "Must not overlap with vLLM server GPUs.")
    # Eval
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Run eval every N steps (0 to disable)")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Subsample N test examples for eval (default: full test set)")
    # MBE dynamics logging
    parser.add_argument("--mbe_log", action="store_true",
                        help="Log MBE dynamics (correct vs incorrect) to JSONL during training")
    parser.add_argument("--mbe_log_steps", type=int, default=1,
                        help="Log MBE every N reward-function calls (1 = every step)")
    parser.add_argument("--mbe_log_sample_k", type=int, default=4,
                        help="Max rollouts to analyse per logged step")
    # MBE reward
    parser.add_argument("--mbe_reward", action="store_true",
                        help="Add scaled MBE reward: min(mbe, clip) / scale")
    parser.add_argument("--gated_mbe_reward", action="store_true",
                        help="Add correctness-gated MBE reward (MBE only for correct rollouts)")
    parser.add_argument("--mbe_scale", type=float, default=40.0,
                        help="MBE reward denominator (default 40.0 → max ~0.05)")
    parser.add_argument("--mbe_clip", type=float, default=2.0,
                        help="MBE value clipped before scaling")
    # Prefix-conditioned rollout exploration (PCRE)
    parser.add_argument("--prefix_rollout", action="store_true",
                        help="Enable prefix-conditioned rollout exploration")
    parser.add_argument("--prefix_augment_prob", type=float, default=0.3,
                        help="Fraction of training examples to replace with prefix-augmented ones")
    parser.add_argument("--prefix_buffer_size", type=int, default=500,
                        help="Max rollouts stored in the prefix buffer")
    parser.add_argument("--prefix_min_frac", type=float, default=0.15,
                        help="Min fraction of completion to keep as prefix")
    parser.add_argument("--prefix_max_frac", type=float, default=0.75,
                        help="Max fraction of completion to keep as prefix")
    parser.add_argument("--prefix_from_correct", type=str, default="all",
                        choices=["all", "correct", "incorrect"],
                        help="Sample prefixes from: all / correct / incorrect rollouts")
    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA (PEFT) instead of full fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    train_dataset, test_dataset = load_gsm8k()

    # Tokenizer needed early if prefix rollout is enabled (for prompt formatting)
    if args.prefix_rollout:
        _tok_for_prefix = AutoTokenizer.from_pretrained(args.model)
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

    # LoRA config
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

    # Explicitly load model to avoid occupying vLLM server GPUs.
    # - Single GPU: pin to --train_device
    # - Multi-GPU (accelerate): load to CPU, let accelerate handle placement
    if not args.no_vllm and args.vllm_mode == "server":
        num_processes = int(os.environ.get("WORLD_SIZE", "1"))
        if num_processes > 1:
            # accelerate multi-GPU: load to CPU, accelerate places per rank
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{args.train_device}"},
            )
    else:
        model = args.model  # let TRL handle device placement for colocate/no-vllm

    # MBE dynamics logger (logs as side-effect, zero reward signal)
    mbe_logger = None
    if args.mbe_log:
        from src.mbe_logger import MBEDynamicsLogger
        tokenizer_for_log = AutoTokenizer.from_pretrained(args.model)
        log_path = os.path.join(args.output_dir, "mbe_dynamics.jsonl")
        mbe_logger = MBEDynamicsLogger(
            tokenizer_for_log,
            log_path=log_path,
            log_steps=args.mbe_log_steps,
            sample_k=args.mbe_log_sample_k,
        )
        print(f"MBE dynamics logger enabled → {log_path}  "
              f"(every {args.mbe_log_steps} steps, {args.mbe_log_sample_k} samples/step)")

    # Prefix rollout buffer (created before reward_funcs so collector can reference it)
    prefix_buffer = None
    prefix_dataset = None
    if args.prefix_rollout:
        from src.prefix_rollout import PrefixRolloutBuffer, PrefixRolloutCollector, PrefixAugmentedDataset
        _from_correct_map = {"all": None, "correct": True, "incorrect": False}
        prefix_buffer = PrefixRolloutBuffer(
            max_size=args.prefix_buffer_size,
            min_prefix_frac=args.prefix_min_frac,
            max_prefix_frac=args.prefix_max_frac,
        )
        prefix_dataset = PrefixAugmentedDataset(
            train_dataset,
            prefix_buffer,
            _tok_for_prefix,
            augment_prob=args.prefix_augment_prob,
            from_correct=_from_correct_map[args.prefix_from_correct],
        )
        print(
            f"Prefix rollout enabled: augment_prob={args.prefix_augment_prob}, "
            f"buffer_size={args.prefix_buffer_size}, "
            f"prefix_frac=[{args.prefix_min_frac}, {args.prefix_max_frac}], "
            f"from={args.prefix_from_correct}"
        )

    # Build reward function list
    reward_funcs = [correctness_reward, format_reward]
    mbe_reward_obj = None

    if args.mbe_log and mbe_logger is not None:
        reward_funcs.append(mbe_logger.as_reward(correctness_fn=correctness_reward))

    if args.prefix_rollout and prefix_buffer is not None:
        prefix_collector = PrefixRolloutCollector(
            prefix_buffer, correctness_fn=correctness_reward
        )
        reward_funcs.append(prefix_collector)

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
        print(f"MBE reward enabled: {'gated' if args.gated_mbe_reward else 'plain'}, "
              f"scale={args.mbe_scale}, clip={args.mbe_clip}")

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
        train_dataset=prefix_dataset if prefix_dataset is not None else train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Bind model ref for MBE forward passes
    if mbe_logger is not None:
        mbe_logger.set_model(trainer.model)
    if mbe_reward_obj is not None:
        mbe_reward_obj.set_model(trainer.model)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()