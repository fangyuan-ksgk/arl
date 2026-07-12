"""Terminal-Bench GRPO training on TRL 1.7 GRPOTrainer (Request 3f).

Architecture:
  - Single-turn: standard GRPOTrainer generation (no rollout_func needed) —
    the model emits ONE ```bash block per prompt.
  - Reward (tbench_reward.py): extract the bash block, execute it in a fresh
    proot sandbox (sandbox.py — WITH the process-group reaping fix), then run
    the task's own pytest verifier; reward = fraction of tests passed. Rewards
    run on CPU workers: the whole batch is fanned out over a thread pool sized
    by env TBENCH_REWARD_WORKERS (default 8).
  - GRPOTrainer + colocated vLLM (vllm_mode="colocate", NO sleep mode —
    broken with trl 1.7 + vllm 0.23 on this box).
  - Default model Qwen/Qwen2.5-Coder-3B-Instruct + LoRA-32.

Prereqs (checked at startup, clear error if missing):
  proot, ~/tbench-venv (pytest+pyyaml), ~/terminal-bench clone,
  dataset built by build_tbench_trl.py.

Launch (see run_tbench_trl.sh):
  CUDA_VISIBLE_DEVICES=0 python trl_tbench.py
First GPU run should be conservative: --max_steps 2 --num_generations 4.
"""

import argparse
import os
import sys

from datasets import load_dataset

import sandbox
from tbench_reward import tbench_reward

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    ap = argparse.ArgumentParser(description="Terminal-Bench GRPO on TRL")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/data/tbench_trl"))
    ap.add_argument("--output_dir", default=os.path.expanduser("~/ckpts/tbench_trl"))
    # GRPO
    ap.add_argument("--num_generations", type=int, default=8,
                    help="samples per task per step (SkyRL run used 8)")
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL coefficient (0 disables the ref model)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=20.0,
                    help="32 tasks/epoch -> many epochs per run")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    # lengths (trl 1.7 has no max_prompt_length; prompt + completion must fit
    # vllm_max_model_len — the 32 curated instructions are all < 1k tokens)
    ap.add_argument("--max_completion_length", type=int, default=1024,
                    help="one bash script; 1024 tokens is plenty")
    # LoRA
    ap.add_argument("--lora_rank", type=int, default=32, help="0 = full finetuning")
    ap.add_argument("--optim", default="adamw_torch",
                    help="paged_adamw_8bit for memory-safe full-FT at 4B")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--liger", action="store_true")
    # vLLM
    ap.add_argument("--no_vllm", action="store_true",
                    help="HF generate fallback (CPU debug only)")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    ap.add_argument("--vllm_max_model_len", type=int, default=4096)
    # eval / saving
    ap.add_argument("--eval_steps", type=int, default=0, help="0 = no eval during training")
    ap.add_argument("--save_steps", type=int, default=0, help="0 = no checkpoints")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--run_name", default="tbench_trl")
    return ap.parse_args()


def main():
    args = parse_args()

    # Fail fast, with setup instructions, if the sandbox stack is missing —
    # otherwise every reward would be a silent 0 and GRPO would learn nothing.
    try:
        sandbox.check_available()
    except sandbox.SandboxUnavailable as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)

    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path = os.path.join(args.data_dir, "validation.parquet")
    if not os.path.exists(train_path):
        print(f"FATAL: {train_path} missing — run build_tbench_trl.py first",
              file=sys.stderr)
        sys.exit(2)
    data_files = {"train": train_path}
    if os.path.exists(val_path):
        data_files["validation"] = val_path
    ds = load_dataset("parquet", data_files=data_files)

    # Import trl lazily so --help / prereq errors work without CUDA-adjacent deps.
    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        save_only_model=True,   # storage class-fix 07-10: full trainer state = 25GB/ckpt
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        # Qwen3 thinking-mode fix (07-08): <think> exhausted the completion
        # budgets sized for non-thinking models -> clip 0.95+ -> reward 0 ->
        # zero-variance GRPO starvation. Match grpo_code's protocol.
        chat_template_kwargs={"enable_thinking": False},
        # optimization
        learning_rate=args.lr,
        max_grad_norm=0.5,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        # lengths
        max_completion_length=args.max_completion_length,
        # sampling
        temperature=args.temperature,
        top_p=args.top_p,
        # vLLM colocate; sleep mode MUST stay off (broken with trl1.7+vllm0.23)
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=False,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_len,
        # a truncated script is still a (bad) attempt — keep it in the loss
        mask_truncated_completions=False,
        # logging / saving / eval
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps or 500,
        eval_strategy="steps" if (args.eval_steps > 0 and "validation" in ds) else "no",
        eval_steps=args.eval_steps or 500,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,
        report_to=args.report_to,
    )

    peft_config = None
    if args.liger:
        # fused linear+CE removes the fp32-logits materialization
        # (bs x seq x 151k vocab) — the 4B full-FT memory wall
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3()
        print("[tbench] liger kernel applied (qwen3 fused ops + FLCE)", flush=True)

    if args.lora_rank > 0:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=tbench_reward,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation") if config.eval_strategy != "no" else None,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
