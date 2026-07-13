"""SearchR1-mini GRPO training on TRL 1.7 with a custom multi-turn rollout_func.

Architecture:
  - GRPOTrainer + colocated vLLM (vllm_mode="colocate", NO sleep mode — broken
    with trl 1.7 + vllm 0.23 on this box).
  - rollout_func (searchr1_rollout.py) drives the SearchR1 protocol:
    <search>q</search> -> mini_retrieval_server /retrieve -> <information>docs>
    injected as env_mask=0 tokens (excluded from the GRPO loss), up to
    --max_turns rounds, final <answer>...</answer>.
  - reward = qa_em (exact match vs ground_truth["target"], qa_em.py).

Prereqs: dataset built by build_searchr1_trl.py; mini retriever running, e.g.
  python ../skyrl_terminal/mini_retrieval_server.py \
      --corpus ~/data/searchr1_trl/corpus.jsonl --port 8000 --device cuda

Launch (see run_searchr1_trl.sh):
  CUDA_VISIBLE_DEVICES=0 python trl_searchr1.py
"""

import argparse
import os

from datasets import load_dataset

from qa_em import qa_em_reward
from retrieval_client import make_retrieve_fn
from searchr1_rollout import STOP_STRINGS, make_rollout_func

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")  # rollout_func is experimental in trl 1.7
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    ap = argparse.ArgumentParser(description="SearchR1-mini GRPO on TRL")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/data/searchr1_trl"))
    ap.add_argument("--output_dir", default=os.path.expanduser("~/ckpts/searchr1_trl"))
    # retriever / rollout
    ap.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--per_turn_max_tokens", type=int, default=500)  # SkyRL max_generate_length
    ap.add_argument("--max_completion_length", type=int, default=2048)  # total budget incl. <information>
    # GRPO
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 disables ref model)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    # LoRA
    ap.add_argument("--lora_rank", type=int, default=32, help="0 = full finetuning")
    ap.add_argument("--optim", default="adamw_torch",
                    help="paged_adamw_8bit for memory-safe full-FT at 4B")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    # vLLM
    ap.add_argument("--no_vllm", action="store_true", help="HF generate fallback (CPU debug only)")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    ap.add_argument("--vllm_max_model_len", type=int, default=6144)
    # eval / saving
    ap.add_argument("--eval_steps", type=int, default=0, help="0 = no eval during training")
    ap.add_argument("--save_steps", type=int, default=0, help="0 = no checkpoints")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--run_name", default="searchr1_trl")
    return ap.parse_args()


def main():
    args = parse_args()
    # Import trl lazily so --help works without initializing CUDA-adjacent deps.
    from trl import GRPOConfig, GRPOTrainer

    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path = os.path.join(args.data_dir, "validation.parquet")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} missing — run build_searchr1_trl.py first")
    data_files = {"train": train_path}
    if os.path.exists(val_path):
        data_files["validation"] = val_path
    ds = load_dataset("parquet", data_files=data_files)

    config = GRPOConfig(
        save_only_model=True,   # storage class-fix 07-10
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
        # lengths: prompts are tokenized untruncated inside the rollout;
        # max_completion_length is the TOTAL multi-turn completion budget.
        max_completion_length=args.max_completion_length,
        # sampling
        temperature=args.temperature,
        top_p=args.top_p,
        # SearchR1 protocol: stop at tag closes, keep the tags in the output so
        # the rollout can parse them and train on them.
        generation_kwargs={"stop": list(STOP_STRINGS), "include_stop_str_in_output": True},
        # vLLM colocate; sleep mode MUST stay off (broken with trl1.7+vllm0.23)
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=False,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_len,
        # keep truncated (budget-exhausted) rollouts in the loss
        mask_truncated_completions=False,
        # logging / saving / eval
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps or 500,
        eval_strategy="steps" if (args.eval_steps > 0 and "validation" in ds) else "no",
        eval_steps=args.eval_steps or 500,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,
        report_to=args.report_to,
    )

    peft_config = None
    if args.lora_rank > 0:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    rollout_func = make_rollout_func(
        retrieve_fn=make_retrieve_fn(url=args.search_url, topk=args.topk),
        max_turns=args.max_turns,
        per_turn_max_tokens=args.per_turn_max_tokens,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=qa_em_reward,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation") if config.eval_strategy != "no" else None,
        peft_config=peft_config,
        rollout_func=rollout_func,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
