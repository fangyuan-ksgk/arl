"""
IBRL (GRPO + MBE) on Wordle hacking-game environment

Single-GPU:
    python scripts/ibrl_wordle.py                                    # 16 episodes, 1 epoch
    python scripts/ibrl_wordle.py --n_episodes 64 --lambda_mbe 0.05

Multi-GPU:
    accelerate launch --num_processes 4 scripts/ibrl_wordle.py --n_episodes 64 --lambda_mbe 0.05
    accelerate launch --config_file scripts/configs/multi_gpu.yaml scripts/ibrl_wordle.py

DeepSpeed ZeRO-2:
    accelerate launch --config_file scripts/configs/deepspeed_zero2.yaml scripts/ibrl_wordle.py
"""

import argparse
import os
import sys
import torch
import anthropic
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import PartialState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load API key from api.txt if ANTHROPIC_API_KEY is not already set
if not os.environ.get("ANTHROPIC_API_KEY"):
    _api_txt = os.path.join(os.path.dirname(__file__), "..", "api.txt")
    if os.path.exists(_api_txt):
        for line in open(_api_txt):
            line = line.strip()
            if line and "=" in line:
                _key, _val = line.split("=", 1)
                os.environ.setdefault("ANTHROPIC_API_KEY", _val.strip('" '))

from src.ibrl import IBRLTrainer, IBRLConfig
from src.wordle_env import (
    WordleEnv, EnvConfig, extract_guess,
    batch_rollout, collect_training_data, DEFAULT_WORD_BANK,
)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


# ---------------------------------------------------------------------------
# Oracle (Claude) & Learner (Qwen)
# ---------------------------------------------------------------------------
def make_oracle_fn(model_name: str):
    client = anthropic.Anthropic()

    def oracle_fn(system_prompt: str, question: str) -> str:
        resp = client.messages.create(
            model=model_name,
            max_tokens=64,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return resp.content[0].text

    return oracle_fn


def make_learner_fn(model, tokenizer):
    def learner_fn(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=256, temperature=0.7, do_sample=True,
                top_p=0.9, top_k=50,
            )
        return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return learner_fn


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------
def wordle_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        comp_text = completion[0]["content"] if isinstance(completion, list) else completion

        if "what is the secret word" in prompt.lower() or "reply with your guess" in prompt.lower():
            guess = extract_guess(comp_text)
            if guess in DEFAULT_WORD_BANK:
                rewards.append(0.5)
            elif len(guess) > 2:
                rewards.append(0.1)
            else:
                rewards.append(-0.5)
        else:
            text = comp_text.strip()
            score = 0.0
            if "?" in text:
                score += 0.3
            if len(text) < 200:
                score += 0.2
            if len(text) < 80:
                score += 0.2
            if "<think>" in text.lower():
                score -= 0.3
            rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Build training dataset from rollouts
# ---------------------------------------------------------------------------
def build_dataset(env, learner_fn, model, tokenizer, n_episodes: int):
    print(f"Collecting {n_episodes} episodes...")
    states = batch_rollout(env, learner_fn=learner_fn, model=model, tokenizer=tokenizer, batch_size=n_episodes)
    samples = collect_training_data(states)
    prompts = [s["prompt"] for s in samples]

    n_questions = sum(1 for s in samples if s["type"] == "question")
    n_guesses = sum(1 for s in samples if s["type"] == "guess")
    avg_reward = sum(s["reward"] for s in samples) / len(samples)
    print(f"  {len(prompts)} prompts ({n_questions} questions, {n_guesses} guesses), avg reward: {avg_reward:.4f}")

    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IBRL (GRPO+MBE) on Wordle")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--oracle_model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--output_dir", type=str, default="ibrl_wordle_output")
    parser.add_argument("--n_episodes", type=int, default=16)
    parser.add_argument("--max_questions", type=int, default=5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--report_to", type=str, default="none")
    # IBRL-specific
    parser.add_argument("--lambda_mbe", type=float, default=0.01)
    parser.add_argument("--mbe_layer", type=int, default=-1)
    parser.add_argument("--mbe_patch_size", type=int, default=8)
    args = parser.parse_args()

    # Load learner model (no device_map — Accelerate handles placement)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    learner_llm = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    )

    # Build dataset from rollouts (rank 0 only to avoid duplicate API calls)
    state = PartialState()
    if state.is_main_process:
        oracle_fn = make_oracle_fn(args.oracle_model)
        env_config = EnvConfig(max_questions=args.max_questions, exact_match_bonus=1.0, log_prob_weight=0.1)
        env = WordleEnv(oracle_fn=oracle_fn, config=env_config)
        rollout_model = learner_llm.to("cuda")
        learner_fn = make_learner_fn(rollout_model, tokenizer)
        train_dataset = build_dataset(env, learner_fn, rollout_model, tokenizer, args.n_episodes)
        train_dataset.save_to_disk("/tmp/_wordle_ibrl_dataset")
        rollout_model.cpu()
        torch.cuda.empty_cache()
    state.wait_for_everyone()
    if not state.is_main_process:
        from datasets import load_from_disk
        train_dataset = load_from_disk("/tmp/_wordle_ibrl_dataset")

    # Configure IBRL
    config = IBRLConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        report_to=args.report_to,
        # IBRL fields
        lambda_mbe=args.lambda_mbe,
        mbe_layer=args.mbe_layer,
        mbe_patch_size=args.mbe_patch_size,
    )

    trainer = IBRLTrainer(
        model=learner_llm,
        reward_funcs=wordle_reward_fn,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print(f"Training: {len(train_dataset)} prompts, {args.num_generations} generations each")
    print(f"IBRL: lambda_mbe={args.lambda_mbe}, mbe_layer={args.mbe_layer}, patch_size={args.mbe_patch_size}")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
