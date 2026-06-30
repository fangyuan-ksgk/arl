"""
GRPO on MATH (Hendrycks) with Qwen3-0.6B + vLLM

Single-GPU:
    python script/grpo_math.py                        # smoke test (20 steps)
    python script/grpo_math.py --max_steps -1         # full run (1 epoch)

Server mode (2-GPU):
    GPU 0: trl vllm-serve --model Qwen/Qwen3-0.6B --port 8000
    GPU 1: CUDA_VISIBLE_DEVICES=1 python script/grpo_math.py \
               --use_vllm --vllm_mode server --vllm_server_port 8000 --max_steps 200

[TBD]. Dr.GRPO | Dr.GRPO + MBE velocity reward + virtual max reward 
"""

import argparse
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
from trl import GRPOTrainer, GRPOConfig

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


class SaveAtStepsCallback(TrainerCallback):
    """Force a checkpoint save at an explicit set of global steps (e.g. the merge step P)."""
    def __init__(self, steps):
        self.steps = set(int(s) for s in steps)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step in self.steps:
            control.should_save = True
        return control


def load_math_any():
    """Load Hendrycks MATH, trying known HF mirrors (the original repo is often gated)."""
    last = None
    # FULL official Hendrycks MATH = 7500 train / 5000 test (disjoint). DLG mirror first; the nlile
    # mirror has only a 500-test (MATH-500) and a non-standard 12000-train, so it goes LAST.
    for repo, cfg in [("DigitalLearningGmbH/MATH-lighteval", None),
                      ("hendrycks/competition_math", None),
                      ("EleutherAI/hendrycks_math", "default"),
                      ("nlile/hendrycks-MATH-benchmark", None)]:
        try:
            return load_dataset(repo) if cfg is None else load_dataset(repo, cfg)
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(f"Could not load any MATH mirror: {last}")


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
        rewards.append(0.2 if has_boxed else 0.0)   # R1d: format reward magnitude 0.2
    return rewards


class VirtualRolloutGRPOTrainer(GRPOTrainer):
    """GRPOTrainer + virtual-rollout advantage shaping (anti reward-hacking under length penalties /
    MBE-velocity rewards). Inserts one no-gradient virtual rollout into each GRPO group BEFORE the
    z-score so the real rollouts are scored against an ideal answer rather than each other — this
    is the necessary complement to the MBE-velocity reward (which otherwise has a length-only contrast
    to hack). No-op when virtual_rollout_mode is None/'none'. Ported verbatim from
    grpo_gsm8k.FastEvalGRPOTrainer (math math identical; see src/arsenal.virtual_rollout_advantages)."""

    def _calculate_rewards(self, *args, **kwargs):
        rpf = super()._calculate_rewards(*args, **kwargs)
        self._last_rewards_per_func = rpf            # (B_gathered, n_funcs)
        return rpf

    def _local_rewards_per_func(self, out):
        rpf = getattr(self, "_last_rewards_per_func", None)
        adv = out.get("advantages")
        if rpf is None or adv is None:
            return None
        Bp = adv.shape[0]
        lo = self.accelerator.process_index * Bp     # same slice TRL applies
        return rpf[lo:lo + Bp]

    def _virtual_rollout_advantages(self, out, local):
        from src.arsenal import virtual_rollout_advantages
        adv = out.get("advantages")
        names = self.reward_func_names
        rewards = local.sum(dim=1)                    # total reward (sum over funcs)
        if "correctness_reward" in names:
            corrects = (local[:, names.index("correctness_reward")] == 1.0)
        else:
            corrects = torch.zeros_like(rewards, dtype=torch.bool)
        return virtual_rollout_advantages(
            rewards, corrects, self.num_generations,
            max_reward=getattr(self, "virtual_max_reward", 1.2),
            mode=self.virtual_rollout_mode,
        ).to(adv) # here we are assuming 'mbe velocity reward' doesn't exist, otherwise we should have 1.4 as max reward

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if getattr(self, "virtual_rollout_mode", None) and self.model.training:
            local = self._local_rewards_per_func(out)
            if local is not None and out.get("advantages") is not None:
                out["advantages"] = self._virtual_rollout_advantages(out, local)
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


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
    # Show level / subject distribution if the columns exist (mirror schemas vary:
    # original MATH has 'type'; the EleutherAI/nlile mirrors use 'subject').
    from collections import Counter
    cols = train_dataset.column_names
    if "level" in cols:
        print(f"  Levels: {dict(sorted(Counter(train_dataset['level']).items(), key=lambda x: str(x[0])))}")
    subj_col = "type" if "type" in cols else ("subject" if "subject" in cols else None)
    if subj_col:
        print(f"  {subj_col.capitalize()}s: {dict(sorted(Counter(train_dataset[subj_col]).items()))}")
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
    parser.add_argument("--loss_type", type=str, default="dapo",
                        choices=["grpo", "dapo", "bnpo", "dr_grpo"],
                        help="'dr_grpo' = Dr.GRPO (constant-length normalization; unbiased length).")
    parser.add_argument("--scale_rewards", type=str, default="group",
                        choices=["group", "batch", "none"],
                        help="'none' = Dr.GRPO (no advantage std division).")
    parser.add_argument("--mask_truncated_completions", action="store_true",
                        help="Exclude truncated completions from the loss (anti length-collapse).")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="Use 'constant' for branch-train-merge rounds.")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Controls train data ordering (branch identity).")
    parser.add_argument("--subject", type=str, default=None,
                        help="Train only on this MATH subject (e.g. 'Algebra'). None = all subjects "
                             "(generalist). Enables domain-specialized mixture-of-LoRA experts.")
    parser.add_argument("--save_steps_list", type=str, default=None,
                        help="Comma-separated steps to force-save a checkpoint (e.g. the merge step P).")
    parser.add_argument("--max_steps", type=int, default=20, help="-1 for full epoch")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--no_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["colocate", "server"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--vllm_max_model_len", type=int, default=0,
                        help="Cap vLLM context (prompt+gen) to bound KV-cache memory. 0=use model native. "
                             "Set for big models (e.g. 8B) so KV fits a small colocate pool.")
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
    # MBE velocity reward (trajectory-level diversity-growth; ported from grpo_gsm8k). R1c: the base
    # MBE reward was removed — only the velocity reward remains. Opt-in; enable with --mbe_velocity_reward.
    parser.add_argument("--mbe_velocity_reward", action=argparse.BooleanOptionalAction, default=False,
                        help="Add length-normalised MBE velocity reward (clip((trace[-1]-trace[0])/log(min(T,D)),±clip)/scale)")
    parser.add_argument("--mbe_velocity_scale", type=float, default=5.0)   # R1d: clip(1.0)/scale=5 -> max |reward|=0.2
    parser.add_argument("--mbe_velocity_clip", type=float, default=1.0)
    parser.add_argument("--mbe_velocity_stride", type=int, default=8)
    parser.add_argument("--mbe_velocity_layers", type=str, default="-1")
    parser.add_argument("--mbe_velocity_mode", type=str, default="trajectory",
                        choices=["trajectory", "rollercoaster"])
    # Virtual-rollout advantage shaping — the NECESSARY complement to MBE velocity (anti reward-hacking).
    parser.add_argument("--virtual_rollout", type=str, default="none",
                        choices=["none", "insert_max", "insert_min", "insert_max_min",
                                 "insert_max_all_incorrect", "insert_max_mixed"],
                        help="Insert a virtual max-reward rollout per GRPO group before z-score "
                             "(insert_max recommended; mitigates length/velocity reward hacking).")
    parser.add_argument("--virtual_max_reward", type=float, default=1.2,
                        help="Reward value of the inserted virtual rollout (sit above the real max).")
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    train_dataset, test_dataset = load_math_dataset()
    if args.subject:
        subj_col = "subject" if "subject" in train_dataset.column_names else "type"
        before = len(train_dataset)
        train_dataset = train_dataset.filter(lambda ex: ex[subj_col] == args.subject)
        print(f"Subject filter '{args.subject}': {before} -> {len(train_dataset)} train examples")
        assert len(train_dataset) > 0, f"no train examples for subject {args.subject}"

    config_kwargs = dict(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        loss_type=args.loss_type,
        scale_rewards=(False if args.scale_rewards == "none" else args.scale_rewards),
        mask_truncated_completions=args.mask_truncated_completions,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_strategy,
        report_to=args.report_to,
        seed=args.seed,
    )
    if args.max_steps > 0:
        config_kwargs["max_steps"] = args.max_steps
    if not args.no_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "colocate":
            config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
            if args.vllm_max_model_len > 0:
                config_kwargs["vllm_max_model_length"] = args.vllm_max_model_len
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
    reward_funcs = [correctness_reward, format_reward]   # R1c: base MBE reward removed (velocity only)

    # MBE velocity reward (length-normalised endpoint diff of the growing-prefix MBE trace).
    mbe_velo_reward_obj = None
    if args.mbe_velocity_reward:
        from src.mbe_reward import MBEVeloReward
        velo_layers = [int(x) for x in args.mbe_velocity_layers.split(",") if x.strip()]
        mbe_velo_reward_obj = MBEVeloReward(
            AutoTokenizer.from_pretrained(args.model),
            layers=velo_layers, stride=args.mbe_velocity_stride,
            scale=args.mbe_velocity_scale, clip=args.mbe_velocity_clip, mode=args.mbe_velocity_mode,
        )
        reward_funcs.append(mbe_velo_reward_obj)
        print(f"MBE velocity reward enabled: mode={args.mbe_velocity_mode}, scale={args.mbe_velocity_scale}, "
              f"clip=±{args.mbe_velocity_clip}, stride={args.mbe_velocity_stride}, layers={velo_layers}")

    # Eval dataset
    eval_dataset = None
    if args.eval_steps > 0:
        eval_dataset = test_dataset
        if args.eval_samples is not None:
            eval_dataset = test_dataset.select(range(min(args.eval_samples, len(test_dataset))))
        print(f"Eval enabled: {len(eval_dataset)} samples every {args.eval_steps} steps")

    callbacks = []
    if args.save_steps_list:
        save_steps = [int(s) for s in args.save_steps_list.split(",") if s.strip()]
        callbacks.append(SaveAtStepsCallback(save_steps))
        print(f"Forced checkpoint saves at steps: {sorted(set(save_steps))} → {args.output_dir}/checkpoint-<step>")

    trainer = VirtualRolloutGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    # Virtual-rollout advantage shaping (no-op unless --virtual_rollout != none).
    trainer.virtual_rollout_mode = None if args.virtual_rollout == "none" else args.virtual_rollout
    trainer.virtual_max_reward = args.virtual_max_reward
    if trainer.virtual_rollout_mode:
        print(f"Virtual-rollout advantage shaping: mode={trainer.virtual_rollout_mode}, "
              f"max_reward={trainer.virtual_max_reward}")

    if mbe_velo_reward_obj is not None:
        mbe_velo_reward_obj.set_model(trainer.model)

    trainer.train()
    if args.save_strategy != "no":
        trainer.save_model(args.output_dir)
        print(f"Training complete. Model saved to {args.output_dir}")
    else:
        print("Training complete. (--save_strategy=no)")


if __name__ == "__main__":
    main()
