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
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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


def _completion_text(completion) -> str:
    """Normalise completion to plain text regardless of TRL format.

    TRL passes completions as list-of-dicts when the prompt is a message list,
    but as a plain string when the prompt is already a pre-formatted string
    (e.g. from PrefixAugmentedDataset).
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
        return str(first)
    return str(completion)


def correctness_reward(completions, gold_answer, **kwargs):
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        text = _completion_text(completion)
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
        text = _completion_text(completion)
        has_format = bool(re.search(r"####\s*[\d,\.\-]+", text))
        rewards.append(0.5 if has_format else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Rollout logging
# ---------------------------------------------------------------------------
# Reward-fn shim that dumps rollouts to JSONL. Returns 0.0 reward (no-op for
# the optimizer). Routed by `EvalFlagCallback` to either the train log or the
# eval log via the `in_eval` flag. Mirrors the pattern in
# `script/run_game24_one.py` (Game24 driver) but adapted to GSM8K fields.
class GSM8KRolloutLogger:
    __name__ = "rollout_logger"

    def __init__(self, train_path: Path, eval_path: Path, tokenizer):
        self.train_path = train_path
        self.eval_path  = eval_path
        self.tokenizer  = tokenizer
        self.in_eval    = False
        self.train_step = 0
        self.eval_step  = 0
        # Trainer.state.global_step at the moment eval was triggered.
        # Stamped by the driver's EvalFlagCallback; -1 before first step.
        self.global_step = -1

    def __call__(self, completions, gold_answer, **_):
        path = self.eval_path if self.in_eval else self.train_path
        step = self.eval_step if self.in_eval else self.train_step
        with path.open("a") as f:
            for i, (c, gold) in enumerate(zip(completions, gold_answer)):
                text      = _completion_text(c)
                predicted = extract_answer_from_completion(text)
                try:
                    correct = float(predicted) == float(gold)
                except (ValueError, TypeError):
                    correct = False
                n_tok = len(self.tokenizer.encode(text, add_special_tokens=False))
                # Rationale = everything before the answer marker (####). If no
                # marker, the whole completion is treated as rationale.
                m_ans = re.search(r"####", text)
                cot_text  = text[: m_ans.start()] if m_ans else text
                n_cot_tok = len(self.tokenizer.encode(cot_text, add_special_tokens=False))
                f.write(json.dumps({
                    "step":              step,
                    "idx":               i,
                    "gold_answer":       str(gold),
                    "predicted_answer":  predicted,
                    "completion":        text,
                    "correct":           bool(correct),
                    "n_tokens":          int(n_tok),
                    "n_cot_tokens":      int(n_cot_tok),
                    "has_answer_marker": bool(m_ans),
                    "split":             "eval" if self.in_eval else "train",
                    "global_step":       int(self.global_step),
                }) + "\n")
        if self.in_eval:
            self.eval_step  += 1
        else:
            self.train_step += 1
        return [0.0] * len(completions)


class FastEvalGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with a fast eval path.

    TRL's default `prediction_step` runs `compute_loss`, which triggers two
    full-model forwards per eval batch (one for `old_per_token_logps` inside
    `_generate_and_score_completions`, one for the policy logp/entropy inside
    `_compute_loss`). At vocab≈151k and `max_completion_length` long enough
    to matter, that local forward dominates wall-clock; evaluating the full
    GSM8K test set (1319 samples × `num_generations` rollouts) takes ~2h on a
    single GPU even with vLLM driving generation.

    For our use-case eval only needs (a) completions and (b) reward-function
    callbacks (so the rollout logger fires). We don't backprop, don't need
    importance-sampling ratios, and we report no loss metric. So skip both
    forwards entirely: generate via vLLM, call reward funcs, return 0.

    `num_generations` still works because TRL's eval sampler
    (`_get_eval_sampler`) already repeats each prompt `num_generations_eval`
    times, and `_generate` returns one completion per repeated prompt.
    """

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        prompts = [x["prompt"] for x in inputs]
        (
            _prompt_ids_list,
            completion_ids_list,
            _tool_mask_list,
            completions,
            _num_items_in_batch,
            _sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Merge rollout_func extras into inputs the same way the parent does
        # before _calculate_rewards, so any reward fn that reads extras still
        # works on the fast eval path.
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Fires reward funcs (including the rollout logger shim).
        self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        loss = torch.zeros((), device=self.accelerator.device)
        return loss, None, None


class EvalFlagCallback(TrainerCallback):
    """Routes reward-fn calls to the right rollout log around eval loops.

    `on_prediction_step` fires per eval batch → flags `in_eval=True`.
    `on_step_begin` fires at the start of each training step → clears it.
    That ordering routes every reward-fn call within an eval pass to
    `eval_rollout.jsonl`, and everything else to `rollouts.jsonl`.
    """
    def __init__(self, logger: GSM8KRolloutLogger):
        self.logger = logger

    def on_prediction_step(self, args, state, control, **kw):
        self.logger.in_eval     = True
        self.logger.global_step = state.global_step

    def on_step_begin(self, args, state, control, **kw):
        self.logger.in_eval     = False
        self.logger.global_step = state.global_step

    def on_evaluate(self, args, state, control, **kw):
        self.logger.in_eval = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")

    def extract_gold_answer(answer_text: str) -> str:
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
        if numbers:
            return numbers[-1].replace(",", "")
        return ""

    answer_format_instruction = (
        "Solve the problem step by step inside <think>...</think>. "
        "After </think>, give a brief final explanation and end your response "
        "with a line of the exact form:\n#### <number>\n"
        "where <number> is the final numeric answer with no units, no commas, "
        "and no extra text."
    )

    def format_example(example):
        user_content = (
            f"{example['question']}\n\n{answer_format_instruction} /think"
        )
        example["prompt"] = [{"role": "user", "content": user_content}]
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
    # MBE velocity reward (trajectory-level).
    # raw_velocity = MBE(hidden_states[:prompt_len+T_comp]) − MBE(hidden_states[:prompt_len])
    #              = MBE(query + response)             − MBE(query)
    # Pass --no-mbe_velocity_reward to disable.
    parser.add_argument("--mbe_velocity_reward",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Add length-normalised MBE velocity reward: clip((trace[-1]-trace[0])/log(min(T,D)), ±clip)/scale")
    parser.add_argument("--mbe_velocity_scale", type=float, default=4.0,
                        help="MBE velocity reward denominator (default 4.0 → max |reward| ≈ 0.25)")
    parser.add_argument("--mbe_velocity_clip", type=float, default=1.0,
                        help="Two-sided clip on length-normalised velocity (∈ ~[-1, +1])")
    parser.add_argument("--mbe_velocity_stride", type=int, default=8,
                        help="Sampling stride for the running-prefix MBE trace "
                             "(every `stride` tokens we record MBE of all tokens so far)")
    parser.add_argument("--mbe_velocity_layers", type=str, default="-1",
                        help="Comma-separated hidden-layer indices for MBE velocity "
                             "(default '-1' = last layer only; e.g. '-1,-2' averages "
                             "the top two layers, '0,-1' averages embedding + final).")
    parser.add_argument("--mbe_velocity_mode", type=str, default="trajectory",
                        choices=["trajectory", "rollercoaster"],
                        help="trajectory: raw_v = MBE(query+response) - MBE(query). "
                             "rollercoaster: raw_v = sum of positive per-stride MBE jumps "
                             "(rewards continual diversity growth, ignores drawdowns).")
    # InvLogLength baseline: same denominator as MBE velocity, raw_v ablated to 1.
    # Use to isolate the length-pressure component of MBE velocity (see analysis 2026-05-27).
    parser.add_argument("--inv_log_length_reward", action="store_true",
                        help="Pure 1/log(min(T_comp, D)) baseline reward — "
                             "the denominator of MBE velocity with raw_v=1. "
                             "Sweep against MBE velocity to test whether the "
                             "diversity numerator adds signal beyond length normalisation.")
    parser.add_argument("--inv_log_length_scale", type=float, default=4.0,
                        help="InvLogLength reward denominator (default 4.0 matches MBE velocity)")
    parser.add_argument("--inv_log_length_clip", type=float, default=1.0,
                        help="Two-sided clip on 1/log(min(T_comp, D))")
    # Correctness-gated InvLogLength: sign of the reward flipped by whether the
    # rollout's answer matches the gold answer.
    # Operationalises the asymmetric shaping hypothesis (analysis note
    # 2026-05-27): longer CoT for failed cases, shorter for successful ones.
    parser.add_argument("--gated_inv_log_length_reward", action="store_true",
                        help="Add correctness-gated InvLogLength reward: "
                             "reward = clip(1/log(min(T,D)),±clip) / scale_correct "
                             "if correct, else / scale_incorrect. With "
                             "scale_correct=+0.1, scale_incorrect=-0.1: correct "
                             "rollouts get pushed shorter, incorrect ones longer.")
    parser.add_argument("--gated_inv_log_length_scale_correct",   type=float, default=0.1,
                        help="Reward divisor for correct rollouts (default 0.1 → w_correct=+10)")
    parser.add_argument("--gated_inv_log_length_scale_incorrect", type=float, default=-0.1,
                        help="Reward divisor for incorrect rollouts (default -0.1 → w_incorrect=-10). "
                             "Set to 0 to disable the incorrect arm (= pure 'reward short when right').")
    parser.add_argument("--gated_inv_log_length_clip",            type=float, default=1.0,
                        help="Two-sided clip on 1/log(min(T_comp, D))")
    # Rationale-internal velocity rewards. Per-token velocity X(o_{t+1}) − X(o_t)
    # summed (telescoping) into endpoint diff X(o_last) − X(o_first), then
    # length-normalised by log(min(T_comp, D)). Negative scale flips sign.
    parser.add_argument("--entropy_velocity_reward", action="store_true",
                        help="Rationale-internal entropy velocity: "
                             "clip((H(o_last) − H(o_first)) / log(min(T,D)), ±clip) / scale")
    parser.add_argument("--entropy_velocity_scale", type=float, default=4.0)
    parser.add_argument("--entropy_velocity_clip",  type=float, default=1.0)
    parser.add_argument("--entropy_velocity_marker", type=str, default="####",
                        help="Rationale/answer separator for entropy velocity (defaults to GSM8K convention).")
    parser.add_argument("--entropy_velocity_aggregation", type=str, default="rollercoaster",
                        choices=["rollercoaster", "trajectory"],
                        help="How to aggregate per-token entropy deltas over the rationale. "
                             "'rollercoaster' (default) sums positive jumps only; "
                             "'trajectory' uses the endpoint diff H(o_last) − H(o_first).")
    parser.add_argument("--perplexity_velocity_reward", action="store_true",
                        help="Rationale-internal perplexity (NLL) velocity: "
                             "clip((NLL(o_last) − NLL(o_first)) / log(min(T,D)), ±clip) / scale")
    parser.add_argument("--perplexity_velocity_scale", type=float, default=4.0)
    parser.add_argument("--perplexity_velocity_clip",  type=float, default=1.0)
    parser.add_argument("--perplexity_velocity_marker", type=str, default="####",
                        help="Rationale/answer separator for perplexity velocity (defaults to GSM8K convention).")
    parser.add_argument("--perplexity_velocity_aggregation", type=str, default="rollercoaster",
                        choices=["rollercoaster", "trajectory"],
                        help="How to aggregate per-token NLL deltas over the rationale. "
                             "'rollercoaster' (default) sums positive jumps only; "
                             "'trajectory' uses the endpoint diff NLL(o_last) − NLL(o_first).")
    # Predictive velocity — two forward passes per rollout: log p(a|q,o) − log p(a|q).
    # Splits completion on `--predictive_marker` (default "####", GSM8K convention).
    parser.add_argument("--predictive_velocity_reward", action="store_true",
                        help="Add length-normalised predictive velocity reward: "
                             "clip((log p(a|q,o) − log p(a|q)) / log(min(T,D)), ±clip) / scale")
    parser.add_argument("--predictive_velocity_scale", type=float, default=4.0)
    parser.add_argument("--predictive_velocity_clip",  type=float, default=1.0)
    parser.add_argument("--predictive_marker", type=str, default="####",
                        help="Substring that separates rationale from answer in the completion.")
    # Entropy density — phase contrast "reason hard, then commit".
    # raw_v = mean(H over rationale)  −  mean(H over answer)
    # High reward at positive scale = rationale is uncertain, answer is
    # decisive. Length-normalised by log(min(T_comp, D)).
    parser.add_argument("--entropy_density_reward", action="store_true",
                        help="Add entropy density contrast reward: "
                             "clip((mean H_o − mean H_a) / log(min(T,D)), ±clip) / scale")
    parser.add_argument("--entropy_density_scale",    type=float, default=4.0)
    parser.add_argument("--entropy_density_clip",     type=float, default=1.0)
    parser.add_argument("--entropy_density_marker", type=str, default="####",
                        help="Rationale/answer separator (defaults to GSM8K convention).")
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
        # Pre-training baseline eval at global_step=0.
        config_kwargs["eval_on_start"] = True
        # Eval forward materialises [batch, seq, vocab] logits and intermediate
        # softmax tensors. Pin per-device eval batch to one generation group —
        # the smallest value that satisfies TRL's eval_batch % num_generations
        # == 0 requirement and keeps eval-side memory comparable to train.
        config_kwargs["per_device_eval_batch_size"] = args.num_generations

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

    # Rollout recorder: lightweight text-only logger, zero forward-pass overhead.
    # Run script/compute_mbe.py after training to get full MBE/CE traces.
    mbe_logger = None
    if args.mbe_log:
        from src.mbe_logger import RolloutRecorder
        tokenizer_for_log = AutoTokenizer.from_pretrained(args.model)
        log_path = os.path.join(args.output_dir, "rollouts.jsonl")
        mbe_logger = RolloutRecorder(
            tokenizer_for_log,
            log_path=log_path,
            log_steps=args.mbe_log_steps,
            sample_k=args.mbe_log_sample_k,
        )
        print(f"Rollout recorder enabled → {log_path}  "
              f"(every {args.mbe_log_steps} steps, {args.mbe_log_sample_k} samples/step)"
              f"\nRun script/compute_mbe.py after training to compute MBE traces.")

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

    # MBE velocity reward (length-normalised endpoint diff of growing-prefix MBE trace).
    # Estimator chain: src/mbe_reward.py:compute_mbe_running_trace (kernel-trick K=h hᵀ
    #   + 2D cumsum, identical math to mbe_reverse_gram, no D×D matrices materialised).
    mbe_velo_reward_obj = None
    if args.mbe_velocity_reward:
        from src.mbe_reward import MBEVeloReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        velo_layers = [int(x) for x in args.mbe_velocity_layers.split(",") if x.strip()]
        mbe_velo_reward_obj = MBEVeloReward(
            tokenizer,
            layers=velo_layers,
            stride=args.mbe_velocity_stride,
            scale=args.mbe_velocity_scale,
            clip=args.mbe_velocity_clip,
            mode=args.mbe_velocity_mode,
        )
        reward_funcs.append(mbe_velo_reward_obj)
        print(f"MBE velocity reward enabled: mode={args.mbe_velocity_mode}, "
              f"scale={args.mbe_velocity_scale}, clip=±{args.mbe_velocity_clip}, "
              f"stride={args.mbe_velocity_stride}, layers={velo_layers}")

    # InvLogLength baseline (no MBE forward pass; just tokenize + log).
    inv_log_len_reward_obj = None
    if args.inv_log_length_reward:
        from src.mbe_reward import InvLogLengthReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        inv_log_len_reward_obj = InvLogLengthReward(
            tokenizer,
            stride=args.mbe_velocity_stride,             # share the guard threshold
            scale=args.inv_log_length_scale,
            clip=args.inv_log_length_clip,
        )
        reward_funcs.append(inv_log_len_reward_obj)
        print(f"InvLogLength reward enabled: scale={args.inv_log_length_scale}, "
              f"clip=±{args.inv_log_length_clip}, stride={args.mbe_velocity_stride}")

    # Correctness-gated InvLogLength (asymmetric shaping: sign flipped by correctness).
    gated_inv_log_len_obj = None
    if args.gated_inv_log_length_reward:
        from src.mbe_reward import CorrectnessGatedInvLogLengthReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        sc_inc = args.gated_inv_log_length_scale_incorrect
        gated_inv_log_len_obj = CorrectnessGatedInvLogLengthReward(
            tokenizer,
            stride=args.mbe_velocity_stride,
            scale_correct=args.gated_inv_log_length_scale_correct,
            scale_incorrect=(None if sc_inc == 0.0 else sc_inc),
            clip=args.gated_inv_log_length_clip,
        )
        reward_funcs.append(gated_inv_log_len_obj)
        print(f"Gated InvLogLength reward enabled: "
              f"scale_correct={args.gated_inv_log_length_scale_correct}, "
              f"scale_incorrect={sc_inc}, "
              f"clip=±{args.gated_inv_log_length_clip}, stride={args.mbe_velocity_stride}")

    # Entropy / perplexity / predictive velocity rewards. All three share the
    # MBE velocity guard threshold (--mbe_velocity_stride) so guard-failed
    # rollouts behave consistently across rewards.
    entropy_velo_obj = None
    if args.entropy_velocity_reward:
        from src.mbe_reward import EntropyVeloReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        entropy_velo_obj = EntropyVeloReward(
            tokenizer,
            stride=args.mbe_velocity_stride,
            scale=args.entropy_velocity_scale,
            clip=args.entropy_velocity_clip,
            marker=args.entropy_velocity_marker,
            aggregation=args.entropy_velocity_aggregation,
        )
        reward_funcs.append(entropy_velo_obj)
        print(f"Entropy velocity reward enabled: scale={args.entropy_velocity_scale}, "
              f"clip=±{args.entropy_velocity_clip}, marker='{args.entropy_velocity_marker}', "
              f"aggregation='{args.entropy_velocity_aggregation}'")

    perplexity_velo_obj = None
    if args.perplexity_velocity_reward:
        from src.mbe_reward import PerplexityVeloReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        perplexity_velo_obj = PerplexityVeloReward(
            tokenizer,
            stride=args.mbe_velocity_stride,
            scale=args.perplexity_velocity_scale,
            clip=args.perplexity_velocity_clip,
            marker=args.perplexity_velocity_marker,
            aggregation=args.perplexity_velocity_aggregation,
        )
        reward_funcs.append(perplexity_velo_obj)
        print(f"Perplexity velocity reward enabled: scale={args.perplexity_velocity_scale}, "
              f"clip=±{args.perplexity_velocity_clip}, marker='{args.perplexity_velocity_marker}', "
              f"aggregation='{args.perplexity_velocity_aggregation}'")

    entropy_density_obj = None
    if args.entropy_density_reward:
        from src.mbe_reward import EntropyDensityReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        entropy_density_obj = EntropyDensityReward(
            tokenizer,
            stride=args.mbe_velocity_stride,
            scale=args.entropy_density_scale,
            clip=args.entropy_density_clip,
            marker=args.entropy_density_marker,
        )
        reward_funcs.append(entropy_density_obj)
        print(f"Entropy density reward enabled: scale={args.entropy_density_scale}, "
              f"clip=±{args.entropy_density_clip}, marker='{args.entropy_density_marker}'")

    predictive_velo_obj = None
    if args.predictive_velocity_reward:
        from src.mbe_reward import PredictiveVeloReward
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        predictive_velo_obj = PredictiveVeloReward(
            tokenizer,
            stride=args.mbe_velocity_stride,
            scale=args.predictive_velocity_scale,
            clip=args.predictive_velocity_clip,
            marker=args.predictive_marker,
        )
        reward_funcs.append(predictive_velo_obj)
        print(f"Predictive velocity reward enabled: scale={args.predictive_velocity_scale}, "
              f"clip=±{args.predictive_velocity_clip}, marker='{args.predictive_marker}'")

    eval_dataset = None
    if args.eval_steps > 0:
        eval_dataset = test_dataset
        if args.eval_samples is not None:
            eval_dataset = test_dataset.select(range(min(args.eval_samples, len(test_dataset))))
        print(f"Eval enabled: {len(eval_dataset)} samples every {args.eval_steps} steps")

    # Rollout logger: dumps every train/eval rollout to JSONL via the reward-fn
    # shim pattern. Returns 0.0 reward (no-op for the optimizer). Routed to the
    # eval log during eval loops via `EvalFlagCallback`.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rollout_path = out_dir / "rollouts.jsonl"
    eval_rollout_path  = out_dir / "eval_rollout.jsonl"
    train_rollout_path.write_text("")
    eval_rollout_path.write_text("")
    rollout_logger = GSM8KRolloutLogger(
        train_rollout_path, eval_rollout_path,
        AutoTokenizer.from_pretrained(args.model),
    )
    reward_funcs.append(rollout_logger)
    print(f"Rollout logger enabled → train: {train_rollout_path}\n"
          f"                         eval:  {eval_rollout_path}")

    trainer = FastEvalGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=prefix_dataset if prefix_dataset is not None else train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[EvalFlagCallback(rollout_logger)],
    )

    # Bind model ref for MBE forward passes (MBEDynamicsLogger only, not RolloutRecorder)
    if mbe_logger is not None and hasattr(mbe_logger, "set_model"):
        mbe_logger.set_model(trainer.model)
    if mbe_reward_obj is not None:
        mbe_reward_obj.set_model(trainer.model)
    if mbe_velo_reward_obj is not None:
        mbe_velo_reward_obj.set_model(trainer.model)
    if inv_log_len_reward_obj is not None:
        inv_log_len_reward_obj.set_model(trainer.model)
    if entropy_velo_obj is not None:
        entropy_velo_obj.set_model(trainer.model)
    if perplexity_velo_obj is not None:
        perplexity_velo_obj.set_model(trainer.model)
    if entropy_density_obj is not None:
        entropy_density_obj.set_model(trainer.model)
    if predictive_velo_obj is not None:
        predictive_velo_obj.set_model(trainer.model)

    trainer.train()
    if args.save_strategy != "no":
        trainer.save_model(args.output_dir)
        print(f"Training complete. Model saved to {args.output_dir}")
    else:
        print(f"Training complete. (--save_strategy=no, model not saved.)")


if __name__ == "__main__":
    main()