"""
Dr.GRPO on GSM8K / MATH with vLLM (colocate by default).

Minimal rebuild of grpo_gsm8k.py / grpo_math.py keeping only:
  * Dr.GRPO defaults        (loss_type=dr_grpo, scale_rewards=none)
  * correctness + format rewards (per-dataset answer checking)
  * the MBE velocity reward gadget (src/mbe_reward.py, deferred model binding)
  * fast greedy+sample eval with JSONL rollout logging
    (rollouts.jsonl / eval_rollout.jsonl — same schema the forgetting
     analysis scripts consume)

Usage:
    python script/grpo.py                                  # GSM8K smoke (20 steps)
    python script/grpo.py --dataset math --max_steps -1    # MATH, full epoch
    python script/grpo.py --no-mbe_velocity_reward         # correctness+format only
    python script/grpo.py --forgettable_indices forgotten.json \
        --replay_fraction 0.25                             # interleaved replay
"""

import argparse
import contextlib
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

from src.complement_grpo import ComplementGRPOTrainer
from src.interleave_grpo import InterleavedGRPOTrainer, load_forgettable_indices

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# Colocate mode: eval-generation fragmentation + fp32-logits backward peaks
# need the expandable allocator (see CLAUDE.md GRPO speed notes).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _completion_text(completion) -> str:
    """Normalise a TRL completion (message list or plain string) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        return first.get("content", "") if isinstance(first, dict) else str(first)
    return str(completion)


# ---------------------------------------------------------------------------
# GSM8K: `#### <number>` convention
# ---------------------------------------------------------------------------
def gsm8k_extract(text: str) -> str:
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def gsm8k_equal(predicted: str, gold: str) -> bool:
    try:
        return float(predicted) == float(gold)
    except (ValueError, TypeError):
        return False


def gsm8k_has_format(text: str) -> bool:
    return bool(re.search(r"####\s*[\d,\.\-]+", text))


def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")

    instruction = (
        "Solve the problem step by step inside <think>...</think>. "
        "After </think>, give a brief final explanation and end your response "
        "with a line of the exact form:\n#### <number>\n"
        "where <number> is the final numeric answer with no units, no commas, "
        "and no extra text."
    )

    def format_example(example):
        example["prompt"] = [{"role": "user", "content":
                              f"{example['question']}\n\n{instruction} /think"}]
        example["gold_answer"] = gsm8k_extract(example["answer"])
        return example

    return (dataset["train"].map(format_example),
            dataset["test"].map(format_example))


# ---------------------------------------------------------------------------
# MATH: `\boxed{...}` convention
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str:
    """Extract the last \\boxed{...} content, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
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


def math_extract(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)  # GSM8K-style fallback
    return match.group(1).strip() if match else ""


def _normalize_math_answer(ans: str) -> str:
    ans = ans.strip().strip("$")
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", ans)
    ans = re.sub(r"\s+", "", ans)
    return ans.rstrip(".")


def math_equal(predicted: str, gold: str) -> bool:
    if not predicted or not gold:
        return False
    p, g = _normalize_math_answer(predicted), _normalize_math_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    try:
        pv = eval(p, {"__builtins__": {}})  # noqa: S307
        gv = eval(g, {"__builtins__": {}})  # noqa: S307
        return abs(float(pv) - float(gv)) < 1e-6
    except Exception:
        return False


def math_has_format(text: str) -> bool:
    return bool(extract_boxed(text))


def load_math():
    """Hendrycks MATH via working mirrors ('hendrycks/competition_math' is gone)."""
    last = None
    for repo, cfg in [("DigitalLearningGmbH/MATH-lighteval", None),
                      ("EleutherAI/hendrycks_math", "default"),
                      ("nlile/hendrycks-MATH-benchmark", None)]:
        try:
            dataset = load_dataset(repo) if cfg is None else load_dataset(repo, cfg)
            break
        except Exception as e:  # noqa
            last = e
    else:
        raise RuntimeError(f"Could not load any MATH mirror: {last}")

    system = ("Solve the following math problem step by step. "
              "Put your final answer in \\boxed{}.")

    def format_example(example):
        example["prompt"] = [{"role": "system", "content": system},
                             {"role": "user", "content": example["problem"]}]
        example["gold_answer"] = extract_boxed(example["solution"])
        return example

    return (dataset["train"].map(format_example),
            dataset["test"].map(format_example))


DOMAINS = {
    "gsm8k": dict(load=load_gsm8k, extract=gsm8k_extract, equal=gsm8k_equal,
                  has_format=gsm8k_has_format,
                  max_completion_length=1024, per_device_train_batch_size=8),
    # Long proofs → fp32-logits memory wall: keep the micro-batch tiny.
    "math": dict(load=load_math, extract=math_extract, equal=math_equal,
                 has_format=math_has_format,
                 max_completion_length=3072, per_device_train_batch_size=2),
}


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def make_reward_funcs(domain):
    extract, equal, has_format = domain["extract"], domain["equal"], domain["has_format"]

    def correctness_reward(completions, gold_answer, **kwargs):
        return [1.0 if equal(extract(_completion_text(c)), g) else 0.0
                for c, g in zip(completions, gold_answer)]

    def format_reward(completions, **kwargs):
        return [0.5 if has_format(_completion_text(c)) else 0.0
                for c in completions]

    return [correctness_reward, format_reward]


# ---------------------------------------------------------------------------
# Rollout logging (reward-fn shim, returns 0.0 — no-op for the optimizer).
# Routed to rollouts.jsonl / eval_rollout.jsonl by EvalFlagCallback.
# ---------------------------------------------------------------------------
class RolloutLogger:
    __name__ = "rollout_logger"

    def __init__(self, train_path: Path, eval_path: Path, tokenizer, domain):
        self.train_path = train_path
        self.eval_path = eval_path
        self.tokenizer = tokenizer
        self.extract = domain["extract"]
        self.equal = domain["equal"]
        self.in_eval = False
        self.decoding = "sample"     # flipped by FastEvalGRPOTrainer per pass
        self.temperature = None
        self.train_step = 0
        self.eval_step = 0
        self.global_step = -1        # stamped by EvalFlagCallback

    def __call__(self, completions, gold_answer, **_):
        path = self.eval_path if self.in_eval else self.train_path
        step = self.eval_step if self.in_eval else self.train_step
        with path.open("a") as f:
            for i, (c, gold) in enumerate(zip(completions, gold_answer)):
                text = _completion_text(c)
                predicted = self.extract(text)
                n_tok = len(self.tokenizer.encode(text, add_special_tokens=False))
                m_ans = re.search(r"####|\\boxed\{", text)
                cot_text = text[: m_ans.start()] if m_ans else text
                n_cot_tok = len(self.tokenizer.encode(cot_text, add_special_tokens=False))
                f.write(json.dumps({
                    "step": step,
                    "idx": i,
                    "gold_answer": str(gold),
                    "predicted_answer": predicted,
                    "completion": text,
                    "correct": bool(self.equal(predicted, gold)),
                    "n_tokens": int(n_tok),
                    "n_cot_tokens": int(n_cot_tok),
                    "has_answer_marker": bool(m_ans),
                    "split": "eval" if self.in_eval else "train",
                    "decoding": self.decoding,
                    "temperature": (None if self.temperature is None
                                    else float(self.temperature)),
                    "global_step": int(self.global_step),
                }) + "\n")
        if self.in_eval:
            self.eval_step += 1
        else:
            self.train_step += 1
        return [0.0] * len(completions)


class FastEvalGRPOTrainer(GRPOTrainer):
    """GRPOTrainer whose eval skips the local forward pass entirely: one
    sampled pass@K generation + one greedy pass@1 generation per eval batch,
    both logged via the rollout logger / reward funcs."""

    @contextlib.contextmanager
    def _greedy_eval(self):
        """Temporarily force greedy decoding (T=0) and 1 generation/prompt."""
        old_neval = self.num_generations_eval
        self.num_generations_eval = 1
        vg = getattr(self, "vllm_generation", None)
        gc = getattr(self, "generation_config", None)
        old_vg_t = getattr(vg, "temperature", None) if vg is not None else None
        old_gc = (gc.temperature, gc.do_sample) if gc is not None else None
        if vg is not None:
            vg.temperature = 0.0
        if gc is not None:
            gc.do_sample = False
        try:
            yield
        finally:
            self.num_generations_eval = old_neval
            if vg is not None:
                vg.temperature = old_vg_t
            if gc is not None:
                gc.temperature, gc.do_sample = old_gc

    def _current_temperature(self):
        vg = getattr(self, "vllm_generation", None)
        if vg is not None and getattr(vg, "temperature", None) is not None:
            return float(vg.temperature)
        gc = getattr(self, "generation_config", None)
        if gc is not None and getattr(gc, "temperature", None) is not None:
            return float(gc.temperature)
        return getattr(getattr(self, "args", None), "temperature", None)

    def _eval_generate(self, prompts, inputs, *, decoding):
        logger = getattr(self, "_rollout_logger", None)
        if logger is not None:
            logger.decoding = decoding
            logger.temperature = self._current_temperature()
        _t = time.time()
        result = self._generate(prompts)
        completion_ids_list = result[1]
        completions = result[3]
        extra_fields = result[-1] if isinstance(result[-1], dict) else {}
        dt = time.time() - _t
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values
        if decoding == "greedy":
            if logger is not None:
                logger(completions=completions,
                       gold_answer=[x["gold_answer"] for x in inputs])
        else:
            self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        return dt, len(completions)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        prompts = [x["prompt"] for x in inputs]
        if not hasattr(self, "_eval_call_idx"):
            self._eval_call_idx = 0
            self._eval_t0 = time.time()
        self._eval_call_idx += 1

        # Pass 1: sampled pass@K (configured eval temperature).
        dt, n_gen = self._eval_generate(prompts, inputs, decoding="sample")

        # Pass 2: greedy pass@1 — the eval sampler repeats each prompt G times;
        # slice back to uniques so we get one deterministic completion each.
        G = max(1, self.num_generations_eval)
        uniq_inputs = inputs[::G]
        uniq_prompts = prompts[::G]
        with self._greedy_eval():
            dt_g, _ = self._eval_generate(uniq_prompts, uniq_inputs, decoding="greedy")

        elapsed = time.time() - self._eval_t0
        n_total = len(self.eval_dataset) if self.eval_dataset is not None else None
        bs_unique = len(prompts) // G
        if n_total:
            done = self._eval_call_idx * bs_unique
            eta = elapsed * (n_total - done) / max(1, done)
            print(f"[eval] call {self._eval_call_idx}: {n_gen} sample + {bs_unique} greedy "
                  f"in {dt:.1f}s+{dt_g:.1f}s | ~{done}/{n_total} prompts | "
                  f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m", flush=True)
        else:
            print(f"[eval] call {self._eval_call_idx}: {n_gen} sample + {bs_unique} greedy "
                  f"in {dt:.1f}s+{dt_g:.1f}s", flush=True)

        return torch.zeros((), device=self.accelerator.device), None, None


class ComplementFastEvalGRPOTrainer(
    FastEvalGRPOTrainer,
    ComplementGRPOTrainer,
    InterleavedGRPOTrainer,
):
    """Fast evaluation plus replay and shortest-correct self-distillation."""


class EvalFlagCallback(TrainerCallback):
    """Routes reward-fn calls to the right rollout log around eval loops."""

    def __init__(self, logger: RolloutLogger):
        self.logger = logger

    def on_prediction_step(self, args, state, control, **kw):
        self.logger.in_eval = True
        self.logger.global_step = state.global_step

    def on_step_begin(self, args, state, control, **kw):
        self.logger.in_eval = False
        self.logger.global_step = state.global_step

    def on_evaluate(self, args, state, control, **kw):
        self.logger.in_eval = False
        # Big eval generation batches fragment the allocator badly enough that
        # the next training backward (fp32 logits) can OOM in colocate mode.
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dr.GRPO on GSM8K / MATH")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=sorted(DOMAINS))
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: output/grpo_<dataset>")
    # Dr.GRPO defaults (override only for ablations).
    parser.add_argument("--loss_type", type=str, default="dr_grpo")
    parser.add_argument("--scale_rewards", type=str, default="none",
                        choices=["none", "group", "batch"],
                        help="Dr.GRPO = no reward scaling (default).")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Default per dataset: gsm8k 1024, math 3072.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                        help="Default per dataset: gsm8k 8, math 2 (fp32-logits wall).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=20, help="-1 for one full epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    # vLLM (colocate default; sleep mode stays OFF — broken weight sync in
    # trl 1.7 + vllm 0.23, see CLAUDE.md).
    parser.add_argument("--no_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["colocate", "server"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--train_device", type=int, default=0,
                        help="(server mode, single process) CUDA index for training.")
    # Checkpointing
    parser.add_argument("--save_strategy", type=str, default="no",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=None)
    # Eval
    parser.add_argument("--eval_steps", type=int, default=50, help="0 to disable")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Subsample N test examples (default: full test set)")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Default num_generations*16; must be a multiple of "
                             "num_generations. Bigger = better vLLM batching.")
    # Forgettable-query interleaving. The file contains train-dataset row
    # indices, or per-index correctness trajectories (see load_forgettable_indices).
    parser.add_argument("--forgettable_indices", type=str, default=None,
                        help="JSON/JSONL forgettable-query indices or trajectories.")
    parser.add_argument("--replay_fraction", type=float, default=0.25,
                        help="Fraction of unique prompt groups in every generation "
                             "batch replaced by forgettable queries.")
    parser.add_argument("--replay_sft_weight", type=float, default=0.1,
                        help="Weight of shortest-correct SFT loss on replay queries; "
                             "0 disables self-distillation.")
    parser.add_argument("--replay_sft_batch_size", type=int, default=1,
                        help="Replay queries independently sampled for the SFT "
                             "objective per GRPO generation step.")
    # MBE velocity reward gadget
    parser.add_argument("--mbe_velocity_reward",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Length-normalised MBE velocity reward: "
                             "clip(raw_v / log(min(T,D)), ±clip) / scale")
    parser.add_argument("--mbe_velocity_mode", type=str, default="trajectory",
                        choices=["trajectory", "rollercoaster"],
                        help="trajectory: MBE(query+response) - MBE(query). "
                             "rollercoaster: sum of positive per-stride MBE jumps.")
    parser.add_argument("--mbe_velocity_scale", type=float, default=4.0)
    parser.add_argument("--mbe_velocity_clip", type=float, default=1.0)
    parser.add_argument("--mbe_velocity_stride", type=int, default=8)
    parser.add_argument("--mbe_velocity_layers", type=str, default="-1",
                        help="Comma-separated hidden-layer indices ('-1' = last layer).")
    args = parser.parse_args()

    domain = DOMAINS[args.dataset]
    if args.output_dir is None:
        args.output_dir = f"output/grpo_{args.dataset}"
    if args.max_completion_length is None:
        args.max_completion_length = domain["max_completion_length"]
    if args.per_device_train_batch_size is None:
        args.per_device_train_batch_size = domain["per_device_train_batch_size"]

    train_dataset, test_dataset = domain["load"]()
    print(f"{args.dataset}: train {len(train_dataset)}, test {len(test_dataset)}")
    replay_indices = (
        load_forgettable_indices(args.forgettable_indices)
        if args.forgettable_indices else []
    )
    if args.forgettable_indices and not replay_indices:
        raise ValueError(f"No forgettable indices found in {args.forgettable_indices}")
    if replay_indices:
        print(f"Interleaved replay: {len(replay_indices)} queries, "
              f"{args.replay_fraction:.0%} of every generation batch; "
              f"shortest-correct SFT weight={args.replay_sft_weight}")

    config_kwargs = dict(
        output_dir=args.output_dir,
        loss_type=args.loss_type,
        scale_rewards=(False if args.scale_rewards == "none" else args.scale_rewards),
        num_generations=args.num_generations,
        temperature=1.0,  # target cache samples t=1, matching the GRPO policy
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_strategy,
        report_to=args.report_to,
        seed=args.seed,
    )
    if args.max_steps > 0:
        config_kwargs["max_steps"] = args.max_steps
    if args.save_strategy == "steps":
        config_kwargs["save_steps"] = args.save_steps
    if args.save_total_limit is not None:
        config_kwargs["save_total_limit"] = args.save_total_limit
    if not args.no_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "colocate":
            config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        else:
            config_kwargs["vllm_server_host"] = args.vllm_server_host
            config_kwargs["vllm_server_port"] = args.vllm_server_port

    eval_dataset = None
    if args.eval_steps > 0:
        eval_dataset = test_dataset
        if args.eval_samples is not None:
            eval_dataset = test_dataset.select(
                range(min(args.eval_samples, len(test_dataset))))
        config_kwargs["eval_strategy"] = "steps"
        config_kwargs["eval_steps"] = args.eval_steps
        config_kwargs["eval_on_start"] = True   # pre-training baseline at step 0
        # FastEval skips the local forward, so vLLM throughput is the eval
        # constraint: send big batches per call. Clamp to the eval set's row
        # count (n_eval * G) or TRL's repeat sampler wraps and re-evaluates.
        config_kwargs["per_device_eval_batch_size"] = min(
            args.eval_batch_size or args.num_generations * 16,
            len(eval_dataset) * args.num_generations,
        )
        print(f"Eval: {len(eval_dataset)} samples every {args.eval_steps} steps")

    config = GRPOConfig(**config_kwargs)

    # Colocate / no-vllm: pass the path and let TRL place the model. Server
    # mode (single process): pin training off the vLLM GPUs.
    model = args.model
    if not args.no_vllm and args.vllm_mode == "server" \
            and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{args.train_device}"},
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    reward_funcs = make_reward_funcs(domain)

    mbe_velo = None
    if args.mbe_velocity_reward:
        from src.mbe_reward import MBEVeloReward
        layers = [int(x) for x in args.mbe_velocity_layers.split(",") if x.strip()]
        mbe_velo = MBEVeloReward(
            tokenizer,
            layers=layers,
            stride=args.mbe_velocity_stride,
            scale=args.mbe_velocity_scale,
            clip=args.mbe_velocity_clip,
            mode=args.mbe_velocity_mode,
        )
        reward_funcs.append(mbe_velo)
        print(f"MBE velocity reward: mode={args.mbe_velocity_mode}, "
              f"scale={args.mbe_velocity_scale}, clip=±{args.mbe_velocity_clip}, "
              f"stride={args.mbe_velocity_stride}, layers={layers}")

    # Per-rank rollout logs (DDP appends can't interleave mid-line);
    # merge with `cat rollouts*.jsonl`, the `step` field orders them.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _rank = int(os.environ.get("RANK", "0"))
    _suffix = f".rank{_rank}" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else ""
    train_rollout_path = out_dir / f"rollouts{_suffix}.jsonl"
    eval_rollout_path = out_dir / f"eval_rollout{_suffix}.jsonl"
    train_rollout_path.write_text("")
    eval_rollout_path.write_text("")
    rollout_logger = RolloutLogger(train_rollout_path, eval_rollout_path,
                                   tokenizer, domain)
    reward_funcs.append(rollout_logger)
    print(f"Rollout logs → {train_rollout_path} / {eval_rollout_path}")

    def replay_correctness(completion, example):
        return domain["equal"](
            domain["extract"](completion),
            example["gold_answer"],
        )

    trainer = ComplementFastEvalGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EvalFlagCallback(rollout_logger)],
        replay_indices=replay_indices,
        replay_fraction=args.replay_fraction,
        correctness_fn=replay_correctness,
        sft_replay_indices=replay_indices,
        sft_weight=(args.replay_sft_weight if replay_indices else 0.0),
        replay_sft_batch_size=args.replay_sft_batch_size,
    )
    trainer._rollout_logger = rollout_logger

    # Deferred binding: MBE needs hidden-state forwards on the live training
    # model — bind AFTER trainer init (see CLAUDE.md MBE Reward).
    if mbe_velo is not None:
        mbe_velo.set_model(trainer.model)

    trainer.train()
    if args.save_strategy != "no":
        trainer.save_model(args.output_dir)
        print(f"Done. Model saved to {args.output_dir}")
    else:
        print("Done. (--save_strategy=no, model not saved.)")


if __name__ == "__main__":
    main()
