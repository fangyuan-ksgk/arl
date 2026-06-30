"""
GRPO on MBPP (code generation) with execution-based reward

Single-GPU:
    python script/grpo_code.py                        # smoke test (20 steps)
    python script/grpo_code.py --max_steps -1         # full run

Server mode (2-GPU):
    GPU 0: trl vllm-serve --model Qwen/Qwen3-0.6B --port 8000
    GPU 1: CUDA_VISIBLE_DEVICES=1 python script/grpo_code.py \
               --use_vllm --vllm_mode server --vllm_server_port 8000 --max_steps 200

Datasets supported:
    --dataset mbpp       (default) Google MBPP — 374 train / 500 test
    --dataset apps       APPS (introductory level) — larger, harder
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


# ---------------------------------------------------------------------------
# Code extraction from model completions
# ---------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """Extract Python code from completion. Tries ```python blocks first, then raw."""
    # Try fenced code blocks
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Try to find code after common markers
    for marker in ["```", "def ", "import "]:
        idx = text.find(marker)
        if idx != -1:
            candidate = text[idx:].strip()
            if candidate.startswith("```"):
                candidate = candidate[3:].strip()
                if candidate.startswith("python"):
                    candidate = candidate[6:].strip()
            return candidate
    # Last resort: return everything after </think> if present
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + 8:].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Safe code execution
# ---------------------------------------------------------------------------
def execute_code_with_tests(code: str, tests: list[str], timeout: int = 10) -> bool:
    """
    Execute code + test assertions in a subprocess with timeout.
    Returns True if all tests pass, False otherwise.
    """
    # Build the full script: code + tests
    test_block = "\n".join(tests)
    full_script = f"{code}\n\n{test_block}\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_script)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def correctness_reward(completions, test_list, **kwargs):
    """Execute code and check test assertions. 1.0 if all pass, 0.0 otherwise."""
    rewards = []
    for completion, tests in zip(completions, test_list):
        text = completion[0]["content"]
        code = extract_code(text)
        if not code:
            rewards.append(0.0)
            continue
        passed = execute_code_with_tests(code, tests)
        rewards.append(1.0 if passed else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Reward for producing a fenced code block."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        has_code_block = bool(re.search(r"```(?:python)?\s*\n.+?```", text, re.DOTALL))
        rewards.append(0.5 if has_code_block else 0.0)
    return rewards


def syntax_reward(completions, **kwargs):
    """Reward for syntactically valid Python (can be compiled)."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        code = extract_code(text)
        if not code:
            rewards.append(0.0)
            continue
        try:
            compile(code, "<string>", "exec")
            rewards.append(0.25)
        except SyntaxError:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write a Python function to solve the given task. "
    "Put your code in a ```python code block."
)


def load_mbpp():
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    # sanitized split: train (120), prompt (10), test (257)
    # Use full split for more data
    full = load_dataset("google-research-datasets/mbpp", "full")

    def format_example(example):
        example["prompt"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["text"]},
        ]
        # test_list is already a list of assert strings
        return example

    # full split: task_id 1-600 train, 601-700 test, 701-974 validation
    # Use train + validation for training, test for eval
    train_ids = set(range(1, 601)) | set(range(701, 975))
    test_ids = set(range(601, 701))

    all_data = full["train"]  # full split puts everything in "train"
    train_data = all_data.filter(lambda x: x["task_id"] in train_ids)
    test_data = all_data.filter(lambda x: x["task_id"] in test_ids)

    train_dataset = train_data.map(format_example)
    test_dataset = test_data.map(format_example)
    print(f"MBPP — Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset


def load_apps():
    dataset = load_dataset("codeparrot/apps", trust_remote_code=True)
    # Filter to introductory difficulty only (most tractable for small models)
    import json

    def format_example(example):
        example["prompt"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        # Parse test cases from JSON string
        try:
            test_cases = json.loads(example["input_output"])
            inputs = test_cases.get("inputs", [])
            outputs = test_cases.get("outputs", [])
            # Build assert-style tests for stdin/stdout problems
            tests = []
            for inp, out in zip(inputs[:5], outputs[:5]):  # cap at 5 tests
                tests.append(
                    f'assert solution({repr(inp.strip())}) == {repr(out.strip())}'
                )
            example["test_list"] = tests
        except (json.JSONDecodeError, KeyError):
            example["test_list"] = []
        return example

    train_dataset = dataset["train"].map(format_example)
    test_dataset = dataset["test"].map(format_example)
    # Remove examples with no tests
    train_dataset = train_dataset.filter(lambda x: len(x["test_list"]) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x["test_list"]) > 0)
    print(f"APPS (all levels) — Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
class VirtualRolloutGRPOTrainer(GRPOTrainer):
    """GRPOTrainer + virtual-rollout advantage shaping (insert-max etc.) — complements MBE velocity.
    No-op when virtual_rollout_mode is None. See src/arsenal.virtual_rollout_advantages."""
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


def main():
    parser = argparse.ArgumentParser(description="GRPO on code generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", type=str, default="mbpp",
                        choices=["mbpp", "apps"])
    parser.add_argument("--output_dir", type=str, default="grpo_code_output")
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
    parser.add_argument("--train_device", type=int, default=0)
    # Eval
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Run eval every N steps (0 to disable)")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Subsample N test examples (default: full test set)")
    # MBE reward
    parser.add_argument("--mbe_reward", action="store_true")
    parser.add_argument("--gated_mbe_reward", action="store_true")
    parser.add_argument("--mbe_scale", type=float, default=5.0)
    parser.add_argument("--mbe_clip", type=float, default=2.0)
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=512)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
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
    args = parser.parse_args()

    # Load dataset
    if args.dataset == "mbpp":
        train_dataset, test_dataset = load_mbpp()
    else:
        train_dataset, test_dataset = load_apps()

    # Config
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

    # Model
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

    # Rewards
    reward_funcs = [correctness_reward, format_reward, syntax_reward]
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

    # Eval
    eval_dataset = None
    if args.eval_steps > 0:
        eval_dataset = test_dataset
        if args.eval_samples is not None:
            eval_dataset = test_dataset.select(range(min(args.eval_samples, len(test_dataset))))
        print(f"Eval: {len(eval_dataset)} samples every {args.eval_steps} steps")

    trainer = VirtualRolloutGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
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
