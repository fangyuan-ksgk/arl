"""Q8 minimal — low-VRAM full-FT GRPO on Qwen3-4B for MATH and MBPP.

The whole point is the memory stack (all five matter; drop one and a 4B
full-FT GRPO run does not fit under ~40GB):
  1. liger kernel (fused RMSNorm/RoPE/SwiGLU; ~20-25GB saved on activations)
  2. gradient_checkpointing=True
  3. optim=paged_adamw_8bit  (Adam fp32 for 4B alone is 32GB; 8-bit is ~8GB)
  4. bs1 x grad-accum 32 (effective 32), completion cap 1024
  5. colocate vLLM at gpu_memory_utilization 0.15 + enable_thinking=False
     (thinking mode exhausts completion budgets -> clip -> reward 0 ->
     zero-variance GRPO starvation; this bit us for a week)
Measured on our runs: ~54GB peak with bs2/ga16 on 80GB; the defaults below
are sized for a single 40-48GB card. save_only_model=True or optimizer
states eat ~25GB per checkpoint on disk.

  python grpo_domain_lowvram.py --domain math --out out_math_4b
  python grpo_domain_lowvram.py --domain mbpp --out out_mbpp_4b --seed 2
Deps: torch, transformers, datasets, trl, vllm, liger-kernel, bitsandbytes.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile

import torch

MODEL = "Qwen/Qwen3-4B"


# ---------------------------------------------------------------- MATH
def extract_boxed(text):
    """Last \\boxed{...} content, nested-brace aware."""
    out, i = None, 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx < 0:
            break
        depth, j = 1, idx + len("\\boxed{")
        start = j
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out, i = text[start:j - 1], j
    return (out or "").strip()


def math_domain():
    from datasets import load_dataset
    last = None
    for repo in ["DigitalLearningGmbH/MATH-lighteval",
                 "nlile/hendrycks-MATH-benchmark"]:
        try:
            ds = load_dataset(repo)
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"no MATH mirror reachable: {last}")

    def fmt(ex):
        ex["prompt"] = [{"role": "user", "content":
                         ex["problem"] + "\n\nPut your final answer in \\boxed{}."}]
        ex["gold"] = extract_boxed(ex["solution"])
        return ex

    def reward(completions, gold, **kw):
        return [1.0 if extract_boxed(c[0]["content"] if isinstance(c, list) else c)
                == g and g else 0.0 for c, g in zip(completions, gold)]

    return ds["train"].map(fmt), reward


# ---------------------------------------------------------------- MBPP
def _run_script(path, timeout):
    proc = subprocess.Popen([sys.executable, path], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return False


def exec_with_tests(code, tests, timeout=6.0):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + "\n".join(tests) + "\n")
        path = f.name
    try:
        return _run_script(path, timeout)
    finally:
        os.unlink(path)


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


def mbpp_domain():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    def fmt(ex):
        ex["prompt"] = [{"role": "user", "content":
                         f"{ex['prompt'] if 'prompt' in ex else ex['text']}\n\n"
                         f"Your solution must pass tests like:\n{ex['test_list'][0]}\n"
                         "Answer with a single ```python code block."}]
        ex["tests"] = ex["test_list"]
        return ex

    def reward(completions, tests, **kw):
        out = []
        for c, t in zip(completions, tests):
            txt = c[0]["content"] if isinstance(c, list) else c
            out.append(1.0 if exec_with_tests(extract_code(txt), t) else 0.0)
        return out

    return ds["train"].map(fmt), reward


DOMAINS = {"math": math_domain, "mbpp": mbpp_domain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ga", type=int, default=32)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion", type=int, default=1024)
    ap.add_argument("--vllm_util", type=float, default=0.15)
    ap.add_argument("--no_liger", action="store_true")
    a = ap.parse_args()

    if not a.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3()

    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    train_ds, reward = DOMAINS[a.domain]()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    cfg = GRPOConfig(
        output_dir=a.out, max_steps=a.max_steps, seed=a.seed,
        save_strategy="steps", save_steps=a.save_every, save_only_model=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        chat_template_kwargs={"enable_thinking": False},
        num_generations=a.num_generations, max_completion_length=a.max_completion,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.ga,
        learning_rate=a.lr, max_grad_norm=0.5, bf16=True,
        logging_steps=5, report_to="none",
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_util,
        loss_type="dr_grpo", scale_rewards=False, remove_unused_columns=False)
    GRPOTrainer(model=model, reward_funcs=[reward], args=cfg,
                train_dataset=train_ds).train()
    print(f"[q8min] done -> {a.out}")


if __name__ == "__main__":
    main()
