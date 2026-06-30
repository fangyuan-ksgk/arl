"""Evaluate ONE checkpoint on the Hendrycks MATH test set with vLLM (greedy / pass@1).

Mirrors tmp-merge-distill/eval_gsm8k.py (same output JSON schema: accuracy, n, correct[],
records[]) so the Local-SGD driver's union/avg "lottery gap" logic is task-agnostic. The only
differences are the MATH prompt format (\\boxed{}) and answer comparison, both imported verbatim
from script/grpo_math.py so eval matches the training recipe.

Usage:
    python script/eval_math.py --model_path <dir> --out r.json --max_tokens 2048
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "script"))
from grpo_math import (extract_boxed, extract_answer_from_completion, answers_equal,  # noqa: E402
                       SYSTEM_PROMPT, load_math_any)


def build_prompts(tokenizer, problems):
    prompts = []
    for prob in problems:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prob}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True))
    return prompts


def main():
    p = argparse.ArgumentParser(description="vLLM MATH eval for one checkpoint")
    p.add_argument("--model_path", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pass_k", type=int, default=1, help="pass@k: sample k completions, correct if any correct")
    p.add_argument("--pass_temp", type=float, default=0.8, help="sampling temperature when pass_k>1")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_dir = args.model_path
    label = args.label or os.path.basename(os.path.normpath(model_dir))

    ds = load_math_any()["test"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    problems = ds["problem"]
    golds = [extract_boxed(s) for s in ds["solution"]]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = build_prompts(tokenizer, problems)

    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, seed=args.seed)
    passk = max(1, args.pass_k)
    temp = args.pass_temp if passk > 1 else args.temperature
    sampling = SamplingParams(temperature=temp, top_p=1.0, max_tokens=args.max_tokens, n=passk,
                              seed=(args.seed if temp > 0 else None))
    outputs = llm.generate(prompts, sampling)

    records, correct_flags, n_truncated = [], [], 0
    for i, out in enumerate(outputs):
        sample_ok = [answers_equal(extract_answer_from_completion(c.text), golds[i]) for c in out.outputs]
        ok = any(sample_ok)                                    # pass@k (k=1 -> greedy)
        comp = out.outputs[0]
        pred = extract_answer_from_completion(comp.text)
        truncated = comp.finish_reason == "length"
        n_truncated += int(truncated)
        correct_flags.append(ok)
        records.append({"idx": i, "gold": golds[i], "pred": pred, "correct": ok,
                        "pass1": sample_ok[0], "n_correct_of_k": int(sum(sample_ok)), "k": passk,
                        "has_marker": bool(extract_boxed(comp.text)), "truncated": truncated,
                        "n_gen_tokens": sum(len(c.token_ids) for c in out.outputs) / len(out.outputs)})

    n = len(correct_flags)
    accuracy = sum(correct_flags) / n if n else 0.0
    result = {"label": label, "model_dir": model_dir, "n": n, "accuracy": accuracy,
              "n_correct": int(sum(correct_flags)), "n_truncated": n_truncated,
              "max_tokens": args.max_tokens, "temperature": args.temperature,
              "correct": correct_flags, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result))
    print(f"[{label}] MATH acc={accuracy:.4f} ({result['n_correct']}/{n}) "
          f"truncated={n_truncated} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
