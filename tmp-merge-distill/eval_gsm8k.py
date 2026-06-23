"""Evaluate ONE GSM8K checkpoint with vLLM (greedy / pass@1) and dump per-question
results to JSON. Shared eval engine for both `merge_and_eval.py` and `onpolicy_distill.py`.

One model per process: each vLLM engine gets a clean CUDA context. (Sequential in-process
vLLM loads corrupt transformers' global config state on this stack, so the merge sweep
shells out to this script once per merged model.)

Prompt / answer conventions are copied verbatim from the GRPO training recipe
(`script/grpo_gsm8k.py`) so eval matches the format the checkpoints were trained under:
Qwen3 thinking mode via the trailing " /think", answer marked by a final `#### <number>`.

Eval hygiene (matters for merge comparisons): use --max_tokens 2048 and the strict `####`
extraction. The GSM8K greedy@1 noise floor is ~+/-0.015; gaps smaller than ~1.5 points are
not meaningful. Short context (1024) inflates truncation and adds noise.

Usage:
    python repro/eval_gsm8k.py --model_path <dir> --out results.json --max_tokens 2048
    python repro/eval_gsm8k.py --repo Ksgk-fy/arl-gsm8k-multiseed \
        --subfolder seed0/checkpoint-200 --out seed0.json --max_tokens 2048
"""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Same instruction string the checkpoints were GRPO-trained with.
ANSWER_FORMAT_INSTRUCTION = (
    "Solve the problem step by step inside <think>...</think>. "
    "After </think>, give a brief final explanation and end your response "
    "with a line of the exact form:\n#### <number>\n"
    "where <number> is the final numeric answer with no units, no commas, "
    "and no extra text."
)


def extract_answer(text: str) -> str:
    """Pull the numeric answer from a completion: prefer the `#### <n>` marker."""
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def extract_gold(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    return numbers[-1].replace(",", "") if numbers else ""


def is_correct(pred: str, gold: str) -> bool:
    try:
        return float(pred) == float(gold)
    except (ValueError, TypeError):
        return False


def build_prompts(tokenizer, questions):
    """Render GSM8K questions to chat-templated prompt strings (training format).

    Imported by merge_and_eval.py / onpolicy_distill.py so on-policy rollouts and the
    final eval see byte-identical prompts.
    """
    prompts = []
    for q in questions:
        messages = [{"role": "user", "content": f"{q}\n\n{ANSWER_FORMAT_INSTRUCTION} /think"}]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def main():
    p = argparse.ArgumentParser(description="vLLM GSM8K eval for one checkpoint")
    p.add_argument("--repo", default="Ksgk-fy/arl-gsm8k-multiseed")
    p.add_argument("--subfolder", default=None, help="e.g. seed0/checkpoint-200")
    p.add_argument("--model_path", default=None, help="Local model dir (overrides --repo/--subfolder).")
    p.add_argument("--label", default=None, help="Name recorded in the output JSON.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--limit", type=int, default=None, help="Eval only the first N test questions (debug).")
    p.add_argument("--max_tokens", type=int, default=2048, help="Max generated tokens (use 2048 for clean comparisons).")
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy pass@1.")
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.model_path:
        model_dir = args.model_path
    else:
        if not args.subfolder:
            p.error("either --model_path or --subfolder is required")
        from huggingface_hub import snapshot_download
        snap = snapshot_download(args.repo, allow_patterns=[f"{args.subfolder}/*"])
        model_dir = os.path.join(snap, args.subfolder)
    label = args.label or args.subfolder or os.path.basename(os.path.normpath(model_dir))

    ds = load_dataset("openai/gsm8k", "main")["test"]   # 1319-question GSM8K val set
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    questions = ds["question"]
    golds = [extract_gold(a) for a in ds["answer"]]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = build_prompts(tokenizer, questions)

    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, seed=args.seed)
    sampling = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
                              seed=args.seed if args.temperature > 0 else None)
    outputs = llm.generate(prompts, sampling)

    records, correct_flags, n_truncated = [], [], 0
    for i, out in enumerate(outputs):
        comp = out.outputs[0]
        pred = extract_answer(comp.text)
        ok = is_correct(pred, golds[i])
        truncated = comp.finish_reason == "length"
        n_truncated += int(truncated)
        correct_flags.append(ok)
        records.append({"idx": i, "gold": golds[i], "pred": pred, "correct": ok,
                        "has_marker": bool(re.search(r"####\s*[\d,\.\-]+", comp.text)),
                        "truncated": truncated, "n_gen_tokens": len(comp.token_ids)})

    n = len(correct_flags)
    accuracy = sum(correct_flags) / n if n else 0.0
    result = {"label": label, "model_dir": model_dir, "n": n, "accuracy": accuracy,
              "n_correct": int(sum(correct_flags)), "n_truncated": n_truncated,
              "max_tokens": args.max_tokens, "temperature": args.temperature,
              "correct": correct_flags, "records": records}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result))
    print(f"[{label}] acc={accuracy:.4f} ({result['n_correct']}/{n}) "
          f"truncated={n_truncated} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
