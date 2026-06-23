"""Build the master-teacher source for on-policy distillation.

For each of the 8 seed checkpoints, generate greedy (T=0) solutions over the first
`--limit` GSM8K *train* queries and record per-query correctness. on-policy distillation
then picks, per query, the master = the seed that solves it (preferring the strongest
seed), and trains the student against that master.

Each seed runs in its own subprocess so every vLLM engine gets a clean CUDA context
(sequential in-process vLLM loads corrupt transformers' global config on this stack).

Output: <out_dir>/seed{S}.json, each with records[].solutions[0].{text,correct}.
This reproduces output/gsm8k_distill/greedy/ (already present in the repo).

Usage:
  python repro/gen_masters.py --out_dir output/gsm8k_distill/greedy --limit 2000
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REPO = "Ksgk-fy/arl-gsm8k-multiseed"
STEP = 200


def worker(seed, out_path, limit):
    """One isolated vLLM process: greedy solutions for a single seed checkpoint."""
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    sys.path.insert(0, HERE)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from eval_gsm8k import build_prompts, extract_answer, extract_gold, is_correct

    snap = snapshot_download(REPO, allow_patterns=[f"seed{seed}/checkpoint-{STEP}/*"])
    model_dir = os.path.join(snap, f"seed{seed}", f"checkpoint-{STEP}")

    ds = load_dataset("openai/gsm8k", "main")["train"]
    indices = list(range(min(limit, len(ds)))) if limit else list(range(len(ds)))
    rows = ds.select(indices)
    questions = rows["question"]
    golds = [extract_gold(a) for a in rows["answer"]]

    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = build_prompts(tok, questions)
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, enforce_eager=True, seed=0)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024))

    records, n_correct = [], 0
    for j, out in enumerate(outputs):
        comp = out.outputs[0]
        pred = extract_answer(comp.text)
        ok = is_correct(pred, golds[j])
        n_correct += int(ok)
        records.append({"idx": int(indices[j]), "question": questions[j], "gold": golds[j],
                        "solutions": [{"text": comp.text, "pred": pred, "correct": ok,
                                       "truncated": comp.finish_reason == "length"}]})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"seed": seed, "n_queries": len(records),
                                          "n_correct": n_correct, "records": records}))
    print(f"[seed{seed}] greedy train acc={n_correct/max(1,len(records)):.4f} -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Generate per-seed greedy masters for on-policy distillation")
    ap.add_argument("--out_dir", default=os.path.join(PROJECT, "output/gsm8k_distill/greedy"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--limit", type=int, default=2000, help="First N train queries per seed.")
    ap.add_argument("--_seed", type=int, default=None, help=argparse.SUPPRESS)  # internal worker flag
    ap.add_argument("--_out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._seed is not None:           # worker mode
        worker(args._seed, args._out, args.limit)
        return

    env = dict(os.environ, PYTHONPATH=f"{PROJECT}:{HERE}", HF_HUB_ENABLE_HF_TRANSFER="1")
    for s in args.seeds:
        out_path = os.path.join(args.out_dir, f"seed{s}.json")
        if os.path.exists(out_path):
            print(f"[seed{s}] cached -> {out_path}", flush=True)
            continue
        subprocess.run([sys.executable, os.path.abspath(__file__), "--_seed", str(s),
                        "--_out", out_path, "--limit", str(args.limit)], env=env, check=True)
    print(f"DONE masters -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
