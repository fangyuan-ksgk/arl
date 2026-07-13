"""Q8 minimal — low-VRAM lottery-gap evaluation across a run's checkpoints.

Greedy-decodes every checkpoint on the domain's test set (MATH-500 or MBPP
test), scores, and prints per-ckpt accuracy + union-of-ever-solved + the gap.
Low-VRAM knobs: --vllm_util (default 0.6), --enforce_eager (drops CUDA-graph
memory), --max_model_len. A 4B eval fits ~14GB with
`--vllm_util 0.3 --enforce_eager`.

Reference (our runs): MATH n=500 final 70.0 / union 82.4; MBPP n=100 (seed 2)
final 46 / union 62. Records land in <run>/records/ with FULL completions
(mirrors q9_minimal/eval_gap.py so the same harvest logic applies).

  python eval_gap_domain.py --domain math --run out_math_4b
"""
import argparse
import json
import re
from pathlib import Path

from grpo_domain_lowvram import exec_with_tests, extract_boxed, extract_code


def math_test():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    return [{"prompt": [{"role": "user", "content":
                         r["problem"] + "\n\nPut your final answer in \\boxed{}."}],
             "check": ("boxed", extract_boxed(r["solution"]) or r["answer"])}
            for r in ds]


def mbpp_test():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")["test"]
    return [{"prompt": [{"role": "user", "content":
                         f"{r['prompt'] if 'prompt' in r else r['text']}\n\n"
                         f"Your solution must pass tests like:\n{r['test_list'][0]}\n"
                         "Answer with a single ```python code block."}],
             "check": ("exec", r["test_list"])} for r in ds]


def score(text, check):
    kind, ref = check
    if kind == "boxed":
        return extract_boxed(text) == ref and bool(ref)
    return exec_with_tests(extract_code(text), ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["math", "mbpp"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=None, help="cap test set size")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--vllm_util", type=float, default=0.6)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--max_model_len", type=int, default=4096)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    data = (math_test if a.domain == "math" else mbpp_test)()
    if a.n:
        data = data[:a.n]
    ckpts = sorted(int(re.search(r"checkpoint-(\d+)", str(p)).group(1))
                   for p in Path(a.run).glob("checkpoint-*")
                   if (p / "model.safetensors").exists()
                   or list(p.glob("model-0*.safetensors")))
    rec_dir = Path(a.run) / "records"
    rec_dir.mkdir(exist_ok=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    for c in ckpts:
        f = rec_dir / f"step{c}_test.jsonl"
        if f.exists() and sum(1 for _ in open(f)) == len(data):
            continue
        path = f"{a.run}/checkpoint-{c}"
        tok = AutoTokenizer.from_pretrained(path)
        llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=a.vllm_util,
                  enforce_eager=a.enforce_eager, max_model_len=a.max_model_len,
                  disable_log_stats=True)
        prompts = [tok.apply_chat_template(d["prompt"], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False) for d in data]
        outs = llm.generate(prompts, sp)
        with open(f, "w") as fh:
            for i, (d, o) in enumerate(zip(data, outs)):
                txt = o.outputs[0].text
                fh.write(json.dumps({"idx": i, "step": c,
                                     "correct": bool(score(txt, d["check"])),
                                     "completion": txt}) + "\n")
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        print(f"[eval] ckpt-{c} done", flush=True)

    ever, series = {}, []
    for c in ckpts:
        rec = {json.loads(l)["idx"]: json.loads(l)["correct"]
               for l in open(rec_dir / f"step{c}_test.jsonl")}
        series.append(100 * sum(rec.values()) / len(rec))
        for i, ok in rec.items():
            ever[i] = ever.get(i, False) or ok
    union = 100 * sum(ever.values()) / len(ever)
    print(f"[gap] {a.domain}: per-ckpt {[f'{x:.1f}' for x in series]}")
    print(f"[gap] final {series[-1]:.1f} | union {union:.1f} "
          f"| LOTTERY GAP {union - series[-1]:.1f}")


if __name__ == "__main__":
    main()
