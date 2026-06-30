"""pass@1 eval on HumanEval+ / MBPP+ via vLLM generation + sandboxed test execution.
Cross-task generalization probe for the code GRPO pipeline (train MBPP/APPS -> eval HumanEval+).

    python script/eval_code.py --model_path Qwen/Qwen3-1.7B --bench humanevalplus --out out.json
"""
import argparse, json, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

INSTR = ("You are an expert Python programmer. Complete the function below. "
         "Think step by step, then give the FULL function in a single ```python code block.\n\n")


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip()


def run_one(code, prompt, test, entry_point, timeout=12):
    if entry_point and f"def {entry_point}" not in code:
        code = prompt.rstrip() + "\n" + code            # completion was just the body
    program = code + "\n\n" + test + (f"\n\ncheck({entry_point})\n" if entry_point else "\n")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(program); p = f.name
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
    except Exception:
        ok = False
    finally:
        try: os.unlink(p)
        except OSError: pass
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--bench", default="humanevalplus", choices=["humanevalplus", "mbppplus"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                    help="Qwen3 thinking mode (default off — code completion; thinking overflows the budget)")
    a = ap.parse_args()

    repo = "evalplus/humanevalplus" if a.bench == "humanevalplus" else "evalplus/mbppplus"
    ds = load_dataset(repo)["test"]
    if a.limit: ds = ds.select(range(min(a.limit, len(ds))))

    tok = AutoTokenizer.from_pretrained(a.model_path)
    def make_prompt(ex):
        body = ex["prompt"] if a.bench == "humanevalplus" else (
            f"{ex['prompt']}\nYour function must satisfy:\n" + "\n".join(ex.get("test_list", [])[:3]))
        msgs = [{"role": "user", "content": INSTR + body}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=a.think)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompts = [make_prompt(ex) for ex in ds]

    llm = LLM(model=a.model_path, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, enforce_eager=True)
    sp = SamplingParams(temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens)
    outs = llm.generate(prompts, sp)

    jobs = []
    for i, ex in enumerate(ds):
        code = extract_code(outs[i].outputs[0].text)
        ep = ex.get("entry_point") if a.bench == "humanevalplus" else None
        test = ex["test"] if a.bench == "humanevalplus" else ex["test"]
        jobs.append((code, ex["prompt"], test, ep))
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda j: run_one(*j), jobs))

    n = len(results); acc = sum(results) / n
    recs = [{"task_id": ds[i].get("task_id", i), "correct": bool(results[i]),
             "n_gen_tokens": len(outs[i].outputs[0].token_ids)} for i in range(n)]
    json.dump({"bench": a.bench, "model": a.model_path, "n": n, "pass@1": acc, "records": recs},
              open(a.out, "w"), indent=2)
    print(f"[code-eval] {a.bench} pass@1={acc:.4f} ({sum(results)}/{n}) -> {a.out}")


if __name__ == "__main__":
    main()
