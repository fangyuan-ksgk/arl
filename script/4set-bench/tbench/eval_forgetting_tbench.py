"""Per-checkpoint Terminal-Bench forgetting eval (Q8 Phase A backfill).

For each checkpoint-N under --run: vLLM greedy generation for the 32 curated
val tasks, each scored by ITS OWN task verifier (proot sandbox + pytest, via
tbench_trl.tbench_reward._score_one — the same scoring as training). Writes
output/forgetting/<run_basename>/step{N}_test.jsonl {idx, correct, pred} —
standard schema, forgetting_viz-compatible.
POWER NOTE: n=32 → per-ckpt acc granularity 3.1%; per-domain lottery deltas
under ~15pt unreadable here — cross-domain pattern carries the claim.

Usage: python script/eval_forgetting_tbench.py --run output/q8a_tbench_s1"""
import argparse, glob, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # tbench_reward does `import sandbox` (sibling-relative)
from tbench_reward import _score_one
from sandbox import check_available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="/home/claudeuser/data/tbench_trl/validation.parquet")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=1024)
    a = ap.parse_args()

    check_available()                      # fail fast if proot/verifier venv gone
    import pandas as pd
    df = pd.read_parquet(a.data)
    msgs = [list(p) for p in df["prompt"]]

    run = Path(a.run)
    out = Path("output/forgetting") / run.name
    out.mkdir(parents=True, exist_ok=True)
    ckpts = sorted((int(re.search(r"checkpoint-(\d+)", str(p)).group(1))
                    for p in (p for p in run.glob("checkpoint-*") if (p / "model.safetensors").exists() or (p / "adapter_config.json").exists())), key=int)
    print(f"[tb-eval] {run.name}: {len(ckpts)} ckpts x {len(df)} val tasks", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    first = f"{a.run}/checkpoint-{ckpts[0]}"
    is_lora = (Path(first) / "adapter_config.json").exists()
    llm = None
    if is_lora:
        base = json.load(open(Path(first) / "adapter_config.json"))["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(base)
        llm = LLM(model=base, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
                  enable_lora=True, max_lora_rank=64, disable_log_stats=True)
        print(f"[tb-eval] LoRA mode: base={base}", flush=True)
    for c in ckpts:
        f = out / f"step{c}_test.jsonl"
        if f.exists() and sum(1 for _ in open(f)) == len(df):
            continue
        lora_req = None
        if is_lora:
            lora_req = LoRARequest(f"ck{c}", c, f"{a.run}/checkpoint-{c}")
        else:
            tok = AutoTokenizer.from_pretrained(f"{a.run}/checkpoint-{c}")
            llm = LLM(model=f"{a.run}/checkpoint-{c}", dtype="bfloat16",
                      gpu_memory_utilization=a.gpu_mem, disable_log_stats=True)
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        outs = llm.generate(prompts, sp, lora_request=lora_req)
        rows = []
        for i, o in enumerate(outs):
            text = o.outputs[0].text
            score = _score_one(text, df.iloc[i]["task_path"],
                               float(df.iloc[i]["exec_timeout"]),
                               float(df.iloc[i]["verify_timeout"]))
            rows.append({"idx": i, "step": c, "split": "test",
                         "correct": bool(score >= 1.0), "pred": text[-300:]})
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        acc = 100 * sum(r["correct"] for r in rows) / len(rows)
        print(f"[tb-eval] step {c}: {acc:.1f} ({sum(r['correct'] for r in rows)}/{len(rows)})", flush=True)
        if not is_lora:
            del llm
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
