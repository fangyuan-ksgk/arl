"""Per-checkpoint SearchR1 forgetting eval (Q8 Phase A backfill).

For each checkpoint-N under --run: load in vLLM (greedy), run the SAME
multi-turn search loop as training (up to 4 turns; <search>q</search> answered
by the retrieval server, injected as <information>), score EM on the val
split. Writes output/forgetting/<run_basename>/step{N}_test.jsonl
{idx, correct, pred} — same schema as eval_forgetting_domain, so
forgetting_viz --splits test --no_lottery works unchanged.

Usage:
  python script/eval_forgetting_searchr1.py --run output/q8a_searchr1_s1 [--n 300]
Requires the mini retrieval server on :8000 (CPU one is long-running)."""
import argparse, glob, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from searchr1_rollout import parse_search_query, has_answer
from retrieval_client import make_retrieve_fn
from qa_em import extract_solution, em_check


def rollout_batch(llm, tok, prompts_msgs, retrieve_fn, max_turns=4, per_turn=500,
                  lora_req=None):
    """Greedy multi-turn rollout for a batch; returns final full texts."""
    from vllm import SamplingParams
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in prompts_msgs]
    done = [False] * len(texts)
    for _ in range(max_turns):
        idxs = [i for i, d in enumerate(done) if not d]
        if not idxs:
            break
        outs = llm.generate([texts[i] for i in idxs],
                            SamplingParams(temperature=0.0, max_tokens=per_turn,
                                           stop=["</search>", "</answer>"],
                                           include_stop_str_in_output=True),
                            lora_request=lora_req)
        for i, o in zip(idxs, outs):
            seg = o.outputs[0].text
            texts[i] += seg
            if has_answer(seg):
                done[i] = True
                continue
            q = parse_search_query(seg)
            if q is None:
                done[i] = True            # neither answer nor search — dead end
                continue
            texts[i] += "\n<information>" + retrieve_fn(q) + "</information>\n"
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="/home/claudeuser/data/searchr1_trl/validation.parquet")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--save_completions", action="store_true")
    ap.add_argument("--idx_file", default=None, help="JSON list of dataset indices to eval (subset)")
    a = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(a.data).head(a.n)
    msgs = [list(p) for p in df["prompt"]]
    golds = [g["target"] for g in df["ground_truth"]]
    ids = list(range(len(msgs)))
    if a.idx_file:
        keep = sorted(set(json.load(open(a.idx_file))))
        msgs = [msgs[i] for i in keep]
        golds = [golds[i] for i in keep]
        ids = keep
        print(f"[sr1-eval] subset: {len(ids)} queries from {a.idx_file}", flush=True)
    retrieve_fn = make_retrieve_fn()
    retrieve_fn("healthcheck")            # fail fast if the server is down

    run = Path(a.run)
    out = Path("output/forgetting") / run.name
    out.mkdir(parents=True, exist_ok=True)
    ckpts = sorted((int(re.search(r"checkpoint-(\d+)", str(p)).group(1))
                    for p in (p for p in run.glob("checkpoint-*") if (p / "model.safetensors").exists() or (p / "adapter_config.json").exists())), key=int)
    print(f"[sr1-eval] {run.name}: {len(ckpts)} ckpts x {len(df)} val queries", flush=True)

    # LoRA-adapter checkpoints (searchr1/tbench train at lora_rank 32 → 260MB
    # adapter dirs): load the BASE engine once with enable_lora and swap
    # adapters per ckpt — 14x fewer engine loads than full-model dirs.
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    first = f"{a.run}/checkpoint-{ckpts[0]}"
    is_lora = (Path(first) / "adapter_config.json").exists()
    llm = None
    if is_lora:
        base = json.load(open(Path(first) / "adapter_config.json"))["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(base)
        llm = LLM(model=base, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
                  enable_lora=True, max_lora_rank=64, disable_log_stats=True)
        print(f"[sr1-eval] LoRA mode: base={base}", flush=True)
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
        rows = []
        for b0 in range(0, len(msgs), a.batch):
            texts = rollout_batch(llm, tok, msgs[b0:b0 + a.batch], retrieve_fn,
                                  lora_req=lora_req)
            for j, t in enumerate(texts):
                i = b0 + j
                pred = extract_solution(t) or ""
                rec = {"idx": ids[i], "step": c, "split": "test",
                       "correct": bool(em_check(pred, golds[i])), "pred": pred}
                if a.save_completions:
                    rec["completion"] = t
                rows.append(rec)
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        acc = 100 * sum(r["correct"] for r in rows) / len(rows)
        print(f"[sr1-eval] step {c}: EM {acc:.1f}", flush=True)
        if not is_lora:                 # LoRA mode reuses one base engine
            del llm
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
