"""Seed-scaling at scale (LoRA + MATH): the capacity-wall test for bigger models.

For Qwen3-1.7B/4B/8B we train with LoRA (frees optimizer memory). Each seed trains an adapter;
we then merge each adapter into the base to get a full model, and run the same comparison as the
0.6B seed-scaling: per-seed acc, soup (dense weight-merge of the full models), majority-vote
(output mixture), and union (oracle ceiling) as a function of K. 

Target: 
    report avg acc | union acc | soup acc | majority vote acc, validate union acc > soup acc > avg acc
    probe on what config improves soup acc the most

LoRA branches save adapter_model.safetensors; eval/soup need full weights, so we peft-merge each
adapter into the base once, up front.

Usage:
    python script/run_seed_scaling_lora.py --model Qwen/Qwen3-1.7B --n_seeds 4 --steps 100 \
        --out output/seed_scaling_math_1p7b
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); PROJECT = os.path.dirname(HERE)
GRPO_MATH = os.path.join(HERE, "grpo_math.py")
EVAL_MATH = os.path.join(HERE, "eval_math.py")
sys.path.insert(0, HERE)
import merge_soup  # noqa: E402


def base_env(gpu, port):
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{PROJECT}:{e.get('PYTHONPATH','')}"; e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["MASTER_ADDR"] = "127.0.0.1"; e["MASTER_PORT"] = str(port)
    e.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1"); e.setdefault("TOKENIZERS_PARALLELISM", "false")
    e.setdefault("VLLM_LOGGING_LEVEL", "WARNING"); e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return e


def gpu_free(g):
    try:
        return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits","-i",str(g)],text=True).strip().splitlines()[0])
    except Exception:
        return 1<<30

def wait_free(gpus, need=55000, timeout=300):
    t0=time.time()
    while time.time()-t0<timeout:
        if all(gpu_free(g)>=need for g in gpus): return
        time.sleep(5)

# [Question 1]. we are not using a seperate device for each seed's training right? 
#               my question is if we'd like to do server-model training, how is this script 

def train_seed(seed, gpu, out_dir, a):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, GRPO_MATH, "--model", a.model, "--output_dir", out_dir,
           "--max_steps", str(a.steps), "--num_generations", str(a.num_gen),
           "--max_completion_length", str(a.maxlen),
           "--per_device_train_batch_size", str(a.num_gen), "--gradient_accumulation_steps", str(a.grad_accum),
           "--learning_rate", str(a.lr), "--lr_scheduler_type", a.lr_scheduler, "--warmup_steps", str(a.warmup),
           "--loss_type", "dr_grpo", "--scale_rewards", "none",
           *(["--mask_truncated_completions"] if a.mask_truncated else []),
           "--use_lora", "--lora_r", str(a.lora_r), "--lora_alpha", str(2*a.lora_r),
           "--logging_steps", "20", "--use_vllm", "--vllm_mode", "colocate",
           "--vllm_gpu_memory_utilization", str(a.vllm_mem), "--gradient_checkpointing",
           "--save_strategy", "steps", "--save_steps_list", str(a.steps), "--eval_steps", "0",
           "--seed", str(seed), "--report_to", "none"]
    if a.vllm_max_model_len > 0:
        cmd += ["--vllm_max_model_len", str(a.vllm_max_model_len)]
    log = open(os.path.join(out_dir, "train.log"), "w")
    return subprocess.Popen(cmd, env=base_env(gpu, 29810 + seed), stdout=log, stderr=subprocess.STDOUT), log


def lora_to_full(adapter_dir, base_model, full_dir):
    """Merge a LoRA adapter into the base and save a full model dir (for eval + soup)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    Path(full_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(full_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(full_dir)
    del base, merged
    torch.cuda.empty_cache()
    return full_dir


def evaluate(model_dir, gpu, out_json, a):
    cmd = [sys.executable, EVAL_MATH, "--model_path", model_dir, "--out", out_json,
           "--max_tokens", str(a.maxlen), "--temperature", "0.0",
           "--gpu_memory_utilization", "0.85"]
    p = subprocess.run(cmd, env=base_env(gpu, 29910), capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[eval] FAIL {model_dir}\n{p.stderr[-1200:]}"); return None
    return json.loads(Path(out_json).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--out", default=os.path.join(PROJECT, "output/seed_scaling_math_1p7b"))
    ap.add_argument("--num_gen", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--mask_truncated", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr_scheduler", default="constant")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--vllm_mem", type=float, default=0.25)
    ap.add_argument("--vllm_max_model_len", type=int, default=0,
                    help="Cap vLLM context for big models (8B) so KV fits a small colocate pool. 0=native.")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- train N LoRA seeds in waves ----
    adapters = {}
    for w0 in range(0, a.n_seeds, len(gpus)):
        wave = list(range(w0, min(w0 + len(gpus), a.n_seeds)))
        wait_free(gpus[:len(wave)]); print(f"[train] wave {wave}", flush=True)
        procs = [(s, str(out / f"seed{s}"), *train_seed(s, gpus[i], str(out / f"seed{s}"), a)) for i, s in enumerate(wave)]
        for s, d, p, log in procs:
            rc = p.wait(); log.close()
            ad = os.path.join(d, f"checkpoint-{a.steps}")
            ok = rc == 0 and os.path.exists(os.path.join(ad, "adapter_model.safetensors"))
            print(f"[train] seed{s} rc={rc} adapter_ok={ok}", flush=True)
            if ok: adapters[s] = ad
    seeds = sorted(adapters)

    # ---- merge each adapter -> full model ----
    fulls = {}
    for s in seeds:
        fulls[s] = lora_to_full(adapters[s], a.model, str(out / f"full_seed{s}"))
        print(f"[merge] seed{s} adapter -> full", flush=True)

    # ---- eval each seed ----
    wait_free([gpus[0]]); correct, accs = {}, {}
    for s in seeds:
        ev = evaluate(fulls[s], gpus[0], str(out / f"eval_seed{s}.json"), a)
        if ev: correct[s] = ev["correct"]; accs[s] = ev["accuracy"]; print(f"[eval] seed{s} acc={ev['accuracy']:.4f}", flush=True)

    # ---- soup(K) eval + majority/union(K) ----
    n = len(next(iter(correct.values())))
    recs = {s: json.load(open(out / f"eval_seed{s}.json"))["records"] for s in seeds}
    def is_corr(p, g):
        try: return float(p) == float(g)
        except: return str(p) == str(g)
    rows = []
    for K in range(1, len(seeds) + 1):
        sub = seeds[:K]
        avg = sum(accs[s] for s in sub) / K
        union = sum(1 for i in range(n) if any(correct[s][i] for s in sub)) / n
        # majority vote over predicted answers
        maj = 0
        for i in range(n):
            preds = [str(recs[s][i]["pred"]) for s in sub if recs[s][i]["pred"] not in ("", "None")]
            if preds and is_corr(Counter(preds).most_common(1)[0][0], recs[sub[0]][i]["gold"]): maj += 1
        maj /= n
        soup = None
        if K >= 2:
            mdir = str(out / f"soup_K{K}"); merge_soup.merge([fulls[s] for s in sub], mdir)
            wait_free([gpus[0]]); ev = evaluate(mdir, gpus[0], str(out / f"eval_soup_K{K}.json"), a)
            soup = ev["accuracy"] if ev else None
        rows.append({"K": K, "avg": avg, "union": union, "gap": union - avg, "majority": maj, "soup": soup})
        (out / "results.json").write_text(json.dumps({"model": a.model, "rows": rows, "per_seed_acc": accs}, indent=2))

    print(f"\n=== SEED SCALING ({a.model}, MATH, LoRA) ===", flush=True)
    print(f"{'K':>2} {'avg':>7} {'soup':>7} {'majority':>8} {'union':>7}", flush=True)
    for r in rows:
        soup = f"{r['soup']:.4f}" if r['soup'] is not None else "  -  "
        print(f"{r['K']:>2} {r['avg']:.4f} {soup:>7} {r['majority']:.4f}  {r['union']:.4f}", flush=True)


if __name__ == "__main__":
    main()
