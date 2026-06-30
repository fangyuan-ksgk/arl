"""Seed-scaling experiment: does merge beat avg, and does the lottery gap reach >20pt, as the
number of independently-trained seeds K grows?

Motivation: the K=3 branch-merge sweep found merged ≈ avg (no soup benefit) and only a ~12pt
union−avg gap. Hypothesis (user's single-GPU experience): souping HELPS and the gap is LARGE only
with enough seeds (~8); K=3 under-samples the diversity. This script trains N independent Dr.GRPO
seeds from base, evals each, then reports avg(K), union(K), and soup(K) for K=1..N.

Pipeline per seed: full Dr.GRPO recipe (the locked Item-0 config), 120 steps from base, distinct
seed → distinct data order + sampling RNG. 3 seeds train concurrently (one per A100), in waves.

Usage:
    python script/run_seed_scaling.py --n_seeds 8 --steps 120 --out output/seed_scaling_gsm8k
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
GRPO = os.path.join(HERE, "grpo_gsm8k.py")
EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_gsm8k.py")
sys.path.insert(0, HERE)
import merge_soup  # noqa: E402


def env(gpu, port):
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{PROJECT}:{e.get('PYTHONPATH','')}"
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["MASTER_ADDR"] = "127.0.0.1"; e["MASTER_PORT"] = str(port)
    e.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    e.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return e


def gpu_free(gpu):
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
            text=True).strip().splitlines()[0])
    except Exception:
        return 1 << 30


def wait_free(gpus, need=60000, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(gpu_free(g) >= need for g in gpus):
            return
        time.sleep(5)


def train_seed(seed, gpu, out_dir, steps, a, lr_override=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lr = lr_override if lr_override is not None else a.lr
    cmd = [sys.executable, GRPO, "--model", a.base_model, "--output_dir", out_dir,
           "--max_steps", str(steps), "--num_generations", "8", "--max_completion_length", "1024",
           "--per_device_train_batch_size", "8", "--gradient_accumulation_steps", str(a.grad_accum),
           "--learning_rate", str(lr), "--lr_scheduler_type", a.lr_scheduler, "--warmup_steps", str(a.warmup),
           "--loss_type", "dr_grpo", "--scale_rewards", "none",
           "--logging_steps", "20", "--use_vllm", "--vllm_mode", "colocate",
           "--vllm_gpu_memory_utilization", "0.35", "--gradient_checkpointing",
           "--save_strategy", "steps", "--save_steps_list", str(steps), "--eval_steps", "0",
           "--no-mbe_velocity_reward", "--seed", str(seed), "--report_to", "none"]
    if a.mask_truncated:
        cmd.append("--mask_truncated_completions")
    log = open(os.path.join(out_dir, "train.log"), "w")
    return subprocess.Popen(cmd, env=env(gpu, 29800 + seed), stdout=log, stderr=subprocess.STDOUT), log


def evaluate(model_dir, gpu, out_json, a):
    # EVAL AT TRAINING BUDGET (1024) — eval budget must match training (user rule 2026-06-29).
    cmd = [sys.executable, EVAL, "--model_path", model_dir, "--out", out_json,
           "--max_tokens", "1024", "--temperature", "0.0", "--max_model_len", "1536",
           "--gpu_memory_utilization", "0.85"]
    if a.eval_limit:
        cmd += ["--limit", str(a.eval_limit)]
    p = subprocess.run(cmd, env=env(gpu, 29900), capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[eval] FAIL {model_dir}\n{p.stderr[-1200:]}", flush=True); return None
    return json.loads(Path(out_json).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default=os.path.join(PROJECT, "output/seed_scaling_gsm8k"))
    ap.add_argument("--eval_limit", type=int, default=0, help="0 = full 1319")
    # rectified-config knobs (compare.md): higher lr + smaller batch + mask off -> more divergence
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--mask_truncated", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr_scheduler", default="constant", help="constant | linear | cosine (decay tames lr5e-6 collapse)")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--lr_list", default="", help="comma-sep per-seed LRs (heterogeneous seeds, Exp#2); empty=use --lr")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    # heterogeneous per-seed LR (Exp#2): seed s uses lr_list[s] if provided, else a.lr
    lr_list = [float(x) for x in a.lr_list.split(",")] if a.lr_list else None

    # ---- Phase 1: train N seeds in waves of len(gpus) ----
    ckpts = {}
    for w0 in range(0, a.n_seeds, len(gpus)):
        wave = list(range(w0, min(w0 + len(gpus), a.n_seeds)))
        wait_free(gpus[:len(wave)])
        print(f"[train] wave seeds {wave}", flush=True)
        procs = []
        for i, s in enumerate(wave):
            d = str(out / f"seed{s}")
            lr_s = lr_list[s] if (lr_list and s < len(lr_list)) else None
            p, log = train_seed(s, gpus[i], d, a.steps, a, lr_override=lr_s)
            procs.append((s, d, p, log))
        for s, d, p, log in procs:
            rc = p.wait(); log.close()
            ck = os.path.join(d, f"checkpoint-{a.steps}")
            ok = rc == 0 and os.path.exists(os.path.join(ck, "model.safetensors"))
            print(f"[train] seed{s} rc={rc} ckpt_ok={ok}", flush=True)
            if ok:
                ckpts[s] = ck
    seeds = sorted(ckpts)
    print(f"[train] done; trained seeds={seeds}", flush=True)

    # ---- Phase 2: eval each seed individually (full set) ----
    wait_free([gpus[0]])
    correct, accs = {}, {}
    for s in seeds:
        ev = evaluate(ckpts[s], gpus[0], str(out / f"eval_seed{s}.json"), a)
        if ev:
            correct[s] = ev["correct"]; accs[s] = ev["accuracy"]
            print(f"[eval] seed{s} acc={ev['accuracy']:.4f}", flush=True)

    # ---- Phase 3: soup(K) for K=2..N (cumulative: first K seeds) + eval ----
    soup_acc = {}
    Ks = [k for k in range(2, len(seeds) + 1)]
    for K in Ks:
        subset = seeds[:K]
        mdir = str(out / f"soup_K{K}")
        merge_soup.merge([ckpts[s] for s in subset], mdir)
        wait_free([gpus[0]])
        ev = evaluate(mdir, gpus[0], str(out / f"eval_soup_K{K}.json"), a)
        if ev:
            soup_acc[K] = ev["accuracy"]
            print(f"[soup] K={K} soup_acc={ev['accuracy']:.4f}", flush=True)

    # ---- Phase 4: union(K)/avg(K) from per-seed correct arrays (cheap) ----
    rows = []
    n = len(next(iter(correct.values())))
    for K in range(1, len(seeds) + 1):
        sub = seeds[:K]
        avg = sum(accs[s] for s in sub) / K
        union = sum(1 for i in range(n) if any(correct[s][i] for s in sub)) / n
        rows.append({"K": K, "avg": avg, "union": union, "gap": union - avg,
                     "soup": soup_acc.get(K), "soup_minus_avg": (soup_acc[K] - avg) if K in soup_acc else None})
    result = {"config": vars(a), "seeds": seeds, "per_seed_acc": accs, "rows": rows}
    res_path.write_text(json.dumps(result, indent=2))

    print("\n=== SEED SCALING (GSM8K Dr.GRPO) ===", flush=True)
    print(f"{'K':>2} {'avg':>7} {'union':>7} {'gap':>7} {'soup':>7} {'soup-avg':>9}", flush=True)
    for r in rows:
        soup = f"{r['soup']:.4f}" if r['soup'] is not None else "  -  "
        sma = f"{r['soup_minus_avg']:+.4f}" if r['soup_minus_avg'] is not None else "  -  "
        print(f"{r['K']:>2} {r['avg']:.4f} {r['union']:.4f} {r['gap']:+.4f} {soup:>7} {sma:>9}", flush=True)
    print(f"-> {res_path}", flush=True)


if __name__ == "__main__":
    main()
