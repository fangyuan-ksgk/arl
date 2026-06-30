"""Branch-train-merge GRPO with period P

    P = 1    ->  merge every step          == the "undelayed / fully-synchronized" baseline
    P = T    ->  merge once, at the very end == independent seeds + a single final soup
                                               (the pure "lottery gap" endpoint)
    1<P<T    ->  delayed synchronization     == what we hope is optimal

Usage (one P setting):
    python script/local_sgd_grpo.py --tag gsm8k_P4 --period 4 --total_steps 120 \
        --gpus 0,1,2 --seeds 0,1,2 --out output/local_sgd/gsm8k_P4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import merge_soup 
import diloco_merge

# Per-task wiring: inner GRPO script, eval script, family (train-cmd template), budgets, metric.
# Reward gadgets (MBE-velocity, virtual-rollout) are OFF by default and controlled per-run via
# --mbe_velocity_reward / --virtual_rollout (R1d), passed through to every branch.

# R1g: per-task wiring. family selects the train-cmd template (the inner scripts have divergent CLIs);
# maxlen/eval_tok are the per-task train + eval budgets (eval budget == train budget, the eval rule);
# metric is the eval-JSON accuracy key; majority=False where output-vote is undefined (code/vlm).
TASKS = {
    "gsm8k": {"script": os.path.join(HERE, "grpo_gsm8k.py"), "eval": os.path.join(HERE, "eval_gsm8k.py"),
              "family": "text", "maxlen": 1024, "eval_tok": 1024,
              "metric": "accuracy", "majority": True, "eval_flags": [], "train_extra": [], "branch_flags": []},
    "math":  {"script": os.path.join(HERE, "grpo_math.py"), "eval": os.path.join(HERE, "eval_math.py"),
              "family": "text", "maxlen": 3072, "eval_tok": 3072,
              "metric": "accuracy", "majority": True, "eval_flags": [], "train_extra": [], "branch_flags": []},
    "mbpp":  {"script": os.path.join(HERE, "grpo_code.py"), "eval": os.path.join(HERE, "eval_code.py"),
              "family": "code", "maxlen": 1024, "eval_tok": 1024,
              "metric": "pass@1", "majority": False, "eval_flags": ["--bench", "humanevalplus", "--no-think"],
              "train_extra": ["--dataset", "mbppplus"], "branch_flags": []},
    "geo8k": {"script": os.path.join(HERE, "grpo_geometry.py"), "eval": os.path.join(HERE, "eval_geometry.py"),
              "family": "vlm", "maxlen": 1024, "eval_tok": 1024,
              "metric": "acc", "majority": False, "eval_flags": [], "train_extra": ["--no_geoqa"],
              "branch_flags": []},   # geo8k: Qwen3.5-4B + native-vLLM (.venv-vllm: vllm 0.24 + transformers 5.13.dev0)
}


def gpu_free_mib(gpu):
    """Return free MiB on one GPU via nvidia-smi (driver-level, sees other processes)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu)], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 1 << 30  # if query fails, don't block


def wait_gpu_free(gpus, need_mib=60000, timeout=180):
    """Block until every GPU in `gpus` has >= need_mib free. CUDA frees lazily after a
    process exits; launching the next stage too early OOMs (claude.md orphan-VRAM trap)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(gpu_free_mib(g) >= need_mib for g in gpus):
            return True
        time.sleep(3)
    frees = {g: gpu_free_mib(g) for g in gpus}
    print(f"[gpu] WARN: still not free after {timeout}s: {frees}", flush=True)
    return False


def kill_orphans(gpus):
    """Kill stale compute procs ON THE GPUs THIS RUN WILL USE (GPU-scoped via nvidia-smi compute-apps),
    so it clears our own GPUs without touching other jobs running on other GPUs."""
    for g in gpus:
        try:
            pids = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits", "-i", str(g)],
                text=True).split()
            for pid in pids:
                if pid.strip().isdigit():
                    subprocess.run(["kill", "-9", pid.strip()], stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(2)


def sh_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT}:{env.get('PYTHONPATH','')}"
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # reduce allocator fragmentation: training + colocate vLLM share one GPU
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _launch_vllm_server(gpu, model, port, a, out_dir):
    """R1d: start one trl vLLM server (server mode) on `gpu`; return Popen once /health responds."""
    import urllib.request
    env = sh_env(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = ["trl", "vllm-serve", "--model", model, "--port", str(port),
           "--gpu_memory_utilization", "0.9"]
    proc = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(out_dir, "vllm_server.log"), "w"),
                            stderr=subprocess.STDOUT)
    for _ in range(180):
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health/", timeout=2); return proc
        except Exception:
            time.sleep(2)
    return proc


def build_train_cmd(tc, a, init_model, out_dir, steps, seed, vllm_flags, reward_flags):
    """Per-family train command (the inner GRPO scripts have divergent CLIs — R1g)."""
    py = sys.executable
    lora = ["--use_lora", "--lora_r", str(a.lora_r)] if a.use_lora else []
    base = [py, tc["script"], "--model", init_model, "--output_dir", out_dir,
            "--max_steps", str(steps), "--num_generations", str(a.num_generations),
            "--max_completion_length", str(a.maxlen), "--learning_rate", str(a.learning_rate)]
    if tc["family"] == "vlm":   # grpo_geometry: --grad_accum, always-LoRA, native-vLLM qwen3_5 (vLLM 0.24)
        return base + ["--grad_accum", str(a.grad_accum), "--lora_r", str(a.lora_r),
                       "--use_vllm", "--vllm_model_impl", "vllm",
                       "--vllm_gpu_mem", str(a.vllm_gpu_mem)] + tc["train_extra"] + tc["branch_flags"]
    cmd = base + ["--per_device_train_batch_size", str(a.num_generations),
                  "--gradient_accumulation_steps", str(a.grad_accum),
                  "--gradient_checkpointing", "--eval_steps", "0"] + lora + vllm_flags + reward_flags
    if tc["family"] == "text":  # grpo_gsm8k / grpo_math: full Dr.GRPO recipe + masking + save-at-step
        cmd += ["--lr_scheduler_type", a.lr_scheduler, "--warmup_steps", str(a.warmup),
                "--loss_type", a.loss_type, "--scale_rewards", a.scale_rewards, "--logging_steps", "5",
                "--save_strategy", "steps", "--save_steps_list", str(steps), "--seed", str(seed),
                "--report_to", "none"] + (["--mask_truncated_completions"] if a.mask_truncated else [])
    # family == "code" (grpo_code): subset CLI; reward_flags (velocity/virtual) already appended above
    return cmd + tc["train_extra"] + tc["branch_flags"]


def launch_branch(train_gpu, serve_gpu, init_model, out_dir, steps, seed, a, port):
    """Start one GRPO branch on `train_gpu`. In server mode (R1f) a dedicated vLLM server runs on
    `serve_gpu` (2 GPUs/branch — avoids the model×max_new_tokens colocate OOM). Returns (proc, log, server)."""
    env = sh_env()
    env["CUDA_VISIBLE_DEVICES"] = str(train_gpu)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tc = TASKS[a.task]
    server = None
    if a.no_vllm:  # HF generation (archs vLLM can't serve, e.g. gemma4); text/code honor --no-use_vllm
        vllm_flags = ["--no-use_vllm"]
    elif a.vllm_mode == "server" and tc["family"] != "vlm":
        sport = port + 1000
        server = _launch_vllm_server(serve_gpu, init_model, sport, a, out_dir)
        vllm_flags = ["--use_vllm", "--vllm_mode", "server",
                      "--vllm_server_host", "127.0.0.1", "--vllm_server_port", str(sport)]
    else:  # colocate (default); vlm family ignores these and forces HF-gen in build_train_cmd
        vllm_flags = ["--use_vllm", "--vllm_mode", "colocate",
                      "--vllm_gpu_memory_utilization", str(a.vllm_gpu_mem)]
    reward_flags = (["--mbe_velocity_reward", "--mbe_velocity_scale", str(a.mbe_velocity_scale),
                     "--mbe_velocity_clip", str(a.mbe_velocity_clip)]
                    if a.mbe_velocity_reward else ["--no-mbe_velocity_reward"])
    if a.virtual_rollout != "none":
        reward_flags += ["--virtual_rollout", a.virtual_rollout, "--virtual_max_reward", str(a.virtual_max_reward)]
    cmd = build_train_cmd(tc, a, init_model, out_dir, steps, seed, vllm_flags, reward_flags)
    log = open(os.path.join(out_dir, "train.log"), "w")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log, server


def run_eval(model_dir, gpu, out_json, a, limit=None, pass_k=1):
    """Greedy eval at the task's budget (== train budget). pass_k>1 -> pass@k (text tasks only).
    Per-family CLI (text/code use --max_tokens; vlm uses --max_new_tokens). Normalizes metric->'accuracy'."""
    tc = TASKS[a.task]
    env = sh_env(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    py = sys.executable
    mt = a.eval_tok
    lim = ["--limit", str(limit)] if limit else []
    pk = ["--pass_k", str(pass_k)] if (pass_k > 1 and tc["family"] == "text") else []
    if tc["family"] == "vlm":
        cmd = [py, tc["eval"], "--model_path", model_dir, "--out", out_json,
               "--max_new_tokens", str(mt)] + lim + tc["eval_flags"]
    elif tc["family"] == "code":
        cmd = [py, tc["eval"], "--model_path", model_dir, "--out", out_json,
               "--max_tokens", str(mt), "--max_model_len", str(mt + 1024),
               "--gpu_memory_utilization", "0.85"] + lim + tc["eval_flags"]
    else:  # text
        cmd = [py, tc["eval"], "--model_path", model_dir, "--out", out_json,
               "--max_tokens", str(mt), "--temperature", "0.0", "--max_model_len", str(mt + 1024),
               "--gpu_memory_utilization", "0.85"] + lim + pk
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[eval] FAILED {model_dir}\n{p.stderr[-1500:]}", flush=True)
        return None
    ev = json.loads(Path(out_json).read_text())
    if "accuracy" not in ev and tc["metric"] in ev:
        ev["accuracy"] = ev[tc["metric"]]            # normalize pass@1 / acc -> accuracy
    return ev


def main():
    ap = argparse.ArgumentParser(description="Branch-train-merge (Local SGD) GRPO")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="Run output dir")
    ap.add_argument("--task", default="gsm8k", choices=list(TASKS),
                    help="Which GRPO task: gsm8k or math.")
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--period", type=int, required=True, help="Sync period P (steps between merges)")
    ap.add_argument("--total_steps", type=int, required=True, help="Total optimizer steps per branch T")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--seeds", default="0,1,2", help="One seed per branch (K = len)")
    # training hyperparams (kept close to the multiseed lottery-gap regime)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=0,
                    help="0 = use the task's default budget (gsm8k 1024, math 3072, ...) — R1g.")
    ap.add_argument("--learning_rate", type=float, default=2e-6)   # validated stable to 120 steps (Item 0)
    ap.add_argument("--lr_scheduler", default="constant", help="constant|linear|cosine (per-round)")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--vllm_gpu_mem", type=float, default=0.35)
    # R1d: vLLM mode passthrough. 'server' launches one trl vllm-serve per branch (same GPU, port base+idx)
    # and connects the branch's GRPO run to it; 'colocate' keeps vLLM in-process (default).
    ap.add_argument("--vllm_mode", default="colocate", choices=["colocate", "server"])
    ap.add_argument("--vllm_server_base_port", type=int, default=8100)
    # R1d: pass the reward-shaping gadgets through to each branch's GRPO run.
    ap.add_argument("--mbe_velocity_reward", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mbe_velocity_scale", type=float, default=5.0)
    ap.add_argument("--mbe_velocity_clip", type=float, default=1.0)
    ap.add_argument("--virtual_rollout", default="none",
                    choices=["none", "insert_max", "insert_min", "insert_max_min",
                             "insert_max_all_incorrect", "insert_max_mixed"])
    ap.add_argument("--virtual_max_reward", type=float, default=1.2)
    ap.add_argument("--use_lora", action="store_true", help="R1f: optional LoRA branches.")
    ap.add_argument("--lora_r", type=int, default=512)
    # vLLM-unsupported archs (e.g. google/gemma-4-E4B-it [gemma4]): force HF generation in the branches.
    ap.add_argument("--no_vllm", action="store_true",
                    help="Force HF generation in the branches (archs vLLM can't serve, e.g. gemma4).")
    # Dr.GRPO recipe (Item 0): unbiased length + no std scaling + anti length-collapse
    ap.add_argument("--loss_type", default="dr_grpo", choices=["grpo", "dapo", "bnpo", "dr_grpo"])
    ap.add_argument("--scale_rewards", default="none", choices=["group", "batch", "none"])
    ap.add_argument("--mask_truncated", action=argparse.BooleanOptionalAction, default=True,
                    help="Mask truncated completions from the loss (prevents CoT length collapse).")
    # eval cadence
    ap.add_argument("--eval_limit", type=int, default=0, help="Per-round eval subset size (0=skip per-round eval)")
    ap.add_argument("--final_eval_limit", type=int, default=0, help="Final eval size (0=full 1319)")
    ap.add_argument("--final_pass_k", type=int, default=8,
                    help="pass@k of the final merged model (text tasks; 0/1=skip). Costs k x generation.")
    ap.add_argument("--seed_round_offset", type=int, default=1000,
                    help="Per-round seed = base_seed*offset + round, so each round sees fresh data ordering.")
    # merge step (Item 2): soup == memoryless; diloco == memory-ful outer optimizer
    ap.add_argument("--merge_mode", default="soup",
                    choices=["soup", "diloco", "diloco_decoupled"],
                    help="soup = uniform average (memoryless). diloco = outer SGD+Nesterov momentum "
                         "(memory). diloco_decoupled = decoupled-momentum variant.")
    ap.add_argument("--outer_lr", type=float, default=0.7, help="DiLoCo outer LR (soup uses 1.0).")
    ap.add_argument("--outer_momentum", type=float, default=0.9, help="DiLoCo outer momentum mu.")
    ap.add_argument("--no_nesterov", action="store_true", help="Use heavy-ball instead of Nesterov.")
    a = ap.parse_args()

    gpus = [int(x) for x in a.gpus.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    K = len(seeds)
    tc = TASKS[a.task]
    a.maxlen = a.max_completion_length if a.max_completion_length > 0 else tc["maxlen"]
    a.eval_tok = tc["eval_tok"]
    # R1f GPU layout: colocate = 1 GPU/branch; server = 2 GPUs/branch (vLLM serve + training).
    if a.vllm_mode == "server":
        assert len(gpus) >= 2 * K, f"server mode needs 2 GPUs/branch -> {2*K} for K={K} (got {len(gpus)})"
        branch_gpus = [(gpus[2 * b + 1], gpus[2 * b]) for b in range(K)]   # (train_gpu, serve_gpu)
    else:
        assert K <= len(gpus), f"need >= {K} gpus for {K} branches"
        branch_gpus = [(gpus[b], None) for b in range(K)]
    P, T = a.period, a.total_steps
    n_rounds = (T + P - 1) // P

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {"tag": a.tag, "config": vars(a), "K": K, "n_rounds": n_rounds, "rounds": []}
    results_path = out / "results.json"

    kill_orphans(gpus)   # GPU-scoped: only clears the GPUs this run uses (won't touch other jobs)
    print(f"=== Local-SGD GRPO  tag={a.tag}  P={P}  T={T}  K={K}  rounds={n_rounds} ===", flush=True)
    print(f"    seeds={seeds} gpus={gpus[:K]} lr={a.learning_rate} ng={a.num_generations} ga={a.grad_accum}", flush=True)

    # Eval at a fixed set of step-milestones (NOT every round) so eval cost is independent
    # of P: small-P runs have many rounds but we still eval only ~6 times + the final step.
    n_milestones = max(1, min(6, n_rounds))
    milestones = sorted(set(
        [int(round(T * k / n_milestones)) for k in range(1, n_milestones + 1)] + [T]
    ))
    init_model = a.base_model
    prev_merged_weights = None       # path to delete once consumed
    done_steps = 0
    t_start = time.time()

    for r in range(n_rounds):
        this_P = min(P, T - done_steps)
        round_dir = out / f"round{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        is_final = (r == n_rounds - 1)
        t_r = time.time()

        # ensure GPUs are clean before launching this round's branches (prior eval freed)
        used_gpus = sorted({g for pair in branch_gpus for g in pair if g is not None})
        wait_gpu_free(used_gpus)

        # ---- launch K branches in parallel (1 GPU/branch colocate, 2 GPUs/branch server) ----
        procs, logs, ckpts, servers = [], [], [], []
        for b in range(K):
            bdir = str(round_dir / f"branch{b}")
            seed = seeds[b] * a.seed_round_offset + r
            port = 29500 + (r * K + b) % 2000
            train_gpu, serve_gpu = branch_gpus[b]
            p, log, server = launch_branch(train_gpu, serve_gpu, init_model, bdir, this_P, seed, a, port)
            procs.append(p); logs.append(log)
            if server is not None:
                servers.append(server)
            ckpts.append(os.path.join(bdir, f"checkpoint-{this_P}"))
        rcs = [p.wait() for p in procs]
        for s in servers:           # R1d: tear down per-branch vLLM servers
            s.terminate()
        for log in logs:
            log.close()
        if any(rc != 0 for rc in rcs):
            print(f"[round {r}] branch failure rcs={rcs}; see {round_dir}/branch*/train.log", flush=True)
            results["error"] = f"round {r} branch rcs={rcs}"
            results_path.write_text(json.dumps(results, indent=2))
            sys.exit(1)
        for c in ckpts:
            if not os.path.exists(os.path.join(c, "model.safetensors")):
                print(f"[round {r}] missing checkpoint {c}", flush=True)
                results["error"] = f"round {r} missing {c}"
                results_path.write_text(json.dumps(results, indent=2))
                sys.exit(1)

        # branch processes have exited; wait for their VRAM to actually free before eval
        wait_gpu_free([gpus[0]])

        # ---- merge: uniform soup (memoryless) or DiLoCo outer step (memory-ful) ----
        merged_dir = str(round_dir / "merged")
        if a.merge_mode == "soup":
            merge_soup.merge(ckpts, merged_dir)
        else:
            diloco_merge.outer_merge(
                prev_global=init_model, ckpt_dirs=ckpts, out_dir=merged_dir,
                momentum_path=str(out / "outer_momentum.pt"),
                outer_lr=a.outer_lr, momentum=a.outer_momentum,
                nesterov=(not a.no_nesterov),
                decoupled=(a.merge_mode == "diloco_decoupled"))
        done_steps += this_P

        rec = {"round": r, "steps_this_round": this_P, "cum_steps": done_steps}

        # ---- eval merged (+ branches on the final round, for the lottery gap) ----
        hit_milestone = any(done_steps >= m for m in milestones if m not in results.get("_evaled", []))
        do_eval = is_final or (a.eval_limit > 0 and hit_milestone)
        if do_eval:
            results.setdefault("_evaled", [])
            results["_evaled"] += [m for m in milestones if done_steps >= m and m not in results["_evaled"]]
            limit = (a.final_eval_limit or None) if is_final else a.eval_limit
            ev = run_eval(merged_dir, gpus[0], str(round_dir / "eval_merged.json"), a, limit=limit)
            rec["merged_acc"] = ev["accuracy"] if ev else None
            rec["merged_n"] = ev["n"] if ev else None
            if ev and ev.get("records"):
                import statistics as _st
                rec["mean_cot_len"] = round(_st.mean(r.get("n_gen_tokens", 0) for r in ev["records"]), 1)
            if is_final and ev:
                # merged-model pass@k (text tasks only; sample k, any-correct)
                if a.final_pass_k > 1 and tc["family"] == "text":
                    ev8 = run_eval(merged_dir, gpus[0], str(round_dir / "eval_merged_passk.json"),
                                   a, limit=limit, pass_k=a.final_pass_k)
                    rec["merged_pass_k"] = ev8["accuracy"] if ev8 else None
                    rec["pass_k"] = a.final_pass_k
                # per-branch eval on the SAME question set -> avg & union (lottery gap)
                branch_correct, branch_accs, branch_preds = [], [], []
                for b in range(K):
                    bev = run_eval(ckpts[b], gpus[0], str(round_dir / f"eval_branch{b}.json"), a, limit=limit)
                    if bev:
                        branch_accs.append(bev["accuracy"])
                        branch_correct.append(bev.get("correct"))
                        recs = bev.get("records")
                        branch_preds.append([r.get("pred") for r in recs] if recs else None)
                if branch_accs and all(c is not None for c in branch_correct):
                    n = len(branch_correct[0])
                    union = sum(any(branch_correct[b][i] for b in range(len(branch_correct)))
                                for i in range(n)) / n
                    avg = sum(branch_accs) / len(branch_accs)
                    rec["branch_accs"] = branch_accs
                    rec["branch_avg_acc"] = avg
                    rec["branch_union_acc"] = union
                    rec["merged_minus_avg"] = (rec["merged_acc"] - avg) if rec["merged_acc"] else None
                    rec["lottery_gap"] = union - avg
                    # R1e: majority-vote baseline — only where output-vote is defined (text tasks; numeric pred)
                    if tc["majority"] and ev.get("records") and all(p is not None for p in branch_preds):
                        from collections import Counter
                        golds = [rr["gold"] for rr in ev["records"]]
                        def _eq(p, g):
                            try: return abs(float(p) - float(g)) < 1e-6
                            except (ValueError, TypeError): return str(p) == str(g)
                        maj = 0
                        for i in range(n):
                            cands = [str(branch_preds[b][i]) for b in range(len(branch_preds))
                                     if branch_preds[b][i] not in (None, "", "None")]
                            if cands and _eq(Counter(cands).most_common(1)[0][0], golds[i]):
                                maj += 1
                        rec["majority_acc"] = maj / n
                        rec["majority_minus_merged"] = (maj / n - rec["merged_acc"]) if rec["merged_acc"] else None

        rec["round_time_s"] = round(time.time() - t_r, 1)
        results["rounds"].append(rec)
        results["elapsed_s"] = round(time.time() - t_start, 1)
        results_path.write_text(json.dumps(results, indent=2))
        msg = f"[round {r}/{n_rounds-1}] steps={done_steps}/{T} merged_acc={rec.get('merged_acc')}"
        if "branch_avg_acc" in rec:
            msg += f" | avg={rec['branch_avg_acc']:.4f} union={rec['branch_union_acc']:.4f}"
            if rec.get("majority_acc") is not None:
                msg += f" majority={rec['majority_acc']:.4f}"
        msg += f" | {rec['round_time_s']}s"
        print(msg, flush=True)

        # ---- disk hygiene: keep final-round branches; delete others + stale optimizer state ----
        for c in ckpts:
            opt = os.path.join(c, "optimizer.pt")
            if os.path.exists(opt):
                os.remove(opt)
        if not is_final:
            for b in range(K):
                shutil.rmtree(round_dir / f"branch{b}", ignore_errors=True)
        # delete the previous round's merged weights once this round consumed them as init
        if prev_merged_weights and os.path.isdir(prev_merged_weights):
            shutil.rmtree(prev_merged_weights, ignore_errors=True)
        prev_merged_weights = merged_dir
        init_model = merged_dir

    # ---- one-row results.csv: config + final-round metrics + training time ----
    import csv
    fr = results["rounds"][-1] if results["rounds"] else {}
    row = {
        # (1) config
        "tag": a.tag, "task": a.task, "base_model": a.base_model, "K": K,
        "period_P": P, "total_steps_T": T, "merge_mode": a.merge_mode,
        "lr": a.learning_rate, "num_generations": a.num_generations, "grad_accum": a.grad_accum,
        "maxlen": a.maxlen, "lr_scheduler": a.lr_scheduler, "loss_type": a.loss_type,
        "scale_rewards": a.scale_rewards, "mask_truncated": a.mask_truncated,
        "use_lora": a.use_lora, "lora_r": (a.lora_r if a.use_lora else None),
        "mbe_velocity_reward": a.mbe_velocity_reward, "virtual_rollout": a.virtual_rollout,
        # (2) results (final round)
        "avg_acc": fr.get("branch_avg_acc"), "union_acc": fr.get("branch_union_acc"),
        "merge_acc": fr.get("merged_acc"), "majority_acc": fr.get("majority_acc"),
        "lottery_gap": fr.get("lottery_gap"), "mean_cot_len": fr.get("mean_cot_len"),
        f"pass@{a.final_pass_k}": fr.get("merged_pass_k"), "eval_n": fr.get("merged_n"),
        # (3) training time
        "train_time_s": results.get("elapsed_s"),
        "train_time_h": round((results.get("elapsed_s") or 0) / 3600, 3),
    }
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)

    print(f"=== DONE {a.tag}: {json.dumps(results['rounds'][-1])} ===", flush=True)
    print(f"    results -> {results_path}", flush=True)
    print(f"    summary CSV -> {csv_path}", flush=True)
    print("    | " + " | ".join(f"{k}={row[k]}" for k in
          ["avg_acc", "union_acc", "merge_acc", "majority_acc", "mean_cot_len",
           f"pass@{a.final_pass_k}", "train_time_h"] if row.get(k) is not None), flush=True)


if __name__ == "__main__":
    main()
