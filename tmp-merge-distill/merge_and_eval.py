"""Deliverable 1 — weight-merging methods + evaluation.

Loads base Qwen3-0.6B + the 8 GRPO seed checkpoints once, builds each merged model from
the per-seed task vectors tau_i = W_i - W0, and evaluates greedy@1 on the full GSM8K val
set (1319 questions, 2048 tokens) by shelling out to repro/eval_gsm8k.py. Results are
written incrementally to <out>/results.json.

Methods (all operate per parameter tensor on the stacked task vectors T = (N, *)):
  soup            W0 + mean(T)                              <- the robust baseline
  weighted_soup   W0 + sum(w_i T_i), w_i proportional to seed val acc
  task_arith_0.8  W0 + 0.8*mean(T)        (scaling >1 collapses; <1 is mild)
  ties_0.5        trim to top-50% |.|, sign-elect, mean of agreeing entries
  sign_consensus  ties with density 1.0 (no trim)
  dare_0.5        Bernoulli-drop 50% of each tau, rescale by 1/0.5, then mean
  dare_ties_0.5   dare then ties_0.5
  breadcrumbs     drop smallest 10% AND largest 1% |.|, then mean
  model_stock     W0 + t*mean(T), t = N*c/(1+(N-1)*c), c = mean pairwise cos (per tensor)
  selective       soup on weight MATRICES, keep base on norms+embeddings
                  (our finding: matrices carry noise, norms/embeds carry the signal)

Reference numbers (full eval, 2048 tok, A100):
  seed baseline   mean 70.6%, best 72.8%
  soup            77.3%   <- +6.6 over seed mean; no method below beats it meaningfully
  dare_0.5        77.3%   (matches soup but truncates ~5x more -> unstable)
  weighted_soup   76.7% | selective 76.7% | task_arith_0.8 76.4% | breadcrumbs 76.3%
  sign_consensus  76.0% (1271 trunc) | model_stock 71.4% | ties_0.5 71.9% (degenerate)
  dare_ties_0.5   63.7% (fully broken: runaway-length generations)

Usage:
  python repro/merge_and_eval.py                       # all methods, full eval
  python repro/merge_and_eval.py --methods soup selective
  python repro/merge_and_eval.py --limit 200           # quick smoke (first 200 questions)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
EVAL = os.path.join(HERE, "eval_gsm8k.py")

REPO = "Ksgk-fy/arl-gsm8k-multiseed"
BASE = "Qwen/Qwen3-0.6B"
STEP = 200                       # seed checkpoint step (200 = final GRPO checkpoint)
SEEDS = list(range(8))
# Per-seed greedy@1 GSM8K accuracy (used by weighted_soup). Update if you re-eval the seeds.
SEED_ACC = {0: 0.7195, 1: 0.7218, 2: 0.7180, 3: 0.6801, 4: 0.6793, 5: 0.6816, 6: 0.7233, 7: 0.7278}
# Parameter roles that carry "signal" -> selective merge keeps the base value for these.
NORMS_EMBED = ["embed_tokens", "input_layernorm", "post_attention_layernorm",
               "self_attn.q_norm", "self_attn.k_norm", "norm"]

ALL_METHODS = ["soup", "weighted_soup", "task_arith_0.8", "ties_0.5", "sign_consensus",
               "dare_0.5", "dare_ties_0.5", "breadcrumbs", "model_stock", "selective"]


def role_of(name: str) -> str:
    """Collapse a parameter name to its submodule role (layer index removed)."""
    n = re.sub(r"\.layers\.\d+\.", ".layers.N.", name)
    n = n.replace("model.layers.N.", "").replace("model.", "")
    return n.replace(".weight", "")


def main():
    ap = argparse.ArgumentParser(description="Merge-methods sweep + GSM8K eval")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--out", default=os.path.join(PROJECT, "output/gsm8k_merge"))
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="Eval only the first N questions (smoke test).")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(16)   # cap threads: default over-subscribes on many-core hosts
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    # Merge math runs on GPU when available: the sort/topk methods (ties, dare_ties,
    # sign_consensus, breadcrumbs) are minutes on CPU but ~seconds on GPU. TAU stays on
    # CPU (8x the model in fp32) and we push only the per-tensor op to the GPU, so the
    # persistent GPU footprint is ~0 and never contends with the eval subprocess.
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"merge device: {DEV}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    env = dict(os.environ, PYTHONPATH=f"{PROJECT}:{HERE}", HF_HUB_ENABLE_HF_TRANSFER="1")

    # --- load base + seeds once, stack task vectors tau_i = W_i - W0 (kept on CPU) ---
    base = {k: v.float() for k, v in load_file(hf_hub_download(BASE, "model.safetensors")).items()}
    snap = snapshot_download(REPO, allow_patterns=[f"seed{s}/checkpoint-{STEP}/*" for s in SEEDS])
    seeds = [load_file(os.path.join(snap, f"seed{s}", f"checkpoint-{STEP}", "model.safetensors")) for s in SEEDS]
    names = [k for k in base if k in seeds[0]]
    TAU = {k: torch.stack([seeds[i][k].float() - base[k] for i in range(len(SEEDS))], 0) for k in names}

    ref = os.path.join(snap, f"seed{SEEDS[0]}", f"checkpoint-{STEP}")
    template = AutoModelForCausalLM.from_pretrained(ref, dtype=torch.bfloat16)
    AutoTokenizer.from_pretrained(ref).save_pretrained(work)
    g = torch.Generator(device=DEV).manual_seed(0)   # fixed seed -> DARE drop masks are reproducible

    def ties(T, density=0.5, lam=1.0):
        N = T.shape[0]; flat = T.reshape(N, -1); k = max(1, int(flat.shape[1] * density))
        tr = torch.zeros_like(flat)
        for i in range(N):
            idx = torch.topk(flat[i].abs(), k).indices
            tr[i, idx] = flat[i, idx]
        sign = torch.sign(tr.sum(0))
        agree = (torch.sign(tr) == sign) & (tr != 0)
        return (lam * (tr * agree).sum(0) / agree.sum(0).clamp(min=1)).reshape(T.shape[1:])

    def dare_stack(T, p=0.5):
        mask = (torch.rand(T.shape, generator=g, device=T.device) > p).float()
        return (T * mask) / (1 - p)

    def breadcrumbs(T, low=0.10, high=0.01):
        N = T.shape[0]; flat = T.reshape(N, -1); out_ = torch.zeros_like(flat); D = flat.shape[1]
        for i in range(N):
            order = torch.argsort(flat[i].abs())
            keep = order[int(D * low):D - int(D * high)]
            out_[i, keep] = flat[i, keep]
        return out_.mean(0).reshape(T.shape[1:])

    def model_stock(T):
        N = T.shape[0]; flat = T.reshape(N, -1)
        fn = torch.nn.functional.normalize(flat, dim=1); cm = fn @ fn.t()
        ti = torch.triu_indices(N, N, 1)
        c = cm[ti[0], ti[1]].mean().clamp(min=0).item()
        t = N * c / (1 + (N - 1) * c) if c > 0 else 0.0
        return (t * flat.mean(0)).reshape(T.shape[1:])

    def delta(method, k):
        T = TAU[k].to(DEV)   # push this tensor to GPU for the heavy op; result moved back in build()
        if method == "soup":
            return T.mean(0)
        if method == "weighted_soup":
            w = torch.tensor([SEED_ACC[s] for s in SEEDS]); w = w / w.sum()
            return (T * w.view(-1, *([1] * (T.dim() - 1)))).sum(0)
        if method == "task_arith_0.8":
            return 0.8 * T.mean(0)
        if method == "ties_0.5":
            return ties(T, 0.5)
        if method == "sign_consensus":
            return ties(T, 1.0)
        if method == "dare_0.5":
            return dare_stack(T, 0.5).mean(0)
        if method == "dare_ties_0.5":
            return ties(dare_stack(T, 0.5), 0.5)
        if method == "breadcrumbs":
            return breadcrumbs(T)
        if method == "model_stock":
            return model_stock(T)
        if method == "selective":
            return T.mean(0) if role_of(k) not in NORMS_EMBED else torch.zeros_like(T[0])
        raise ValueError(method)

    def build(method):
        sd = {k: (base[k].to(DEV) + delta(method, k)).to(torch.bfloat16).cpu() for k in names}
        if DEV == "cuda":
            torch.cuda.empty_cache()   # release transient GPU blocks before vLLM eval grabs the GPU
        return sd

    def evaluate(method):
        template.load_state_dict(build(method), strict=False)
        template.save_pretrained(work)
        eo = out / "tmp_eval.json"
        cmd = [sys.executable, EVAL, "--model_path", str(work), "--label", method, "--out", str(eo),
               "--max_tokens", str(args.max_tokens), "--temperature", "0.0", "--max_model_len", "3072",
               "--gpu_memory_utilization", str(args.gpu_memory_utilization)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stderr[-2000:]); return None
        ev = json.loads(eo.read_text())
        rec = {"method": method, "acc": ev["accuracy"], "n": ev["n"],
               "n_truncated": ev.get("n_truncated"), "max_tokens": args.max_tokens}
        res = json.loads(results_path.read_text()) if results_path.exists() else []
        res = [r for r in res if r["method"] != method] + [rec]
        results_path.write_text(json.dumps(res, indent=2))
        print(f"  {method:<16} acc={ev['accuracy']:.4f} trunc={ev.get('n_truncated')}", flush=True)
        return ev["accuracy"]

    print(f"== merge methods sweep ({args.max_tokens} tok, {len(args.methods)} methods) ==", flush=True)
    for m in args.methods:
        evaluate(m)
    print(f"DONE -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
