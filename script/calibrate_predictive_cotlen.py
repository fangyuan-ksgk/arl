"""Calibrate --predictive_velocity_scale for the new norm_mode='cot_len' reward.

Why: in GRPO the *absolute* reward scale mostly cancels (advantage is divided
by the per-group std). What matters is the cot_len reward's WITHIN-GROUP std
relative to correctness's. This script measures both on a sample of existing
rollouts and recommends a scale so the cot_len term is a chosen fraction of the
correctness signal.

Method:
  1. Load Qwen3-1.7B (same as the sweep) + tokenizer.
  2. Reconstruct prompts from the GSM8K test set (eval rollouts are dataset-
     ordered, NUM_GEN per question) and pair them with logged completions.
  3. For each rollout compute the *raw* (pre-clip, pre-scale) predictive-velocity
     value under norm_mode in {cot_len, log_total} via the actual reward kernel.
  4. Group by query, compute mean within-group std for each, and the within-group
     std of correctness (0/1). Recommend scale = std(cot_len) / (target * std_correct).

Usage (CPU/MPS friendly — keep the sample small):
    python script/calibrate_predictive_cotlen.py \
        --rollout_jsonl logs/sweep_mbe_velocity_think/predvelo_w10/eval_rollout.jsonl \
        --step 0 --n_groups 16
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.mbe_reward import _compute_predictive_velocity_for_completion

ANSWER_FORMAT_INSTRUCTION = (
    "Solve the problem step by step inside <think>...</think>. "
    "After </think>, give a brief final explanation and end your response "
    "with a line of the exact form:\n#### <number>\n"
    "where <number> is the final numeric answer with no units, no commas, "
    "and no extra text."
)


def build_prompt(question: str):
    user_content = f"{question}\n\n{ANSWER_FORMAT_INSTRUCTION} /think"
    return [{"role": "user", "content": user_content}]


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_jsonl", type=str,
                    default="logs/sweep_mbe_velocity_think/predvelo_w10/eval_rollout.jsonl")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--step", type=int, default=0,
                    help="global_step block to sample from.")
    ap.add_argument("--n_groups", type=int, default=16,
                    help="number of (query) groups to sample; 8 rollouts each.")
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--marker", type=str, default="####")
    ap.add_argument("--target_frac", type=float, default=1.0,
                    help="desired cot_len within-group std as a fraction of "
                         "correctness within-group std (1.0 = parity).")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device={device}  model={args.model}")

    # --- load rollouts, reconstruct query_id by dataset order -----------------
    path = PROJECT / args.rollout_jsonl if not Path(args.rollout_jsonl).is_absolute() \
        else Path(args.rollout_jsonl)
    recs_all = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    pos = defaultdict(int)
    for r in recs_all:
        gs = r["global_step"]
        r["query_id"] = pos[gs] // args.num_gen
        pos[gs] += 1
    recs = [r for r in recs_all
            if r["global_step"] == args.step and r["query_id"] < args.n_groups]
    if not recs:
        raise SystemExit(f"no rollouts at step={args.step}")
    print(f"sampled {len(recs)} rollouts across {args.n_groups} groups "
          f"at step {args.step}")

    # --- prompts from GSM8K test set (same order as eval) ---------------------
    test = load_dataset("openai/gsm8k", "main")["test"]
    qids = sorted({r["query_id"] for r in recs})
    prompts = {q: build_prompt(test[q]["question"]) for q in qids}

    # --- load model -----------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    # --- compute raw values ---------------------------------------------------
    by_group = defaultdict(lambda: {"cot_len": [], "log_total": [], "correct": []})
    for i, r in enumerate(recs):
        prompt = prompts[r["query_id"]]
        comp = r["completion"]
        v_cot = _compute_predictive_velocity_for_completion(
            model, tok, prompt, comp, stride=args.stride,
            marker=args.marker, norm_mode="cot_len")
        v_log = _compute_predictive_velocity_for_completion(
            model, tok, prompt, comp, stride=args.stride,
            marker=args.marker, norm_mode="log_total")
        g = by_group[r["query_id"]]
        g["cot_len"].append(v_cot)
        g["log_total"].append(v_log)
        g["correct"].append(1.0 if r["correct"] else 0.0)
        if (i + 1) % 16 == 0:
            print(f"  ...{i + 1}/{len(recs)}")

    # --- aggregate ------------------------------------------------------------
    def within_group_std(key):
        stds = [np.std(g[key], ddof=0) for g in by_group.values() if len(g[key]) > 1]
        return float(np.mean(stds)) if stds else 0.0

    def pooled(key):
        vals = [v for g in by_group.values() for v in g[key]]
        return float(np.mean(vals)), float(np.std(vals))

    cot_wg = within_group_std("cot_len")
    log_wg = within_group_std("log_total")
    cor_wg = within_group_std("correct")
    cot_mean, cot_std = pooled("cot_len")
    log_mean, log_std = pooled("log_total")
    cot_absmax = max(abs(v) for g in by_group.values() for v in g["cot_len"])

    print("\n================ calibration =================")
    print(f"cot_len   raw value:  mean={cot_mean:+.5f}  pooled_std={cot_std:.5f}  "
          f"|max|={cot_absmax:.5f}")
    print(f"log_total raw value:  mean={log_mean:+.5f}  pooled_std={log_std:.5f}")
    print(f"\nwithin-group std (the part GRPO actually uses):")
    print(f"  cot_len    : {cot_wg:.5f}")
    print(f"  log_total  : {log_wg:.5f}   (note: clipped at 1.0 in practice)")
    print(f"  correctness: {cor_wg:.5f}")

    if cor_wg > 0 and cot_wg > 0:
        scale = cot_wg / (args.target_frac * cor_wg)
        print(f"\nRecommended --predictive_velocity_scale = {scale:.4f}  "
              f"(=> cot_len within-group std ≈ {args.target_frac:.2f}× correctness)")
        print(f"  weight w = 1/scale ≈ {1.0 / scale:.2f}")
        print(f"  clip check: raw |max|={cot_absmax:.5f} ≪ clip=1.0 → clip never binds (good)")
        for frac in (0.5, 0.25):
            print(f"  for {frac:.2f}× correctness: scale ≈ {cot_wg / (frac * cor_wg):.4f}")
    else:
        print("\nDegenerate: zero within-group variance somewhere; widen the sample.")


if __name__ == "__main__":
    main()
