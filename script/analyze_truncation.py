"""
Causal test #1 for the predictive-velocity reward: paragraph truncation.

Hypothesis: paragraphs with low cumulative v_t are dispensable; dropping
them should preserve accuracy substantially better than dropping random
paragraphs, and dropping HIGH-v_t paragraphs should crater accuracy.

Procedure per correct rollout (verify_24(expr) is True):
  1. Score the rollout against its own (correct) expression with
     compute_vt_batched -> per-token v_t.
  2. Split the rollout's CoT (the text BEFORE the '#### ...' marker) into
     paragraphs on '\\n\\n'. Per-paragraph reward = sum of v_t over the
     paragraph's token span.
  3. For each drop fraction f in --drop-fracs and each arm in
     {low, rand, high}: build a truncated CoT by dropping floor(f*P)
     paragraphs (lowest-reward / random / highest-reward).
  4. Re-prompt the policy with `prompt + truncated_CoT` (NO answer marker
     included) and let it generate up to --max-new-tokens. Verify with
     verify_24.

Outputs a JSON summary + a PNG with three lines (low / rand / high) per
model class. Run once per (rollouts.jsonl, model) triple; aggregate with
the matching figure in the manuscript.

Usage:
    python script/analyze_truncation.py \\
        --rollouts logs/game24_sweep1/len512/Qwen__Qwen3-1.7B/rollouts.jsonl \\
        --scorer-model Qwen/Qwen3-1.7B \\
        --policy-model Qwen/Qwen3-1.7B \\
        --drop-fracs 0,0.1,0.25,0.5,0.75 \\
        --n-rollouts 200 \\
        --output output/truncation/Qwen3-1.7B.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game24utils import verify_24, extract_expr, to_chat
from src.velocity import compute_vt_batched


# ---------------------------------------------------------------------------
# Paragraph parsing
# ---------------------------------------------------------------------------
def split_cot_and_answer(completion: str) -> Tuple[str, str]:
    """Return (cot_text, answer_block). The answer_block starts at the first
    '####' (or at the end if missing) so it is never included in the CoT we
    truncate."""
    marker = completion.find("####")
    if marker < 0:
        return completion, ""
    return completion[:marker], completion[marker:]


def paragraph_spans(cot: str) -> List[Tuple[int, int]]:
    """Return [(start_char, end_char_exclusive)] for each non-empty paragraph
    split on blank lines. Char indices reference `cot`."""
    spans, i, n = [], 0, len(cot)
    while i < n:
        # consume blank-line separators
        while i < n and cot[i] == "\n":
            i += 1
        if i >= n:
            break
        # find next blank-line boundary
        j = cot.find("\n\n", i)
        end = n if j < 0 else j
        if cot[i:end].strip():
            spans.append((i, end))
        i = end + 2 if j >= 0 else n
    return spans


def char_to_token_span(
    tokenizer, prompt_text: str, full_text: str, char_lo: int, char_hi: int
) -> Tuple[int, int]:
    """Map a char range inside the COMPLETION (offset 0 = first CoT char) to a
    token range inside the encoded completion (no prompt). Inclusive lower,
    exclusive upper."""
    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]  # [(s, e), ...]
    lo = next((k for k, (s, e) in enumerate(offsets) if e > char_lo), len(offsets))
    hi = next((k for k, (s, e) in enumerate(offsets) if s >= char_hi), len(offsets))
    return lo, hi


# ---------------------------------------------------------------------------
# Per-rollout paragraph reward
# ---------------------------------------------------------------------------
def paragraph_rewards(
    cot: str,
    spans: List[Tuple[int, int]],
    vt: np.ndarray,
    tokenizer,
) -> List[float]:
    """Sum v_t over each paragraph's token span. v_t is aligned with the
    tokenized COMPLETION (the same one passed to compute_vt_batched)."""
    rewards = []
    for char_lo, char_hi in spans:
        lo, hi = char_to_token_span(tokenizer, "", cot, char_lo, char_hi)
        hi = min(hi, len(vt))
        rewards.append(float(vt[lo:hi].sum()) if hi > lo else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Truncated regeneration & verification
# ---------------------------------------------------------------------------
def build_truncated_cot(
    cot: str, spans: List[Tuple[int, int]], keep_idx: List[int]
) -> str:
    keep_idx = sorted(keep_idx)
    return "\n\n".join(cot[s:e].strip() for s, e in (spans[i] for i in keep_idx))


@torch.no_grad()
def regenerate_and_verify(
    policy_model,
    tokenizer,
    prompt_text: str,
    truncated_cot: str,
    numbers: List[int],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> bool:
    """Concatenate prompt + truncated CoT (no answer marker), generate, and
    verify the produced answer with verify_24."""
    seed = prompt_text + ("\n\n" + truncated_cot if truncated_cot else "")
    ids = tokenizer(seed, return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to(policy_model.device) for k, v in ids.items()}
    out = policy_model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        pad_token_id=tokenizer.eos_token_id,
    )
    new_text = tokenizer.decode(
        out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True
    )
    expr = extract_expr(new_text)
    return verify_24(list(numbers), expr)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def select_keep(
    rewards: List[float], drop_frac: float, arm: str, rng: random.Random
) -> List[int]:
    P = len(rewards)
    n_drop = int(P * drop_frac)
    if n_drop <= 0:
        return list(range(P))
    if n_drop >= P:
        return []
    order = sorted(range(P), key=lambda i: rewards[i])  # ascending
    if arm == "low":
        drop = set(order[:n_drop])
    elif arm == "high":
        drop = set(order[-n_drop:])
    elif arm == "rand":
        drop = set(rng.sample(range(P), n_drop))
    else:
        raise ValueError(arm)
    return [i for i in range(P) if i not in drop]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", required=True, type=Path)
    p.add_argument("--scorer-model", required=True,
                   help="Model used as the reference for v_t computation.")
    p.add_argument("--policy-model", default=None,
                   help="Model used to regenerate continuations. Defaults to "
                        "--scorer-model so we test the SAME policy that wrote "
                        "the CoT.")
    p.add_argument("--n-rollouts", type=int, default=200,
                   help="Cap on the number of correct rollouts to test.")
    p.add_argument("--drop-fracs", type=str, default="0,0.1,0.25,0.5,0.75")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--scorer-micro-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    drop_fracs = [float(x) for x in args.drop_fracs.split(",")]
    rng = random.Random(args.seed)

    # 1. Load + filter to correct rollouts ---------------------------------
    rows = [json.loads(l) for l in args.rollouts.read_text().splitlines() if l.strip()]
    correct = [
        r for r in rows
        if r.get("completion")
        and verify_24(list(r["numbers"]), extract_expr(r["completion"]))
    ]
    rng.shuffle(correct)
    correct = correct[: args.n_rollouts]
    print(f"[trunc] {len(correct)} correct rollouts kept from {args.rollouts.name}")

    # 2. Load models -------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.scorer_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[trunc] loading scorer {args.scorer_model}")
    scorer = AutoModelForCausalLM.from_pretrained(
        args.scorer_model, dtype=dtype
    ).to(device).eval()
    if args.policy_model and args.policy_model != args.scorer_model:
        print(f"[trunc] loading policy  {args.policy_model}")
        policy = AutoModelForCausalLM.from_pretrained(
            args.policy_model, dtype=dtype
        ).to(device).eval()
    else:
        policy = scorer

    # 3. Score per-token v_t for every correct rollout vs its own expr ----
    prompts, completions, refs = [], [], []
    cots_and_spans, prompt_texts = [], []
    for r in correct:
        cot, _ = split_cot_and_answer(r["completion"])
        spans = paragraph_spans(cot)
        if len(spans) < 2:
            cots_and_spans.append(None)
            continue
        puzzle = {"numbers": list(r["numbers"]),
                  "solutions": [extract_expr(r["completion"])]}
        prompt = tok.apply_chat_template(
            to_chat(puzzle)["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
        completions.append(cot)
        refs.append(extract_expr(r["completion"]))
        cots_and_spans.append((cot, spans))
        prompt_texts.append(prompt)

    print(f"[trunc] scoring v_t for {len(prompts)} rollouts ...")
    t0 = time.time()
    scored = compute_vt_batched(
        prompts, completions, refs, scorer, tok,
        micro_batch_size=args.scorer_micro_batch,
        strip_answer_marker=False,  # we already stripped it
    )
    print(f"[trunc] scoring done in {time.time()-t0:.0f}s")

    # 4. Truncation arms ---------------------------------------------------
    arms = ["low", "rand", "high"]
    # results[arm][frac] -> list[bool]
    results = {a: {f: [] for f in drop_fracs} for a in arms}
    # also record baseline (f=0) regeneration with FULL CoT once
    baseline_full = []

    scored_iter = iter(scored)
    pt_iter = iter(prompt_texts)
    for r, item in zip(correct, cots_and_spans):
        if item is None:
            continue
        cot, spans = item
        s = next(scored_iter)
        prompt_text = next(pt_iter)
        vt = np.asarray(s["vt"])
        if len(vt) == 0:
            continue
        rewards = paragraph_rewards(cot, spans, vt, tok)

        # baseline: full CoT, regenerate continuation
        baseline_full.append(
            regenerate_and_verify(
                policy, tok, prompt_text, cot,
                numbers=list(r["numbers"]),
                max_new_tokens=args.max_new_tokens,
            )
        )

        for f in drop_fracs:
            for arm in arms:
                keep = select_keep(rewards, f, arm, rng)
                trunc = build_truncated_cot(cot, spans, keep)
                ok = regenerate_and_verify(
                    policy, tok, prompt_text, trunc,
                    numbers=list(r["numbers"]),
                    max_new_tokens=args.max_new_tokens,
                )
                results[arm][f].append(ok)

    # 5. Aggregate + write -------------------------------------------------
    summary = {
        "rollouts": str(args.rollouts),
        "scorer_model": args.scorer_model,
        "policy_model": args.policy_model or args.scorer_model,
        "n_rollouts_evaluated": len(baseline_full),
        "drop_fracs": drop_fracs,
        "baseline_full_regen_acc": float(np.mean(baseline_full)) if baseline_full else None,
        "acc": {
            arm: {str(f): float(np.mean(results[arm][f])) if results[arm][f] else None
                  for f in drop_fracs}
            for arm in arms
        },
        "n_per_cell": {
            arm: {str(f): len(results[arm][f]) for f in drop_fracs}
            for arm in arms
        },
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[trunc] wrote {args.output}")
    print(json.dumps(summary["acc"], indent=2))


if __name__ == "__main__":
    main()
