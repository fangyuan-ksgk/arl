"""
Causal test #2 for the predictive-velocity reward: CoT selection.

Hypothesis: among multiple CoTs proposed for the same puzzle by diverse
API-based LLMs, the one with the highest predictive R_T is more likely to
land on a correct answer than a random pick, and at least competitive
with simple majority vote.

Procedure:
  1. Sample puzzles from build_puzzle_pool (or load from --puzzles).
  2. For each puzzle, query each --provider API n_per_provider times at
     a fixed (moderate) temperature -> a pool of K = sum(n_per_provider)
     candidate completions per puzzle.
  3. Score each (prompt, completion, ref) triple with compute_vt_batched
     against every enumerated canonical solution -> R_T per candidate is
     max over canonical solutions (mirrors rescore_vt.py).
  4. Selection rules:
       random      : uniform pick over the K candidates
       majority    : pick the candidate whose answer expression is the
                     mode of {extract_expr(c) for c in candidates}
       argmax_R_T  : pick the candidate with the highest R_T
       per_api_best: argmax_R_T restricted to each provider's own samples
                     (oracle upper bound per provider)
       any_correct : exists-test (upper bound for selection methods)
  5. Aggregate accuracy across puzzles. Write JSON.

API keys: read from environment as usual
  - OPENAI_API_KEY     for provider=openai
  - ANTHROPIC_API_KEY  for provider=anthropic

Usage:
    python script/analyze_cot_selection.py \\
        --n-puzzles 30 \\
        --providers openai:gpt-4o-mini=4,anthropic:claude-3-5-haiku=4 \\
        --scorer-model Qwen/Qwen3-1.7B \\
        --temperature 0.7 \\
        --output output/cot_selection/cmp.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game24utils import (
    build_puzzle_pool,
    enumerate_solutions,
    extract_expr,
    to_chat,
    verify_24,
)
from src.velocity import compute_vt_batched


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are solving the Game-24 puzzle. Use each of the four given numbers "
    "exactly once with + - * / and parentheses to make 24. Show your "
    "reasoning, then on a new line write '#### <expression>' where "
    "<expression> evaluates to 24."
)


def user_prompt(numbers: List[int]) -> str:
    return (
        f"Numbers: {','.join(map(str, numbers))}\n"
        "Use each number exactly once. End with a line of the form\n"
        "#### <expression>"
    )


def call_openai(model: str, numbers: List[int], temperature: float) -> str:
    from openai import OpenAI  # lazy import
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(numbers)},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def call_anthropic(model: str, numbers: List[int], temperature: float) -> str:
    import anthropic  # lazy import
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt(numbers)}],
    )
    return resp.content[0].text


DISPATCH = {"openai": call_openai, "anthropic": call_anthropic}


def parse_providers(spec: str) -> List[Tuple[str, str, int]]:
    """'openai:gpt-4o-mini=4,anthropic:claude-3-5-haiku=4' ->
    [('openai','gpt-4o-mini',4), ('anthropic','claude-3-5-haiku',4)]"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        head, _, count = item.partition("=")
        provider, _, model = head.partition(":")
        out.append((provider.strip(), model.strip(), int(count)))
    return out


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------
def select_random(candidates: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    return rng.choice(candidates)


def select_majority(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    answers = [extract_expr(c["completion"]) for c in candidates]
    counts = Counter(a for a in answers if a)
    if not counts:
        return candidates[0]
    top = counts.most_common(1)[0][0]
    return next(c for c in candidates if extract_expr(c["completion"]) == top)


def select_argmax_rt(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(candidates, key=lambda c: c["R_T"] if c["R_T"] is not None else -1e18)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-puzzles", type=int, default=30)
    p.add_argument("--puzzles", type=Path, default=None,
                   help="Optional JSONL of {numbers: [...]} rows. Defaults to "
                        "build_puzzle_pool sampled to --n-puzzles.")
    p.add_argument("--providers", required=True,
                   help="'openai:gpt-4o-mini=4,anthropic:claude-3-5-haiku=4'")
    p.add_argument("--scorer-model", required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--scorer-micro-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=None,
                   help="Optional path; if exists, skip API calls and reuse "
                        "previously-collected candidates.")
    return p.parse_args()


def collect_candidates(
    puzzles: List[Dict[str, Any]],
    providers: List[Tuple[str, str, int]],
    temperature: float,
) -> List[List[Dict[str, Any]]]:
    """Returns one list of candidate dicts per puzzle.
    Each candidate has {provider, model, completion}."""
    all_cands: List[List[Dict[str, Any]]] = []
    for p_i, puz in enumerate(puzzles):
        nums = list(puz["numbers"])
        cands = []
        for provider, model, n in providers:
            fn = DISPATCH[provider]
            for k in range(n):
                t0 = time.time()
                try:
                    text = fn(model, nums, temperature)
                except Exception as e:
                    print(f"  [{provider}/{model}] FAILED puzzle {p_i} #{k}: {e}")
                    continue
                cands.append({"provider": provider, "model": model,
                              "completion": text})
                print(f"  [{provider}/{model}] puzzle {p_i} #{k} "
                      f"({time.time()-t0:.1f}s)")
        all_cands.append(cands)
        print(f"[gather] puzzle {p_i+1}/{len(puzzles)} -> {len(cands)} candidates")
    return all_cands


def score_candidates(
    puzzles: List[Dict[str, Any]],
    candidates: List[List[Dict[str, Any]]],
    scorer_model: str,
    micro_batch: int,
) -> None:
    """Mutates `candidates` in place, adding 'R_T' to each."""
    tok = AutoTokenizer.from_pretrained(scorer_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(scorer_model, dtype=dtype).to(device).eval()

    prompts, comps, refs = [], [], []
    # flat-index bookkeeping: for each candidate, the range [start, end) in
    # refs that covers this candidate's K canonical solutions.
    spans: List[Tuple[int, int]] = []
    for puz, cands in zip(puzzles, candidates):
        sols = list(enumerate_solutions(tuple(puz["numbers"])))
        if not sols:
            spans.extend([(0, 0)] * len(cands))
            continue
        puzzle_for_chat = {"numbers": list(puz["numbers"]), "solutions": sols}
        prompt = tok.apply_chat_template(
            to_chat(puzzle_for_chat)["prompt"],
            tokenize=False, add_generation_prompt=True,
        )
        for c in cands:
            start = len(refs)
            prompts.extend([prompt] * len(sols))
            comps.extend([c["completion"]] * len(sols))
            refs.extend(sols)
            spans.append((start, start + len(sols)))

    if not prompts:
        return
    print(f"[score] forwarding {len(prompts)} (cand, sol) pairs ...")
    t0 = time.time()
    scored = compute_vt_batched(prompts, comps, refs, model, tok,
                                micro_batch_size=micro_batch)
    print(f"[score] done in {time.time()-t0:.0f}s")

    flat_i = 0
    for cands in candidates:
        for c in cands:
            s, e = spans[flat_i]; flat_i += 1
            if s == e:
                c["R_T"] = None
                continue
            rts = [scored[k]["R_T"] for k in range(s, e)]
            c["R_T"] = float(max(rts))


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    providers = parse_providers(args.providers)

    # 1. Puzzles -----------------------------------------------------------
    if args.puzzles and args.puzzles.exists():
        puzzles = [json.loads(l) for l in args.puzzles.read_text().splitlines() if l.strip()]
    else:
        pool = build_puzzle_pool(max_n=9)
        rng.shuffle(pool)
        puzzles = pool[: args.n_puzzles]
    print(f"[main] {len(puzzles)} puzzles")

    # 2. Candidates (with optional cache) ----------------------------------
    if args.cache and args.cache.exists():
        print(f"[main] loading cached candidates from {args.cache}")
        cached = json.loads(args.cache.read_text())
        candidates = cached["candidates"]
        puzzles = cached["puzzles"]
    else:
        # require API keys present
        if any(p == "openai" for p, _, _ in providers) and not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY missing")
        if any(p == "anthropic" for p, _, _ in providers) and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY missing")
        candidates = collect_candidates(puzzles, providers, args.temperature)
        if args.cache:
            args.cache.write_text(json.dumps(
                {"puzzles": puzzles, "candidates": candidates}, indent=1))
            print(f"[main] cached candidates -> {args.cache}")

    # 3. Score R_T --------------------------------------------------------
    score_candidates(puzzles, candidates, args.scorer_model, args.scorer_micro_batch)

    # 4. Evaluate selection rules -----------------------------------------
    rule_correct = {"random": [], "majority": [], "argmax_R_T": [], "any_correct": []}
    per_api_best: Dict[str, List[bool]] = {}

    for puz, cands in zip(puzzles, candidates):
        nums = list(puz["numbers"])
        if not cands:
            continue

        def ok(c): return verify_24(nums, extract_expr(c["completion"]))

        rule_correct["random"].append(ok(select_random(cands, rng)))
        rule_correct["majority"].append(ok(select_majority(cands)))
        rule_correct["argmax_R_T"].append(ok(select_argmax_rt(cands)))
        rule_correct["any_correct"].append(any(ok(c) for c in cands))

        by_api: Dict[str, List[Dict[str, Any]]] = {}
        for c in cands:
            by_api.setdefault(f'{c["provider"]}/{c["model"]}', []).append(c)
        for api, sub in by_api.items():
            per_api_best.setdefault(api, []).append(ok(select_argmax_rt(sub)))

    summary = {
        "n_puzzles": len(puzzles),
        "providers": [f"{p}:{m}={n}" for p, m, n in providers],
        "temperature": args.temperature,
        "scorer_model": args.scorer_model,
        "acc": {k: float(np.mean(v)) if v else None for k, v in rule_correct.items()},
        "per_api_best_acc": {k: float(np.mean(v)) for k, v in per_api_best.items()},
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
