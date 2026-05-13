"""
Game-of-24 utilities for the velocity-reward experiments.

Contents
--------
- Verifier and brute-force solver
- Puzzle-pool builder with difficulty bucketing
- Proportion-based train/eval/probe split
- Chat-template prompt formatting
- Trajectory-level reward functions for GRPO

Everything here is plain Python with no heavy dependencies beyond `datasets`,
so it imports fast and is easy to unit-test.
"""

from __future__ import annotations

import itertools
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset

__all__ = [
    # solving / verification
    "TARGET", "safe_eval", "verify_24", "enumerate_solutions",
    # dataset construction
    "build_puzzle_pool", "bucket_by_difficulty", "make_splits",
    # prompt formatting
    "SYSTEM_PROMPT", "to_chat", "build_datasets",
    # rewards
    "completion_text", "_text", "extract_expr",
    "correctness_reward", "format_reward",
]


# ---------------------------------------------------------------------------
# 1. Verifier + brute-force solver
# ---------------------------------------------------------------------------
TARGET = 24
_EPS = 1e-6
_ALLOWED = set("0123456789+-*/(). ")


def safe_eval(expr: str) -> Optional[float]:
    """Evaluate a numeric expression restricted to digits, + - * / ( ) and spaces."""
    if not expr or any(c not in _ALLOWED for c in expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed
    except Exception:
        return None


def verify_24(numbers: Sequence[int], expr: str) -> bool:
    """True iff `expr` uses each integer in `numbers` exactly once and evaluates to 24."""
    val = safe_eval(expr)
    if val is None or abs(val - TARGET) > _EPS:
        return False
    used = [int(x) for x in re.findall(r"\d+", expr)]
    return sorted(used) == sorted(numbers)


_TEMPLATES = (
    "(({a}{o1}{b}){o2}{c}){o3}{d}",
    "({a}{o1}({b}{o2}{c})){o3}{d}",
    "({a}{o1}{b}){o2}({c}{o3}{d})",
    "{a}{o1}(({b}{o2}{c}){o3}{d})",
    "{a}{o1}({b}{o2}({c}{o3}{d}))",
)
_OPS = ("+", "-", "*", "/")


def enumerate_solutions(numbers: Tuple[int, ...], max_solutions: int = 50) -> List[str]:
    """All distinct (string-canonical) expressions over `numbers` that evaluate to 24."""
    sols: set[str] = set()
    for perm in set(itertools.permutations(numbers)):
        a, b, c, d = perm
        for op1, op2, op3 in itertools.product(_OPS, repeat=3):
            for tmpl in _TEMPLATES:
                expr = tmpl.format(a=a, b=b, c=c, d=d, o1=op1, o2=op2, o3=op3)
                if verify_24(list(numbers), expr):
                    sols.add(expr)
                    if len(sols) >= max_solutions:
                        return list(sols)
    return list(sols)


# ---------------------------------------------------------------------------
# 2. Puzzle pool + difficulty bucketing
# ---------------------------------------------------------------------------
def build_puzzle_pool(max_n: int = 9, max_solutions: int = 50) -> List[Dict[str, Any]]:
    """Enumerate all solvable 4-tuples drawn from {1, ..., max_n} with replacement."""
    pool: List[Dict[str, Any]] = []
    for tup in itertools.combinations_with_replacement(range(1, max_n + 1), 4):
        sols = enumerate_solutions(tup, max_solutions=max_solutions)
        if sols:
            pool.append({
                "numbers":     list(tup),
                "solutions":   sols,
                "n_solutions": len(sols),
            })
    return pool


def bucket_by_difficulty(
    pool: List[Dict[str, Any]],
    easy_min: int = 8,
    hard_max: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split pool into (easy, medium, hard) by #solutions."""
    easy   = [p for p in pool if p["n_solutions"] >= easy_min]
    medium = [p for p in pool if hard_max < p["n_solutions"] < easy_min]
    hard   = [p for p in pool if p["n_solutions"] <= hard_max]
    return easy, medium, hard


def _split(lst: List[Any], frac: float) -> Tuple[List[Any], List[Any]]:
    n = int(round(len(lst) * frac))
    return (lst[:-n] if n else lst, lst[-n:] if n else [])


def make_splits(
    easy: List[Dict[str, Any]],
    medium: List[Dict[str, Any]],
    hard: List[Dict[str, Any]],
    *,
    eval_frac: float = 0.10,
    probe_frac: float = 0.40,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Proportion-based train / eval / probe split.

    - `eval_frac` of easy + medium becomes the eval set (in-distribution).
    - `probe_frac` of HARD puzzles is held out as the D3 zero-pass@K probe.
    - All remaining puzzles form the training set, which is shuffled.
    """
    rng = rng or random
    rng.shuffle(easy); rng.shuffle(medium); rng.shuffle(hard)

    easy_tr,   easy_ev    = _split(easy,   eval_frac)
    medium_tr, medium_ev  = _split(medium, eval_frac)
    hard_tr,   hard_probe = _split(hard,   probe_frac)

    train = easy_tr + medium_tr + hard_tr
    rng.shuffle(train)
    eval_ = easy_ev + medium_ev
    return train, eval_, hard_probe


# ---------------------------------------------------------------------------
# 3. Prompt formatting
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You play the Game of 24. Given four numbers, you must write a single "
    "arithmetic expression using each number exactly once with + - * / and "
    "parentheses that evaluates to 24.\n\n"
    "Think step by step. First reason about how to combine the numbers, then "
    "on the final line output only the expression after '#### '.\n"
    "Example final line: '#### (3+5)*(7-4)'."
)


def to_chat(puzzle: Dict[str, Any]) -> Dict[str, Any]:
    """Render one puzzle into the chat-prompt format expected by TRL's GRPOTrainer."""
    num_str = ",".join(map(str, puzzle["numbers"]))
    user = (
        f"Given numbers: {num_str}. Make 24. "
        "Think step by step, then give your final expression on the last line after '#### '."
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        "numbers":   puzzle["numbers"],
        "solutions": puzzle["solutions"],
    }


def build_datasets(
    train_puzzles: List[Dict[str, Any]],
    eval_puzzles: List[Dict[str, Any]],
    probe_puzzles: List[Dict[str, Any]],
) -> Tuple[Dataset, Dataset, Dataset]:
    """Wrap puzzle lists as HuggingFace `Dataset`s ready for `GRPOTrainer`."""
    return (
        Dataset.from_list([to_chat(p) for p in train_puzzles]),
        Dataset.from_list([to_chat(p) for p in eval_puzzles]),
        Dataset.from_list([to_chat(p) for p in probe_puzzles]),
    )


# ---------------------------------------------------------------------------
# 4. Reward functions (trajectory-level — D2 is structural)
# ---------------------------------------------------------------------------
def completion_text(completion: Any) -> str:
    """Normalise a TRL completion (str OR list[dict]) to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


# Short alias used by the rollout logger and other diagnostic call sites.
_text = completion_text


def extract_expr(text: str) -> str:
    """Return the expression after the last '#### ' marker, or empty string."""
    m = re.search(r"####\s*(.+?)\s*$", text.strip())
    return m.group(1).strip() if m else ""


def correctness_reward(completions, numbers, **_) -> List[float]:
    rewards = []
    for c, nums in zip(completions, numbers):
        rewards.append(1.0 if verify_24(list(nums), extract_expr(completion_text(c))) else 0.0)
    return rewards


def format_reward(completions, **_) -> List[float]:
    return [0.2 if re.search(r"####\s*\S", completion_text(c)) else 0.0 for c in completions]
