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

import ast
import itertools
import json
import random
import re
import warnings
from pathlib import Path
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
    "completion_text", "_text", "extract_expr", "extract_expr_candidates",
    "normalize_math", "correctness_reward", "format_reward",
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
        # Suppress SyntaxWarning that Python's compiler emits for malformed
        # model outputs like "24(2+5)" before they get caught as TypeError.
        # The except below already classifies these as unparseable → None.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed
    except Exception:
        return None


def verify_24(numbers: Sequence[int], expr: str, target: float = TARGET) -> bool:
    """True iff `expr` uses each integer in `numbers` exactly once and evaluates
    to `target` (default :data:`TARGET` = 24). Pass `target` to study easier/
    harder variants (e.g. a 2-number "Game of 10") without monkeypatching."""
    val = safe_eval(expr)
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return False
    if abs(val - target) > _EPS:
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


def _ast_key(expr: str) -> Optional[str]:
    """Canonical AST dump of `expr`, used for dedup. None if unparseable."""
    try:
        return ast.dump(ast.parse(expr, mode="eval").body)
    except Exception:
        return None


def enumerate_solutions(
    numbers: Tuple[int, ...],
    max_solutions: Optional[int] = None,
    target: float = TARGET,
) -> List[str]:
    """All distinct AST-canonical expressions over `numbers` that evaluate to `target`.

    Brute-forces every (permutation × op-triple × bracket-template) candidate,
    filters by `verify_24`, and de-duplicates by AST canonical form so that
    parenthesization-only and chain-direction-only differences (e.g.
    `((3*8)+1)-1` and `3*8+1-1`) collapse to a single solution. Operand
    order remains significant — `8*(1+1+1)` and `(1+1+1)*8` are distinct.

    `max_solutions` is an optional safety cap (default: no cap). The candidate
    space is bounded (≤ 24 perms × 64 op-triples × 5 templates = 7680), so
    enumeration is cheap; the cap exists only to defend pathological callers.
    """
    by_ast: Dict[str, str] = {}   # ast_key -> representative source string
    for perm in set(itertools.permutations(numbers)):
        a, b, c, d = perm
        for op1, op2, op3 in itertools.product(_OPS, repeat=3):
            for tmpl in _TEMPLATES:
                expr = tmpl.format(a=a, b=b, c=c, d=d, o1=op1, o2=op2, o3=op3)
                if not verify_24(list(numbers), expr, target):
                    continue
                k = _ast_key(expr)
                if k is None or k in by_ast:
                    continue
                by_ast[k] = expr
                if max_solutions is not None and len(by_ast) >= max_solutions:
                    return list(by_ast.values())
    return list(by_ast.values())


# ---------------------------------------------------------------------------
# 2. Puzzle pool + difficulty bucketing
# ---------------------------------------------------------------------------
def build_puzzle_pool(
    max_n: int = 9,
    max_solutions: Optional[int] = None,
    target: float = TARGET,
) -> List[Dict[str, Any]]:
    """Enumerate all solvable 4-tuples drawn from {1, ..., max_n} with replacement.

    `max_solutions=None` (default) returns true AST-canonical solution counts
    per puzzle. Pass an int to cap (legacy behaviour). Note: switching from
    the previous capped string-dedup to uncapped AST-dedup changes the
    `n_solutions` distribution → re-tune `bucket_by_difficulty` thresholds.
    """
    pool: List[Dict[str, Any]] = []
    for tup in itertools.combinations_with_replacement(range(1, max_n + 1), 4):
        sols = enumerate_solutions(tup, max_solutions=max_solutions, target=target)
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
        "After thinking, give your final expression on the last line after '#### '."
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


# Arithmetic candidate: must start with a digit or '(' then ASCII math chars.
# '.' and newline are NOT in the class, so a match never crosses lines and never
# swallows letters → an answer can't be reward-hacked out of CoT prose.
_MATH_RE = re.compile(r"[0-9(][0-9+\-*/(). ]*")


def normalize_math(s: str) -> str:
    """Map common LaTeX/markdown math notation onto ``safe_eval``'s ASCII
    alphabet so cosmetic formatting (×, \\times, $…$, **bold**) doesn't sink an
    otherwise-correct expression. Correctness measures MATH, not notation —
    notation compliance is ``format_reward``'s job."""
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", " ")
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/")
    s = s.replace("$", " ")
    return s


def extract_expr_candidates(text: str) -> List[str]:
    """All arithmetic substrings in the post-``####`` region of the post-
    ``</think>`` tail.

    Requiring the answer to live after ``####`` mirrors the reward-hacking guard
    in ``velocity._find_answer_marker``: a literal ``####`` inside the CoT (or
    bare CoT arithmetic when no marker is emitted) must not be picked up. Each
    candidate is notation-normalized, has its trailing ``= 24`` stripped, and
    must contain at least one digit and one operator. Empty list on miss.
    """
    tail = text.split("</think>")[-1]
    if "####" not in tail:
        return []                                  # no answer marker → no credit
    region = normalize_math(tail.split("####")[-1])  # text after the LAST '####'
    cands: List[str] = []
    for m in _MATH_RE.findall(region):
        c = m.strip().rstrip("=").strip()          # drop trailing "= 24"
        if c and re.search(r"\d", c) and re.search(r"[+\-*/]", c):
            cands.append(c)
    return cands


def extract_expr(text: str) -> str:
    """Best-effort single expression for display/logging: the first candidate
    after ``####`` (notation-normalized), or the empty string on miss. For
    correctness use ``correctness_reward`` / ``extract_expr_candidates``, which
    consider every candidate."""
    cands = extract_expr_candidates(text)
    return cands[0] if cands else ""


def correctness_reward(completions, numbers, target: float = TARGET, **_) -> List[float]:
    rewards = []
    for c, nums in zip(completions, numbers):
        cands = extract_expr_candidates(completion_text(c))
        rewards.append(1.0 if any(verify_24(list(nums), e, target) for e in cands) else 0.0)
    return rewards


def format_reward(completions, **_) -> List[float]:
    return [0.2 if re.search(r"####\s*\S", completion_text(c)) else 0.0 for c in completions]


# ============================
# Logging Rollout Statistics
# ============================
#
# Reward-fn shim that dumps rollouts to JSONL. Used by the non-velocity
# drivers (`script/run_game24_one.py`, `script/run_game24_deepspeed.py`)
# that run vanilla `GRPOTrainer`. For `PerTokenAdvantageTrainer` (velocity
# route) rollout logging is built in — see `_log_rollouts` there. The two
# paths intentionally produce records with the same `split / global_step /
# idx / completion / correct / n_tokens` core fields so downstream readers
# can treat them interchangeably.


class RolloutLogger:
    __name__ = "rollout_logger"

    def __init__(self, train_path: Path, eval_path: Path, tokenizer):
        self.train_path = train_path
        self.eval_path = eval_path
        self.in_eval = False
        self.train_step = 0
        self.eval_step = 0
        # Trainer.state.global_step at the moment eval was triggered.
        # Stamped by the driver's EvalFlagCallback; -1 before first step.
        self.global_step = -1
        # "sample" (T=1 pass@K) or "greedy" (T=0 pass@1). Flipped by the
        # fast-eval trainer around each generation pass.
        self.decoding = "sample"
        # Sampling temperature of the current pass (~0.0 for greedy, the
        # configured eval temperature for sample). Stamped by the trainer.
        self.temperature = None
        # Name of the validation split being evaluated (e.g. "eval", "probe").
        # Stamped by the trainer's `evaluate` override for dict eval datasets.
        self.eval_dataset_name = "eval"
        self.tokenizer = tokenizer

    def __call__(self, completions, numbers, **_):
        path = self.eval_path if self.in_eval else self.train_path
        step = self.eval_step if self.in_eval else self.train_step
        with path.open("a") as f:
            for i, (c, nums) in enumerate(zip(completions, numbers)):
                text = _text(c)
                cands = extract_expr_candidates(text)
                # `correct` matches correctness_reward (any candidate verifies);
                # `expr` is the first verifying one, else the first candidate.
                correct = any(verify_24(list(nums), e) for e in cands)
                expr = next((e for e in cands if verify_24(list(nums), e)),
                            cands[0] if cands else "")
                n_tok = len(self.tokenizer.encode(text, add_special_tokens=False))

                think_idx = text.find("</think>")
                m_ans = re.search(r"####", text)
                if think_idx >= 0:
                    cot_text = text[:think_idx]
                elif m_ans is not None:
                    cot_text = text[: m_ans.start()]
                else:
                    cot_text = text
                n_cot_tok = len(self.tokenizer.encode(cot_text, add_special_tokens=False))
                f.write(json.dumps({
                    "step": step, "idx": i, "numbers": list(nums),
                    "completion": text, "expr": expr,
                    "correct": bool(correct),
                    "n_tokens": int(n_tok),
                    "n_cot_tokens": int(n_cot_tok),
                    "has_answer_marker": bool(m_ans),
                    "has_think_close": bool(think_idx >= 0),
                    "split": "eval" if self.in_eval else "train",
                    "eval_dataset": (self.eval_dataset_name if self.in_eval else None),
                    "decoding": self.decoding,
                    "temperature": (None if self.temperature is None
                                    else float(self.temperature)),
                    "global_step": int(self.global_step),
                }) + "\n")
        if self.in_eval:
            self.eval_step += 1
        else:
            self.train_step += 1
        return [0.0] * len(completions)