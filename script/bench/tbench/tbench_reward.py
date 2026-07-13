"""TRL GRPOTrainer reward function for Terminal-Bench tasks.

Single-turn protocol: the model answers with ONE ```bash fenced block; the
reward function extracts it, executes it in a fresh proot sandbox, then runs
the task's own pytest verifier. Reward = fraction of tests passed (0..1).

TRL hands reward functions the WHOLE generation batch at once, and each
sandbox rollout costs seconds of wall time on the CPU, so scoring is fanned
out over a thread pool (subprocess-bound => threads are fine). Pool size comes
from env var TBENCH_REWARD_WORKERS (default 8).

Dataset columns (task_path, exec_timeout, verify_timeout, ...) arrive as
per-sample kwarg lists — see GRPOTrainer._calculate_rewards.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import sandbox

logger = logging.getLogger("tbench_reward")

DEFAULT_EXEC_TIMEOUT = 45.0
DEFAULT_VERIFY_TIMEOUT = 60.0

_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK_RE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)


def extract_bash(text: str) -> Optional[str]:
    """Return the LAST ```bash fenced block (fallback: last generic fenced
    block). None if the completion contains no fenced code — those score 0.0
    without touching the sandbox (never execute free prose)."""
    matches = _BASH_BLOCK_RE.findall(text) or _ANY_BLOCK_RE.findall(text)
    if not matches:
        return None
    script = matches[-1].strip()
    return script or None


def _completion_text(completion) -> str:
    """TRL conversational completions are [{'role': ..., 'content': ...}]."""
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content", "") or "" for m in completion if isinstance(m, dict))


def _score_one(text: str, task_path: str, exec_timeout, verify_timeout) -> float:
    script = extract_bash(text)
    if script is None:
        return 0.0
    try:
        res = sandbox.score_completion(
            script,
            str(task_path),
            exec_timeout=float(exec_timeout or DEFAULT_EXEC_TIMEOUT),
            verify_timeout=float(verify_timeout or DEFAULT_VERIFY_TIMEOUT),
        )
        return float(res.reward)
    except sandbox.SandboxUnavailable:
        raise  # misconfiguration: fail loudly, do not train on silent zeros
    except Exception as e:  # a weird agent script must never kill training
        logger.warning("scoring failed for %s: %s", task_path, e)
        return 0.0


def tbench_reward(
    prompts=None,
    completions=None,
    completion_ids=None,
    task_path=None,
    exec_timeout=None,
    verify_timeout=None,
    **kwargs,
):
    """TRL reward function: batch of completions -> list of rewards in order."""
    n = len(completions)
    texts = [_completion_text(c) for c in completions]
    exec_ts = list(exec_timeout) if exec_timeout is not None else [None] * n
    verify_ts = list(verify_timeout) if verify_timeout is not None else [None] * n
    if task_path is None:
        raise ValueError("dataset must carry a 'task_path' column (build_tbench_trl.py)")

    workers = max(1, int(os.environ.get("TBENCH_REWARD_WORKERS", "8")))
    with ThreadPoolExecutor(max_workers=min(workers, n)) as pool:
        rewards = list(pool.map(_score_one, texts, task_path, exec_ts, verify_ts))

    log_metric = kwargs.get("log_metric")
    if log_metric is not None:
        log_metric("tbench/frac_fully_solved", sum(1.0 for r in rewards if r >= 1.0) / n)
        log_metric("tbench/frac_no_bash_block",
                   sum(1.0 for t in texts if extract_bash(t) is None) / n)
    return rewards
