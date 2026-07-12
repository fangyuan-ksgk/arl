"""Build the Terminal-Bench GRPO dataset in a TRL/HF-datasets-friendly format.

Port of arl/skyrl_terminal/build_dataset.py (SkyRL schema) to the flat-column
schema TRL's GRPOTrainer expects: a conversational `prompt` column plus extra
columns that reach the reward function as kwargs.

Single-turn setup (SkyRL goal-1): the prompt is the task instruction + an
environment description; the agent answers with ONE bash script; reward =
fraction of the task's own pytest tests passing in the proot sandbox.

Task selection comes from the curated arl/skyrl_terminal/local_tasks.json
(32 of 241 tasks whose reference solution.sh fully passes in the base proot
sandbox — no Docker-image deps). Paths in that file are re-resolved against
--tasks_dir, so a fresh `git clone laude-institute/terminal-bench` works.

    python build_tbench_trl.py                # -> ~/data/tbench_trl/{train,validation}.parquet
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

# Same prompt as the validated SkyRL run (skyrl_terminal/build_dataset.py).
SYSTEM_PROMPT = (
    "You are an expert software engineer operating a Linux shell. "
    "You are given a task to accomplish in a fresh environment whose working "
    "directory is /app. Solve it by writing a SINGLE bash script that, when "
    "executed from /app, fully accomplishes the task.\n\n"
    "Rules:\n"
    "- Output ONLY one ```bash code block containing the script. No prose.\n"
    "- You may use coreutils, grep/sed/awk, and `python3` (standard library only).\n"
    "- Write all required output files to the exact paths requested (under /app).\n"
    "- The script runs non-interactively; do not wait for input."
)


def resolve_task_path(recorded_path: str, tasks_dir: str) -> str:
    """local_tasks.json paths were recorded on another box; re-root them."""
    if os.path.isdir(recorded_path):
        return recorded_path
    candidate = os.path.join(tasks_dir, Path(recorded_path).name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"task {Path(recorded_path).name!r} not found at {recorded_path!r} or "
        f"{candidate!r} — clone the tasks first:\n"
        "  git clone --depth 1 https://github.com/laude-institute/terminal-bench "
        "~/terminal-bench"
    )


def load_instruction(task_path: str):
    y = yaml.safe_load((Path(task_path) / "task.yaml").read_text())
    return y["instruction"].strip(), y.get("difficulty", "?"), y.get("category", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_tasks",
                    default="/home/claudeuser/arl/skyrl_terminal/local_tasks.json")
    ap.add_argument("--tasks_dir",
                    default=os.path.expanduser("~/terminal-bench/original-tasks"))
    ap.add_argument("--output_dir", default=os.path.expanduser("~/data/tbench_trl"))
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="task names to exclude from training")
    ap.add_argument("--only", nargs="*", default=None,
                    help="if set, only use these task names")
    ap.add_argument("--exec_timeout", type=int, default=45,
                    help="seconds before a runaway agent script is reaped")
    ap.add_argument("--verify_timeout", type=int, default=60)
    args = ap.parse_args()

    kept = json.loads(Path(args.local_tasks).read_text())["kept"]
    if args.only:
        kept = [k for k in kept if k["name"] in args.only]
    kept = [k for k in kept if k["name"] not in set(args.exclude)]

    rows = []
    for k in kept:
        task_path = resolve_task_path(k["path"], args.tasks_dir)
        instr, diff, cat = load_instruction(task_path)
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instr},
            ],
            # extra columns below reach tbench_reward as per-sample kwargs
            "task_path": task_path,
            "task_name": k["name"],
            "difficulty": diff,
            "category": cat,
            "exec_timeout": args.exec_timeout,
            "verify_timeout": args.verify_timeout,
            "data_source": "terminal_bench",
        })

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    # Train and val use the same task set (in-distribution RL fit + per-task
    # eval), exactly like the validated SkyRL goal-1 run.
    df.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    df.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
    print(f"Wrote {len(rows)} tasks -> {args.output_dir}")
    print("difficulty:", Counter(r["difficulty"] for r in rows))
    print("tasks:", [r["task_name"] for r in rows])


if __name__ == "__main__":
    main()
