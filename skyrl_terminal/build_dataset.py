"""
Build SkyRL train/val parquet datasets from the locally-validated Terminal-Bench
tasks (output of curate_tasks.py).

Each row follows SkyRL's schema: a chat `prompt`, `env_class="terminal"`, a
`reward_spec`, and `extra_info` carrying the task directory the env will sandbox.
"""
import argparse
import json
import os
from pathlib import Path

import datasets
import yaml

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


def load_instruction(task_path: str) -> tuple[str, str, str]:
    y = yaml.safe_load((Path(task_path) / "task.yaml").read_text())
    return y["instruction"].strip(), y.get("difficulty", "?"), y.get("category", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_tasks", default="/home/claudeuser/arl/skyrl_terminal/local_tasks.json")
    ap.add_argument("--output_dir", default=os.path.expanduser("~/data/terminal_bench"))
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="task names to exclude from training")
    ap.add_argument("--only", nargs="*", default=None,
                    help="if set, only use these task names")
    ap.add_argument("--exec_timeout", type=int, default=45,
                    help="seconds before a runaway agent script is killed")
    ap.add_argument("--verify_timeout", type=int, default=60)
    args = ap.parse_args()

    kept = json.loads(Path(args.local_tasks).read_text())["kept"]
    if args.only:
        kept = [k for k in kept if k["name"] in args.only]
    kept = [k for k in kept if k["name"] not in set(args.exclude)]

    rows = []
    for k in kept:
        instr, diff, cat = load_instruction(k["path"])
        rows.append({
            "data_source": "terminal_bench",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instr},
            ],
            "env_class": "terminal",
            "reward_spec": {"method": "rule", "ground_truth": ""},
            "extra_info": {
                "task_path": k["path"],
                "task_name": k["name"],
                "difficulty": diff,
                "category": cat,
                "exec_timeout": args.exec_timeout,
                "verify_timeout": args.verify_timeout,
            },
        })

    os.makedirs(args.output_dir, exist_ok=True)
    ds = datasets.Dataset.from_list(rows)
    # Train and val use the same task set (in-distribution RL fit + per-task eval).
    ds.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    ds.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
    print(f"Wrote {len(rows)} tasks to {args.output_dir}")
    from collections import Counter
    print("difficulty:", Counter(r["extra_info"]["difficulty"] for r in rows))
    print("tasks:", [r["extra_info"]["task_name"] for r in rows])


if __name__ == "__main__":
    main()
