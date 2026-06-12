"""
Build train/val parquet for the ToyBox agentic GRPO env from the task pack in
``skyrl_gym.envs.toybox.tasks``. 🎮

Each row is a single-prompt RL example: a shared system message teaching the
action protocol + the task's themed prompt. The hidden answer key lives in the
env (referenced by ``extra_info.task_id``), never in the prompt.

    /home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/build_toybox_dataset.py \
        --output_dir ~/data/toybox --repeat 4
"""

import argparse
import os
import sys

# Make the in-repo skyrl_gym importable without installing.
sys.path.insert(0, "/home/claudeuser/SkyRL/skyrl-gym")
from skyrl_gym.envs.toybox import tasks as toytasks  # noqa: E402

SYSTEM_PROMPT = (
    "You are ToyBot, a playful coding agent solving small puzzles in a sandbox. 🎮\n\n"
    "Each turn you may act by emitting fenced code blocks, which are executed in a "
    "throwaway working directory and whose output is returned to you:\n"
    "  • ```python ... ```  — runs a Python snippet (your latest one is saved as solution.py)\n"
    "  • ```bash ... ```    — runs a shell command\n\n"
    "To finish a task, do ONE of:\n"
    "  • submit a final result as <answer>YOUR_ANSWER</answer>, or\n"
    "  • write TASK_COMPLETE once the required file/output is in place.\n\n"
    "Think briefly, then act. Keep snippets small and verify your work before finishing."
)


def make_row(task, split, idx):
    user = task["prompt"]
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "env_class": "toybox",
        "reward_spec": {"method": "rule", "ground_truth": task["id"]},
        "extra_info": {
            "split": split,
            "index": idx,
            "task_id": task["id"],
            "title": task["title"],
            "difficulty": task["difficulty"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="~/data/toybox")
    ap.add_argument("--repeat", type=int, default=1,
                    help="duplicate the task pack N times in the train split for bigger batches")
    args = ap.parse_args()
    out = os.path.expanduser(args.output_dir)
    os.makedirs(out, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    train_rows = []
    for r in range(args.repeat):
        for i, t in enumerate(toytasks.TASKS):
            train_rows.append(make_row(t, "train", r * len(toytasks.TASKS) + i))
    val_rows = [make_row(t, "val", i) for i, t in enumerate(toytasks.TASKS)]

    pq.write_table(pa.Table.from_pylist(train_rows), os.path.join(out, "train.parquet"))
    pq.write_table(pa.Table.from_pylist(val_rows), os.path.join(out, "validation.parquet"))
    print(f"Wrote {len(train_rows)} train rows ({args.repeat}x{len(toytasks.TASKS)}) "
          f"and {len(val_rows)} val rows to {out}")
    print("Tasks:", ", ".join(f"{t['theme']}{t['id']}" for t in toytasks.TASKS))


if __name__ == "__main__":
    main()
