"""
Curate Terminal-Bench tasks that are faithfully reproducible in the local,
container-free proot sandbox.

A task is kept iff running its own ``solution.sh`` inside the sandbox makes its
pytest verifier fully pass (all_passed). This guarantees the reward signal used
for RL is correct: the canonical solution scores 1.0.

Usage:
    python curate_tasks.py --tasks_dir /home/claudeuser/terminal-bench/original-tasks \
        --out /home/claudeuser/arl/skyrl_terminal/local_tasks.json --workers 32
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/claudeuser/SkyRL/skyrl-gym/skyrl_gym/envs/terminal")
import sandbox  # noqa: E402

import yaml  # noqa: E402


def task_meta(task_dir: Path):
    try:
        y = yaml.safe_load((task_dir / "task.yaml").read_text())
        return y.get("difficulty"), y.get("category"), (y.get("instruction") or "").strip()
    except Exception:
        return None, None, ""


def check_one(task_dir_str: str):
    task_dir = Path(task_dir_str)
    t0 = time.time()
    diff, cat, instr = task_meta(task_dir)
    try:
        r = sandbox.validate_task(task_dir, solve_timeout=120, verify_timeout=120)
        return {
            "name": task_dir.name,
            "path": str(task_dir),
            "difficulty": diff,
            "category": cat,
            "instruction_len": len(instr),
            "n_passed": r.n_passed,
            "n_total": r.n_total,
            "reward": r.reward,
            "all_passed": r.all_passed,
            "secs": round(time.time() - t0, 1),
        }
    except Exception as e:
        return {"name": task_dir.name, "path": str(task_dir), "error": str(e)[:200],
                "all_passed": False, "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_dir", default="/home/claudeuser/terminal-bench/original-tasks")
    ap.add_argument("--out", default="/home/claudeuser/arl/skyrl_terminal/local_tasks.json")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = sorted(p for p in Path(args.tasks_dir).iterdir() if (p / "task.yaml").exists())
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Scanning {len(tasks)} tasks with {args.workers} workers...", flush=True)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(check_one, str(t)): t.name for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            if res.get("all_passed"):
                print(f"  [{done}/{len(tasks)}] ✓ KEEP {res['name']} "
                      f"({res.get('difficulty')}/{res.get('category')})", flush=True)
            elif done % 25 == 0:
                print(f"  [{done}/{len(tasks)}] scanned...", flush=True)

    kept = [r for r in results if r.get("all_passed")]
    kept.sort(key=lambda r: (str(r.get("difficulty")), r["name"]))
    results.sort(key=lambda r: r["name"])
    Path(args.out).write_text(json.dumps({"kept": kept, "all": results}, indent=2))
    print(f"\nKept {len(kept)}/{len(tasks)} local-compatible tasks -> {args.out}")
    from collections import Counter
    print("By difficulty:", Counter(r.get("difficulty") for r in kept))
    print("By category:", Counter(r.get("category") for r in kept))
    print("Tasks:", [r["name"] for r in kept])


if __name__ == "__main__":
    main()
