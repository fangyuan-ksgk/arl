"""
Turn a SkyRL run into clean figures + a CSV — no more grepping console soup.

Reads two structured sources for a run:
  * eval curve  <- ~/exports/<run>/dumped_evals/global_step_N_evals/aggregated_results.jsonl
  * train curve <- ordered metric lines in /tmp/<run>.<user>.log
Writes  ~/exports/<run>/<run>_curves.png  and  <run>_metrics.csv, prints a summary.

    /home/claudeuser/tbench-venv/bin/python arl/skyrl_terminal/plot_run.py --run searchr1_mini_fixed
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPORTS = os.environ.get("SKYRL_EXPORTS", os.path.expanduser("~/exports"))

# train metrics parsed in-order from the console log (one value per step)
TRAIN_KEYS = {
    "reward": r"avg_final_rewards:\s*([0-9.]+)",
    "loss": r"'final_loss':\s*([0-9.eE-]+)",
    "grad_norm": r"'grad_norm':\s*([0-9.eE-]+)",
    "entropy": r"'policy_entropy':\s*([0-9.eE-]+)",
}


def read_eval_curve(run):
    rows = []
    for d in glob.glob(f"{EXPORTS}/{run}/dumped_evals/global_step_*_evals"):
        m = re.search(r"global_step_(\d+)_evals", d)
        agg = os.path.join(d, "aggregated_results.jsonl")
        if not m or not os.path.exists(agg):
            continue
        data = json.load(open(agg))
        rows.append((int(m.group(1)),
                     data.get("eval/all/pass_at_1"),
                     data.get("eval/all/avg_score")))
    return sorted(rows)


def read_train_curve(run):
    logs = glob.glob(f"/tmp/{run}.*.log") + [f"/tmp/{run}.log"]
    log = next((l for l in logs if os.path.exists(l)), None)
    series = {k: [] for k in TRAIN_KEYS}
    if log:
        text = open(log, errors="ignore").read()
        for k, pat in TRAIN_KEYS.items():
            series[k] = [float(x) for x in re.findall(pat, text)]
    return log, series


def plot(run):
    ev = read_eval_curve(run)
    log, tr = read_train_curve(run)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f"SkyRL run: {run}", fontsize=14, fontweight="bold")

    # (1) eval accuracy
    ax = axes[0, 0]
    if ev:
        steps = [s for s, _, _ in ev]
        ax.plot(steps, [p for _, p, _ in ev], "o-", label="pass@1", color="#1a6")
        ax.plot(steps, [a for _, _, a in ev], "s--", label="avg_score", color="#888")
        b, f = ev[0][1], ev[-1][1]
        ax.set_title(f"eval pass@1   {b:.3f} → {f:.3f}  ({'+' if f >= b else ''}{f-b:.3f})")
        ax.legend(); ax.set_xlabel("global_step"); ax.grid(alpha=.3)
    else:
        ax.text(.5, .5, "no eval dumps", ha="center"); ax.set_title("eval")

    # (2) train reward
    ax = axes[0, 1]
    if tr["reward"]:
        ax.plot(range(1, len(tr["reward"]) + 1), tr["reward"], color="#06c")
        ax.set_title("train mean reward"); ax.set_xlabel("step"); ax.grid(alpha=.3)
    else:
        ax.set_title("train reward (none)")

    # (3) loss + grad_norm (twin axis)
    ax = axes[1, 0]
    if tr["loss"]:
        ax.plot(range(1, len(tr["loss"]) + 1), tr["loss"], color="#c30", label="loss")
        ax.set_ylabel("loss", color="#c30"); ax.set_xlabel("step"); ax.grid(alpha=.3)
        if tr["grad_norm"]:
            ax2 = ax.twinx()
            ax2.plot(range(1, len(tr["grad_norm"]) + 1), tr["grad_norm"], color="#093", alpha=.6, label="grad_norm")
            ax2.set_ylabel("grad_norm", color="#093")
        ax.set_title("policy loss / grad norm")
    else:
        ax.set_title("loss (none)")

    # (4) entropy
    ax = axes[1, 1]
    if tr["entropy"]:
        ax.plot(range(1, len(tr["entropy"]) + 1), tr["entropy"], color="#90c")
        ax.set_title("policy entropy"); ax.set_xlabel("step"); ax.grid(alpha=.3)
    else:
        ax.set_title("entropy (none)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = f"{EXPORTS}/{run}/{run}_curves.png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=130)
    plt.close(fig)

    # CSV (per-step train metrics + eval where available)
    out_csv = f"{EXPORTS}/{run}/{run}_metrics.csv"
    n = max([len(v) for v in tr.values()] + [0])
    eval_by_step = {s: (p, a) for s, p, a in ev}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "reward", "loss", "grad_norm", "entropy", "eval_pass@1", "eval_avg_score"])
        for i in range(n):
            s = i + 1
            ep, ea = eval_by_step.get(s, ("", ""))
            w.writerow([s] + [tr[k][i] if i < len(tr[k]) else "" for k in ("reward", "loss", "grad_norm", "entropy")] + [ep, ea])

    print(f"[plot_run] {run}: {len(ev)} evals, {n} train steps")
    if ev:
        print(f"[plot_run] eval pass@1: {ev[0][1]:.3f} (baseline) -> {ev[-1][1]:.3f} (final)")
    print(f"[plot_run] wrote {out_png}")
    print(f"[plot_run] wrote {out_csv}")
    return out_png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="RUN_NAME (e.g. searchr1_mini_fixed)")
    plot(ap.parse_args().run)
