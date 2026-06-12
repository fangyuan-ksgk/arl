"""
Generate figures for the demonstration notebook:
  1. pipeline_architecture.png  — the SkyRL + local-proot-sandbox GRPO loop
  2. terminal_curve.png         — Goal 1 GRPO learning curves (from logs)
  3. terminal_heatmap.png       — Goal 1 per-task score across eval steps
  4. geo3k_curve.png            — Goal 2 GRPO learning curve (from logs)

Run:  python make_figures.py --out_dir figures
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import parse_results as pr

FIGDIR = Path(__file__).parent / "figures"


def _box(ax, xy, w, h, text, fc, ec="#222", fs=10, tc="#111"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, wrap=True)


def _arrow(ax, p0, p1, color="#333", style="-|>", lw=1.8, rad=0.0, label=None, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                                 zorder=1, linestyle=ls))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=8, color=color)


def architecture(out):
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.set_title("One GRPO step — Terminal-Bench on SkyRL with a container-free proot sandbox (1× A100-80GB)",
                 fontsize=13, weight="bold", pad=10)

    def num(ax, x, y, n, c):
        ax.add_patch(plt.Circle((x, y), 0.17, color=c, zorder=5))
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=6)

    BLUE, GREEN, YELLOW, RED, GREENL = "#dbeafe", "#dcfce7", "#fef9c3", "#fee2e2", "#16a34a"
    # Top row: the rollout flow, left -> right
    _box(ax, (0.3, 5.5), 2.9, 1.4, "vLLM policy\n(colocated, sleep/wake)\nQwen2.5-Coder-3B\nsample n=8 scripts", "#e0e7ff", fs=9)
    _box(ax, (3.9, 5.5), 2.9, 1.4, "skyrl-gym env\n'terminal'\nextract ```bash```\nfrom each rollout", GREEN, fs=9)
    _box(ax, (7.5, 5.5), 2.9, 1.4, "proot /app sandbox\nproot -b $TMP:/app\nrun bash; writes\nland in /app", YELLOW, fs=9)
    _box(ax, (11.0, 5.5), 2.2, 1.4, "pytest verifier\ntask's tests/\non /app →\nreward∈[0,1]", RED, fs=9)
    for i, x in enumerate([0.15, 3.75, 7.35, 10.85], start=1):
        num(ax, x, 6.95, i, "#1e3a8a")
    num(ax, 10.85, 6.95, 4, "#b91c1c")
    _arrow(ax, (3.2, 6.2), (3.9, 6.2), label="prompt")
    _arrow(ax, (6.8, 6.2), (7.5, 6.2), label="bash")
    _arrow(ax, (10.4, 6.2), (11.0, 6.2), label="verify")

    # GRPO feedback loop (bottom)
    _box(ax, (4.2, 3.0), 5.1, 1.2,
         "GRPO   —   8 rewards per prompt\nadvantage = (r − mean)/std   →   policy gradient (FSDP)", BLUE, fs=10)
    num(ax, 4.05, 4.05, 5, GREENL)
    _box(ax, (0.3, 3.0), 2.9, 1.2, "FSDP policy update\nthen NCCL weight\nsync → vLLM", "#c7d2fe", fs=9)
    num(ax, 0.15, 4.05, 6, GREENL)
    # arrows: verifier -> grpo -> fsdp -> vLLM
    _arrow(ax, (12.1, 5.5), (12.1, 3.6), color=GREENL)
    _arrow(ax, (12.1, 3.6), (9.3, 3.6), color=GREENL, label="rewards")
    _arrow(ax, (4.2, 3.6), (3.2, 3.6), color=GREENL)
    _arrow(ax, (1.75, 4.2), (1.75, 5.5), color=GREENL, label="updated weights")

    ax.text(6.75, 1.55,
            "Why proot?  No CAP_SYS_ADMIN and user namespaces are blocked here, so Docker / rootless Docker / bubblewrap all fail.\n"
            "proot is a userspace ptrace chroot needing zero privilege: it binds a per-rollout temp dir to the guest path /app, so the\n"
            "agent's writes and the verifier's absolute /app/... assertions hit the same isolated dir.  The tests/ never enter the sandbox.",
            ha="center", va="center", fontsize=9.5, color="#334155",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", ec="#cbd5e1"))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def terminal_curve(log, exp, out):
    cdf = pr.parse_console_log(log)
    cur = pr.eval_curve(cdf)
    if cur.empty:
        print("no eval curve yet"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = {"eval/all/avg_score": ("mean test-pass fraction", "#2563eb"),
            "eval/all/environment/all_passed": ("tasks fully solved", "#16a34a"),
            "eval/all/pass_at_1": ("pass@1 (any partial)", "#9333ea")}
    for k, (lab, c) in cmap.items():
        if k in cur.columns:
            ax.plot(cur["step"], cur[k], "-o", color=c, label=lab)
    ax.set_xlabel("GRPO step"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Goal 1 — Terminal-Bench GRPO (Qwen2.5-Coder-3B)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def terminal_heatmap(exp, out):
    edf = pr.load_eval_dumps(exp)
    if edf.empty:
        print("no eval dumps yet"); return
    piv = edf.pivot_table(index="task_name", columns="step", values="score")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("eval @ GRPO step"); ax.set_title("Goal 1 — per-task score")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction passed")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def geo3k_curve(log, out):
    cdf = pr.parse_console_log(log)
    sub = cdf[cdf.metric.isin(["eval/all/avg_score", "eval/all/pass_at_1",
                               "eval/geometry3k/avg_score"])]
    if sub.empty:
        print("no geo3k eval yet"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, c in [("eval/all/avg_score", "#2563eb"), ("eval/all/pass_at_1", "#9333ea")]:
        s = sub[sub.metric == k]
        if not s.empty:
            ax.plot(s["step"], s["value"], "-o", color=c, label=k.split("/")[-1])
    ax.set_xlabel("GRPO step"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Goal 2 — Geometry-3K VLM GRPO (Qwen3-VL-8B + LoRA)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(FIGDIR))
    ap.add_argument("--terminal_log", default="/tmp/terminal_grpo.log")
    ap.add_argument("--terminal_exp", default="/home/claudeuser/exports/terminal_v2")
    ap.add_argument("--geo3k_log", default="/tmp/geo3k_grpo.log")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    od = Path(args.out_dir)
    architecture(od / "pipeline_architecture.png")
    terminal_curve(args.terminal_log, args.terminal_exp, od / "terminal_curve.png")
    terminal_heatmap(args.terminal_exp, od / "terminal_heatmap.png")
    geo3k_curve(args.geo3k_log, od / "geo3k_curve.png")
