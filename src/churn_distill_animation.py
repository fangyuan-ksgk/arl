"""Animate checkpoint churn and per-query teacher distillation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap


def load_eval_churn(run_dir: str | Path, max_step: int = 225):
    """Load the real rank-local GSM8K greedy-eval correctness matrix."""
    rows = {}
    for rank, path in enumerate(sorted(Path(run_dir).glob("eval_rollout.rank*.jsonl"))):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                step = record["global_step"]
                if (
                    step <= max_step
                    and record["split"] == "eval"
                    and record["decoding"] == "greedy"
                ):
                    rows[(rank, record["idx"], step)] = bool(record["correct"])

    steps = sorted({step for _, _, step in rows})
    queries = sorted({(rank, idx) for rank, idx, _ in rows})
    matrix = np.array(
        [[rows[(rank, idx, step)] for step in steps] for rank, idx in queries],
        dtype=bool,
    )
    return np.asarray(steps), matrix


def _query_grid(values, fill=-1):
    side = int(np.ceil(np.sqrt(len(values))))
    grid = np.full(side * side, fill, dtype=np.asarray(values).dtype)
    grid[: len(values)] = values
    return grid.reshape(side, side)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def make_churn_distill_animation(
    run_dir: str | Path,
    output: str | Path | None = None,
    *,
    fps: int = 2,
):
    """Build the two-act animation; optionally save it as a GIF."""
    steps, correct = load_eval_churn(run_dir)
    current = 100 * correct.mean(axis=0)
    ever = 100 * np.maximum.accumulate(correct, axis=1).mean(axis=0)
    gap = ever - current
    last_solved = np.where(correct, np.arange(len(steps)), -1).max(axis=1)

    # Hold key moments without duplicating data frames in the update function.
    frames = list(range(len(steps))) + [len(steps) - 1] * 3
    frames += [len(steps) + i for i in range(9)]

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#F8FAFC")
    status_cmap = ListedColormap(["#CBD5E1", "#22C55E", "#F43F5E"])
    status_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], status_cmap.N)
    teacher_cmap = plt.colormaps["turbo"].copy()
    teacher_cmap.set_under("#CBD5E1")

    def draw_churn(frame):
        fig.clear()
        gs = fig.add_gridspec(
            1, 2, width_ratios=(1.22, 1), left=0.06, right=0.96,
            top=0.84, bottom=0.13, wspace=0.22,
        )
        ax_grid, ax_curve = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        solved_before = np.maximum.accumulate(correct[:, : frame + 1], axis=1)[:, -1]
        status = np.where(correct[:, frame], 1, np.where(solved_before, 2, 0))
        ax_grid.imshow(
            _query_grid(status),
            cmap=status_cmap,
            norm=status_norm,
            interpolation="nearest",
        )
        ax_grid.set_title("Each square is one validation query", fontweight="bold")
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        for spine in ax_grid.spines.values():
            spine.set_visible(False)

        ax_curve.plot(steps[: frame + 1], current[: frame + 1], "o-",
                      color="#16A34A", lw=2.5, label="correct now")
        ax_curve.plot(steps[: frame + 1], ever[: frame + 1], "o-",
                      color="#334155", lw=2.5, label="ever solved")
        ax_curve.fill_between(
            steps[: frame + 1], current[: frame + 1], ever[: frame + 1],
            color="#FDA4AF", alpha=0.45, label="forgotten",
        )
        ax_curve.set(xlim=(steps[0], steps[-1]), ylim=(45, 100),
                     xlabel="training step", ylabel="queries (%)")
        ax_curve.legend(loc="lower right", frameon=False)
        _style(ax_curve)

        fig.suptitle(
            f"Checkpoint {steps[frame]}: the solved set keeps churning",
            y=0.95, fontsize=20, fontweight="bold", color="#0F172A",
        )
        fig.text(
            0.5, 0.885,
            f"current accuracy  {current[frame]:.1f}%     "
            f"ever solved  {ever[frame]:.1f}%     "
            f"forgetting gap  {gap[frame]:.1f} pt",
            ha="center", fontsize=13, color="#475569",
        )
        fig.text(
            0.08, 0.055,
            "■ correct now", color="#16A34A", fontsize=11, fontweight="bold",
        )
        fig.text(
            0.21, 0.055,
            "■ solved before, wrong now", color="#E11D48",
            fontsize=11, fontweight="bold",
        )
        fig.text(
            0.44, 0.055,
            "■ never solved yet", color="#94A3B8", fontsize=11, fontweight="bold",
        )

    def draw_distillation(stage):
        fig.clear()
        gs = fig.add_gridspec(
            1, 2, width_ratios=(1.15, 1), left=0.06, right=0.96,
            top=0.84, bottom=0.13, wspace=0.24,
        )
        ax_grid, ax_metrics = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        ax_grid.imshow(
            _query_grid(last_solved),
            cmap=teacher_cmap,
            vmin=0,
            vmax=len(steps) - 1,
            interpolation="nearest",
        )
        ax_grid.set_title("Choose a teacher checkpoint per query", fontweight="bold")
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        for spine in ax_grid.spines.values():
            spine.set_visible(False)
        ax_grid.text(
            0.5, -0.08,
            "color = checkpoint that solved this query   ·   gray = never solved",
            transform=ax_grid.transAxes, ha="center", color="#64748B",
        )

        progress = min(1.0, stage / 6)
        accuracy = 70.9 + progress * (73.4 - 70.9)
        distilled_gap = 20.7 + progress * (16.1 - 20.7)
        y = np.arange(2)
        ax_metrics.barh(y, [70.9, 20.7], color="#CBD5E1", height=0.35)
        ax_metrics.barh(
            y + 0.36, [accuracy, distilled_gap],
            color=["#2563EB", "#8B5CF6"], height=0.35,
        )
        ax_metrics.set(
            yticks=y + 0.18,
            yticklabels=["Accuracy ↑", "Forgetting gap ↓"],
            xlim=(0, 80),
            xlabel="percentage points",
        )
        ax_metrics.invert_yaxis()
        ax_metrics.legend(
            ["Dr.GRPO", "logit distillation"],
            loc="lower right", frameon=False,
        )
        _style(ax_metrics)
        ax_metrics.text(70.9 + 1, -0.18, "70.9", va="center", color="#475569")
        ax_metrics.text(
            accuracy + 1, 0.18, f"{accuracy:.1f}",
            va="center", color="#1D4ED8", fontweight="bold",
        )
        ax_metrics.text(20.7 + 1, 0.82, "20.7", va="center", color="#475569")
        ax_metrics.text(
            distilled_gap + 1, 1.18, f"{distilled_gap:.1f}",
            va="center", color="#6D28D9", fontweight="bold",
        )

        fig.suptitle(
            "Distill each query from its best checkpoint",
            y=0.95, fontsize=20, fontweight="bold", color="#0F172A",
        )
        fig.text(
            0.5, 0.885,
            "Temporal diversity becomes one student: no multi-run ensemble needed",
            ha="center", fontsize=13, color="#475569",
        )
        fig.text(
            0.5, 0.045,
            "Churn trajectories are measured; post-distillation metrics are aggregate evals.",
            ha="center", fontsize=10, color="#64748B",
        )

    def update(frame):
        if frame < len(steps):
            draw_churn(frame)
        else:
            draw_distillation(frame - len(steps))

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    if output is not None:
        animation.save(output, writer=PillowWriter(fps=fps), dpi=110)
    return animation
