"""
Post-hoc diagnostic probes for Game-of-24 GRPO rollouts.

Each probe returns ``(metrics_dict, matplotlib.Figure | None)`` so callers can
both aggregate scalar metrics across runs and save figures to disk.

Probes
------
- ``d1_length_diversity``     — D1 collapse / mode collapse signature
- ``coverage_probe``          — answer-buffer motivator (NOT D2)
- ``d3_pass_rate_by_bucket``  — zero-pass@K dead-zone test, with
                                a found-vs-total-solutions panel
- ``rt_dynamics``             — per-eval-cycle 2-panel R_t figure
                                (one (correct, incorrect) pair + global mean ±1σ)
- ``rt_progress``             — mean R_T per eval cycle, split by correctness;
                                summarises how training reshapes the
                                decoding-velocity reward over time.

The R_T probes consume the augmented ``eval_rollout.jsonl`` produced by
``script/run_game24_one.py`` (R_T, R_per_token, cumR_resampled). The
underlying kernel lives in ``src/velocity.py``
(``compute_vt_batched``, ``compute_cot_perplexity``).
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

__all__ = [
    "load_rollouts",
    "d1_length_diversity",
    "coverage_probe",
    "d3_pass_rate_by_bucket",
    "rt_dynamics",
    "rt_progress",
]


# ---------------------------------------------------------------------------
# 0. Common helpers
# ---------------------------------------------------------------------------
def load_rollouts(path: Path) -> pd.DataFrame:
    """Load the JSONL rollout log into a DataFrame keyed by sorted-numbers tuple."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if len(df):
        df["key"] = df["numbers"].apply(lambda x: tuple(sorted(x)))
    return df


def _norm_edit(a: str, b: str) -> float:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 1.0 - sm.ratio()


def _canon(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def _puzzle_index(puzzles: List[Dict[str, Any]]) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    return {tuple(sorted(p["numbers"])): p for p in puzzles}


# ---------------------------------------------------------------------------
# D1 — CoT length + correct-rollout diversity
# ---------------------------------------------------------------------------
def d1_length_diversity(df: pd.DataFrame) -> Tuple[Dict[str, float], Figure]:
    if len(df) == 0:
        return {}, None

    # Prefer CoT-only length when the logger recorded it; older runs only
    # have the full-completion ``n_tokens`` field.
    len_col = "n_cot_tokens" if "n_cot_tokens" in df.columns else "n_tokens"

    agg = df.groupby("step").agg(
        n_tokens_mean=(len_col, "mean"),
        n_tokens_p90=(len_col, lambda s: float(np.percentile(s, 90))),
        acc=("correct", "mean"),
    ).reset_index()

    ed_rows = []
    for (step, key), g in df[df.correct].groupby(["step", "key"]):
        texts = g.completion.tolist()
        if len(texts) < 2:
            continue
        if len(texts) > 20:
            import random
            texts = random.sample(texts, 20)
        pairs = [_norm_edit(texts[i], texts[j])
                 for i in range(len(texts)) for j in range(i + 1, len(texts))]
        ed_rows.append({"step": step, "key": key,
                        "mean_edit": float(np.mean(pairs))})
    edf = pd.DataFrame(ed_rows)
    # Aggregate puzzles → one point per step. mean over (step, puzzle) cells
    # gives the average "within-puzzle diversity of correct CoTs" at that step.
    edf_step = (edf.groupby("step")["mean_edit"]
                   .agg(["mean", "std", "count"]).reset_index()
                if len(edf) else pd.DataFrame(columns=["step","mean","std","count"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(agg.step, agg.n_tokens_mean, label="mean")
    axes[0].plot(agg.step, agg.n_tokens_p90, label="p90", linestyle="--")
    axes[0].set(xlabel="step", ylabel="CoT tokens", title="D1 · CoT length")
    axes[0].legend()

    for ok, label in [(True, "correct"), (False, "incorrect")]:
        sub = df[df.correct == ok].groupby("step")[len_col].mean()
        if len(sub):
            axes[1].plot(sub.index, sub.values, label=label)
    axes[1].set(xlabel="step", ylabel=f"mean {len_col}",
                title="D1 · length, split by correctness")
    axes[1].legend()

    if len(edf_step):
        x = edf_step["step"].values
        mu = edf_step["mean"].values
        sd = edf_step["std"].fillna(0.0).values
        axes[2].plot(x, mu, color="#264653", linewidth=2,
                     label="mean over puzzles")
        axes[2].fill_between(x, mu - sd, mu + sd, color="#264653", alpha=0.15,
                             label="±1σ across puzzles")
        axes[2].set(xlabel="step", ylabel="within-puzzle mean pairwise edit-dist",
                    title="D1 · diversity of correct CoTs\n(collapse → 0)")
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "not enough correct rollouts\non same puzzle",
                     ha="center", va="center")
        axes[2].axis("off")
    fig.tight_layout()

    metrics = {
        "d1_mean_tokens_first":  float(agg.n_tokens_mean.iloc[0])  if len(agg) else float("nan"),
        "d1_mean_tokens_last":   float(agg.n_tokens_mean.iloc[-1]) if len(agg) else float("nan"),
        "d1_mean_tokens_delta":  float(agg.n_tokens_mean.iloc[-1] - agg.n_tokens_mean.iloc[0]) if len(agg) > 1 else float("nan"),
        "d1_edit_distance_last": float(edf_step["mean"].iloc[-1]) if len(edf_step) else float("nan"),
    }
    return metrics, fig


# ---------------------------------------------------------------------------
# Diversity probe — coverage of answer buffer A_q (NOT D2)
# ---------------------------------------------------------------------------
def coverage_probe(df: pd.DataFrame, puzzles: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Optional[Figure]]:
    if len(df) == 0:
        return {}, None
    idx = _puzzle_index(puzzles)
    rows = []
    for key, g in df[df.correct].groupby("key"):
        p = idx.get(key)
        if p is None:
            continue
        unique = len(set(_canon(e) for e in g.expr))
        rows.append({"unique_found": unique,
                     "total_solutions": p["n_solutions"],
                     "coverage": unique / p["n_solutions"]})
    if not rows:
        return {"coverage_mean": 0.0, "n_solved_puzzles": 0}, None

    ddf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(ddf.coverage, bins=20)
    ax.set(xlabel="fraction of a puzzle's solutions the model produced",
           ylabel="# puzzles",
           title="Coverage probe — motivates answer buffer A_q")
    fig.tight_layout()
    return {
        "coverage_mean":    float(ddf.coverage.mean()),
        "unique_per_puzzle": float(ddf.unique_found.mean()),
        "n_solved_puzzles":  int(len(ddf)),
    }, fig


# ---------------------------------------------------------------------------
# D3 — pass-rate by difficulty bucket
# ---------------------------------------------------------------------------
def d3_pass_rate_by_bucket(df: pd.DataFrame, puzzles: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Optional[Figure]]:
    """Two-panel D3.

    Left  — fraction of puzzles "ever solved" per difficulty bucket (binary).
    Right — for each bucket, the average number of *distinct* canonical
            solutions the model produced per puzzle, vs the average number
            of total valid solutions the puzzle admits. Closes the loop on
            "hard puzzles have fewer solutions" — even a 100 % pass-rate on
            easy puzzles can hide low solution coverage.
    """
    if len(df) == 0:
        return {}, None
    info = (
        pd.DataFrame([{"key": tuple(sorted(p["numbers"])),
                       "n_solutions": p["n_solutions"]} for p in puzzles])
        .drop_duplicates("key").set_index("key")
    )

    # Distinct correct expressions found per puzzle (canonicalised).
    correct_df = df[df.correct & df.expr.notna()].copy()
    if len(correct_df):
        correct_df["canon"] = correct_df["expr"].astype(str).map(_canon)
        found = (correct_df.groupby("key")["canon"]
                           .nunique().rename("n_found"))
    else:
        found = pd.Series(dtype=int, name="n_found")

    solved = df.groupby("key").correct.any().rename("ever_solved")
    joined = info.join(solved).join(found).fillna(
        {"ever_solved": False, "n_found": 0})
    joined["n_found"] = joined["n_found"].astype(int)

    bucket = pd.cut(joined.n_solutions, [0, 2, 7, 1000],
                    labels=["hard (≤2)", "med (3-7)", "easy (≥8)"])
    pass_grouped = joined.groupby(bucket).ever_solved.agg(["mean", "count"])
    sol_grouped  = joined.groupby(bucket).agg(
        mean_found=("n_found", "mean"),
        mean_total=("n_solutions", "mean"),
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 3.5))

    # --- left: pass-rate bars (existing) ---
    ax_l.bar(range(len(pass_grouped)), pass_grouped["mean"].values,
             tick_label=[str(i) for i in pass_grouped.index])
    ax_l.set(ylabel="fraction ever solved", ylim=(0, 1.05),
             title="D3 · pass-rate by difficulty bucket")
    for i, (m, n) in enumerate(zip(pass_grouped["mean"], pass_grouped["count"])):
        ax_l.text(i, m + 0.02, f"{m:.0%}\n(n={n})", ha="center", fontsize=8)

    # --- right: avg found vs avg total solutions (grouped bars) ---
    x = np.arange(len(sol_grouped))
    w = 0.38
    ax_r.bar(x - w / 2, sol_grouped["mean_found"], w,
             label="avg distinct found", color="#2a9d8f")
    ax_r.bar(x + w / 2, sol_grouped["mean_total"], w,
             label="avg total solutions", color="#9b9b9b")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([str(i) for i in sol_grouped.index])
    ax_r.set(ylabel="# canonical solutions / puzzle",
             title="D3 · solution coverage (found / total)")
    ax_r.legend(fontsize=8)
    for i, (f_, t_) in enumerate(zip(sol_grouped["mean_found"], sol_grouped["mean_total"])):
        ratio = f_ / t_ if t_ > 0 else float("nan")
        ax_r.text(i, max(f_, t_) + 0.05, f"{ratio:.0%}",
                  ha="center", fontsize=8, color="#264653")

    fig.tight_layout()

    metrics = {f"d3_passrate_{label}": float(m)
               for label, m in zip(pass_grouped.index.astype(str), pass_grouped["mean"])}
    metrics["d3_passrate_overall"] = float(joined.ever_solved.mean())
    for label, row in sol_grouped.iterrows():
        metrics[f"d3_mean_found_{label}"] = float(row["mean_found"])
        metrics[f"d3_mean_total_{label}"] = float(row["mean_total"])
    metrics["d3_coverage_overall"] = (
        float(sol_grouped["mean_found"].sum() / sol_grouped["mean_total"].sum())
        if sol_grouped["mean_total"].sum() > 0 else float("nan")
    )
    return metrics, fig


# ---------------------------------------------------------------------------
# R_T — decoding-velocity reward dynamics (consumes augmented eval JSONL)
# ---------------------------------------------------------------------------
def _stack_cumR(df_class: pd.DataFrame) -> np.ndarray:
    """Stack `cumR_resampled` arrays into (N, n_pts). Skips None / empty rows."""
    if len(df_class) == 0:
        return np.empty((0, 0))
    arrs = [np.asarray(x, dtype=float)
            for x in df_class["cumR_resampled"].tolist()
            if x is not None and len(x)]
    return np.array(arrs) if arrs else np.empty((0, 0))


def rt_dynamics(
    df: pd.DataFrame,
    step_idx: int,
    *,
    pair_seed: int = 0,
) -> Tuple[Dict[str, float], Optional[Figure]]:
    """Two-panel R_t figure for one eval cycle (`global_step == step_idx`).

    Left  — one randomly-sampled (correct, incorrect) pair: cumulative R_t
            over normalised CoT position (t / T).
    Right — global mean cumulative R_t across ALL rollouts at this step,
            ±1σ band, split by correctness.

    Requires the eval-log augmentation written by
    ``script/run_game24_one.py --score-vt`` (default on): each row must
    carry ``R_T`` and ``cumR_resampled``.
    """
    if "cumR_resampled" not in df.columns or "global_step" not in df.columns:
        return {}, None
    at = df[(df.global_step == step_idx) & df.cumR_resampled.notna()]
    if len(at) == 0:
        return {}, None
    correct   = at[at.correct]
    incorrect = at[~at.correct]

    R_c, R_i = _stack_cumR(correct), _stack_cumR(incorrect)
    n_pts = R_c.shape[1] if len(R_c) else (R_i.shape[1] if len(R_i) else 0)
    if n_pts == 0:
        return {}, None
    x = np.linspace(0.0, 1.0, n_pts)

    pair_c = correct.sample(1,   random_state=pair_seed).iloc[0] if len(correct)   else None
    pair_i = incorrect.sample(1, random_state=pair_seed).iloc[0] if len(incorrect) else None

    fig, (ax_pair, ax_avg) = plt.subplots(1, 2, figsize=(14, 4.5))

    for row, lbl, col in [(pair_c, "correct", "#2a9d8f"),
                          (pair_i, "incorrect", "#e76f51")]:
        if row is None:
            continue
        R = np.asarray(row.cumR_resampled, dtype=float)
        T = row.get("n_cot_tokens", row.get("n_tokens", "?"))
        ax_pair.plot(x, R, color=col, linewidth=2,
                     label=f"{lbl}  (R_T={row.R_T:+.2f}, T={T})")
    ax_pair.axhline(0, color="k", linestyle=":", alpha=0.4)
    ax_pair.set(xlabel="normalised CoT position (t / T)",
                ylabel="cumulative R_t",
                title=f"R_t · individual pair  (step={step_idx})")
    ax_pair.legend(fontsize=9); ax_pair.grid(alpha=0.3)

    for R, lbl, col in [(R_c, "correct", "#2a9d8f"), (R_i, "incorrect", "#e76f51")]:
        if len(R) == 0:
            continue
        mu, sd = R.mean(0), R.std(0)
        ax_avg.plot(x, mu, color=col, linewidth=2, label=f"{lbl}  (n={len(R)})")
        ax_avg.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15)
    ax_avg.axhline(0, color="k", linestyle=":", alpha=0.4)
    ax_avg.set(xlabel="normalised CoT position (t / T)",
               ylabel="mean cumulative R_t",
               title=f"R_t · global average  (step={step_idx})")
    ax_avg.legend(fontsize=9); ax_avg.grid(alpha=0.3)
    fig.tight_layout()

    metrics = {
        f"rt_step{step_idx}_R_T_correct_mean":   float(correct.R_T.mean())   if len(correct)   else float("nan"),
        f"rt_step{step_idx}_R_T_incorrect_mean": float(incorrect.R_T.mean()) if len(incorrect) else float("nan"),
        f"rt_step{step_idx}_n_correct":          int(len(correct)),
        f"rt_step{step_idx}_n_incorrect":        int(len(incorrect)),
    }
    return metrics, fig


def rt_progress(df: pd.DataFrame) -> Tuple[Dict[str, float], Optional[Figure]]:
    """Mean R_T per eval cycle, split by correctness — training-time view.

    One line per class over `global_step`, with ±1σ bands. Shows whether
    training drives correct-rollout R_T upward and how the gap between
    classes evolves.
    """
    if "R_T" not in df.columns or "global_step" not in df.columns:
        return {}, None
    sub = df[df.R_T.notna()]
    if len(sub) == 0:
        return {}, None
    grouped = (sub.groupby(["global_step", "correct"])["R_T"]
                  .agg(["mean", "std", "count"]).reset_index())

    fig, ax = plt.subplots(figsize=(7, 4))
    for cf, col, lbl in [(True, "#2a9d8f", "correct"),
                         (False, "#e76f51", "incorrect")]:
        g = grouped[grouped.correct == cf].sort_values("global_step")
        if len(g) == 0:
            continue
        x  = g.global_step.values
        mu = g["mean"].values
        sd = g["std"].fillna(0.0).values
        ax.plot(x, mu, marker="o", color=col,
                label=f"{lbl} (n={int(g['count'].sum())})")
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15)
    ax.axhline(0, color="k", linestyle=":", alpha=0.4)
    ax.set(xlabel="global_step", ylabel="mean R_T",
           title="R_T progress over eval cycles")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()

    overall = grouped.groupby("correct")["mean"].mean()
    metrics: Dict[str, float] = {
        "rt_progress_R_T_correct_mean":   float(overall.get(True,  float("nan"))),
        "rt_progress_R_T_incorrect_mean": float(overall.get(False, float("nan"))),
    }
    last = grouped.global_step.max()
    last_g = grouped[grouped.global_step == last].set_index("correct")["mean"]
    if True in last_g.index and False in last_g.index:
        metrics["rt_progress_final_gap"] = float(last_g[True] - last_g[False])
    return metrics, fig

