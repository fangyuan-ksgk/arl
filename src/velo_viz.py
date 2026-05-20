"""Velocity-reward visualization helpers.

One ``prepare`` call computes everything the three views need; each view
function then takes the resulting context and renders.

- :func:`prepare`           pre-compute paragraph boundaries, cumulative R(t),
                             ΔR_own per paragraph, sorted final-R_T order, etc.
- :func:`plot_R_t_static`   two-panel matplotlib (curves + sorted final R_T bars)
- :func:`print_paragraphs`  ANSI-colored terminal print of CoT paragraphs + ΔR_own
- :func:`make_animation`    streaming animation (live curves + bars + CoT reveal)
- :func:`plot_lifecycle_river`  stream-graph life-cycle "river" of unique
                                expressions for one Game-of-24 puzzle across
                                eval cycles (consumes ``eval_rollout.jsonl``).
"""
from __future__ import annotations

import ast
import bisect
import json
import re
import textwrap
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


__all__ = ["VizCtx", "prepare", "plot_R_t_static", "print_paragraphs",
           "make_animation", "make_pair_animation",
           "make_pair_animation_from_vt", "plot_lifecycle_river",
           "plot_solution_coverage_dynamics",
           "plot_prefix_sharing_dynamics",
           "plot_R_t_correctness_dynamics",
           "plot_truncation_stats",
           "plot_length_and_diversity",
           "plot_pass_at_k"]


# ────────────────────────────────────────────────────────────────────────────
# pass@k over training
# ────────────────────────────────────────────────────────────────────────────
def plot_pass_at_k(eval_df, *, ks: Tuple[int, ...] = (1, 4, 8),
                   figsize=(7, 4.2), n_boot: int = 300, seed: int = 0,
                   verbose: bool = True):
    """pass@k over GRPO training (95% bootstrap CI).

    Each (eval cycle, puzzle) cell has ``num_generations`` rollouts.
    Per-puzzle unbiased estimator::

        pass@k = 1 - C(n - c, k) / C(n, k),  n = #rollouts, c = #correct

    Cycle-level point = mean across puzzles; CI ribbon = 95% bootstrap
    over puzzles. No extra forward passes needed.

    Returns ``(fig, ax, info)`` with ``info`` = {"pak_long", "pak_df"}.
    """
    from math import comb
    import pandas as pd

    C_K = {1: "#264653", 4: "#2a9d8f", 8: "#e76f51"}

    def _pass_at_k(c: int, n: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)

    rows = []
    for (gs, key), g in eval_df.groupby(["global_step", "key"]):
        n = len(g); c = int(g.correct.sum())
        for k in ks:
            if k > n:
                continue
            rows.append({"global_step": gs, "key": key, "k": k,
                         "pass": _pass_at_k(c, n, k)})
    pak_long = pd.DataFrame(rows)

    def _boot_ci_mean(values):
        v = np.asarray(values, dtype=float)
        if len(v) < 2:
            c = float(v.mean()) if len(v) else float("nan")
            return c, c
        rng = np.random.default_rng(seed)
        boots = np.array([rng.choice(v, size=len(v), replace=True).mean()
                          for _ in range(n_boot)])
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    pak_summary = []
    for (step, k), g in pak_long.groupby(["global_step", "k"]):
        lo, hi = _boot_ci_mean(g["pass"].values)
        pak_summary.append({"step": step, "k": k,
                            "val": float(g["pass"].mean()),
                            "lo": lo, "hi": hi})
    pak_df = pd.DataFrame(pak_summary).sort_values(["k", "step"])

    with plt.rc_context({
        "axes.spines.top":   False, "axes.spines.right": False,
        "axes.grid":         True,  "grid.alpha": 0.25,
        "grid.linestyle":    ":",
        "axes.titleweight":  "bold", "axes.titlesize": 11,
        "axes.labelsize":    10,    "legend.frameon": False,
    }):
        fig, ax = plt.subplots(figsize=figsize)
        for k in ks:
            sub = pak_df[pak_df.k == k]
            if sub.empty:
                continue
            c = C_K.get(k, None)
            ax.plot(sub.step, sub.val, color=c, lw=2, marker="o", ms=5,
                    label=f"pass@{k}")
            ax.fill_between(sub.step, sub.lo, sub.hi,
                            color=c, alpha=0.15, linewidth=0)
        ax.set(xlabel="global_step", ylabel="pass@k", ylim=(0, 1.02),
               title="Accuracy · pass@k over eval cycles (95% CI)")
        ax.legend(loc="best")
        fig.tight_layout()

    if verbose:
        print(pak_df.pivot(index="step", columns="k", values="val")
                    .round(3)
                    .rename_axis(index="global_step"))

    return fig, ax, {"pak_long": pak_long, "pak_df": pak_df}


# ────────────────────────────────────────────────────────────────────────────
# D1 · CoT length & diversity
# ────────────────────────────────────────────────────────────────────────────
def plot_length_and_diversity(eval_df, *, figsize=(16, 4.2),
                              n_boot: int = 300, seed: int = 0):
    """D1 · CoT length & diversity over GRPO training.

    Panels:
      1. Mean CoT length per step (95% bootstrap CI ribbon).
      2. Mean CoT length split by correctness (95% bootstrap CI ribbons).
      3. Mean unique-CoT count per puzzle, split by correctness.
         Per (step, key) we count unique whitespace-normalized completions
         in each class; both counts are over the SAME G rollouts so
         ``n_correct + n_incorrect ≤ G`` (= G when no byte-identical
         duplicates within a class, which is the case here).

    Returns ``(fig, axes, info)`` with ``info`` containing
    ``mean_df``, ``corr_dfs``, ``uniq_dfs``, ``len_col``.
    """
    import pandas as pd

    C_CORRECT, C_INCORRECT, C_MEAN = "#2a9d8f", "#e76f51", "#264653"
    len_col = ("n_cot_tokens" if "n_cot_tokens" in eval_df.columns
               else "n_tokens")

    def _boot_ci(values, fn):
        v = np.asarray(values, dtype=float)
        if len(v) < 2:
            c = float(fn(v)) if len(v) else float("nan")
            return c, c
        rng = np.random.default_rng(seed)
        boots = np.array([fn(rng.choice(v, size=len(v), replace=True))
                          for _ in range(n_boot)])
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    def _per_step(df, value_col, fn):
        rows = []
        for step, g in df.groupby("global_step"):
            v = g[value_col].values
            lo, hi = _boot_ci(v, fn)
            rows.append({"step": step, "val": float(fn(v)),
                         "lo": lo, "hi": hi})
        return pd.DataFrame(rows).sort_values("step")

    # length aggregations
    mean_df  = _per_step(eval_df, len_col, np.mean)
    corr_dfs = {bool(k): _per_step(g, len_col, np.mean)
                for k, g in eval_df.groupby("correct")}

    # per-puzzle unique-CoT counts, split by correctness
    def _norm(s: str) -> str:
        return " ".join(s.split())

    def _split(g):
        ok   = g.correct.astype(bool).values
        cots = g.completion.values
        return pd.Series({
            "n_correct":   len({_norm(c) for c, k in zip(cots, ok) if k}),
            "n_incorrect": len({_norm(c) for c, k in zip(cots, ok) if not k}),
        })

    udf = (eval_df.groupby(["global_step", "key"])
                  .apply(_split, include_groups=False).reset_index())
    uniq_dfs = {
        True:  _per_step(udf.rename(columns={"n_correct":   "v"}), "v", np.mean),
        False: _per_step(udf.rename(columns={"n_incorrect": "v"}), "v", np.mean),
    }

    with plt.rc_context({
        "axes.spines.top":   False, "axes.spines.right": False,
        "axes.grid":         True,  "grid.alpha": 0.25,
        "grid.linestyle":    ":",
        "axes.titleweight":  "bold", "axes.titlesize": 11,
        "axes.labelsize":    10,    "legend.frameon": False,
    }):
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

        ax = axes[0]
        ax.plot(mean_df.step, mean_df.val, color=C_MEAN, lw=2, label="mean")
        ax.fill_between(mean_df.step, mean_df.lo, mean_df.hi,
                        color=C_MEAN, alpha=0.15, linewidth=0)
        ax.set(xlabel="global_step", ylabel=len_col,
               title="CoT length (mean, 95% CI)")
        ax.legend(loc="best")

        ax = axes[1]
        for is_corr, df in corr_dfs.items():
            c = C_CORRECT if is_corr else C_INCORRECT
            ax.plot(df.step, df.val, color=c, lw=2,
                    label="correct" if is_corr else "incorrect")
            ax.fill_between(df.step, df.lo, df.hi,
                            color=c, alpha=0.15, linewidth=0)
        ax.set(xlabel="global_step", ylabel=f"mean {len_col}",
               title="length, split by correctness (95% CI)")
        ax.legend(loc="best")

        ax = axes[2]
        for is_corr in (True, False):
            c   = C_CORRECT if is_corr else C_INCORRECT
            df  = uniq_dfs[is_corr]
            ax.plot(df.step, df.val, color=c, lw=2, marker="o", ms=5,
                    label="correct" if is_corr else "incorrect")
            ax.fill_between(df.step, df.lo, df.hi,
                            color=c, alpha=0.15, linewidth=0)
        ax.set(xlabel="global_step",
               ylabel="unique CoTs per puzzle (mean ± 95% CI)",
               title="unique CoTs per puzzle\n(correct + incorrect ≤ G)")
        ax.legend(loc="best")

        fig.suptitle("D1 · CoT length & diversity over GRPO training",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

    return fig, axes, {"mean_df": mean_df, "corr_dfs": corr_dfs,
                       "uniq_dfs": uniq_dfs, "len_col": len_col}


# ────────────────────────────────────────────────────────────────────────────
# Truncation / structural-completeness stats
# ────────────────────────────────────────────────────────────────────────────
def plot_truncation_stats(eval_df, *, figsize=(16, 4), verbose: bool = True):
    """Truncation statistics from an eval_rollout DataFrame.

    A rollout is "structurally complete" when both markers are present:
    ``</think>`` (reasoning closed) AND ``####`` (parsable answer line).
    If either is missing, the response was almost certainly cut off by
    ``max_completion_length``.

    Splits incorrect rollouts into:
      - ``parsable_wrong``   : both markers, but the expression is wrong
                               → genuine reasoning error
      - ``no_answer_marker`` : ``####`` missing → almost always truncated
      - ``no_think_close``   : ``</think>`` missing but ``####`` present → rare

    Returns (fig, axes, info) where info is a dict with the per-step summary
    DataFrame and the bucket-tagged frame.
    """
    import inspect
    import pandas as pd

    needed = {"has_answer_marker", "has_think_close"}
    assert needed.issubset(eval_df.columns), (
        f"eval_df is missing {needed - set(eval_df.columns)}; "
        "re-run training with the updated RolloutLogger."
    )

    # matplotlib >=3.9 renamed `labels=` to `tick_labels=`.
    _BOXPLOT_LABEL_KW = (
        "tick_labels"
        if "tick_labels" in inspect.signature(plt.Axes.boxplot).parameters
        else "labels"
    )

    trunc_df = eval_df.copy()
    trunc_df["parsable"] = trunc_df.has_answer_marker & trunc_df.has_think_close

    def _bucket(row):
        if row.correct:                  return "correct"
        if not row.has_answer_marker:    return "no_answer_marker"
        if not row.has_think_close:      return "no_think_close"
        return "parsable_wrong"
    trunc_df["bucket"] = trunc_df.apply(_bucket, axis=1)

    order  = ["correct", "parsable_wrong", "no_think_close", "no_answer_marker"]
    colors = {"correct": "#2a9d8f", "parsable_wrong": "#e76f51",
              "no_think_close": "#f4a261", "no_answer_marker": "#264653"}

    comp = (trunc_df.groupby(["global_step", "bucket"]).size()
                    .unstack(fill_value=0)
                    .reindex(columns=order, fill_value=0))
    comp_frac = comp.div(comp.sum(axis=1), axis=0)

    len_col = ("n_cot_tokens" if "n_cot_tokens" in trunc_df.columns
               else "n_tokens")
    by_step = trunc_df.groupby("global_step").agg(
        pct_truncated=("has_answer_marker", lambda s: 1.0 - s.mean()),
        pct_no_think =("has_think_close",   lambda s: 1.0 - s.mean()),
        pct_parsable =("parsable",          "mean"),
        pct_correct  =("correct",           "mean"),
        mean_len     =(len_col,             "mean"),
        p90_len      =(len_col, lambda s: float(np.percentile(s, 90))),
        max_len      =(len_col,             "max"),
    ).reset_index()

    if verbose:
        print("Per eval-cycle structural completeness:")
        print(by_step.assign(
            pct_truncated=lambda d: (d.pct_truncated * 100).round(1),
            pct_no_think =lambda d: (d.pct_no_think  * 100).round(1),
            pct_parsable =lambda d: (d.pct_parsable  * 100).round(1),
            pct_correct  =lambda d: (d.pct_correct   * 100).round(1),
        ).to_string(index=False))

        def _pct(mask):
            return f"{mask.mean()*100:5.1f}%  ({int(mask.sum())}/{len(mask)})"

        print("\nOverall (all eval cycles pooled):")
        print(f"  '####' present     : {_pct(trunc_df.has_answer_marker)}")
        print(f"  '</think>' present : {_pct(trunc_df.has_think_close)}")
        print(f"  parsable (both)    : {_pct(trunc_df.parsable)}")
        print(f"  correct            : {_pct(trunc_df.correct)}")

        inc = trunc_df[~trunc_df.correct]
        if len(inc):
            print(f"\nAmong {len(inc)} incorrect eval rollouts:")
            print(f"  no_answer_marker (truncated)   : {_pct(~inc.has_answer_marker)}")
            print(f"  no_think_close                 : {_pct(~inc.has_think_close & inc.has_answer_marker)}")
            print(f"  parsable_wrong (real error)    : {_pct(inc.parsable)}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # (a) trend lines over training
    axes[0].plot(by_step.global_step, by_step.pct_truncated,
                 label="no '####' (truncated)",
                 color=colors["no_answer_marker"], marker="o")
    axes[0].plot(by_step.global_step, by_step.pct_no_think,
                 label="no '</think>'",
                 color=colors["no_think_close"], marker="o", linestyle="--")
    axes[0].plot(by_step.global_step, by_step.pct_parsable,
                 label="parsable", color="#2a9d8f", marker="o")
    axes[0].plot(by_step.global_step, by_step.pct_correct,
                 label="correct", color="#2a9d8f", marker="s", linestyle=":")
    axes[0].set(xlabel="global_step", ylabel="fraction of rollouts",
                title="Structural completeness over training",
                ylim=(-0.02, 1.02))
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    # (b) stacked composition by failure mode
    axes[1].stackplot(comp_frac.index,
                      *[comp_frac[c] for c in order],
                      labels=order,
                      colors=[colors[c] for c in order], alpha=0.85)
    axes[1].set(xlabel="global_step", ylabel="fraction of rollouts",
                title="Eval-rollout composition by failure mode",
                ylim=(0, 1))
    axes[1].legend(loc="lower right", fontsize=8)

    # (c) length by bucket
    data, labels = [], []
    for b in order:
        vals = trunc_df.loc[trunc_df.bucket == b, len_col].values
        if len(vals):
            data.append(vals); labels.append(b)
    parts = axes[2].boxplot(data, patch_artist=True, showfliers=False,
                            **{_BOXPLOT_LABEL_KW: labels})
    for patch, lbl in zip(parts["boxes"], labels):
        patch.set_facecolor(colors[lbl]); patch.set_alpha(0.65)
    cap = int(trunc_df[len_col].max())
    axes[2].axhline(cap, color="k", linestyle=":", alpha=0.5,
                    label=f"observed max = {cap}")
    axes[2].set(ylabel=len_col,
                title="Length by failure mode\n"
                      "(truncated should pile up at the cap)")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    return fig, axes, {"by_step": by_step, "trunc_df": trunc_df,
                       "comp_frac": comp_frac, "len_col": len_col}


# ────────────────────────────────────────────────────────────────────────────
# context
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class VizCtx:
    """Pre-computed inputs shared across the three visualizers."""
    row:               Dict[str, Any]
    scored:            List[Dict[str, Any]]
    refs:              List[str]
    own_idx:           int
    own_label:         str
    cot_used:          str
    offsets:           List[Tuple[int, int]]
    para_tok:          List[int]
    para_tok_starts:   List[int]
    para_tok_ends:     List[int]
    para_char_ranges:  List[Tuple[int, int]]
    cums:              List[np.ndarray]
    T:                 int
    own_logps:         np.ndarray
    delta_R:           List[float]
    final_R:           np.ndarray
    order_final:       np.ndarray


def prepare(row, scored, refs, own_idx, tokenizer) -> VizCtx:
    """Pre-compute paragraph boundaries (token & char), cum R(t), and ΔR_own.

    Mirrors ``compute_vt_batched``'s answer-marker stripping so token indices
    align with ``scored[i]["logps"]``.
    """
    cot = row["completion"]
    i   = cot.find("####")
    cot_used = cot[:i].rstrip() if i >= 0 else cot

    enc     = tokenizer(cot_used, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    splits = list(re.finditer(r"\n\s*\n", cot_used))
    para_tok = [next((k for k, (a, _) in enumerate(offsets) if a >= m.end()),
                     len(offsets)) for m in splits]
    para_char_ranges, prev = [], 0
    for m in splits:
        para_char_ranges.append((prev, m.start())); prev = m.end()
    para_char_ranges.append((prev, len(cot_used)))

    cums = [s["logps"] - s["logps"][0] for s in scored]
    T    = max(len(c) for c in cums) - 1
    para_tok_starts = [0] + para_tok
    para_tok_ends   = para_tok + [T]

    own_logps = scored[own_idx]["logps"]
    delta_R   = [float(own_logps[min(e, len(own_logps) - 1)] - own_logps[s])
                 for s, e in zip(para_tok_starts, para_tok_ends)]
    final_R     = np.array([s["R_T"] for s in scored])
    order_final = np.argsort(final_R)[::-1]
    own_label   = (f"own ({'correct' if row['correct'] else 'wrong'}): "
                   f"{row['expr']}")

    return VizCtx(
        row=row, scored=scored, refs=refs, own_idx=own_idx, own_label=own_label,
        cot_used=cot_used, offsets=offsets, para_tok=para_tok,
        para_tok_starts=para_tok_starts, para_tok_ends=para_tok_ends,
        para_char_ranges=para_char_ranges, cums=cums, T=T,
        own_logps=own_logps, delta_R=delta_R,
        final_R=final_R, order_final=order_final,
    )


# ────────────────────────────────────────────────────────────────────────────
# (1) static two-panel plot
# ────────────────────────────────────────────────────────────────────────────
def plot_R_t_static(ctx: VizCtx, *, figsize=(12, 4)):
    """Left: cum R(t) per ref (own highlighted). Right: final R_T bars sorted."""
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 2]})

    for i, c in enumerate(ctx.cums):
        is_own = (i == ctx.own_idx)
        ax0.plot(c, color="C3" if is_own else "0.6",
                 lw=2 if is_own else 1, alpha=1.0 if is_own else 0.5,
                 label=ctx.own_label if is_own else None)
    for k, t in enumerate(ctx.para_tok):
        ax0.axvline(t, color="C0", ls=":", lw=0.8, alpha=0.6,
                    label="paragraph end" if k == 0 else None)
    ax0.axhline(0, color="k", lw=0.5)
    ax0.set_xlabel("CoT prefix length t"); ax0.set_ylabel("cum R(t)")
    ax0.set_title(f"R(t) over {len(ctx.scored)} refs  ·  "
                  f"step={ctx.row['global_step']}  idx={ctx.row['idx']}")
    ax0.legend(loc="best", fontsize=8)

    order  = ctx.order_final
    colors = ["C3" if i == ctx.own_idx else "0.6" for i in order]
    ax1.barh(range(len(order)), ctx.final_R[order], color=colors)
    ax1.invert_yaxis()
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([ctx.refs[i] + (" ←own" if i == ctx.own_idx else "")
                         for i in order], fontsize=7)
    ax1.set_xlabel("R_T"); ax1.set_title("final R_T, sorted")
    plt.tight_layout()

    rank = int((ctx.final_R > ctx.final_R[ctx.own_idx]).sum())
    print(f"own answer rank: {rank}/{len(ctx.final_R) - 1}   "
          f"R_T(own)={ctx.final_R[ctx.own_idx]:+.3f}   "
          f"R_T(best)={ctx.final_R.max():+.3f}   "
          f"gap={ctx.final_R.max() - ctx.final_R[ctx.own_idx]:+.3f}   "
          f"·  {len(ctx.para_tok)} paragraphs")
    return fig


# ────────────────────────────────────────────────────────────────────────────
# (2) terminal print
# ────────────────────────────────────────────────────────────────────────────
_ANSI = {"green": "\033[32m", "red": "\033[31m",
         "dim":   "\033[2m",  "reset": "\033[0m"}


def print_paragraphs(ctx: VizCtx) -> None:
    """One block per paragraph: colored header line + indented body."""
    paras  = re.split(r"\n\s*\n", ctx.cot_used)
    bounds = [0] + ctx.para_tok + [len(ctx.own_logps) - 1]
    for k, text in enumerate(paras):
        t0, t1 = bounds[k], bounds[k + 1]
        dR     = float(ctx.own_logps[t1] - ctx.own_logps[t0])
        color  = (_ANSI["green"] if dR > 0.5 else
                  _ANSI["red"]   if dR < -0.5 else _ANSI["dim"])
        head   = f"[¶{k}] tok {t0:>3}→{t1:<3}  ΔR_own={dR:+.2f}"
        print(f"{color}{head}{_ANSI['reset']}")
        print(textwrap.indent(text.strip(), "    "))
        print()


# ────────────────────────────────────────────────────────────────────────────
# (3) streaming animation
# ────────────────────────────────────────────────────────────────────────────
def make_animation(
    ctx: VizCtx,
    *,
    save_path: Optional[str] = None,
    fps: int = 10,
    max_frames: int = 200,
    interval_ms: int = 60,
):
    """Streaming animation: live curves + live bars + paragraph CoT reveal.

    Saves a GIF if ``save_path`` is provided. Returns the
    :class:`matplotlib.animation.FuncAnimation` (figure is closed so it
    won't auto-display in notebooks; use ``HTML(anim.to_jshtml())`` to embed).
    """
    cums, T = ctx.cums, ctx.T
    ymin = min(c.min() for c in cums); ymax = max(c.max() for c in cums)
    char_end = [0] + [b for _, b in ctx.offsets]

    stride = max(1, T // max_frames)
    frames = list(range(0, T + 1, stride)) + [T]

    fig = plt.figure(figsize=(13, 9), facecolor="white")
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 5], width_ratios=[3, 2],
                           hspace=0.18, wspace=0.18)
    ax     = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    txt_ax = fig.add_subplot(gs[1, :])

    # -- top-left: streaming curves
    lines = []
    for i, c in enumerate(cums):
        is_own = (i == ctx.own_idx)
        (ln,) = ax.plot([], [], color="C3" if is_own else "0.65",
                        lw=2 if is_own else 1, alpha=1.0 if is_own else 0.5,
                        label=ctx.own_label if is_own else None)
        lines.append(ln)
    for t in ctx.para_tok:
        ax.axvline(t, color="0.55", ls=":", lw=0.8, alpha=0.7,
                   label="paragraph end" if t == ctx.para_tok[0] else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, T); ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_xlabel("CoT prefix length t"); ax.set_ylabel("cum R(t)")
    ax.set_title(f"R(t) over {len(ctx.scored)} refs · "
                 f"step={ctx.row['global_step']} idx={ctx.row['idx']}",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    cursor = ax.axvline(0, color="k", lw=1, alpha=0.4)

    # -- top-right: live bars in fixed final-R_T order
    rank_of = {int(i): r for r, i in enumerate(ctx.order_final)}
    bars = ax_bar.barh(
        range(len(ctx.scored)), [0] * len(ctx.scored),
        color=["C3" if i == ctx.own_idx else "0.7" for i in ctx.order_final])
    ax_bar.invert_yaxis()
    ax_bar.set_yticks(range(len(ctx.scored)))
    ax_bar.set_yticklabels(
        [ctx.refs[i] + (" ←own" if i == ctx.own_idx else "")
         for i in ctx.order_final], fontsize=6)
    ax_bar.set_xlim(min(ymin - 1, -1), ymax + 1)
    ax_bar.axvline(0, color="k", lw=0.5)
    ax_bar.set_xlabel("R(t) — live")
    ax_bar.set_title(
        f"live R(t) for all {len(ctx.refs)} valid 24-solutions"
        f"\n(rows fixed in final-R_T order)", fontsize=10)

    # -- bottom: paragraph reveal
    WRAP, HEAD_COLOR, BODY_COLOR, BG = 92, "#1a7f37", "#000000", "#ffffff"

    def _render_paragraphs(t):
        txt_ax.clear(); txt_ax.set_facecolor(BG); txt_ax.axis("off")
        txt_ax.set_xlim(0, 1); txt_ax.set_ylim(0, 1)
        y, dy = 0.97, 0.038
        k_now    = max(0, bisect.bisect_right(ctx.para_tok_starts, t) - 1)
        char_now = char_end[min(t, len(char_end) - 1)]
        for k in range(k_now + 1):
            a, b = ctx.para_char_ranges[k]
            is_partial = (k == k_now and char_now < b)
            body = ctx.cot_used[a:char_now] if is_partial else ctx.cot_used[a:b]
            body = body.strip()
            if not body and not is_partial:
                continue
            header = (f"[¶{k}] tok {ctx.para_tok_starts[k]:>3}→"
                      f"{ctx.para_tok_ends[k]:<3}    "
                      f"ΔR_own={ctx.delta_R[k]:+.2f}")
            txt_ax.text(0.01, y, header, color=HEAD_COLOR, family="monospace",
                        fontsize=11, fontweight="bold",
                        transform=txt_ax.transAxes, va="top")
            y -= dy
            wrapped = textwrap.wrap(body, width=WRAP) or [""]
            if is_partial and wrapped:
                wrapped[-1] = wrapped[-1] + "▌"
            for line in wrapped:
                txt_ax.text(0.03, y, line, color=BODY_COLOR, family="monospace",
                            fontsize=10, transform=txt_ax.transAxes, va="top")
                y -= dy
                if y < 0.02:
                    txt_ax.text(0.03, y, "…", color=BODY_COLOR,
                                family="monospace", fontsize=10,
                                transform=txt_ax.transAxes, va="top")
                    return
            y -= 0.012

    def _update(t):
        vals = np.empty(len(cums))
        for i, (ln, c) in enumerate(zip(lines, cums)):
            tt = min(t, len(c) - 1)
            ln.set_data(range(tt + 1), c[: tt + 1])
            vals[i] = c[tt]
        cursor.set_xdata([t, t])
        for i, v in enumerate(vals):
            bars[rank_of[i]].set_width(v)
        _render_paragraphs(t)
        return []

    anim = FuncAnimation(fig, _update, frames=frames,
                         interval=interval_ms, blit=False)
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
    return anim


# ────────────────────────────────────────────────────────────────────────────
# (3b) paired success/failure streaming animation
# ────────────────────────────────────────────────────────────────────────────
def make_pair_animation(
    ctx_correct: VizCtx,
    ctx_incorrect: VizCtx,
    *,
    save_path: Optional[str] = None,
    fps: int = 10,
    max_frames: int = 200,
    interval_ms: int = 60,
    column_titles: Tuple[str, str] = ("✓ correct rollout", "✗ incorrect rollout"),
    figsize: Tuple[float, float] = (20, 9),
):
    """Co-presented streaming animation for one success and one failure rollout.

    Two-column layout, each column mirroring :func:`make_animation`:
    streaming cumR(t) curves + live R(t) bars on top, paragraph reveal below.
    The two rollouts can have different CoT lengths; we drive both columns
    with a shared normalised progress φ ∈ [0, 1] and map φ → t per side.
    """
    ctxs = (ctx_correct, ctx_incorrect)
    Ts   = tuple(c.T for c in ctxs)

    # Shared normalised frame schedule.
    n_frames = min(max_frames, max(Ts) + 1)
    phis     = np.linspace(0.0, 1.0, n_frames)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = fig.add_gridspec(
        2, 4, height_ratios=[3, 5], width_ratios=[3, 2, 3, 2],
        hspace=0.22, wspace=0.22,
    )
    axes_curve = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    axes_bar   = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]
    axes_txt   = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])]

    # Per-column static setup (curves, bars, paragraph dividers).
    per_col: List[Dict[str, Any]] = []
    for col, (ctx, ax, ax_bar, ax_txt, title) in enumerate(zip(
        ctxs, axes_curve, axes_bar, axes_txt, column_titles
    )):
        cums, T = ctx.cums, ctx.T
        ymin = min(c.min() for c in cums); ymax = max(c.max() for c in cums)
        char_end = [0] + [b for _, b in ctx.offsets]

        lines = []
        for i, c in enumerate(cums):
            is_own = (i == ctx.own_idx)
            (ln,) = ax.plot([], [], color="C3" if is_own else "0.65",
                            lw=2 if is_own else 1,
                            alpha=1.0 if is_own else 0.5,
                            label=ctx.own_label if is_own else None)
            lines.append(ln)
        for t in ctx.para_tok:
            ax.axvline(t, color="0.55", ls=":", lw=0.8, alpha=0.7,
                       label="paragraph end" if t == ctx.para_tok[0] else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(0, T); ax.set_ylim(ymin - 1, ymax + 1)
        ax.set_xlabel("CoT prefix length t"); ax.set_ylabel("cum R(t)")
        ax.set_title(f"{title}  ·  step={ctx.row['global_step']} "
                     f"idx={ctx.row['idx']}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        cursor = ax.axvline(0, color="k", lw=1, alpha=0.4)

        rank_of = {int(i): r for r, i in enumerate(ctx.order_final)}
        bars = ax_bar.barh(
            range(len(ctx.scored)), [0] * len(ctx.scored),
            color=["C3" if i == ctx.own_idx else "0.7" for i in ctx.order_final])
        ax_bar.invert_yaxis()
        ax_bar.set_yticks(range(len(ctx.scored)))
        ax_bar.set_yticklabels(
            [ctx.refs[i] + (" ←own" if i == ctx.own_idx else "")
             for i in ctx.order_final], fontsize=6)
        ax_bar.set_xlim(min(ymin - 1, -1), ymax + 1)
        ax_bar.axvline(0, color="k", lw=0.5)
        ax_bar.set_xlabel("R(t) — live")
        ax_bar.set_title(f"live R(t) · {len(ctx.refs)} refs (rows fixed in "
                         f"final-R_T order)", fontsize=9)

        per_col.append({
            "ctx": ctx, "lines": lines, "cursor": cursor, "bars": bars,
            "rank_of": rank_of, "ax_txt": ax_txt, "char_end": char_end,
            "T": T,
        })

    WRAP, HEAD_COLOR, BODY_COLOR, BG = 70, "#1a7f37", "#000000", "#ffffff"

    def _render_paragraphs(slot):
        ctx     = slot["ctx"]
        ax_txt  = slot["ax_txt"]
        char_end = slot["char_end"]
        t        = slot["t"]

        ax_txt.clear(); ax_txt.set_facecolor(BG); ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
        y, dy = 0.97, 0.038
        k_now    = max(0, bisect.bisect_right(ctx.para_tok_starts, t) - 1)
        char_now = char_end[min(t, len(char_end) - 1)]
        for k in range(k_now + 1):
            a, b = ctx.para_char_ranges[k]
            is_partial = (k == k_now and char_now < b)
            body = ctx.cot_used[a:char_now] if is_partial else ctx.cot_used[a:b]
            body = body.strip()
            if not body and not is_partial:
                continue
            header = (f"[¶{k}] tok {ctx.para_tok_starts[k]:>3}→"
                      f"{ctx.para_tok_ends[k]:<3}    "
                      f"ΔR_own={ctx.delta_R[k]:+.2f}")
            ax_txt.text(0.01, y, header, color=HEAD_COLOR, family="monospace",
                        fontsize=10, fontweight="bold",
                        transform=ax_txt.transAxes, va="top")
            y -= dy
            wrapped = textwrap.wrap(body, width=WRAP) or [""]
            if is_partial and wrapped:
                wrapped[-1] = wrapped[-1] + "▌"
            for line in wrapped:
                ax_txt.text(0.03, y, line, color=BODY_COLOR, family="monospace",
                            fontsize=9, transform=ax_txt.transAxes, va="top")
                y -= dy
                if y < 0.02:
                    ax_txt.text(0.03, y, "…", color=BODY_COLOR,
                                family="monospace", fontsize=9,
                                transform=ax_txt.transAxes, va="top")
                    return
            y -= 0.012

    def _update(phi):
        for slot in per_col:
            ctx, T = slot["ctx"], slot["T"]
            t = int(round(phi * T))
            slot["t"] = t
            cums = ctx.cums
            vals = np.empty(len(cums))
            for i, (ln, c) in enumerate(zip(slot["lines"], cums)):
                tt = min(t, len(c) - 1)
                ln.set_data(range(tt + 1), c[: tt + 1])
                vals[i] = c[tt]
            slot["cursor"].set_xdata([t, t])
            for i, v in enumerate(vals):
                slot["bars"][slot["rank_of"][i]].set_width(v)
            _render_paragraphs(slot)
        return []

    anim = FuncAnimation(fig, _update, frames=phis,
                         interval=interval_ms, blit=False)
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
    return anim


# ────────────────────────────────────────────────────────────────────────────
# (3c) vt-sidecar paired animation: no scorer model required
# ────────────────────────────────────────────────────────────────────────────
def _trim_for_anim(completion: str) -> str:
    """Same answer-marker stripping rule as ``prepare``."""
    i = completion.find("####")
    return completion[:i].rstrip() if i >= 0 else completion


def make_pair_animation_from_vt(
    vt_row_correct: Dict[str, Any],
    completion_correct: str,
    vt_row_incorrect: Dict[str, Any],
    completion_incorrect: str,
    *,
    save_path: Optional[str] = None,
    fps: int = 10,
    max_frames: int = 200,
    interval_ms: int = 60,
    column_titles: Tuple[str, str] = ("✓ correct rollout", "✗ incorrect rollout"),
    figsize: Tuple[float, float] = (20, 9),
    short_ref_chars: int = 22,
):
    """Side-by-side success/failure animation driven by the vt sidecar.

    Consumes two rows from ``eval_rollout_vt.jsonl`` (per-ref
    ``cumR_resampled_per_ref`` arrays, on a fixed 100-pt grid in [0, 1])
    plus the corresponding ``completion`` strings from
    ``eval_rollout.jsonl``. No scorer model required.

    Each column shows
      • streaming cumR(φ) curves for every reference (own ref highlighted),
      • live R(φ) bars in fixed final-R_T order,
      • paragraph-by-paragraph reveal of the rollout text.

    The two rollouts share a normalised progress φ ∈ [0, 1] so they stay
    co-aligned even when their CoT lengths differ.
    """
    cols: List[Dict[str, Any]] = []
    for vt_row, completion, title in (
        (vt_row_correct, completion_correct, column_titles[0]),
        (vt_row_incorrect, completion_incorrect, column_titles[1]),
    ):
        cums_raw = vt_row.get("cumR_resampled_per_ref") or []
        R_T = vt_row.get("R_T_per_ref") or []
        cums = [np.asarray(c, dtype=float) for c in cums_raw if c]
        if not cums:
            raise ValueError(
                f"vt row idx={vt_row.get('idx')} has no usable "
                f"cumR_resampled_per_ref entries"
            )
        keep = [i for i, c in enumerate(cums_raw) if c]
        refs = [vt_row["refs"][i] for i in keep]
        R_T_arr = np.asarray([R_T[i] for i in keep], dtype=float)

        own_idx_global = vt_row.get("own_ref_idx", -1)
        own_idx = (keep.index(own_idx_global)
                   if own_idx_global is not None and own_idx_global in keep
                   else int(np.argmax(R_T_arr)))
        order_final = np.argsort(R_T_arr)[::-1]

        cot = _trim_for_anim(completion)
        splits = list(re.finditer(r"\n\s*\n", cot))
        para_char_ranges, prev = [], 0
        for m in splits:
            para_char_ranges.append((prev, m.start())); prev = m.end()
        para_char_ranges.append((prev, len(cot)))
        # paragraph end as a fraction of total chars (drives both the
        # vertical paragraph-divider line and which paragraphs to reveal)
        L = max(len(cot), 1)
        para_phi_ends = [b / L for _, b in para_char_ranges]
        # ΔR_own per paragraph, computed on the resampled grid via
        # nearest-neighbour lookup at the paragraph boundary fractions.
        own_cum = cums[own_idx]
        n_grid  = len(own_cum)
        def _cum_at(phi):
            j = min(int(round(phi * (n_grid - 1))), n_grid - 1)
            return float(own_cum[j])
        para_phi_starts = [0.0] + para_phi_ends[:-1]
        delta_R = [_cum_at(e) - _cum_at(s)
                   for s, e in zip(para_phi_starts, para_phi_ends)]

        own_label = (
            f"own ({'correct' if vt_row['correct'] else 'wrong'}): "
            f"{vt_row['expr']}"
        )

        cols.append({
            "vt_row": vt_row, "title": title,
            "cums": cums, "R_T": R_T_arr, "refs": refs,
            "own_idx": own_idx, "order_final": order_final,
            "own_label": own_label,
            "cot": cot, "para_char_ranges": para_char_ranges,
            "para_phi_ends": para_phi_ends,
            "para_phi_starts": para_phi_starts,
            "delta_R": delta_R,
        })

    n_frames = min(max_frames, 200)
    phis     = np.linspace(0.0, 1.0, n_frames)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = fig.add_gridspec(
        2, 4, height_ratios=[3, 5], width_ratios=[3, 2, 3, 2],
        hspace=0.22, wspace=0.22,
    )

    for col_i, c in enumerate(cols):
        ax     = fig.add_subplot(gs[0, 2 * col_i])
        ax_bar = fig.add_subplot(gs[0, 2 * col_i + 1])
        ax_txt = fig.add_subplot(gs[1, 2 * col_i:2 * col_i + 2])

        cums = c["cums"]
        ymin = float(min(cu.min() for cu in cums))
        ymax = float(max(cu.max() for cu in cums))

        lines = []
        n_grid = len(cums[0])
        x_grid = np.linspace(0.0, 1.0, n_grid)
        for i, cu in enumerate(cums):
            is_own = (i == c["own_idx"])
            (ln,) = ax.plot([], [], color="C3" if is_own else "0.65",
                            lw=2 if is_own else 1,
                            alpha=1.0 if is_own else 0.5,
                            label=c["own_label"] if is_own else None)
            lines.append(ln)
        for k, phi in enumerate(c["para_phi_ends"][:-1]):
            ax.axvline(phi, color="0.55", ls=":", lw=0.8, alpha=0.7,
                       label="paragraph end" if k == 0 else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(ymin - 1, ymax + 1)
        ax.set_xlabel("normalised CoT position (φ = t / T)")
        ax.set_ylabel("cum R(φ)")
        ax.set_title(f"{c['title']}  ·  step={c['vt_row']['global_step']} "
                     f"idx={c['vt_row']['idx']}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        cursor = ax.axvline(0, color="k", lw=1, alpha=0.4)

        rank_of = {int(i): r for r, i in enumerate(c["order_final"])}
        bars = ax_bar.barh(
            range(len(cums)), [0] * len(cums),
            color=["C3" if i == c["own_idx"] else "0.7"
                   for i in c["order_final"]])
        ax_bar.invert_yaxis()
        ax_bar.set_yticks(range(len(cums)))
        ax_bar.set_yticklabels(
            [(c["refs"][i][:short_ref_chars]
              + ("…" if len(c["refs"][i]) > short_ref_chars else ""))
             + (" ←own" if i == c["own_idx"] else "")
             for i in c["order_final"]], fontsize=6)
        ax_bar.set_xlim(min(ymin - 1, -1), ymax + 1)
        ax_bar.axvline(0, color="k", lw=0.5)
        ax_bar.set_xlabel("R(φ) — live")
        ax_bar.set_title(f"live R(φ) · {len(c['refs'])} refs (rows fixed in "
                         f"final-R_T order)", fontsize=9)

        c.update({
            "ax": ax, "ax_bar": ax_bar, "ax_txt": ax_txt,
            "lines": lines, "bars": bars, "rank_of": rank_of,
            "cursor": cursor, "x_grid": x_grid, "n_grid": n_grid,
        })

    WRAP, HEAD_COLOR, BODY_COLOR, BG = 70, "#1a7f37", "#000000", "#ffffff"

    def _render_paragraphs(c, phi):
        ax_txt = c["ax_txt"]
        ax_txt.clear(); ax_txt.set_facecolor(BG); ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
        y, dy = 0.97, 0.038
        char_now = int(round(phi * len(c["cot"])))
        # paragraph index containing char_now
        k_now = 0
        for k, (a, b) in enumerate(c["para_char_ranges"]):
            if a <= char_now <= b:
                k_now = k; break
            if char_now > b:
                k_now = k
        for k in range(k_now + 1):
            a, b = c["para_char_ranges"][k]
            is_partial = (k == k_now and char_now < b)
            body = c["cot"][a:char_now] if is_partial else c["cot"][a:b]
            body = body.strip()
            if not body and not is_partial:
                continue
            header = (f"[¶{k}]  φ {c['para_phi_starts'][k]:.2f}→"
                      f"{c['para_phi_ends'][k]:.2f}    "
                      f"ΔR_own={c['delta_R'][k]:+.2f}")
            ax_txt.text(0.01, y, header, color=HEAD_COLOR, family="monospace",
                        fontsize=10, fontweight="bold",
                        transform=ax_txt.transAxes, va="top")
            y -= dy
            wrapped = textwrap.wrap(body, width=WRAP) or [""]
            if is_partial and wrapped:
                wrapped[-1] = wrapped[-1] + "▌"
            for line in wrapped:
                ax_txt.text(0.03, y, line, color=BODY_COLOR, family="monospace",
                            fontsize=9, transform=ax_txt.transAxes, va="top")
                y -= dy
                if y < 0.02:
                    ax_txt.text(0.03, y, "…", color=BODY_COLOR,
                                family="monospace", fontsize=9,
                                transform=ax_txt.transAxes, va="top")
                    return
            y -= 0.012

    def _update(phi):
        for c in cols:
            n_grid = c["n_grid"]
            j = min(int(round(phi * (n_grid - 1))), n_grid - 1)
            x_seg = c["x_grid"][: j + 1]
            vals = np.empty(len(c["cums"]))
            for i, (ln, cu) in enumerate(zip(c["lines"], c["cums"])):
                ln.set_data(x_seg, cu[: j + 1])
                vals[i] = cu[j]
            c["cursor"].set_xdata([phi, phi])
            for i, v in enumerate(vals):
                c["bars"][c["rank_of"][i]].set_width(v)
            _render_paragraphs(c, phi)
        return []

    anim = FuncAnimation(fig, _update, frames=phis,
                         interval=interval_ms, blit=False)
    plt.close(fig)
    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
    return anim


# ────────────────────────────────────────────────────────────────────────────
# Life-cycle river: per-puzzle phylogeny of unique expressions across cycles
# ────────────────────────────────────────────────────────────────────────────
def _ast_canon(expr: str) -> str:
    """AST canonical dump of a Python arithmetic expression, '' on failure."""
    try:
        return ast.dump(ast.parse(expr, mode="eval").body)
    except Exception:
        return ""


def _smooth_band(xs, ys_top, ys_bot, ax, facecolor, alpha=0.85, zorder=2):
    """Filled band whose top/bottom polylines are smoothed with cubic bezier."""
    n = len(xs)
    if n < 2:
        return
    verts, codes = [], []
    verts.append((xs[0], ys_top[0])); codes.append(MplPath.MOVETO)
    for i in range(n - 1):
        x0, x1 = xs[i], xs[i + 1]; xm = (x0 + x1) / 2.0
        verts += [(xm, ys_top[i]), (xm, ys_top[i + 1]), (x1, ys_top[i + 1])]
        codes += [MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    verts.append((xs[-1], ys_bot[-1])); codes.append(MplPath.LINETO)
    for i in range(n - 1, 0, -1):
        x0, x1 = xs[i], xs[i - 1]; xm = (x0 + x1) / 2.0
        verts += [(xm, ys_bot[i]), (xm, ys_bot[i - 1]), (x1, ys_bot[i - 1])]
        codes += [MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    verts.append((xs[0], ys_top[0])); codes.append(MplPath.CLOSEPOLY)
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=facecolor,
                           edgecolor="white", linewidth=0.6,
                           alpha=alpha, zorder=zorder))


def plot_lifecycle_river(
    eval_df,
    *,
    idx: int = 0,
    numbers: Optional[Tuple[int, ...]] = None,
    max_expr_chars: int = 28,
    lane_half_width: float = 0.42,
    trunk_half_width: float = 0.06,
    legend_loc: str = "upper right",
    figsize_width: float = 13.0,
    verbose: bool = True,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """Stream-graph life-cycle "river" of unique expressions for one puzzle.

    Each unique AST-canonicalized ``expr`` becomes one stream whose:

      - **thickness** at step *s* equals the number of rollouts (out of *G*)
        that emitted it that cycle,
      - **colour** is green (verifier accepted) or red (verifier rejected),
      - phantom-births from the previous cycle's trunk and phantom-deaths
        on the lane's own y create a gentle taper at endpoints,
      - **silent cycles pinch to zero thickness**, so a lane that vanishes
        for a few cycles and reappears looks like a river going underground
        and resurfacing rather than a fat continuous flow.

    The ``"truncated / no expr"`` bucket is shown as a hatched grey band
    below the tree and never participates in the trunk computation.

    Truth source for green/red is ``r.correct`` (the trainer's verifier).
    A ``★`` prefix on a y-tick label additionally flags membership in
    ``enumerate_solutions(numbers)`` — useful as a sanity check on the
    enumeration but not the criterion for "correct".

    Parameters
    ----------
    eval_df
        DataFrame of eval-time rollouts with at minimum columns
        ``key`` (sorted-numbers tuple), ``numbers``, ``global_step``,
        ``expr``, ``correct``.
    idx
        Index into the de-duplicated puzzle list ``sorted(eval_df.key.unique())``.
        Ignored if ``numbers`` is given.
    numbers
        Optional puzzle selector, e.g. ``(1, 1, 3, 8)``. Overrides ``idx``.
    max_expr_chars
        Truncate y-tick expression labels to this many chars.
    lane_half_width, trunk_half_width
        Visual tuning. ``lane_half_width`` is the maximum half-thickness of
        a stream when count == G; ``trunk_half_width`` is the seed thickness
        used at phantom births to keep them visible.
    legend_loc
        Standard matplotlib legend ``loc``. Default ``"upper right"``.
    figsize_width
        Figure width in inches; height is computed from the lane count.
    verbose
        If True, prints lane summaries to stdout.

    Returns
    -------
    (fig, ax, info)
        ``info`` carries the per-lane bookkeeping (counts, birth steps,
        verifier verdicts, ★ membership) so callers can render their own
        side panels without re-walking the dataframe.
    """
    # local import to avoid a hard dep on Game-of-24 utilities for the
    # other visualizers in this module.
    from src.game24utils import enumerate_solutions

    # --- resolve puzzle ---------------------------------------------------
    unique_keys = sorted(eval_df.key.unique())
    if numbers is not None:
        key = tuple(sorted(numbers))
        if key not in unique_keys:
            raise ValueError(
                f"numbers={numbers} (key={key}) not present in eval_df"
            )
    else:
        if not (0 <= idx < len(unique_keys)):
            raise ValueError(
                f"idx={idx} out of range; eval_df has {len(unique_keys)} "
                f"unique puzzles (0..{len(unique_keys)-1})"
            )
        key = unique_keys[idx]

    rows_df = eval_df[eval_df.key == key].copy()
    nums = list(key)
    sols = list(enumerate_solutions(tuple(key)))
    canon_sols = set(_ast_canon(s) for s in sols)
    n_sol = len(sols)
    steps = sorted(rows_df.global_step.unique())
    G_typical = int(rows_df.groupby("global_step").size().mode().iloc[0])
    step_to_x = {s: i for i, s in enumerate(steps)}

    distinct_nums = {tuple(n) for n in rows_df.numbers.apply(tuple)}
    assert len(distinct_nums) == 1, (
        f"rows_df spans multiple number tuples {distinct_nums}; "
        "key-based filtering is broken."
    )

    if verbose:
        print(
            f"Tracing puzzle  numbers={nums}  "
            f"(puzzle #{unique_keys.index(key)} of {len(unique_keys)})"
        )
        print(
            f"|S(p)| = {n_sol}, G = {G_typical}, {len(steps)} eval cycles, "
            f"{len(rows_df)} rollouts total."
        )

    # --- collect per-step counts -----------------------------------------
    NOEXPR = "__noexpr__"
    per_step: Dict[int, Dict[str, Dict[str, Any]]] = {gs: {} for gs in steps}
    first_seen_step: Dict[str, int] = {}
    last_seen_step: Dict[str, int] = {}
    display_by_canon: Dict[str, str] = {}
    correct_by_canon: Dict[str, bool] = {}
    in_S_by_canon: Dict[str, bool] = {}
    for _, r in rows_df.iterrows():
        expr = r.expr if isinstance(r.expr, str) else ""
        verifier_ok = bool(r.correct)
        if not expr:
            c = NOEXPR; disp = "truncated / no expr"; ok = False; in_S = False
        else:
            cn = _ast_canon(expr)
            if not cn:
                c = f"__nocanon__::{expr}"
                disp = expr.strip()[:80]
                ok, in_S = verifier_ok, False
            else:
                c, disp, ok, in_S = cn, expr.strip(), verifier_ok, cn in canon_sols
        bucket = per_step[r.global_step].setdefault(
            c, {"display": disp, "correct": ok, "count": 0, "in_S": in_S}
        )
        bucket["count"] += 1
        display_by_canon.setdefault(c, disp)
        correct_by_canon.setdefault(c, ok)
        in_S_by_canon.setdefault(c, in_S)
        first_seen_step.setdefault(c, int(r.global_step))
        last_seen_step[c] = int(r.global_step)

    # --- order lanes ------------------------------------------------------
    expr_lanes = [c for c in display_by_canon if c != NOEXPR]
    expr_lanes.sort(
        key=lambda c: (
            first_seen_step[c],
            0 if correct_by_canon[c] else 1,
            display_by_canon[c],
        )
    )
    lane_y = {c: i + 1 for i, c in enumerate(expr_lanes)}
    n_lanes = len(expr_lanes)
    has_noexpr = NOEXPR in display_by_canon
    y_noexpr = n_lanes + 1.0

    n_ok = sum(1 for c in expr_lanes if correct_by_canon[c])
    n_bad = n_lanes - n_ok
    n_inS = sum(
        1 for c in expr_lanes if correct_by_canon[c] and in_S_by_canon[c]
    )
    if verbose:
        print(
            f"Distinct expressions (excl. no-expr): {n_lanes}  "
            f"({n_ok} verifier-correct, of which {n_inS} also ∈ "
            f"enumerate_solutions; {n_bad} verifier-wrong)"
        )

    # --- trunk: median y of expression lanes alive at each step ----------
    trunk_y_at: Dict[int, float] = {}
    for gs in steps:
        alive = [lane_y[c] for c in per_step[gs] if c in lane_y]
        trunk_y_at[gs] = float(np.median(alive)) if alive else (n_lanes + 1) / 2.0

    # --- build stream geometry per lane (with silent-cycle pinching) -----
    streams: List[Dict[str, Any]] = []
    for c in expr_lanes:
        y_lane = lane_y[c]
        ok = correct_by_canon[c]
        color = "#2a9d8f" if ok else "#c0392b"
        appearances = [
            (gs, per_step[gs][c]["count"]) for gs in steps if c in per_step[gs]
        ]
        if not appearances:
            continue
        xs, ys_t, ys_b = [], [], []
        # phantom birth
        birth_step = appearances[0][0]
        bi = steps.index(birth_step)
        if bi > 0:
            prev = steps[bi - 1]
            xs.append(step_to_x[prev])
            ys_t.append(trunk_y_at[prev] - trunk_half_width)
            ys_b.append(trunk_y_at[prev] + trunk_half_width)
        else:
            xs.append(step_to_x[birth_step] - 0.6)
            ys_t.append(y_lane - trunk_half_width)
            ys_b.append(y_lane + trunk_half_width)
        # main spine — pinch to zero thickness on every silent intermediate step
        prev_idx: Optional[int] = None
        for gs, cnt in appearances:
            cur_idx = steps.index(gs)
            if prev_idx is not None and cur_idx - prev_idx > 1:
                for k in range(prev_idx + 1, cur_idx):
                    xs.append(step_to_x[steps[k]])
                    ys_t.append(y_lane); ys_b.append(y_lane)
            half = lane_half_width * (cnt / G_typical)
            xs.append(step_to_x[gs])
            ys_t.append(y_lane - half); ys_b.append(y_lane + half)
            prev_idx = cur_idx
        # phantom death: zero on lane's own y at the next cycle
        death_step = appearances[-1][0]
        di = steps.index(death_step)
        if di < len(steps) - 1:
            nxt = steps[di + 1]
            xs.append(step_to_x[nxt])
            ys_t.append(y_lane); ys_b.append(y_lane)
        streams.append({
            "xs": xs, "ys_t": ys_t, "ys_b": ys_b,
            "color": color, "correct": ok, "lane": c,
        })

    # --- figure layout ----------------------------------------------------
    height = max(3.2, 0.55 * n_lanes + (0.6 if has_noexpr else 0.0) + 1.6)
    fig = plt.figure(figsize=(figsize_width, height))
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fbfbfa")

    for c, y in lane_y.items():
        ax.plot([-0.5, len(steps) - 0.5], [y, y],
                color="#dddddd", linewidth=0.6, zorder=0)

    for s in sorted(streams,
                    key=lambda d: (d["correct"], -first_seen_step[d["lane"]])):
        _smooth_band(s["xs"], s["ys_t"], s["ys_b"], ax,
                     facecolor=s["color"], alpha=0.82, zorder=2)

    for gs in steps:
        x = step_to_x[gs]
        for c, info in per_step[gs].items():
            if c not in lane_y:
                continue
            y = lane_y[c]
            cnt = info["count"]
            edge = "#2a9d8f" if info["correct"] else "#c0392b"
            ax.scatter(x, y, s=80, marker="o", facecolor="white",
                       edgecolor=edge, linewidth=1.2, zorder=4)
            ax.text(x, y, str(cnt), color=edge, fontsize=8,
                    ha="center", va="center", fontweight="bold", zorder=5)

    if has_noexpr:
        band_half = 0.32
        ax.add_patch(plt.Rectangle(
            (-0.5, y_noexpr - band_half), len(steps), 2 * band_half,
            facecolor="#eeeeee", edgecolor="#999999",
            hatch="///", linewidth=0.6, alpha=0.55, zorder=1,
        ))
        for gs in steps:
            cnt = per_step[gs].get(NOEXPR, {}).get("count", 0)
            if cnt == 0:
                continue
            x = step_to_x[gs]
            ax.scatter(x, y_noexpr, s=80 + 50 * cnt, marker="X",
                       facecolor="#888888", edgecolor="white", linewidth=0.8,
                       zorder=4)
            ax.text(x, y_noexpr, str(cnt), color="white", fontsize=8,
                    ha="center", va="center", fontweight="bold", zorder=5)

    ax.set_xticks([step_to_x[s] for s in steps])
    ax.set_xticklabels(steps)
    ax.set_xlabel("global_step")
    ax.set_xlim(-0.7, len(steps) - 0.3)

    ytick_y, ytick_lbl, ytick_col = [], [], []
    for c in expr_lanes:
        ytick_y.append(lane_y[c])
        s = display_by_canon[c]
        s = s[:max_expr_chars] + ("…" if len(s) > max_expr_chars else "")
        if correct_by_canon[c] and in_S_by_canon[c]:
            s = "★ " + s
        ytick_lbl.append(s)
        ytick_col.append("#2a9d8f" if correct_by_canon[c] else "#c0392b")
    if has_noexpr:
        ytick_y.append(y_noexpr)
        ytick_lbl.append("truncated / no expr")
        ytick_col.append("#888888")
    ax.set_yticks(ytick_y)
    ax.set_yticklabels(ytick_lbl, fontsize=9)
    for tick, col in zip(ax.get_yticklabels(), ytick_col):
        tick.set_color(col)
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_ylim(y_noexpr + 0.8 if has_noexpr else n_lanes + 0.6, 0.2)

    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", color="#cccccc", alpha=0.6, zorder=0)

    legend = [
        Line2D([0], [0], lw=6, color="#2a9d8f", alpha=0.82,
               label="verifier-correct stream"),
        Line2D([0], [0], lw=6, color="#c0392b", alpha=0.82,
               label="verifier-wrong stream"),
        Line2D([0], [0], marker="X", linestyle="", color="#888888",
               markersize=10, label="truncated / no expr"),
        Line2D([0], [0], linestyle="", marker="*", color="#2a9d8f",
               markersize=10, label="★ also ∈ enumerate_solutions"),
    ]
    ax.legend(handles=legend, loc=legend_loc, fontsize=9, frameon=False)

    n_noexpr_total = sum(
        per_step[s].get(NOEXPR, {}).get("count", 0) for s in steps
    )
    title = (
        f"life-cycle of unique answers  ·  numbers = {nums}   "
        f"(num rollout {G_typical}, total solutions {n_sol})\n"
        f"{n_ok} verifier-correct (★ {n_inS} ∈ S)  ·  "
        f"{n_bad} verifier-wrong"
    )
    if has_noexpr:
        title += f"  ·  {n_noexpr_total} no-expr rollouts"
    ax.set_title(title, fontweight="bold", fontsize=11, loc="left")

    fig.tight_layout()

    if verbose:
        print("\nLane summary (birth-ordered):")
        for c in expr_lanes:
            apps = [
                (gs, per_step[gs][c]["count"])
                for gs in steps if c in per_step[gs]
            ]
            tag = "✓" if correct_by_canon[c] else "✗"
            star = " ★" if (correct_by_canon[c] and in_S_by_canon[c]) else "  "
            print(
                f"  {tag}{star} born@step={first_seen_step[c]}  "
                f"({' '.join(f'{gs}:{n}' for gs, n in apps)})   "
                f"{display_by_canon[c]}"
            )
        if has_noexpr:
            apps = [
                (gs, per_step[gs][NOEXPR]["count"])
                for gs in steps if NOEXPR in per_step[gs]
            ]
            print(
                f"\n  ✗  truncated / no expr   "
                f"({' '.join(f'{gs}:{n}' for gs, n in apps)})"
            )

    info: Dict[str, Any] = {
        "key": key,
        "numbers": nums,
        "steps": steps,
        "G": G_typical,
        "n_canonical_solutions": n_sol,
        "expr_lanes": expr_lanes,
        "display_by_canon": display_by_canon,
        "correct_by_canon": correct_by_canon,
        "in_S_by_canon": in_S_by_canon,
        "first_seen_step": first_seen_step,
        "last_seen_step": last_seen_step,
        "per_step": per_step,
        "n_verifier_correct": n_ok,
        "n_verifier_wrong": n_bad,
        "n_in_S": n_inS,
        "n_noexpr_total": n_noexpr_total,
    }
    return fig, ax, info


# ────────────────────────────────────────────────────────────────────────────
# Solution coverage dynamics (3 panels)
# ────────────────────────────────────────────────────────────────────────────
def _per_step_mean_ci(df, value_col: str, n_boot: int = 300, seed: int = 0):
    """Per-step mean + 95% bootstrap CI (over rows in the group)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    out = []
    for s, g in df.groupby("step"):
        v = g[value_col].values.astype(float)
        mu = float(v.mean()) if len(v) else float("nan")
        if len(v) >= 2:
            boots = np.array([rng.choice(v, size=len(v), replace=True).mean()
                              for _ in range(n_boot)])
            lo = float(np.percentile(boots, 2.5))
            hi = float(np.percentile(boots, 97.5))
        else:
            lo = hi = mu
        out.append({"step": s, "val": mu, "lo": lo, "hi": hi,
                    "n_puzzles": len(v)})
    return pd.DataFrame(out).sort_values("step")


def plot_solution_coverage_dynamics(
    eval_df,
    *,
    n_boot: int = 300,
    seed: int = 0,
    figsize: Tuple[float, float] = (18.0, 4.3),
    verbose: bool = True,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """Three-panel solution-coverage dynamics for a Game-of-24 eval run.

    Per puzzle ``p`` with ``G`` rollouts per cycle and ``|S(p)|`` canonical
    solutions:

      * ``found(s, p)``    = solution indices emitted (verifier-correct,
                             AST-canonicalized) at step *s*
      * ``total_found(p)`` = ``|⋃_s found(s, p)|`` (over all eval cycles)
      * ``div(s, p)``      = ``|found(s, p)| / min(total_found(p), G)``  (panel A)
      * ``cum(s, p)``      = ``|⋃_{s' ≤ s} found(s', p)| / |S(p)|``      (panel B)
      * ``novel_frac(s,p)`` = ``|found(s,p) \\ ⋃_{s'<s} found(s',p)| /
                              total_found(p)``                            (panel C)

    Panel C's denominator is the puzzle's total ever-found over the run, so
    each puzzle's column sums to 1 by construction (asserted). Puzzles with
    ``total_found(p) == 0`` are excluded from panel C only.

    Parameters
    ----------
    eval_df
        Long-form eval rollout DataFrame with columns ``key`` (sorted-numbers
        tuple), ``global_step``, ``expr``, ``correct``.

    Returns
    -------
    (fig, axes, info)
        ``info`` carries the per-step summaries (``cov_mean``, ``cum_mean``,
        ``novel_mean``) and per-row tables (``cov_df``, ``cum_df``,
        ``novel_df``) plus puzzle-level scalars.
    """
    import pandas as pd
    from src.game24utils import enumerate_solutions

    # --- enumerate canonical solutions once per puzzle key ---------------
    keys = sorted(eval_df["key"].unique())
    sols_by_key  = {k: list(enumerate_solutions(tuple(k))) for k in keys}
    canon_by_key = {k: [_ast_canon(s) for s in sols_by_key[k]] for k in keys}
    nsol_by_key  = {k: len(sols_by_key[k]) for k in keys}

    # --- per (step, key) coverage ----------------------------------------
    steps = sorted(eval_df.global_step.unique())
    records: List[Dict[str, Any]] = []
    found_by_key_step: Dict[Tuple[Any, Any], set] = {}
    for (gs, key), g in eval_df.groupby(["global_step", "key"]):
        n_sol = nsol_by_key.get(key, 0)
        if n_sol == 0:
            continue
        canon_targets = canon_by_key[key]
        found: set = set()
        for expr, ok in zip(g.expr, g.correct):
            if not ok or not isinstance(expr, str) or not expr:
                continue
            c = _ast_canon(expr)
            if not c:
                continue
            for i, t in enumerate(canon_targets):
                if c == t:
                    found.add(i); break
        found_by_key_step[(gs, key)] = found
        records.append({"step": gs, "key": key,
                        "n_found": len(found), "n_sol": n_sol,
                        "n_roll": len(g),
                        "cov_g": len(found) / len(g)})
    cov_df = pd.DataFrame(records)

    # --- per-puzzle "total ever-found" -----------------------------------
    total_found_by_key = {k: set() for k in keys}
    for (gs, key), found in found_by_key_step.items():
        total_found_by_key[key] |= found
    total_found_count = {k: len(v) for k, v in total_found_by_key.items()}

    # diversity-collapse ratio: # unique-correct-now / min(total-ever-found, G)
    # The min(., G) cap keeps the ceiling reachable: if a puzzle has more
    # ever-found solutions than rollouts per cycle, the numerator can't
    # exceed G no matter how diverse the model is at that cycle.
    # Puzzles where the model never finds a solution are dropped.
    cov_df["total_found"] = cov_df["key"].map(total_found_count)
    div_df = cov_df[cov_df.total_found > 0].copy()
    div_df["div_denom"] = np.minimum(div_df["total_found"], div_df["n_roll"])
    div_df["div_ratio"] = div_df["n_found"] / div_df["div_denom"]

    # --- per-step cumulative + novelty fraction --------------------------
    cum_records: List[Dict[str, Any]] = []
    novel_records: List[Dict[str, Any]] = []
    cum_found = {k: set() for k in keys}
    for gs in steps:
        for key in keys:
            n_sol = nsol_by_key.get(key, 0)
            if n_sol == 0:
                continue
            cur_found = found_by_key_step.get((gs, key), set())
            n_novel = len(cur_found - cum_found[key])
            tot = total_found_count.get(key, 0)
            if tot > 0:
                novel_records.append({"step": gs, "key": key,
                                      "n_novel": n_novel,
                                      "total_found": tot,
                                      "novel_frac": n_novel / tot})
            cum_found[key] |= cur_found
            cum_records.append({"step": gs, "key": key,
                                "n_found_cum": len(cum_found[key]),
                                "n_sol": n_sol,
                                "cum": len(cum_found[key]) / n_sol})
    cum_df   = pd.DataFrame(cum_records)
    novel_df = pd.DataFrame(novel_records)

    _check = novel_df.groupby("key")["novel_frac"].sum()
    assert np.allclose(_check.values, 1.0, atol=1e-9), (
        f"novel_frac columns don't sum to 1 for some puzzles: "
        f"min={_check.min():.4f}, max={_check.max():.4f}"
    )

    cov_mean   = _per_step_mean_ci(cov_df,   "cov_g",      n_boot=n_boot, seed=seed)
    div_mean   = _per_step_mean_ci(div_df,   "div_ratio",  n_boot=n_boot, seed=seed)
    cum_mean   = _per_step_mean_ci(cum_df,   "cum",        n_boot=n_boot, seed=seed)
    novel_mean = _per_step_mean_ci(novel_df, "novel_frac", n_boot=n_boot, seed=seed)

    G_typical = int(cov_df.n_roll.mode().iloc[0]) if len(cov_df) else 0
    n_with_finds = sum(1 for k in keys if total_found_count[k] > 0)

    if verbose:
        print(f"Puzzles with ≥1 canonical solution: "
              f"{sum(1 for k in keys if nsol_by_key[k] > 0)} / {len(keys)}")
        print(f"Puzzles where the model EVER found ≥1 solution: "
              f"{n_with_finds} / {len(keys)}")
        print(f"|S(p)| stats: min={min(nsol_by_key.values())}, "
              f"max={max(nsol_by_key.values())}, "
              f"mean={np.mean(list(nsol_by_key.values())):.2f}")
        print(f"Rollouts per (step,puzzle): mode={G_typical}")
        print("\nDiversity collapse:  # unique-correct-now / min(# unique-correct-total, G)")
        print(div_mean.round(3).to_string(index=False))
        print("\nCumulative:  # found solutions / # solutions   (over s' ≤ s)")
        print(cum_mean.round(3).to_string(index=False))
        print("\nNovel-fraction:  # novel solutions / # ever-found  (per cycle)")
        print(novel_mean.round(3).to_string(index=False))
        print("Sums to 1.0 across cycles by construction (verified by assert).")

    # --- plot ------------------------------------------------------------
    C_INST  = "#2a9d8f"
    C_CUM   = "#264653"
    C_NOVEL = "#e76f51"

    with plt.rc_context({
        "axes.spines.top":   False, "axes.spines.right": False,
        "axes.grid":         True,  "grid.alpha": 0.25, "grid.linestyle": ":",
        "axes.titleweight":  "bold", "axes.titlesize": 11,
        "axes.labelsize":    10,    "legend.frameon": False,
    }):
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax = axes[0]
        ax.plot(div_mean.step, div_mean.val, color=C_INST, lw=2, marker="o", ms=5)
        ax.fill_between(div_mean.step, div_mean.lo, div_mean.hi,
                        color=C_INST, alpha=0.15, linewidth=0,
                        label="95% bootstrap CI over puzzles")
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.4, label="ceiling = 1")
        ax.set(xlabel="global_step",
               ylabel="# unique-correct-now / min(# unique-correct-total, G)",
               title="(a) Per-cycle correct-answer diversity\n(collapse if ratio drops)",
               ylim=(-0.02, 1.02))
        ax.legend(loc="best")

        ax = axes[1]
        ax.plot(cum_mean.step, cum_mean.val, color=C_CUM, lw=2, marker="o", ms=5)
        ax.fill_between(cum_mean.step, cum_mean.lo, cum_mean.hi,
                        color=C_CUM, alpha=0.15, linewidth=0,
                        label="95% bootstrap CI over puzzles")
        ax.set(xlabel="global_step",
               ylabel="# found solutions / # solutions",
               title="(b) Cumulative canonical-solution coverage",
               ylim=(-0.02, 1.02))
        ax.legend(loc="best")

        ax = axes[2]
        ax.plot(novel_mean.step, novel_mean.val,
                color=C_NOVEL, lw=2, marker="^", ms=6)
        ax.fill_between(novel_mean.step, novel_mean.lo, novel_mean.hi,
                        color=C_NOVEL, alpha=0.15, linewidth=0,
                        label="95% bootstrap CI over puzzles")
        ax.set(xlabel="global_step",
               ylabel="# novel solutions / # found solutions",
               title="(c) Per-cycle novelty fraction",
               ylim=(-0.02, 1.02))
        ax.set_xticks(novel_mean.step)
        ax.legend(loc="best")

        fig.tight_layout()

    info: Dict[str, Any] = {
        "cov_df": cov_df, "div_df": div_df,
        "cum_df": cum_df, "novel_df": novel_df,
        "cov_mean": cov_mean, "div_mean": div_mean,
        "cum_mean": cum_mean, "novel_mean": novel_mean,
        "nsol_by_key": nsol_by_key,
        "total_found_by_key": total_found_by_key,
        "total_found_count": total_found_count,
        "G_typical": G_typical,
        "n_puzzles_with_finds": n_with_finds,
        "steps": steps,
    }
    return fig, axes, info


def _lcp_chars(a: str, b: str) -> int:
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def plot_prefix_sharing_dynamics(
    eval_df,
    *,
    ks: Tuple[int, ...] = (1, 2, 3, 4, 5),
    prefix_len: Optional[int] = None,
    n_boot: int = 300,
    seed: int = 0,
    figsize: Tuple[float, float] = (12.0, 4.3),
    n_examples_per_band: int = 3,
    verbose: bool = True,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """Two-panel within-puzzle prefix-sharing dynamics for a Game-of-24 run.

    Both panels are computed WITHIN ``(global_step, key)`` groups (the G
    rollouts the model produces for a single puzzle). Shaded bands are 95%
    bootstrap CIs over puzzles.

      (b) Mean pairwise LCP per simplified pair-class
          (correct↔correct / correct↔incorrect / incorrect↔incorrect).
          Sets the natural length-scale: when ``prefix_len`` is None it is
          set to ``round(median pairwise LCP / 10) * 10``.

      (a) Top-k prefix coverage at ``prefix_len``: for each ``(step, key)``
          group, bucket rollouts by their first ``prefix_len`` characters and
          report the share of rollouts captured by the top-k buckets,
          averaged over puzzles.

    After the figure ``info["examples"]`` carries representative
    ``(numbers → prefix, count/G)`` samples from the last eval cycle, sampled
    from the high / median / low ends of the per-puzzle top-1 share
    distribution.

    Parameters
    ----------
    eval_df
        Long-form eval rollout DataFrame with columns ``key``, ``global_step``,
        ``completion``, ``correct`` (and optionally ``numbers``).

    Returns
    -------
    (fig, axes, info)
        ``info`` carries ``prefix_len``, the per-step bootstrap summaries
        (``cov_mean_by_k``, ``lcp_mean_by_pair``), the per-puzzle tables
        (``cov_g``, ``per_puzzle``), and ``examples``.
    """
    import pandas as pd

    # --- per-puzzle pairwise LCP per pair-class ---------------------------
    pair_rows: List[Dict[str, Any]] = []
    for (gs, key), g in eval_df.groupby(["global_step", "key"]):
        comps = g.completion.tolist()
        corr  = g.correct.astype(bool).tolist()
        n = len(g)
        for i in range(n):
            for j in range(i + 1, n):
                l = _lcp_chars(comps[i], comps[j])
                if   corr[i] and corr[j]:             tag = "correct ↔ correct"
                elif (not corr[i]) and (not corr[j]): tag = "incorrect ↔ incorrect"
                else:                                 tag = "correct ↔ incorrect"
                pair_rows.append({"step": gs, "key": key, "lcp": l, "pair": tag})
    pair_df = pd.DataFrame(pair_rows)

    # collapse to one observation per (step, pair, key) before bootstrapping
    per_puzzle = (pair_df.groupby(["step", "pair", "key"]).lcp.mean()
                         .reset_index())
    lcp_mean_by_pair = {
        p: _per_step_mean_ci(per_puzzle[per_puzzle.pair == p], "lcp",
                             n_boot=n_boot, seed=seed)
        for p in per_puzzle.pair.unique()
    }

    if prefix_len is None:
        prefix_len = int(round(pair_df.lcp.median() / 10) * 10)

    # --- (a) within-puzzle top-k prefix coverage at prefix_len -----------
    cov_per_group: List[Dict[str, Any]] = []
    for (gs, key), g in eval_df.groupby(["global_step", "key"]):
        prefixes = [c[:prefix_len] for c in g.completion]
        sorted_counts = sorted(Counter(prefixes).values(), reverse=True)
        n_total = len(prefixes)
        row = {"step": gs, "key": key,
               "n": n_total, "n_unique_prefixes": len(sorted_counts)}
        for k in ks:
            row[f"top{k}"] = sum(sorted_counts[:k]) / n_total
        cov_per_group.append(row)
    cov_g = pd.DataFrame(cov_per_group)
    cov_mean_by_k = {k: _per_step_mean_ci(cov_g, f"top{k}",
                                          n_boot=n_boot, seed=seed) for k in ks}

    if verbose:
        print(f"Median pairwise LCP across all (step, key, pair) = "
              f"{pair_df.lcp.median():.1f} chars  →  PREFIX_LEN = {prefix_len}")
        print(f"\nComputed LCP for {len(pair_df):,} pairs across "
              f"{pair_df.step.nunique()} eval cycles.")
        print(f"\nWithin-puzzle top-k {prefix_len}-char prefix coverage "
              f"(mean across {cov_g.key.nunique()} puzzles):")
        summary = pd.DataFrame({"step": cov_mean_by_k[ks[0]].step.values})
        for k in ks:
            summary[f"top{k}"] = cov_mean_by_k[k].val.values
        print(summary.round(3).to_string(index=False))

    # --- figure ----------------------------------------------------------
    PAIR_COLOR = {
        "correct ↔ correct":     "#2a9d8f",
        "correct ↔ incorrect":   "#e9c46a",
        "incorrect ↔ incorrect": "#e76f51",
    }
    K_CMAP = plt.cm.viridis

    with plt.rc_context({
        "axes.spines.top":   False, "axes.spines.right": False,
        "axes.grid":         True,  "grid.alpha": 0.25, "grid.linestyle": ":",
        "axes.titleweight":  "bold", "axes.titlesize": 11,
        "axes.labelsize":    10,    "legend.frameon": False,
    }):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        for i, k in enumerate(ks):
            d = cov_mean_by_k[k]
            color = K_CMAP(0.15 + 0.7 * i / max(1, len(ks) - 1))
            ax.fill_between(d.step, d.lo, d.hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(d.step, d.val, color=color, lw=2, marker="o", ms=5,
                    label=f"top-{k}")
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.4, label="ceiling = 1")
        ax.set(xlabel="global_step",
               ylabel="fraction of rollouts (per puzzle)",
               title=f"(a) Within-puzzle top-k {prefix_len}-char prefix coverage",
               ylim=(-0.02, 1.02))
        ax.set_xticks(cov_mean_by_k[ks[0]].step)
        ax.legend(loc="best", ncol=2)

        ax = axes[1]
        for p in ["correct ↔ correct", "correct ↔ incorrect", "incorrect ↔ incorrect"]:
            if p not in lcp_mean_by_pair:
                continue
            d = lcp_mean_by_pair[p]
            color = PAIR_COLOR[p]
            ax.fill_between(d.step, d.lo, d.hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(d.step, d.val, color=color, lw=2, marker="o", ms=5, label=p)
        ax.axhline(prefix_len, color="k", linestyle=":", alpha=0.4,
                   label=f"PREFIX_LEN = {prefix_len}")
        ax.set(xlabel="global_step",
               ylabel="mean pairwise LCP (chars)",
               title="(b) Within-puzzle mean pairwise LCP per pair-class")
        ax.set_xticks(d.step)
        ax.legend(loc="best")

        fig.tight_layout()

    # --- representative examples from the last eval cycle ----------------
    last_step = int(eval_df.global_step.max())
    last = eval_df[eval_df.global_step == last_step]

    ex_rows: List[Tuple[float, Any, List[Tuple[str, int]]]] = []
    for key, g in last.groupby("key"):
        prefs  = [c[:prefix_len] for c in g.completion]
        counts = Counter(prefs).most_common()
        top1_share = counts[0][1] / len(prefs)
        nums = list(g.numbers.iloc[0]) if "numbers" in g.columns else key
        ex_rows.append((top1_share, nums, counts))
    ex_rows.sort(key=lambda r: r[0], reverse=True)

    n = len(ex_rows)
    m = n_examples_per_band
    picks = (ex_rows[:m]
             + ex_rows[max(0, n // 2 - m // 2): n // 2 - m // 2 + m]
             + ex_rows[-m:])
    examples = [{"numbers": nums, "top1_share": share, "counts": counts}
                for share, nums, counts in picks]

    if verbose and examples:
        G = int(last.groupby("key").size().mode().iloc[0])
        print(f"\n=== Representative {prefix_len}-char prefixes at step {last_step} ===")
        print(f"(top-1 share = fraction of the {G} rollouts that share the most-common prefix)\n")
        for ex in examples:
            counts = ex["counts"]
            print(f"numbers={ex['numbers']}  top-1 share = {ex['top1_share']*100:.0f}%  "
                  f"({len(counts)} unique prefixes)")
            for pref, c in counts[:3]:
                snippet = pref.replace("\n", "\\n")
                print(f"  [{c}/{G}]  {snippet!r}")
            print()

    info: Dict[str, Any] = {
        "prefix_len": prefix_len,
        "ks": tuple(ks),
        "cov_g": cov_g,
        "per_puzzle": per_puzzle,
        "cov_mean_by_k": cov_mean_by_k,
        "lcp_mean_by_pair": lcp_mean_by_pair,
        "median_lcp": float(pair_df.lcp.median()),
        "examples": examples,
        "last_step": last_step,
    }
    return fig, axes, info


def _vt_pick_ref(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Collapse a per-reference vt row to a scalar (R_T, cumR) pair.

    Picks ``own_ref_idx`` for correct rollouts (the canonical solution match)
    and ``best_ref_idx`` otherwise (highest-R_T reference). Returns
    ``(None, None)`` when no valid reference is available.
    """
    own  = row.get("own_ref_idx", -1)
    best = row.get("best_ref_idx", -1)
    idx  = own if own is not None and own >= 0 else best
    if idx is None or idx < 0:
        return None, None
    R_T = row["R_T_per_ref"][idx]
    cum = row["cumR_resampled_per_ref"][idx]
    if R_T is None or not cum:
        return None, None
    return float(R_T), np.asarray(cum, dtype=float)


def plot_R_t_correctness_dynamics(
    vt_path,
    *,
    step: Optional[int] = None,
    n_boot: int = 300,
    seed: int = 0,
    figsize: Tuple[float, float] = (12.0, 4.3),
    verbose: bool = True,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """Two-panel R_t decoding-velocity dynamics for a single eval cycle.

    Reads the sidecar ``eval_rollout_vt.jsonl`` written by
    ``script/visualize_vt.py``: each rollout is scored against EVERY canonical
    24-solution for its puzzle (plus its own expression if AST-distinct), so
    each row carries per-reference arrays. We collapse to a scalar
    ``(R_T, cumR)`` pair via :func:`_vt_pick_ref`.

      (a) Within-puzzle correctness contrast
          ``mean cumR_correct − mean cumR_incorrect`` per relative position,
          averaged over puzzles that contain BOTH classes, with a 95%
          bootstrap CI over those puzzles. Removes the puzzle-difficulty
          confound that panel (b) inherits.

      (b) Global mean cumulative R_t per relative position, ±1σ, split by
          correctness — coarse marginal "progression rate" view; mixes
          puzzle effect with correctness effect since many puzzles are
          all-correct or all-incorrect.

    Parameters
    ----------
    vt_path
        Path to ``eval_rollout_vt.jsonl`` (or any pathlike).
    step
        Eval cycle (``global_step``) to plot. Defaults to the maximum present
        in the file (the vt sidecar is typically a single cycle).

    Returns
    -------
    (fig, axes, info)
        ``info`` carries ``vt_df`` (one row per rollout with the collapsed
        ``R_T`` / ``cumR``), ``contrast`` (per-puzzle Δ-curves used in (a)),
        ``mixed_keys``, ``step``.
    """
    import pandas as pd

    vt_path = Path(vt_path)
    if not vt_path.exists():
        raise FileNotFoundError(
            f"missing {vt_path}; run `python script/visualize_vt.py "
            f"--run-dir {vt_path.parent} --scorer-model <hf-id> --step <gs>`"
        )

    vt_rows = [json.loads(l) for l in vt_path.read_text().splitlines() if l.strip()]
    vt_df = pd.DataFrame([
        {
            "global_step": r["global_step"],
            "idx":         r["idx"],
            "key":         tuple(sorted(r["numbers"])),
            "correct":     bool(r["correct"]),
            "R_T":         _vt_pick_ref(r)[0],
            "cumR":        _vt_pick_ref(r)[1],
        }
        for r in vt_rows
    ])
    vt_df = vt_df[vt_df.cumR.notna()].copy()

    if step is None:
        step = int(vt_df.global_step.max())
    at_step   = vt_df[vt_df.global_step == step]
    correct   = at_step[at_step.correct]
    incorrect = at_step[~at_step.correct]
    mixed_keys = sorted(set(correct.key) & set(incorrect.key))

    if verbose:
        print(f"step={step}  rollouts: correct={len(correct)}  "
              f"incorrect={len(incorrect)}  "
              f"puzzles total={at_step.key.nunique()}  "
              f"with-both={len(mixed_keys)}")

    def _stack(df_class):
        if len(df_class) == 0:
            return np.empty((0, 0))
        return np.array(df_class["cumR"].tolist(), dtype=float)

    R_c, R_i = _stack(correct), _stack(incorrect)
    n_pts = R_c.shape[1] if len(R_c) else (R_i.shape[1] if len(R_i) else 100)
    x = np.linspace(0.0, 1.0, n_pts)

    contrast_per_q = []
    for k in mixed_keys:
        mu_c_q = np.array(correct  [correct.key   == k]["cumR"].tolist()).mean(0)
        mu_i_q = np.array(incorrect[incorrect.key == k]["cumR"].tolist()).mean(0)
        contrast_per_q.append(mu_c_q - mu_i_q)
    contrast = np.asarray(contrast_per_q)
    mu_d = lo_d = hi_d = None
    if len(contrast):
        mu_d = contrast.mean(0)
        rng = np.random.default_rng(seed)
        boots = np.stack([
            contrast[rng.integers(0, len(contrast), len(contrast))].mean(0)
            for _ in range(n_boot)
        ])
        lo_d = np.percentile(boots, 2.5,  axis=0)
        hi_d = np.percentile(boots, 97.5, axis=0)

    with plt.rc_context({
        "axes.spines.top":   False, "axes.spines.right": False,
        "axes.grid":         True,  "grid.alpha": 0.25, "grid.linestyle": ":",
        "axes.titleweight":  "bold", "axes.titlesize": 11,
        "axes.labelsize":    10,    "legend.frameon": False,
    }):
        fig, (ax_diff, ax_avg) = plt.subplots(1, 2, figsize=figsize)

        if len(contrast):
            col_d = "#264653"
            ax_diff.plot(x, mu_d, color=col_d, lw=2,
                         label=f"mean over {len(contrast)} puzzles")
            ax_diff.fill_between(x, lo_d, hi_d, color=col_d, alpha=0.15,
                                 linewidth=0,
                                 label="95% bootstrap CI over puzzles")
        ax_diff.axhline(0, color="k", linestyle=":", alpha=0.4)
        ax_diff.set(xlabel="normalised CoT position (t / T)",
                    ylabel="Δ cumR  (correct − incorrect)",
                    title=f"(a) Within-puzzle correctness contrast  (step={step})")
        ax_diff.legend(loc="best")

        for R, lbl, col in [(R_c, "correct", "#2a9d8f"),
                            (R_i, "incorrect", "#e76f51")]:
            if len(R) == 0:
                continue
            mu, sd = R.mean(0), R.std(0)
            ax_avg.plot(x, mu, color=col, lw=2, label=f"{lbl}  (n={len(R)})")
            ax_avg.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15,
                                linewidth=0)
        ax_avg.axhline(0, color="k", linestyle=":", alpha=0.4)
        ax_avg.set(xlabel="normalised CoT position (t / T)",
                   ylabel="mean cumulative R_t",
                   title=f"(b) R_t · global average  (step={step})")
        ax_avg.legend(loc="best")

        fig.tight_layout()

    info: Dict[str, Any] = {
        "vt_df": vt_df,
        "step": step,
        "mixed_keys": mixed_keys,
        "contrast": contrast,
        "mu_d": mu_d, "lo_d": lo_d, "hi_d": hi_d,
        "R_c": R_c, "R_i": R_i,
    }
    return fig, np.array([ax_diff, ax_avg]), info