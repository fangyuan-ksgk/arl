"""
Post-hoc diagnostic probes for Game-of-24 GRPO rollouts.

Each probe returns ``(metrics_dict, matplotlib.Figure | None)`` so callers can
both aggregate scalar metrics across runs and save figures to disk.

Probes
------
- ``d1_length_diversity``     — D1 collapse / mode collapse signature
- ``coverage_probe``          — answer-buffer motivator (NOT D2)
- ``d3_pass_rate_by_bucket``  — zero-pass@K dead-zone test
- ``d2_vt_overlay``           — D2 structural — v_t vs broadcast advantage
- ``d4_vt_on_failed``         — D4 indiscriminate credit on failed rollouts

The two ``v_t`` probes need a callable ``vt_scorer(prompt_messages,
completion_text, reference_answer) -> (token_strs, vt, logps)``; build one with
``make_vt_scorer(model_name)``.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

__all__ = [
    "load_rollouts",
    "d1_length_diversity",
    "coverage_probe",
    "d3_pass_rate_by_bucket",
    "make_vt_scorer",
    "d2_vt_overlay",
    "d4_vt_on_failed",
    "score_rollout_sample",
    "d2_pair_figures",
    "decoding_reward_stats",
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

    agg = df.groupby("step").agg(
        n_tokens_mean=("n_tokens", "mean"),
        n_tokens_p90=("n_tokens", lambda s: float(np.percentile(s, 90))),
        acc=("correct", "mean"),
    ).reset_index()

    ed_rows = []
    for (step, _key), g in df[df.correct].groupby(["step", "key"]):
        texts = g.completion.tolist()
        if len(texts) < 2:
            continue
        pairs = [_norm_edit(texts[i], texts[j])
                 for i in range(len(texts)) for j in range(i + 1, len(texts))]
        ed_rows.append({"step": step, "mean_edit": float(np.mean(pairs))})
    edf = pd.DataFrame(ed_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(agg.step, agg.n_tokens_mean, label="mean")
    axes[0].plot(agg.step, agg.n_tokens_p90, label="p90", linestyle="--")
    axes[0].set(xlabel="step", ylabel="CoT tokens", title="D1 · CoT length")
    axes[0].legend()

    for ok, label in [(True, "correct"), (False, "incorrect")]:
        sub = df[df.correct == ok].groupby("step").n_tokens.mean()
        if len(sub):
            axes[1].plot(sub.index, sub.values, label=label)
    axes[1].set(xlabel="step", ylabel="mean tokens",
                title="D1 · length, split by correctness")
    axes[1].legend()

    if len(edf):
        axes[2].plot(edf.step, edf.mean_edit)
        axes[2].set(xlabel="step", ylabel="mean pairwise edit-distance",
                    title="D1 · diversity of correct CoTs\n(collapse → 0)")
    else:
        axes[2].text(0.5, 0.5, "not enough correct rollouts\non same puzzle",
                     ha="center", va="center")
        axes[2].axis("off")
    fig.tight_layout()

    metrics = {
        "d1_mean_tokens_first":  float(agg.n_tokens_mean.iloc[0])  if len(agg) else float("nan"),
        "d1_mean_tokens_last":   float(agg.n_tokens_mean.iloc[-1]) if len(agg) else float("nan"),
        "d1_mean_tokens_delta":  float(agg.n_tokens_mean.iloc[-1] - agg.n_tokens_mean.iloc[0]) if len(agg) > 1 else float("nan"),
        "d1_edit_distance_last": float(edf.mean_edit.iloc[-1]) if len(edf) else float("nan"),
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
    if len(df) == 0:
        return {}, None
    info = (
        pd.DataFrame([{"key": tuple(sorted(p["numbers"])),
                       "n_solutions": p["n_solutions"]} for p in puzzles])
        .drop_duplicates("key").set_index("key")
    )
    solved = df.groupby("key").correct.any().rename("ever_solved")
    joined = info.join(solved).fillna({"ever_solved": False})
    bucket = pd.cut(joined.n_solutions, [0, 2, 7, 1000],
                    labels=["hard (≤2)", "med (3-7)", "easy (≥8)"])
    grouped = joined.groupby(bucket).ever_solved.agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(grouped)), grouped["mean"].values,
           tick_label=[str(i) for i in grouped.index])
    ax.set(ylabel="fraction ever solved", ylim=(0, 1.05),
           title="D3 · pass-rate by difficulty bucket")
    for i, (m, n) in enumerate(zip(grouped["mean"], grouped["count"])):
        ax.text(i, m + 0.02, f"{m:.0%}\n(n={n})", ha="center", fontsize=8)
    fig.tight_layout()

    metrics = {f"d3_passrate_{label}": float(m)
               for label, m in zip(grouped.index.astype(str), grouped["mean"])}
    metrics["d3_passrate_overall"] = float(joined.ever_solved.mean())
    return metrics, fig


# ---------------------------------------------------------------------------
# v_t scorer
# ---------------------------------------------------------------------------
def make_vt_scorer(
    model_name: str,
    tokenizer,
    *,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Callable:
    """Returns a callable that scores per-token v_t against a reference answer."""
    from transformers import AutoModelForCausalLM
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    def _split_completion(completion: str) -> Tuple[str, str]:
        m = re.search(r"####\s*(.*)$", completion.strip(), re.DOTALL)
        if not m:
            return completion, ""
        return completion[: m.start()].rstrip(), f"#### {m.group(1).strip()}"

    def _prompt_text(prompt_messages) -> str:
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

    @torch.no_grad()
    def compute_vt(prompt_messages, completion_text: str, reference_answer: str):
        q_text = _prompt_text(prompt_messages)
        cot, _ = _split_completion(completion_text)
        a_text = reference_answer if reference_answer.startswith("####") else f"#### {reference_answer}"

        q_ids = tokenizer(q_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        o_ids = tokenizer(cot,    return_tensors="pt", add_special_tokens=False).input_ids[0]
        a_ids = tokenizer(a_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        T, La = len(o_ids), len(a_ids)
        if T == 0 or La == 0:
            return [], [], []

        logps = []
        for t in range(T + 1):
            ids = torch.cat([q_ids, o_ids[:t], a_ids]).unsqueeze(0).to(device)
            logits = model(ids).logits[0]
            start = len(q_ids) + t
            lp = sum(
                torch.log_softmax(logits[start + i - 1], dim=-1)[a_ids[i]].item()
                for i in range(La)
            )
            logps.append(lp)

        vt = [logps[t] - logps[t - 1] for t in range(1, T + 1)]
        toks = tokenizer.convert_ids_to_tokens(o_ids)
        return toks, vt, logps

    compute_vt.model = model     # expose for cleanup
    return compute_vt


# ---------------------------------------------------------------------------
# D2 — v_t overlay (one correct + one incorrect)
# ---------------------------------------------------------------------------
def d2_vt_overlay(
    df: pd.DataFrame,
    puzzles: List[Dict[str, Any]],
    vt_scorer: Callable,
    to_chat: Callable,
    *,
    seed: int = 1,
) -> Tuple[Dict[str, float], Optional[Figure]]:
    if len(df) == 0:
        return {}, None
    idx = _puzzle_index(puzzles)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    metrics: Dict[str, float] = {}
    for ax, correct_flag, label, broadcast in [
        (axes[0], True,  "correct rollout (GRPO adv ≈ +A)",  +1.0),
        (axes[1], False, "incorrect rollout (GRPO adv ≈ −A)", -1.0),
    ]:
        sub = df[df.correct == correct_flag]
        if len(sub) == 0:
            ax.text(0.5, 0.5, f"no {('correct' if correct_flag else 'incorrect')} rollouts",
                    ha="center", va="center"); ax.axis("off"); continue
        row = sub.sample(1, random_state=seed).iloc[0]
        puzzle = idx.get(row.key)
        if puzzle is None:
            ax.axis("off"); continue
        ref = puzzle["solutions"][0]
        toks, vt, _ = vt_scorer(to_chat(puzzle)["prompt"], row.completion, ref)
        ax.bar(range(len(vt)), vt,
               color=["#2a9d8f" if v >= 0 else "#e76f51" for v in vt])
        ax.axhline(broadcast, color="k", linestyle="--", alpha=0.7,
                   label=f"GRPO broadcast adv = {broadcast:+.2f}")
        ax.set(title=f"{label}  ·  puzzle={row.numbers}",
               xlabel="token position t", ylabel="v_t")
        ax.legend(loc="upper right", fontsize=8)
        s = max(1, len(toks) // 20)
        ax.set_xticks(range(0, len(toks), s))
        ax.set_xticklabels([toks[i].replace("Ġ", "▁")[:6] for i in range(0, len(toks), s)],
                           rotation=45, fontsize=7)
        suffix = "correct" if correct_flag else "incorrect"
        if vt:
            metrics[f"d2_vt_std_{suffix}"]   = float(np.std(vt))
            metrics[f"d2_vt_range_{suffix}"] = float(max(vt) - min(vt))

    fig.suptitle("D2 · per-token v_t vs. GRPO's broadcast advantage", y=1.02)
    fig.tight_layout()
    return metrics, fig


# ---------------------------------------------------------------------------
# D4 — productive tokens inside failed rollouts
# ---------------------------------------------------------------------------
def d4_vt_on_failed(
    df: pd.DataFrame,
    puzzles: List[Dict[str, Any]],
    vt_scorer: Callable,
    to_chat: Callable,
    *,
    n_sample: int = 8,
    seed: int = 0,
) -> Tuple[Dict[str, float], Optional[Figure]]:
    if len(df) == 0:
        return {}, None
    idx = _puzzle_index(puzzles)
    failed = df[~df.correct]
    if len(failed) == 0:
        return {"d4_n_failed_probed": 0}, None
    sample = failed.sample(min(n_sample, len(failed)), random_state=seed)

    rows = []
    for _, r in sample.iterrows():
        p = idx.get(r.key)
        if p is None:
            continue
        toks, vt, _ = vt_scorer(to_chat(p)["prompt"], r.completion, p["solutions"][0])
        if not vt:
            continue
        pos = sum(1 for v in vt if v > 0)
        rows.append({"frac_positive": pos / len(vt),
                     "total_vt": float(sum(vt)),
                     "n_tokens": len(vt)})
    if not rows:
        return {"d4_n_failed_probed": 0}, None

    ddf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(ddf.frac_positive, bins=20)
    ax.set(xlabel="fraction of tokens with v_t > 0 (failed rollouts)",
           ylabel="# rollouts",
           title="D4 · productive tokens inside failed trajectories")
    fig.tight_layout()
    return {
        "d4_n_failed_probed":           int(len(ddf)),
        "d4_mean_frac_positive":        float(ddf.frac_positive.mean()),
        "d4_median_frac_positive":      float(ddf.frac_positive.median()),
    }, fig


# ---------------------------------------------------------------------------
# Cached scoring + pair figures + global decoding-reward statistics
# ---------------------------------------------------------------------------
def score_rollout_sample(
    df: pd.DataFrame,
    puzzles: List[Dict[str, Any]],
    vt_scorer: Callable,
    to_chat: Callable,
    *,
    n_per_class: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score a balanced sample of correct/incorrect rollouts ONCE.

    For each sampled rollout we cache ``(toks, vt, R_T, R_per_token)`` so that
    downstream consumers (pair plots, global stats) don't repeat the
    expensive forward passes. Reference answer is chosen as follows:

    - **correct rollout**: model's own verified expression (`row.expr`) →
      makes cumulative R_t cleanly trend upward.
    - **incorrect rollout**: the puzzle's canonical `solutions[0]`.

    Returns a DataFrame with one row per scored rollout. Columns include:
    ``label`` ('correct' | 'incorrect'), ``key``, ``numbers``, ``completion``,
    ``ref_answer``, ``toks``, ``vt``, ``R_T``, ``n_tokens``, ``R_per_token``,
    ``step``, ``frac_positive``.
    """
    if len(df) == 0:
        return pd.DataFrame()
    idx = _puzzle_index(puzzles)

    records: List[Dict[str, Any]] = []
    for correct_flag, label in [(True, "correct"), (False, "incorrect")]:
        sub = df[df.correct == correct_flag]
        if len(sub) == 0:
            continue
        sample = sub.sample(min(n_per_class, len(sub)), random_state=seed)
        if verbose:
            print(f"  scoring {len(sample)} {label} rollouts...", flush=True)
        for k, (_, r) in enumerate(sample.iterrows()):
            puzzle = idx.get(r.key)
            if puzzle is None:
                continue
            ref = r.expr if (correct_flag and isinstance(r.get("expr"), str) and r.expr.strip())\
                  else puzzle["solutions"][0]
            toks, vt, _ = vt_scorer(to_chat(puzzle)["prompt"], r.completion, ref)
            if not vt:
                continue
            R_T = float(np.sum(vt))
            records.append({
                "label":         label,
                "correct":       bool(correct_flag),
                "key":           r.key,
                "numbers":       r.get("numbers"),
                "step":          int(r.get("step", -1)),
                "completion":    r.completion,
                "ref_answer":    ref,
                "toks":          toks,
                "vt":            vt,
                "n_tokens":      len(vt),
                "R_T":           R_T,
                "R_per_token":   R_T / len(vt),
                "frac_positive": float(sum(1 for v in vt if v > 0)) / len(vt),
            })
            if verbose and (k + 1) % 10 == 0:
                print(f"    {k+1}/{len(sample)}", flush=True)
    return pd.DataFrame(records)


def d2_pair_figures(
    scored: pd.DataFrame,
    output_dir: Path,
    *,
    n_pairs: int = 50,
    seed: int = 1,
) -> int:
    """Save up to ``n_pairs`` figures, each showing one correct + one incorrect
    rollout (v_t bars + cumulative R_t line + GRPO broadcast advantage).

    Pairs are sampled WITHOUT replacement within each class. Returns the number
    of figures actually written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct   = scored[scored.label == "correct"]
    incorrect = scored[scored.label == "incorrect"]
    n = min(n_pairs, len(correct), len(incorrect))
    if n == 0:
        return 0

    rng = np.random.default_rng(seed)
    c_idx = rng.permutation(len(correct))[:n]
    i_idx = rng.permutation(len(incorrect))[:n]

    written = 0
    for k, (ci, ii) in enumerate(zip(c_idx, i_idx)):
        cr = correct.iloc[ci]
        ir = incorrect.iloc[ii]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for ax, row, broadcast, title in [
            (axes[0], cr, +1.0, f"correct rollout (GRPO adv ≈ +A)   ·   puzzle={row_numbers(cr)}"),
            (axes[1], ir, -1.0, f"incorrect rollout (GRPO adv ≈ −A)  ·   puzzle={row_numbers(ir)}"),
        ]:
            _plot_vt_with_cumulative(ax, row["toks"], row["vt"], broadcast, title)
        fig.suptitle(f"pair {k+1}/{n}   ·   "
                     f"R_T(correct)={cr.R_T:+.2f}   R_T(incorrect)={ir.R_T:+.2f}",
                     y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"pair_{k:03d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        written += 1
    return written


def row_numbers(row) -> Any:
    return row.get("numbers")


def _plot_vt_with_cumulative(ax, toks, vt, broadcast, title) -> None:
    ax.bar(range(len(vt)), vt,
           color=["#2a9d8f" if v >= 0 else "#e76f51" for v in vt],
           alpha=0.7, label="per-token v_t")
    ax.axhline(broadcast, color="k", linestyle="--", alpha=0.7,
               label=f"GRPO broadcast adv = {broadcast:+.2f}")
    ax.set(title=title, xlabel="token position t", ylabel="v_t")
    s = max(1, len(toks) // 20)
    ax.set_xticks(range(0, len(toks), s))
    ax.set_xticklabels([toks[i].replace("Ġ", "▁")[:6] for i in range(0, len(toks), s)],
                       rotation=45, fontsize=7)

    R = np.cumsum(vt)
    ax2 = ax.twinx()
    ax2.plot(range(len(R)), R, color="#264653", linewidth=2,
             label=f"cumulative R_t  (R_T = {R[-1]:+.2f})")
    ax2.axhline(0, color="#264653", linestyle=":", alpha=0.4)
    ax2.set_ylabel("R_t = Σ v_{≤t}", color="#264653")
    ax2.tick_params(axis="y", labelcolor="#264653")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)


def decoding_reward_stats(
    scored: pd.DataFrame,
) -> Tuple[Dict[str, float], Optional[Figure]]:
    """Quantify how well total decoding reward R_T discriminates correct from
    incorrect trajectories. Computes:

    - mean / std / median of R_T per class
    - Cohen's d (effect size)
    - ROC AUC treating R_T as a binary classifier of correctness
    - Mann-Whitney U statistic + p-value
    - same metrics for R_T normalised by length (R_T / n_tokens)
    - same metrics for fraction of tokens with v_t > 0

    Produces a 2×2 figure: R_T hist, R_T boxplot, R_per_token boxplot,
    R_T vs step (training-time trend, if step info present).
    """
    if len(scored) == 0:
        return {}, None
    correct   = scored[scored.label == "correct"]
    incorrect = scored[scored.label == "incorrect"]
    if len(correct) == 0 or len(incorrect) == 0:
        return {"dr_n_correct": int(len(correct)),
                "dr_n_incorrect": int(len(incorrect))}, None

    def _summary(name: str, c: np.ndarray, i: np.ndarray) -> Dict[str, float]:
        c, i = np.asarray(c, dtype=float), np.asarray(i, dtype=float)
        pooled = np.sqrt((c.var(ddof=1) + i.var(ddof=1)) / 2) if len(c) > 1 and len(i) > 1 else float("nan")
        cohens_d = (c.mean() - i.mean()) / pooled if pooled and not np.isnan(pooled) else float("nan")

        # ROC AUC via Mann-Whitney U identity: AUC = U / (n_c * n_i)
        try:
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(c, i, alternative="greater")
            auc = float(u) / (len(c) * len(i))
        except Exception:  # scipy missing — fall back to numpy
            ranks = pd.Series(np.concatenate([c, i])).rank().to_numpy()
            r_c = ranks[: len(c)].sum()
            u = r_c - len(c) * (len(c) + 1) / 2
            auc = float(u) / (len(c) * len(i))
            p = float("nan")

        return {
            f"{name}_correct_mean":   float(c.mean()),
            f"{name}_correct_std":    float(c.std(ddof=1)) if len(c) > 1 else 0.0,
            f"{name}_incorrect_mean": float(i.mean()),
            f"{name}_incorrect_std":  float(i.std(ddof=1)) if len(i) > 1 else 0.0,
            f"{name}_gap":            float(c.mean() - i.mean()),
            f"{name}_cohens_d":       float(cohens_d),
            f"{name}_auc":            auc,
            f"{name}_mw_p":           float(p),
        }

    metrics: Dict[str, float] = {
        "dr_n_correct":   int(len(correct)),
        "dr_n_incorrect": int(len(incorrect)),
    }
    metrics.update(_summary("R_T",          correct.R_T,          incorrect.R_T))
    metrics.update(_summary("R_per_token",  correct.R_per_token,  incorrect.R_per_token))
    metrics.update(_summary("frac_pos",     correct.frac_positive, incorrect.frac_positive))
    metrics["len_correct_mean"]   = float(correct.n_tokens.mean())
    metrics["len_incorrect_mean"] = float(incorrect.n_tokens.mean())

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Histogram of R_T
    all_R = pd.concat([correct.R_T, incorrect.R_T])
    bins = np.linspace(float(all_R.min()), float(all_R.max()), 25)
    axes[0, 0].hist(incorrect.R_T, bins=bins, alpha=0.6, label="incorrect", color="#e76f51")
    axes[0, 0].hist(correct.R_T,   bins=bins, alpha=0.6, label="correct",   color="#2a9d8f")
    axes[0, 0].axvline(0, color="k", linestyle=":", alpha=0.5)
    axes[0, 0].set(xlabel="R_T = Σ v_t", ylabel="# rollouts",
                   title=f"R_T distribution   |   AUC={metrics['R_T_auc']:.3f}, "
                         f"d={metrics['R_T_cohens_d']:+.2f}")
    axes[0, 0].legend()

    # (b) Boxplot of R_T
    parts = axes[0, 1].boxplot([incorrect.R_T, correct.R_T],
                               tick_labels=["incorrect", "correct"], patch_artist=True)
    for patch, color in zip(parts["boxes"], ["#e76f51", "#2a9d8f"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    axes[0, 1].axhline(0, color="k", linestyle=":", alpha=0.5)
    axes[0, 1].set(ylabel="R_T",
                   title=f"R_T   |   gap = {metrics['R_T_gap']:+.2f}")

    # (c) Boxplot of length-normalised R_T
    parts = axes[1, 0].boxplot([incorrect.R_per_token, correct.R_per_token],
                               tick_labels=["incorrect", "correct"], patch_artist=True)
    for patch, color in zip(parts["boxes"], ["#e76f51", "#2a9d8f"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    axes[1, 0].axhline(0, color="k", linestyle=":", alpha=0.5)
    axes[1, 0].set(ylabel="R_T / n_tokens",
                   title=f"length-normalised  |   AUC={metrics['R_per_token_auc']:.3f}")

    # (d) R_T vs training step (does the gap grow / shrink as GRPO trains?)
    if scored.step.max() > scored.step.min():
        for label, sub, color in [
            ("correct",   correct,   "#2a9d8f"),
            ("incorrect", incorrect, "#e76f51"),
        ]:
            if len(sub) == 0:
                continue
            binned = (sub.assign(step_bin=pd.cut(sub.step, 8))
                        .groupby("step_bin", observed=True)
                        .R_T.agg(["mean", "count"]).reset_index())
            x = [iv.mid for iv in binned.step_bin]
            axes[1, 1].plot(x, binned["mean"], "o-", label=label, color=color)
        axes[1, 1].axhline(0, color="k", linestyle=":", alpha=0.5)
        axes[1, 1].set(xlabel="training step (binned)", ylabel="mean R_T",
                       title="R_T trend over training")
        axes[1, 1].legend()
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "single training step\n(no trend to plot)",
                        ha="center", va="center")

    fig.suptitle("Decoding reward R_T as a graded surrogate for trajectory correctness",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    return metrics, fig
