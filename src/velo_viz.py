"""Velocity-reward visualization helpers.

One ``prepare`` call computes everything the three views need; each view
function then takes the resulting context and renders.

- :func:`prepare`           pre-compute paragraph boundaries, cumulative R(t),
                             ΔR_own per paragraph, sorted final-R_T order, etc.
- :func:`plot_R_t_static`   two-panel matplotlib (curves + sorted final R_T bars)
- :func:`print_paragraphs`  ANSI-colored terminal print of CoT paragraphs + ΔR_own
- :func:`make_animation`    streaming animation (live curves + bars + CoT reveal)
"""
from __future__ import annotations

import bisect
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


__all__ = ["VizCtx", "prepare", "plot_R_t_static", "print_paragraphs",
           "make_animation"]


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
