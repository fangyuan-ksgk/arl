"""Prefix-trie construction + matplotlib visualization for rollout CoTs.

Public API:
    tokenize(text)                            -> list[str]
    Trie                                      word-level prefix trie with correctness counters
    build_trie_with_correctness(rows)         -> Trie
    draw_compressed_trie(ax, trie, ...)       -> (n_layers, n_leaves, max_depth)
    legend_handles()                          -> list[matplotlib.lines.Line2D]

    MultiStepTrie                             word-level trie with per-step counters
    draw_overlay_trie(ax, multi_trie, steps,  overlay multi-step trie; earlier step = more opaque
                      ...)                    -> (n_layers, n_leaves, max_depth)

Per-node counters on the raw trie:
    _c   : pass-through count       (rollouts whose path includes this node)
    _cc  : pass-through correct count
    _t   : termination count at this node
    _tc  : termination correct count
"""
from __future__ import annotations

import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch


# Edge colour encodes downstream success:
#   green = at least one rollout taking this branch ends correctly
#   red   = every rollout taking this branch ends incorrectly
EDGE_GREEN = "#1e7e34"
EDGE_RED   = "#b00020"


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------
def tokenize(text):
    """Whitespace-split so the trie reflects exactly what the model emitted."""
    return text.split()


def _new_node():
    return {"_c": 0, "_cc": 0, "_t": 0, "_tc": 0, "_adv": 0.0, "_an": 0}


class Trie:
    __slots__ = ("root",)

    def __init__(self):
        self.root = _new_node()

    def insert(self, toks, correct, advantage=None):
        c = int(bool(correct))
        a = None if advantage is None else float(advantage)
        n = self.root
        n["_c"] += 1; n["_cc"] += c
        if a is not None:
            n["_adv"] += a; n["_an"] += 1
        for t in toks:
            n = n.setdefault(t, _new_node())
            n["_c"] += 1; n["_cc"] += c
            if a is not None:
                n["_adv"] += a; n["_an"] += 1
        n["_t"] += 1; n["_tc"] += c


def build_trie_with_correctness(rows):
    """rows: iterable of (token_seq, correct_bool) or (token_seq, correct_bool, advantage)."""
    t = Trie()
    for row in rows:
        if len(row) == 2:
            toks, c = row; t.insert(toks, c)
        else:
            toks, c, a = row; t.insert(toks, c, advantage=a)
    return t


# ---------------------------------------------------------------------------
# Compress unary chains + layout
# ---------------------------------------------------------------------------
def _collapse(raw):
    out = {"tokens": [], "count": raw["_c"], "count_correct": raw["_cc"],
           "terminal": raw["_t"], "terminal_correct": raw["_tc"],
           "adv_sum": raw.get("_adv", 0.0), "adv_n": raw.get("_an", 0),
           "children": {}}
    cur = raw
    while True:
        kids = {k: v for k, v in cur.items() if not k.startswith("_")}
        if len(kids) == 1 and cur["_t"] == 0:
            tok, child = next(iter(kids.items()))
            out["tokens"].append(tok)
            out["count"] = child["_c"];  out["count_correct"] = child["_cc"]
            out["terminal"] = child["_t"];  out["terminal_correct"] = child["_tc"]
            out["adv_sum"] = child.get("_adv", 0.0); out["adv_n"] = child.get("_an", 0)
            cur = child
        else:
            break
    kids = {k: v for k, v in cur.items() if not k.startswith("_")}
    for tok, child in kids.items():
        sub = _collapse(child)
        sub["tokens"].insert(0, tok)
        out["children"][tok] = sub
    return out


def _annotate(tree, layer=0, parent_n_children=1, _state=None):
    if _state is None:
        _state = {"counter": [0], "all": []}
    tree["id"] = _state["counter"][0]
    _state["counter"][0] += 1
    tree["layer"] = layer
    tree["forked_parent"] = parent_n_children > 1
    tree.setdefault("parent_id", None)
    _state["all"].append(tree)
    n_kids = len(tree["children"])
    for child in tree["children"].values():
        child["parent_id"] = tree["id"]
        _annotate(child, layer + 1, parent_n_children=n_kids, _state=_state)
    return _state


def _compute_y(tree):
    counter = [0]
    def walk(node):
        kids = list(node["children"].values())
        if not kids:
            node["y"] = float(counter[0]); counter[0] += 1
        else:
            for k in kids: walk(k)
            node["y"] = sum(k["y"] for k in kids) / len(kids)
    walk(tree)


def _fmt_label(tokens, max_chars=22):
    if not tokens: return ""
    s = " ".join(tokens)
    if len(s) <= max_chars: return s
    head = max_chars // 2 - 2
    tail = max_chars - head - 3
    return s[:head] + "..." + s[-tail:]


def _edge_color(node):
    return EDGE_GREEN if node["count_correct"] > 0 else EDGE_RED


def _node_size(count, n_total):
    base, span = 220, 800
    frac = max(1, count) / max(1, n_total)
    return base + span * frac


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_compressed_trie(ax, raw_trie, max_edge_chars=22, root_prefix_max_chars=60):
    tree = _collapse(raw_trie.root)
    state = _annotate(tree)
    _compute_y(tree)
    by_id = {n["id"]: n for n in state["all"]}
    n_total = state["all"][0]["count"]   # rollouts pass through ROOT = total CoTs

    # Edges
    for node in state["all"]:
        if node["parent_id"] is None: continue
        parent = by_id[node["parent_id"]]
        x0, y0 = parent["layer"], parent["y"]
        x1, y1 = node["layer"], node["y"]
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>",
            color=_edge_color(node),
            lw=1.5, mutation_scale=8,
            shrinkA=14, shrinkB=14, zorder=2,
        ))
        if node["tokens"]:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2,
                    _fmt_label(node["tokens"], max_edge_chars),
                    fontsize=7, ha="center", va="center", parse_math=False, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bbb", lw=0.5))

    # Nodes — size encodes pass-through count; no label inside the circle.
    for node in state["all"]:
        x, y = node["layer"], node["y"]
        is_root = node["parent_id"] is None
        gold = (not is_root) and node["forked_parent"]
        face = "#ffc933" if gold else "white"
        size = _node_size(node["count"], n_total)
        ax.scatter([x], [y], s=size, facecolor=face, edgecolor="black",
                   lw=1.0, zorder=3)

        # Root's collapsed prefix (shared by ALL rollouts) has no incoming edge —
        # render as a label to the left of ROOT so it isn't silently dropped.
        if is_root and node["tokens"]:
            ax.text(x - 0.18, y,
                    _fmt_label(node["tokens"], root_prefix_max_chars),
                    fontsize=7, ha="right", va="center", parse_math=False, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#bbb", lw=0.5))

        # Success rate of rollouts passing through this node.
        if node["count"] > 0:
            rate = node["count_correct"] / node["count"]
            ax.text(x, y - 0.30,
                    f"{node['count_correct']}/{node['count']}  ({rate:.0%})",
                    fontsize=6.5, ha="center", va="top",
                    parse_math=False, color="#222")
            # Mean trajectory advantage (if tracked).
            if node.get("adv_n", 0) > 0:
                mean_adv = node["adv_sum"] / node["adv_n"]
                adv_color = "#1e7e34" if mean_adv > 0 else ("#b00020" if mean_adv < 0 else "#444")
                ax.text(x, y - 0.50,
                        f"adv={mean_adv:+.2f}",
                        fontsize=6.5, ha="center", va="top",
                        parse_math=False, color=adv_color, fontweight="bold")

        # Leaf marker: ✓ / ✗ to the right.
        if not node["children"] and node["terminal"] > 0:
            tc = node["terminal_correct"]
            ti = node["terminal"] - tc
            if tc > 0 and ti == 0:
                mark, color = (f"✓ ×{tc}" if tc > 1 else "✓"), EDGE_GREEN
            elif ti > 0 and tc == 0:
                mark, color = (f"✗ ×{ti}" if ti > 1 else "✗"), EDGE_RED
            else:
                mark, color = f"✓×{tc}  ✗×{ti}", "#444"
            ax.text(x + 0.16, y, mark, fontsize=11, ha="left", va="center",
                    parse_math=False, color=color, fontweight="bold", zorder=5)

    n_layers = max(n["layer"] for n in state["all"]) + 1
    n_leaves = sum(1 for n in state["all"] if not n["children"])
    max_depth = max(n["layer"] for n in state["all"])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    has_root_prefix = bool(state["all"][0]["tokens"])
    left_pad = 1.6 if has_root_prefix else 0.6
    ax.set_xlim(-left_pad, n_layers - 0.4 + 0.45)
    ax.set_ylim(-1.0, max(1.0, n_leaves))
    return n_layers, n_leaves, max_depth


# ---------------------------------------------------------------------------
# Multi-step overlay
# ---------------------------------------------------------------------------
class MultiStepTrie:
    """Prefix trie that tracks counts per `step` key (e.g., global_step).

    Each node stores 4 dicts (step -> int): _c, _cc, _t, _tc.
    """
    __slots__ = ("root",)

    @staticmethod
    def _empty():
        return {"_c": {}, "_cc": {}, "_t": {}, "_tc": {},
                "_adv": {}, "_an": {}}

    def __init__(self):
        self.root = self._empty()

    def insert(self, toks, correct, step, advantage=None):
        c = int(bool(correct))
        a = None if advantage is None else float(advantage)
        n = self.root
        n["_c"][step] = n["_c"].get(step, 0) + 1
        n["_cc"][step] = n["_cc"].get(step, 0) + c
        if a is not None:
            n["_adv"][step] = n["_adv"].get(step, 0.0) + a
            n["_an"][step] = n["_an"].get(step, 0) + 1
        for t in toks:
            n = n.setdefault(t, self._empty())
            n["_c"][step] = n["_c"].get(step, 0) + 1
            n["_cc"][step] = n["_cc"].get(step, 0) + c
            if a is not None:
                n["_adv"][step] = n["_adv"].get(step, 0.0) + a
                n["_an"][step] = n["_an"].get(step, 0) + 1
        n["_t"][step] = n["_t"].get(step, 0) + 1
        n["_tc"][step] = n["_tc"].get(step, 0) + c


def _collapse_multi(raw):
    out = {"tokens": [],
           "count_per_step": dict(raw["_c"]),
           "count_correct_per_step": dict(raw["_cc"]),
           "terminal_per_step": dict(raw["_t"]),
           "terminal_correct_per_step": dict(raw["_tc"]),
           "adv_sum_per_step": dict(raw.get("_adv", {})),
           "adv_n_per_step": dict(raw.get("_an", {})),
           "children": {}}
    cur = raw
    while True:
        kids = {k: v for k, v in cur.items() if not k.startswith("_")}
        if len(kids) == 1 and sum(cur["_t"].values()) == 0:
            tok, child = next(iter(kids.items()))
            out["tokens"].append(tok)
            out["count_per_step"] = dict(child["_c"])
            out["count_correct_per_step"] = dict(child["_cc"])
            out["terminal_per_step"] = dict(child["_t"])
            out["terminal_correct_per_step"] = dict(child["_tc"])
            out["adv_sum_per_step"] = dict(child.get("_adv", {}))
            out["adv_n_per_step"] = dict(child.get("_an", {}))
            cur = child
        else:
            break
    kids = {k: v for k, v in cur.items() if not k.startswith("_")}
    for tok, child in kids.items():
        sub = _collapse_multi(child)
        sub["tokens"].insert(0, tok)
        out["children"][tok] = sub
    return out


def _survival_lw(n_present, n_steps, lw_min=0.5, lw_max=4.0):
    """Edge width scales with how many steps traverse the edge."""
    if n_steps <= 1:
        return lw_max
    return lw_min + (lw_max - lw_min) * ((n_present - 1) / (n_steps - 1))


def _survival_face(n_present, n_steps):
    """Node face color: light yellow when 1 step, deep red when all steps present."""
    if n_present <= 0:
        return "white"
    if n_steps <= 1:
        return cm.get_cmap("YlOrRd")(0.85)
    frac = (n_present - 1) / (n_steps - 1)
    return cm.get_cmap("YlOrRd")(0.18 + 0.72 * frac)


def draw_overlay_trie(ax, multi_trie, steps, max_edge_chars=22,
                      root_prefix_max_chars=60):
    """Overlay multi-step trie on a single shared layout.

    Visual encoding emphasises *survival across optimization*:
        - edge line width  -> number of steps traversing it (thin=1 step, thick=all)
        - edge color       -> green if any step has correct rollouts via this edge,
                              red if no step does
        - node face color  -> YlOrRd by step-survival (pale=1 step, dark=all steps)
        - node outline     -> bold gold ring when the node is a fork point
        - `K/N` label      -> below each node: # steps present / total steps
    """
    steps_sorted = sorted(steps)
    n_steps = len(steps_sorted)

    tree = _collapse_multi(multi_trie.root)
    state = _annotate(tree)
    _compute_y(tree)
    by_id = {n["id"]: n for n in state["all"]}
    n_total = sum(state["all"][0]["count_per_step"].values())

    # Edges — one per merged edge, encoding survival via width.
    for node in state["all"]:
        if node["parent_id"] is None:
            continue
        parent = by_id[node["parent_id"]]
        x0, y0 = parent["layer"], parent["y"]
        x1, y1 = node["layer"], node["y"]
        steps_here = [s for s in steps_sorted
                      if node["count_per_step"].get(s, 0) > 0]
        n_present = len(steps_here)
        any_correct = any(node["count_correct_per_step"].get(s, 0) > 0
                          for s in steps_here)
        color = EDGE_GREEN if any_correct else EDGE_RED
        lw = _survival_lw(n_present, n_steps)
        last_step = steps_sorted[-1]
        in_last = node["count_per_step"].get(last_step, 0) > 0
        linestyle = "solid" if in_last else "dotted"
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>",
            color=color, lw=lw, mutation_scale=8,
            shrinkA=14, shrinkB=14, zorder=2,
            linestyle=linestyle,
        ))
        if node["tokens"]:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2,
                    _fmt_label(node["tokens"], max_edge_chars),
                    fontsize=7, ha="center", va="center", parse_math=False, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bbb", lw=0.5))

    # Nodes — face color encodes survival breadth.
    for node in state["all"]:
        x, y = node["layer"], node["y"]
        is_root = node["parent_id"] is None
        total_cnt = sum(node["count_per_step"].values())
        n_present = sum(1 for s in steps_sorted
                        if node["count_per_step"].get(s, 0) > 0)

        face = _survival_face(n_present, n_steps)
        is_forked = (not is_root) and node["forked_parent"]
        edge_color = "#7a4a00" if is_forked else "black"
        edge_lw = 2.0 if is_forked else 1.0
        size = _node_size(total_cnt, n_total)
        last_step = steps_sorted[-1]
        in_last = node["count_per_step"].get(last_step, 0) > 0 or is_root
        node_linestyle = "solid" if in_last else "dotted"
        ax.scatter([x], [y], s=size, facecolor=face, edgecolor=edge_color,
                   lw=edge_lw, zorder=3, linestyle=node_linestyle)

        if is_root and node["tokens"]:
            ax.text(x - 0.18, y,
                    _fmt_label(node["tokens"], root_prefix_max_chars),
                    fontsize=7, ha="right", va="center", parse_math=False, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#bbb", lw=0.5))

        # Compact K/N label
        if total_cnt > 0:
            ax.text(x, y - 0.28, f"{n_present}/{n_steps}",
                    fontsize=7.5, ha="center", va="top",
                    parse_math=False, color="#111", fontweight="bold")

    n_layers = max(n["layer"] for n in state["all"]) + 1
    n_leaves = sum(1 for n in state["all"] if not n["children"])
    max_depth = max(n["layer"] for n in state["all"])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    has_root_prefix = bool(state["all"][0]["tokens"])
    left_pad = 1.6 if has_root_prefix else 0.6
    ax.set_xlim(-left_pad, n_layers - 0.4 + 0.45)
    ax.set_ylim(-1.0, max(1.0, n_leaves))
    return n_layers, n_leaves, max_depth


def overlay_legend_handles(steps):
    """Legend entries for the survival-encoded overlay."""
    n = len(steps)
    handles = [
        mlines.Line2D([], [], color=EDGE_GREEN, lw=2,
                      label="edge: ≥1 step has correct rollouts here"),
        mlines.Line2D([], [], color=EDGE_RED, lw=2,
                      label="edge: no step has correct rollouts here"),
        mlines.Line2D([], [], color="#666",
                      lw=_survival_lw(1, n), label=f"width: 1/{n} steps (rare)"),
        mlines.Line2D([], [], color="#666",
                      lw=_survival_lw(n, n), label=f"width: {n}/{n} steps (survives)"),
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=_survival_face(1, n),
                      markeredgecolor="black", markersize=10,
                      label=f"node: 1/{n} steps"),
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=_survival_face(n, n),
                      markeredgecolor="black", markersize=10,
                      label=f"node: {n}/{n} steps (survives)"),
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor="white",
                      markeredgecolor="#7a4a00", markeredgewidth=2,
                      markersize=10, label="node outline: fork point"),
        mlines.Line2D([], [], color="#666", lw=1.5, linestyle="dotted",
                      label="dotted: absent at last step (dropped out)"),
        mlines.Line2D([], [], color="#666", lw=1.5, linestyle="solid",
                      label="solid: still present at last step"),
    ]
    return handles


# ---------------------------------------------------------------------------
# Trajectory-level rewards + GRPO group-normalized advantages
# ---------------------------------------------------------------------------
def compute_grpo_advantages(
    df,
    *,
    inv_length_weight: float = 10.0,
    inv_length_clip: float = 1.0,
    inv_length_stride: int = 8,
    inv_length_dim: int = 2048,
    correctness_weight: float = 1.0,
    format_weight: float = 0.5,
    format_regex: str = r"####\s*[\d,\.\-]+",
    n_cot_tokens_col: str = "n_cot_tokens",
    completion_col: str = "completion",
    correct_col: str = "correct",
    step_col: str = "global_step",
    query_col: str = "query_id",
    normalize: bool = True,
    eps: float = 1e-8,
):
    """Add per-rollout `r_inv_length`, `r_correctness`, `r_format`, `reward`,
    and `advantage` columns to `df`.

    Reward components mirror the GSM8K trainer reward functions
    (`script/grpo_gsm8k.py`):

    - **InvLogLength** (`InvLogLengthReward`): `clip(1/log(min(T_comp, D)), ±clip)`,
      gated to 0 when `T_comp < 2 * stride`. We use ``n_cot_tokens`` as
      ``T_comp`` and weight by ``inv_length_weight`` (default 10× — the
      caller's "InvLength × 10").
    - **Correctness** (`correctness_reward`): `1.0 if correct else 0.0`,
      already provided by the rollout logger as the ``correct`` column.
    - **Format** (`format_reward`): `0.5 if regex matches else 0.0`, regex
      matched against the completion text.

    Trajectory-level **advantage** is computed GRPO-style by treating the
    rollouts within each ``(global_step, query_id)`` group as one batch:
        ``advantage_i = (reward_i - mean_g(reward)) / (std_g(reward) + eps)``
    Set ``normalize=False`` to skip the std normalisation.
    """
    import math
    import re
    pat = re.compile(format_regex)

    out = df.copy()

    def _inv_len(T):
        T = int(T)
        if T < 2 * inv_length_stride:
            return 0.0
        v = 1.0 / math.log(min(T, inv_length_dim))
        return max(-inv_length_clip, min(v, inv_length_clip))

    inv_len_r = out[n_cot_tokens_col].apply(_inv_len)
    correct_r = out[correct_col].astype(float)
    fmt_r = out[completion_col].apply(lambda c: 1.0 if pat.search(str(c)) else 0.0)

    out["r_inv_length"] = inv_length_weight * inv_len_r
    out["r_correctness"] = correctness_weight * correct_r
    out["r_format"] = format_weight * fmt_r
    out["reward"] = out["r_inv_length"] + out["r_correctness"] + out["r_format"]

    grp = out.groupby([step_col, query_col])["reward"]
    mean = grp.transform("mean")
    if normalize:
        std = grp.transform("std").fillna(0.0)
        out["advantage"] = (out["reward"] - mean) / (std + eps)
    else:
        out["advantage"] = out["reward"] - mean
    return out


# ---------------------------------------------------------------------------
# Interactive HTML overlay (Plotly)
# ---------------------------------------------------------------------------
def _wrap_text(s, width=80, max_lines=30):
    out, cur = [], ""
    for word in s.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur = cur + " " + word
        else:
            out.append(cur); cur = word
        if len(out) >= max_lines:
            out.append("... [truncated]")
            return out
    if cur:
        out.append(cur)
    return out


def build_interactive_overlay_html(
    df,
    output_path="trie_overlay.html",
    query_ids=None,
    completion_col="completion",
    correct_col="correct",
    step_col="global_step",
    query_col="query_id",
    gold_col="gold_answer",
    advantage_col="advantage",
):
    """Write a self-contained HTML page for interactively browsing prefix tries
    across queries and global steps.

    UI:
      - dropdown: query index
      - dropdown: step (each individual step + "all steps overlay")
      - hover any node: full accumulated prefix, depth (tokens), survival (k/N
        steps), aggregate success rate, and per-step counts/success
      - hover any edge midpoint: the tokens along that edge (full string)
    """
    import json
    try:
        import plotly.graph_objects as go  # noqa: F401
        from plotly.utils import PlotlyJSONEncoder
    except ImportError as e:
        raise ImportError("plotly is required: pip install plotly") from e

    if query_ids is None:
        query_ids = sorted(df[query_col].unique().tolist())

    all_data = {}

    for qid in query_ids:
        sub = df[df[query_col] == qid]
        if sub.empty:
            continue
        steps = sorted(sub[step_col].unique().tolist())
        n_steps = len(steps)
        last_step = steps[-1] if steps else None
        gold = sub.iloc[0][gold_col] if gold_col and gold_col in sub.columns else None

        has_adv = advantage_col is not None and advantage_col in sub.columns
        multi = MultiStepTrie()
        for step in steps:
            sub_s = sub[sub[step_col] == step]
            if has_adv:
                for toks, c, a in zip(sub_s[completion_col].map(tokenize),
                                      sub_s[correct_col],
                                      sub_s[advantage_col]):
                    multi.insert(toks, c, step, advantage=a)
            else:
                for toks, c in zip(sub_s[completion_col].map(tokenize),
                                   sub_s[correct_col]):
                    multi.insert(toks, c, step)

        tree = _collapse_multi(multi.root)
        state = _annotate(tree)
        _compute_y(tree)
        nodes = state["all"]
        by_id = {n["id"]: n for n in nodes}

        prefix_text = {}
        token_depth = {}

        def walk(node, parent_pref, parent_len):
            tok_str = " ".join(node["tokens"])
            if tok_str and parent_pref:
                cur = parent_pref + " " + tok_str
            elif tok_str:
                cur = tok_str
            else:
                cur = parent_pref
            prefix_text[node["id"]] = cur
            token_depth[node["id"]] = parent_len + len(node["tokens"])
            for ch in node["children"].values():
                walk(ch, cur, token_depth[node["id"]])

        walk(tree, "", 0)

        root_count_per_step = state["all"][0]["count_per_step"]

        view_keys = ["all"] + [str(s) for s in steps]
        view_steps = [None] + steps

        all_data[str(qid)] = {
            "gold": "" if gold is None else str(gold),
            "n_steps": n_steps,
            "views": {},
        }

        for view_key, view_step in zip(view_keys, view_steps):
            def present(node, vs=view_step):
                if vs is None:
                    return any(node["count_per_step"].get(s, 0) > 0 for s in steps)
                return node["count_per_step"].get(vs, 0) > 0

            # 4 edge buckets: (color, dashed)
            edges = {("g", False): [[], []], ("g", True): [[], []],
                     ("r", False): [[], []], ("r", True): [[], []]}
            mid_x, mid_y, mid_text = [], [], []

            for node in nodes:
                if node["parent_id"] is None:
                    continue
                if not present(node):
                    continue
                parent = by_id[node["parent_id"]]
                x0, y0 = parent["layer"], parent["y"]
                x1, y1 = node["layer"], node["y"]

                if view_step is None:
                    correct = any(node["count_correct_per_step"].get(s, 0) > 0
                                  for s in steps)
                else:
                    correct = node["count_correct_per_step"].get(view_step, 0) > 0
                color_key = "g" if correct else "r"

                if view_step is None:
                    dashed = (last_step is not None and
                              node["count_per_step"].get(last_step, 0) == 0)
                else:
                    dashed = False

                ex, ey = edges[(color_key, dashed)]
                ex += [x0, x1, None]
                ey += [y0, y1, None]

                edge_tokens = " ".join(node["tokens"]) or "(empty)"
                edge_wrapped = "<br>".join(_wrap_text(edge_tokens, 80, 20))
                mid_x.append((x0 + x1) / 2)
                mid_y.append((y0 + y1) / 2)
                mid_text.append(
                    f"<b>Edge tokens</b> ({len(node['tokens'])}):<br>{edge_wrapped}"
                )

            # Nodes
            nx, ny, nhover, nsize, ncolor, nborder = [], [], [], [], [], []
            if view_step is None:
                n_total = sum(root_count_per_step.values()) or 1
            else:
                n_total = root_count_per_step.get(view_step, 0) or 1

            for node in nodes:
                is_root = node["parent_id"] is None
                if not is_root and not present(node):
                    continue
                nx.append(node["layer"])
                ny.append(node["y"])

                n_present = sum(1 for s in steps
                                if node["count_per_step"].get(s, 0) > 0)

                if view_step is None:
                    total = sum(node["count_per_step"].values())
                    total_correct = sum(node["count_correct_per_step"].values())
                else:
                    total = node["count_per_step"].get(view_step, 0)
                    total_correct = node["count_correct_per_step"].get(view_step, 0)
                rate = (total_correct / total) if total else 0.0

                per_step_lines = []
                for s in steps:
                    c = node["count_per_step"].get(s, 0)
                    cc = node["count_correct_per_step"].get(s, 0)
                    if c > 0:
                        line = f"  step {s}: {cc}/{c} ({cc / c * 100:.0f}%)"
                        if has_adv:
                            an = node.get("adv_n_per_step", {}).get(s, 0)
                            asum = node.get("adv_sum_per_step", {}).get(s, 0.0)
                            if an > 0:
                                line += f"  adv={asum / an:+.2f}"
                        per_step_lines.append(line)
                    else:
                        per_step_lines.append(f"  step {s}: —")

                # Aggregate advantage line (mean over all rollouts traversing this node).
                adv_line = ""
                if has_adv:
                    asum_total = sum(node.get("adv_sum_per_step", {}).values())
                    an_total = sum(node.get("adv_n_per_step", {}).values())
                    if an_total > 0:
                        adv_line = (f"<b>Mean advantage:</b> "
                                    f"{asum_total / an_total:+.3f} "
                                    f"(n={an_total})<br>")

                pref = prefix_text[node["id"]] or "(root, empty prefix)"
                pref_wrapped = "<br>".join(_wrap_text(pref, 90, 40))

                hover = (
                    f"<b>Depth:</b> {token_depth[node['id']]} tokens<br>"
                    f"<b>Survival:</b> {n_present}/{n_steps} steps<br>"
                    f"<b>Aggregate:</b> {total_correct}/{total} "
                    f"({rate * 100:.0f}%)<br>"
                    f"{adv_line}"
                    f"<b>Per-step:</b><br>" + "<br>".join(per_step_lines) +
                    f"<br><br><b>Accumulated prefix:</b><br>{pref_wrapped}"
                )
                nhover.append(hover)
                size = 10 + 28 * (max(1, total) / n_total)
                nsize.append(size)
                ncolor.append(n_present / max(1, n_steps))
                nborder.append("#7a4a00" if (node["forked_parent"] and not is_root)
                               else "black")

            traces = []
            for (color_key, dashed), (ex, ey) in edges.items():
                if not ex:
                    continue
                col = EDGE_GREEN if color_key == "g" else EDGE_RED
                traces.append(dict(
                    type="scatter", x=ex, y=ey, mode="lines",
                    line=dict(color=col, width=2.0,
                              dash="dot" if dashed else "solid"),
                    hoverinfo="skip", showlegend=False,
                ))
            traces.append(dict(
                type="scatter", x=mid_x, y=mid_y, mode="markers",
                marker=dict(size=10, color="rgba(0,0,0,0)"),
                hovertext=mid_text, hoverinfo="text", showlegend=False,
                name="edge-mid",
            ))
            traces.append(dict(
                type="scatter", x=nx, y=ny, mode="markers",
                marker=dict(
                    size=nsize, color=ncolor, colorscale="YlOrRd",
                    cmin=0, cmax=1,
                    line=dict(color=nborder, width=1.2),
                    colorbar=dict(title="survival<br>k/N", thickness=10,
                                  len=0.5, x=1.02, y=0.5),
                ),
                hovertext=nhover, hoverinfo="text", showlegend=False,
                name="nodes",
            ))

            title_bits = [f"Query #{qid}"]
            if gold is not None:
                title_bits.append(f"gold = {gold}")
            if view_step is None:
                title_bits.append(f"view: overlay of all {n_steps} steps")
            else:
                title_bits.append(f"view: step {view_step}")
            all_data[str(qid)]["views"][view_key] = {
                "traces": traces,
                "title": " · ".join(title_bits),
            }

    json_data = json.dumps(all_data, cls=PlotlyJSONEncoder)
    qids_str = json.dumps([str(q) for q in query_ids])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Prefix Trie Overlay</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 12px; color: #222; }}
.controls {{ display: flex; gap: 18px; align-items: center;
             padding: 8px 4px; border-bottom: 1px solid #eee; }}
select {{ font-size: 13px; padding: 4px 8px; }}
#plot {{ width: 100%; height: 88vh; }}
.hint {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<div class="controls">
  <label>Query: <select id="qsel"></select></label>
  <label>Step: <select id="ssel"></select></label>
  <span class="hint">Hover a node: full prefix + per-step stats. Hover an edge: full edge tokens.</span>
</div>
<div id="plot"></div>
<script>
const DATA = {json_data};
const QIDS = {qids_str};
const qsel = document.getElementById('qsel');
const ssel = document.getElementById('ssel');

QIDS.forEach(q => {{
  const o = document.createElement('option');
  o.value = q; o.text = 'q' + q;
  qsel.appendChild(o);
}});

function refreshSteps() {{
  const qid = qsel.value;
  ssel.innerHTML = '';
  const views = DATA[qid].views;
  Object.keys(views).forEach(k => {{
    const o = document.createElement('option');
    o.value = k;
    o.text = (k === 'all') ? 'all steps (overlay)' : ('step ' + k);
    ssel.appendChild(o);
  }});
  render();
}}

function render() {{
  const qid = qsel.value;
  const step = ssel.value;
  if (!step) return;
  const v = DATA[qid].views[step];
  const layout = {{
    title: {{ text: v.title, font: {{ size: 13 }} }},
    hovermode: 'closest',
    xaxis: {{ visible: false, zeroline: false }},
    yaxis: {{ visible: false, zeroline: false }},
    margin: {{ t: 50, l: 10, r: 60, b: 10 }},
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    hoverlabel: {{ bgcolor: 'white', font: {{ family: 'ui-monospace, Menlo, monospace', size: 11 }} }},
  }};
  Plotly.react('plot', v.traces, layout, {{responsive: true}});
}}

qsel.addEventListener('change', refreshSteps);
ssel.addEventListener('change', render);
refreshSteps();
</script>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def legend_handles():
    return [
        mlines.Line2D([], [], color=EDGE_GREEN, lw=2,
                      label="edge: at least one downstream rollout correct"),
        mlines.Line2D([], [], color=EDGE_RED,   lw=2,
                      label="edge: all downstream rollouts incorrect"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#ffc933",
                      markeredgecolor="black", markersize=9, label="sampleable (parent forked)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                      markeredgecolor="black", markersize=9, label="forced-chain"),
        mlines.Line2D([], [], marker="$✓$", color=EDGE_GREEN, markersize=10, lw=0,
                      label="leaf: rollout was correct"),
        mlines.Line2D([], [], marker="$✗$", color=EDGE_RED, markersize=10, lw=0,
                      label="leaf: rollout was incorrect"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                      markeredgecolor="black", markersize=12,
                      label="node area ∝ pass-through count"),
    ]
