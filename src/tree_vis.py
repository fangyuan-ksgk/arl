"""Matplotlib visualization for :class:`tree_trainer.PrefixTrie`.

Single-child chains are collapsed so each drawn node is a branch point. Styling
mirrors ``prefix_trie.draw_compressed_trie``: circular nodes sized by
pass-through ``count``, arrowed edges whose colour encodes downstream success
(green = a correct continuation is reachable, red = none), token chains rendered
in rounded boxes on the edges, and per-node stats printed *below* the node. Fork
nodes are gold; leaves get a ✓ / ✗ marker.

Pass ``score_fn`` to overlay the under-explored-prefix ranking: each node gains a
``score`` / ``rank`` line and is shaded by score (darker = higher).
"""
from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Optional

from .tree_trainer import PrefixTrie


__all__ = ["compress_trie", "visualize_trie"]


# Edge colour encodes downstream success (mirrors prefix_trie.py).
EDGE_GREEN = "#1e7e34"
EDGE_RED = "#b00020"


# ---------------------------------------------------------------------------
# Collapse unary chains + layout
# ---------------------------------------------------------------------------
def compress_trie(root: "PrefixTrie") -> dict:
    """Collapse maximal single-child chains into one display node.

    Returns the display-tree root, a dict with keys ``tokens`` (the collapsed
    chain on the incoming edge), ``node`` (the underlying end-of-chain
    :class:`PrefixTrie`), ``count`` and ``children`` (list of display dicts).
    Storage keeps one token per node so distinct-length shared prefixes stay
    separate; a single-child chain carries no branching information, so for
    display it becomes one edge.
    """
    def _build(node: "PrefixTrie", chain: List[Hashable]) -> dict:
        d = {"tokens": list(chain), "node": node,
             "count": node.count, "children": []}
        for tok, child in node.children.items():
            ch, cur = [tok], child
            while len(cur.children) == 1:          # collapse single-child chain
                (t2, c2), = cur.children.items()
                ch.append(t2)
                cur = c2
            d["children"].append(_build(cur, ch))
        return d

    return _build(root, [])


def _annotate(disp: dict) -> List[dict]:
    """Assign ``id``, ``parent_id``, ``layer`` (depth), ``forked_parent`` and a
    leaf-ordered ``y`` to every display node; return them in DFS order."""
    all_nodes: List[dict] = []
    counter = [0]

    def _walk(node: dict, layer: int, parent_id: Optional[int],
              parent_n_children: int) -> None:
        node["id"] = counter[0]
        counter[0] += 1
        node["layer"] = layer
        node["parent_id"] = parent_id
        node["forked_parent"] = parent_n_children > 1
        all_nodes.append(node)
        for ch in node["children"]:
            _walk(ch, layer + 1, node["id"], len(node["children"]))

    _walk(disp, 0, None, 1)

    leaf_counter = [0]

    def _y(node: dict) -> float:
        kids = node["children"]
        if not kids:
            node["y"] = float(leaf_counter[0])
            leaf_counter[0] += 1
        else:
            node["y"] = sum(_y(k) for k in kids) / len(kids)
        return node["y"]

    _y(disp)
    return all_nodes


def _fmt_label(tokens: List[Hashable], max_chars: int = 22) -> str:
    if not tokens:
        return ""
    s = " ".join(map(str, tokens))
    if len(s) <= max_chars:
        return s
    head = max_chars // 2 - 2
    tail = max_chars - head - 3
    return s[:head] + "..." + s[-tail:]


def _node_size(count: int, n_total: int) -> float:
    base, span = 220, 800
    return base + span * (max(1, count) / max(1, n_total))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def visualize_trie(root: "PrefixTrie", reward_key: str = "correctness_reward",
                   ax=None, figsize=(11, 5), save_path: Optional[str] = None,
                   title: Optional[str] = None, max_edge_chars: int = 22,
                   score_fn: Optional[Callable[["PrefixTrie"], float]] = None):
    """Draw the compressed trie left-to-right and return the matplotlib ``Axes``.

    Edge colour: green if ``best_reward(reward_key) > 0`` for the child (a correct
    continuation is reachable), else red. Below each node: ``count`` and, when the
    key is present, the running ``max`` / ``mean`` reward. When ``score_fn`` is
    given, a ``score`` / ``rank`` line is added (rank 1 = highest) and node faces
    are shaded by score. Pass ``ax`` to draw onto an existing axis; ``save_path``
    to also write a PNG. ``matplotlib`` is imported lazily.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    disp = compress_trie(root)
    all_nodes = _annotate(disp)
    by_id = {n["id"]: n for n in all_nodes}
    n_total = disp["count"] or 1

    def _has_key(n: dict) -> bool:
        return reward_key in n["node"].stats

    def _correct(n: dict) -> bool:
        return _has_key(n) and n["node"].best_reward(reward_key) > 0

    # Per-node score + rank over the displayed/collapsed nodes. ``score_fn`` may
    # return ``None`` to opt a node out of ranking (e.g. leaves are complete
    # rollouts, not prefixes, so they receive no score / rank / shading).
    scores: Dict[int, float] = {}
    ranks: Dict[int, int] = {}
    cmap = None
    lo = span = 0.0
    if score_fn is not None:
        for n in all_nodes:
            s = score_fn(n["node"])
            if s is not None:
                scores[n["id"]] = float(s)
        for r, nid in enumerate(sorted(scores, key=scores.get, reverse=True), 1):
            ranks[nid] = r
        if scores:
            lo, hi = min(scores.values()), max(scores.values())
            span = (hi - lo) or 1.0
            cmap = plt.get_cmap("YlOrRd")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Edges
    for node in all_nodes:
        if node["parent_id"] is None:
            continue
        parent = by_id[node["parent_id"]]
        x0, y0 = parent["layer"], parent["y"]
        x1, y1 = node["layer"], node["y"]
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>",
            color=(EDGE_GREEN if _correct(node) else EDGE_RED),
            lw=1.5, mutation_scale=8, shrinkA=14, shrinkB=14, zorder=2,
        ))
        if node["tokens"]:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2,
                    _fmt_label(node["tokens"], max_edge_chars),
                    fontsize=7, ha="center", va="center", parse_math=False,
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#bbb", lw=0.5))

    # Nodes — size encodes pass-through count.
    for node in all_nodes:
        x, y = node["layer"], node["y"]
        is_root = node["parent_id"] is None
        forked = (not is_root) and node["forked_parent"]
        scored = node["id"] in scores
        if scored:
            face = cmap(0.15 + 0.85 * (scores[node["id"]] - lo) / span)
        else:
            face = "#ffc933" if forked else "white"
        edge_color = "#7a4a00" if forked else "black"
        ax.scatter([x], [y], s=_node_size(node["count"], n_total),
                   facecolor=face, edgecolor=edge_color,
                   lw=(2.0 if forked else 1.0), zorder=3)

        if is_root:
            ax.text(x - 0.04, y, "<root>", fontsize=7, ha="right", va="center",
                    parse_math=False, color="#555", zorder=4)

        # Stats below the node: count + running max / mean of the reward key.
        line = f"n={node['count']}"
        if _has_key(node):
            n = node["node"]
            line += (f"  max={n.best_reward(reward_key):.2f}"
                     f"  mean={n.reward_mean(reward_key):.2f}")
        ax.text(x, y - 0.30, line, fontsize=6.5, ha="center", va="top",
                parse_math=False, color="#222")

        if scored:
            ax.text(x, y - 0.56,
                    f"score={scores[node['id']]:.3f}  rank={ranks[node['id']]}",
                    fontsize=6.5, ha="center", va="top", parse_math=False,
                    color="#7a3a00", fontweight="bold")

        # Leaf marker: ✓ / ✗ to the right, ×count when >1.
        if not node["children"] and _has_key(node):
            c = node["count"]
            if node["node"].reward_mean(reward_key) > 0.5:
                mark, color = (f"✓ ×{c}" if c > 1 else "✓"), EDGE_GREEN
            else:
                mark, color = (f"✗ ×{c}" if c > 1 else "✗"), EDGE_RED
            ax.text(x + 0.10, y, mark, fontsize=11, ha="left", va="center",
                    parse_math=False, color=color, fontweight="bold", zorder=5)

    n_layers = max(n["layer"] for n in all_nodes) + 1
    n_leaves = sum(1 for n in all_nodes if not n["children"])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title or f"Compressed prefix trie ({reward_key})")
    ax.set_xlim(-0.8, n_layers - 0.4 + 0.45)
    ax.set_ylim(-1.0, max(1.0, n_leaves))
    if save_path:
        ax.figure.savefig(save_path, dpi=130, bbox_inches="tight")
    return ax