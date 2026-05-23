"""Prefix injection at rollout time for TRL GRPO (vLLM colocate mode).

The :class:`PrefixInjector` is a ``rollout_func`` for
:class:`trl.GRPOTrainer` — pass it via ``rollout_func=...`` and it will:

1. Per group (default) or per rollout, decide whether to inject a prefix.
2. Sample an accepted CoT from a :class:`PrefixBuffer`, optionally truncate
   it, and **prepend it to the prompt token IDs** sent to vLLM. vLLM
   conditions normally on ``prompt + prefix`` and generates fresh tokens.
3. After generation, the prefix tokens are folded into ``completion_ids``
   (left of the freshly-generated tokens), so the trainer treats them as
   if the policy produced them. The reported ``logprobs`` for prefix
   tokens come from vLLM's ``prompt_logprobs`` (the current policy's own
   logprobs), making the initial importance ratio exactly 1.

Scope
-----
This implementation targets ``vllm_mode="colocate"`` only — the path used
by the Game-of-24 notebook. The trainer's distributed / server / tensor-
parallel branches are NOT exercised here.

Per-rollout ``max_tokens``
--------------------------
``SamplingParams.max_tokens`` is held **constant** across rollouts
(= ``max_completion_length``) so every rollout gets the same fresh-
generation budget regardless of prefix length. The total per-request
length is ``len(prompt) + len(prefix) + max_completion_length`` — make
sure ``vllm_max_model_length`` is set generously.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np


__all__ = [
    "PrefixTrieNode",
    "PrefixTrie",
    "PrefixTrieBuffer",
    "PrefixInjector",
    "visualize_prefix_trie",
]


# ---------------------------------------------------------------------------
# Prefix trie — character-level, depth-capped. Used as the rollout prefix
# cache (OnlineBuffer dedups by *exact* string equality and can't share
# structure across rollouts that diverge late, so it's unfit for prefix
# injection). The trie gives us:
#   • automatic dedup of common prefixes,
#   • per-node visit counts (branch-popularity signal),
#   • bounded memory (~|alphabet| × max_depth nodes per puzzle),
#   • layer-aware uniform sampling over *real* fork points only.
# ---------------------------------------------------------------------------


class PrefixTrieNode:
    __slots__ = ("children", "count", "terminal")

    def __init__(self):
        self.children: dict = {}
        self.count: int = 0      # rollouts passing through this node
        self.terminal: int = 0   # rollouts that ended exactly here (≤ max_depth)


class PrefixTrie:
    """Character-level trie capped at ``max_depth`` characters, with
    layer-aware uniform sampling over real branch points.

    A node is *sampleable* iff its parent had ≥ 2 children (a real fork).
    Single-child chains are absorbed into the edge label and do NOT
    advance the layer counter, so the shared head (e.g. ``<think>``)
    is never returned by :meth:`sample`.
    """

    def __init__(self, max_depth: int = 100):
        self.root = PrefixTrieNode()
        self.max_depth = max_depth

    def add(self, s: str) -> None:
        head = s[: self.max_depth]
        node = self.root
        node.count += 1
        for ch in head:
            child = node.children.get(ch)
            if child is None:
                child = PrefixTrieNode()
                node.children[ch] = child
            node = child
            node.count += 1
        node.terminal += 1

    # ----- layer-aware enumeration / sampling --------------------------
    def layered_prefixes(self, max_layer: int = 3) -> List[Tuple[str, int]]:
        """Return ``(prefix_string, layer)`` for every node whose parent
        had ≥ 2 children, at layer ∈ ``[2, max_layer]``.
        """
        out: List[Tuple[str, int]] = []

        def walk(node, prefix: str, layer: int) -> None:
            if layer >= max_layer:
                return
            is_real_fork = len(node.children) >= 2
            for ch, child in node.children.items():
                new_prefix = prefix + ch
                cur = child
                while len(cur.children) == 1 and cur.terminal == 0:
                    only_ch, only_child = next(iter(cur.children.items()))
                    new_prefix += only_ch
                    cur = only_child
                if is_real_fork:
                    out.append((new_prefix, layer + 1))
                    walk(cur, new_prefix, layer + 1)
                else:
                    walk(cur, new_prefix, layer)  # forced chain → keep layer

        walk(self.root, "", 1)
        return out

    def sample(self, rng, max_layer: int = 3) -> str:
        """Uniform-sample one prefix among all real-branch nodes at
        layer ∈ ``[2, max_layer]``. Returns ``""`` if no candidates exist.
        """
        cands = self.layered_prefixes(max_layer=max_layer)
        if not cands:
            return ""
        return cands[int(rng.integers(0, len(cands)))][0]

    def n_branches_at(self, d: int) -> int:
        """Number of distinct char-level nodes at char-depth ``d``."""
        frontier = [self.root]
        for _ in range(d):
            frontier = [c for n in frontier for c in n.children.values()]
        return len(frontier)


# ---------------------------------------------------------------------------
# PrefixTrieBuffer — a per-query dict of :class:`PrefixTrie`s, exposing the
# same ``add(qk, s) / has(qk) / sample(qk, rng)`` surface as
# :class:`~src.online_buffer.OnlineBuffer` so it is a drop-in replacement
# for the ``prefix_buffer`` slot inside :class:`PrefixInjector` /
# :class:`PerTokenTrainer`.
#
# What gets stored
# ----------------
# We do NOT store full CoTs. ``add(qk, completion_str)`` truncates the
# completion to the first ``max_depth`` characters and inserts that head
# into the per-query trie. So the buffer holds a *forest of prefix heads*,
# one tree per query, with shared structure across rollouts that agree on
# their early tokens.
#
# What ``sample`` returns
# -----------------------
# A *layer-bounded real-fork prefix* — i.e. the trie walks down to a node
# whose parent had ≥ 2 children and stops between layers ``[2, max_layer]``.
# This is already a partial prefix, not a full CoT, so additional uniform
# truncation on top is unnecessary (and would in fact often cut into the
# fork point). Hence :class:`PrefixInjector` defaults ``truncate="none"``.
# ---------------------------------------------------------------------------


class PrefixTrieBuffer:
    """Per-query character-level prefix trie buffer.

    Parameters
    ----------
    max_depth
        Per-rollout char-cap when inserting (only the first ``max_depth``
        chars of any completion are kept; the rest is discarded).
    max_layer
        Sampling depth in *real fork* layers. ``2`` means "sample from the
        first real branch point", ``3`` adds the second, etc. The shared
        deterministic head (e.g. ``<think>``) is layer 1 and is never
        returned. See :meth:`PrefixTrie.layered_prefixes` for the layering
        semantics.
    """

    def __init__(self, *, max_depth: int = 100, max_layer: int = 3):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_layer < 2:
            raise ValueError("max_layer must be >= 2 (layer 1 is the shared head)")
        self.max_depth = max_depth
        self.max_layer = max_layer
        self._tries: Dict[Hashable, PrefixTrie] = {}

    # ------------------------------------------------------------------ writes
    def add(self, query_key: Hashable, completion: Optional[str], count: int = 1) -> bool:
        """Insert ``completion[:max_depth]`` into the trie for ``query_key``.

        ``count`` is accepted for API parity with :class:`OnlineBuffer` but
        only the *presence* matters for trie sampling (each call inserts
        once; visit counts already accumulate inside the trie node).
        Returns ``True`` if the trie now has at least one sampleable node.
        """
        if completion is None:
            return False
        s = completion.strip()
        if not s:
            return False
        trie = self._tries.get(query_key)
        if trie is None:
            trie = PrefixTrie(max_depth=self.max_depth)
            self._tries[query_key] = trie
        for _ in range(max(1, int(count))):
            trie.add(s)
        return self.has(query_key)

    # ------------------------------------------------------------------ reads
    def has(self, query_key: Hashable) -> bool:
        """Whether the per-query trie has ≥ 1 sampleable real-fork node.

        A trie with only a single shared head (no real fork) returns
        ``False`` so the injector skips injection rather than returning an
        empty prefix.
        """
        trie = self._tries.get(query_key)
        if trie is None:
            return False
        return bool(trie.layered_prefixes(max_layer=self.max_layer))

    def sample(
        self,
        query_key: Hashable,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[str]:
        """Sample one real-fork prefix for ``query_key`` (or ``None``)."""
        trie = self._tries.get(query_key)
        if trie is None:
            return None
        if rng is None:
            rng = np.random.default_rng()
        s = trie.sample(rng, max_layer=self.max_layer)
        return s if s else None

    # ------------------------------------------------------------------ stats
    def num_queries(self) -> int:
        return sum(1 for t in self._tries.values() if t.root.count > 0)

    def __len__(self) -> int:
        """Total CoTs ingested across all per-query tries."""
        return sum(t.root.count for t in self._tries.values())

    def stats(self) -> Dict[str, float]:
        sizes = [t.root.count for t in self._tries.values() if t.root.count > 0]
        sampleable = [
            len(t.layered_prefixes(max_layer=self.max_layer))
            for t in self._tries.values()
        ]
        return {
            "n_queries":            float(len(sizes)),
            "n_cots_inserted":      float(sum(sizes)),
            "median_cots_per_q":    float(np.median(sizes)) if sizes else 0.0,
            "max_cots_per_q":       float(max(sizes)) if sizes else 0.0,
            "median_forks_per_q":   float(np.median(sampleable)) if sampleable else 0.0,
            "n_queries_sampleable": float(sum(1 for n in sampleable if n > 0)),
        }


class PrefixInjector:
    """Rollout-time prefix injector (colocate-mode TRL rollout_func).

    Parameters
    ----------
    buffer
        Any object exposing ``has(qk) -> bool`` and
        ``sample(qk, rng=...) -> Optional[str]``. Recommended:
        :class:`PrefixTrieBuffer`, which stores per-query *prefix heads*
        (CoTs truncated to ``max_depth`` chars) in a radix trie and samples
        a layer-bounded real-fork prefix. :class:`OnlineBuffer` also works
        but it stores full strings with no shared structure across
        rollouts and dedups only by exact equality, so use
        ``truncate="uniform"`` in that case to randomise the cut.
    query_key_fn
        ``(prompt_str) -> Hashable`` mapping the chat-templated prompt
        string to a buffer key. Must agree with the online-update side of
        the buffer.
    p_inject
        Probability of injecting a prefix into a given group (or rollout
        if ``share_within_group=False``).
    truncate
        ``"none"`` (default, correct for :class:`PrefixTrieBuffer`) — use
        the sampled string verbatim. The trie sample is *already* a
        layer-bounded prefix, so no extra cut is needed.
        ``"uniform"`` — additionally sample a uniform truncation length in
        ``[1, len(toks)]``. Sensible only with full-CoT buffers like
        :class:`OnlineBuffer`; on a trie sample it tends to chop below the
        real fork point and is therefore counter-productive.
    share_within_group
        ``True`` (recommended) — one prefix decision per group of
        ``num_generations`` rollouts. Keeps the GRPO within-group baseline
        comparing apples to apples.
        ``False`` — independent prefix decision per rollout; noisier
        baseline but more diversity per step.
    rng
        ``np.random.Generator``. Default: fresh non-reproducible.
    """

    def __init__(
        self,
        buffer,
        *,
        query_key_fn: Callable[[str], Hashable],
        p_inject: float = 0.5,
        truncate: str = "none",
        share_within_group: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        if truncate not in ("none", "uniform"):
            raise ValueError(f"unknown truncate: {truncate!r}")
        if not 0.0 <= p_inject <= 1.0:
            raise ValueError(f"p_inject must be in [0, 1], got {p_inject}")
        self.buffer = buffer
        self.query_key_fn = query_key_fn
        self.p_inject = p_inject
        self.truncate = truncate
        self.share_within_group = share_within_group
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ helpers
    def _sample_prefix_ids(self, qk, tokenizer):
        # buffer.sample(...) returns either:
        #   • a full CoT string (OnlineBuffer)                   -> needs random cut
        #   • a layer-bounded real-fork prefix (PrefixTrieBuffer) -> no cut needed
        # The trie path is the recommended one; see PrefixTrieBuffer.
        prefix_str = self.buffer.sample(qk, rng=self.rng)
        if not prefix_str:
            return []
        ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
        if not ids:
            return []
        if self.truncate == "uniform":
            L = int(self.rng.integers(1, len(ids) + 1))
            ids = ids[:L]
        return list(ids)

    # ------------------------------------------------------------------ main
    def __call__(self, prompts, trainer):
        """The rollout_func entry point. ``prompts`` are already
        chat-templated strings (TRL handles the conversion in colocate
        mode before calling us).
        """
        from vllm import SamplingParams, TokensPrompt

        if trainer.args.vllm_mode != "colocate":
            raise NotImplementedError(
                "PrefixInjector currently supports vllm_mode='colocate' only"
            )
        llm = trainer.vllm_generation.llm
        tokenizer = trainer.processing_class
        G = trainer.num_generations

        base_sp_kw = dict(
            n=1,
            temperature=trainer.temperature,
            top_p=trainer.top_p,
            top_k=trainer.top_k if trainer.top_k is not None else -1,
            min_p=0.0 if trainer.min_p is None else trainer.min_p,
            repetition_penalty=trainer.repetition_penalty,
            max_tokens=trainer.max_completion_length,
            logprobs=0,           # logprob of the sampled token
            prompt_logprobs=0,    # logprob of each prompt token under the policy
        )

        # Build per-rollout (token_prompt, sampling_params).
        flat_token_prompts: list = []
        flat_sps: list = []
        flat_prefix_lens: list = []
        flat_bare_lens: list = []

        for prompt_str in prompts:
            p_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
            p_ids = list(p_ids)
            qk = self.query_key_fn(prompt_str)

            if self.share_within_group:
                inject = (self.rng.random() < self.p_inject) and self.buffer.has(qk)
                shared = self._sample_prefix_ids(qk, tokenizer) if inject else []
                prefix_choices = [shared] * G
            else:
                prefix_choices = [
                    self._sample_prefix_ids(qk, tokenizer)
                    if (self.rng.random() < self.p_inject and self.buffer.has(qk))
                    else []
                    for _ in range(G)
                ]

            for prefix in prefix_choices:
                flat_token_prompts.append(TokensPrompt(prompt_token_ids=p_ids + prefix))
                flat_sps.append(SamplingParams(**base_sp_kw))
                flat_prefix_lens.append(len(prefix))
                flat_bare_lens.append(len(p_ids))

        # vLLM generate. Accepts list of SamplingParams aligned with prompts.
        outputs = llm.generate(flat_token_prompts, sampling_params=flat_sps, use_tqdm=False)

        prompt_ids_out: list = []
        completion_ids_out: list = []
        logprobs_out: list = []

        for out, plen, blen in zip(outputs, flat_prefix_lens, flat_bare_lens):
            gen = out.outputs[0]
            gen_ids = list(gen.token_ids)
            full_prompt = list(out.prompt_token_ids)
            bare_ids = full_prompt[:blen]
            prefix_ids = full_prompt[blen:blen + plen]
            completion = prefix_ids + gen_ids

            # logprobs for prefix tokens: from vLLM's prompt_logprobs (first
            # entry is None — the BOS / first prompt token under-defined).
            # Take only the prefix slice.
            prefix_lps: list = []
            if plen > 0 and out.prompt_logprobs is not None:
                prompt_lp_seq = out.prompt_logprobs[blen:blen + plen]
                for entry in prompt_lp_seq:
                    if entry is None:
                        prefix_lps.append(0.0)
                    else:
                        # Pick the entry for the actual token id at this position.
                        # vLLM returns dict {token_id: Logprob}; we want the one
                        # corresponding to the prompt's own token. Since
                        # prompt_logprobs=0 only returns the sampled (= actual)
                        # token, the dict has one entry — take it.
                        prefix_lps.append(float(next(iter(entry.values())).logprob))
            else:
                prefix_lps = [0.0] * plen

            gen_lps: list = []
            if gen.logprobs is not None:
                for tok_lps in gen.logprobs:
                    if tok_lps:
                        gen_lps.append(float(next(iter(tok_lps.values())).logprob))
                    else:
                        gen_lps.append(0.0)
            if len(gen_lps) != len(gen_ids):
                gen_lps = [0.0] * len(gen_ids)

            prompt_ids_out.append(bare_ids)
            completion_ids_out.append(completion)
            logprobs_out.append(prefix_lps + gen_lps)

        return {
            "prompt_ids": prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
        }


# ---------------------------------------------------------------------------
# Visualization — render a PrefixTrie as a compressed (radix) tree with
# edge labels showing the absorbed character runs, nodes colored by
# whether they are sampleable, and a legend mapping each node id to its
# full prefix string (so you can read long labels even if the edge text
# is truncated).
# ---------------------------------------------------------------------------


def visualize_prefix_trie(
    trie: "PrefixTrie",
    *,
    ax=None,
    max_label_chars: int = 28,
    orientation: str = "horizontal",
    figsize: tuple = (16, 9),
    title: Optional[str] = None,
    show_legend_table: bool = True,
):
    """Render ``trie`` as a compressed tree.

    Parameters
    ----------
    trie
        The :class:`PrefixTrie` to draw.
    ax
        Optional matplotlib axes. If ``None``, a new figure is created.
    max_label_chars
        Edge-label truncation budget. Set higher for fewer ellipses.
    orientation
        ``"horizontal"`` (root on the left, leaves on the right — best
        for long text labels) or ``"vertical"`` (root on top).
    figsize
        Figure size when ``ax`` is ``None``.
    title
        Optional title.
    show_legend_table
        If ``True``, prints a node-id → full-prefix mapping below the
        diagram (helpful when edge labels are truncated).
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    # ---- build compressed graph ----
    G = nx.DiGraph()
    edge_labels: dict = {}
    edge_full: dict = {}
    node_layer: dict = {}
    node_sampleable: dict = {}
    node_prefix: dict = {0: ""}
    counter = [0]

    def fresh():
        counter[0] += 1
        return counter[0]

    def visit(node, parent_id, layer, prefix):
        node_layer.setdefault(parent_id, layer)
        node_sampleable.setdefault(parent_id, False)
        is_real_fork = len(node.children) >= 2
        for ch, child in node.children.items():
            label_chars = [ch]
            cur = child
            while len(cur.children) == 1 and cur.terminal == 0:
                only_ch, only_child = next(iter(cur.children.items()))
                label_chars.append(only_ch)
                cur = only_child
            cid = fresh()
            label_full = "".join(label_chars)
            new_prefix = prefix + label_full
            node_prefix[cid] = new_prefix

            if len(label_full) <= max_label_chars:
                label = label_full
            else:
                keep = max(4, (max_label_chars - 1) // 2)
                label = label_full[:keep] + "…" + label_full[-keep:]

            G.add_edge(parent_id, cid)
            edge_labels[(parent_id, cid)] = repr(label)[1:-1]  # show escapes
            edge_full[(parent_id, cid)] = label_full
            new_layer = layer + 1 if is_real_fork else layer
            node_sampleable[cid] = is_real_fork
            visit(cur, cid, new_layer, new_prefix)

    G.add_node(0)
    visit(trie.root, 0, 1, "")

    # ---- layout ----
    horiz = orientation.startswith("h")

    def hierarchy_pos(root=0, breadth=4.0, depth_gap=1.2):
        pos = {}

        def place(n, depth, lo, hi):
            mid = (lo + hi) / 2
            pos[n] = (depth * depth_gap, mid) if horiz else (mid, -depth * depth_gap)
            kids = list(G.successors(n))
            if not kids:
                return
            span = hi - lo
            step = span / len(kids)
            for i, k in enumerate(kids):
                place(k, depth + 1, lo + i * step, lo + (i + 1) * step)

        place(root, 0, -breadth / 2, breadth / 2)
        return pos

    pos = hierarchy_pos()

    # ---- styling ----
    def edge_color(v):
        lyr = node_layer[v]
        return {2: "#d62728", 3: "#2ca02c", 4: "#1f77b4"}.get(lyr, "#888888")

    edge_colors = [edge_color(v) for (_, v) in G.edges()]
    node_colors = [
        "#ffd966" if node_sampleable.get(n, False) else "#ffffff" for n in G.nodes
    ]
    node_labels = {n: ("ROOT" if n == 0 else f"L{node_layer[n]}\nn{n}") for n in G.nodes}

    # ---- draw ----
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors, width=2.5, arrows=True, arrowsize=18,
        node_size=2200, min_target_margin=18, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, edgecolors="black",
        node_size=2200, linewidths=1.5, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=13, rotate=False,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.95),
        label_pos=0.5, ax=ax,
    )

    if title is None:
        title = (
            f"Compressed prefix trie  ({trie.root.count} CoTs, "
            f"max_depth={trie.max_depth})\n"
            "gold = sampleable (parent forked) · white = forced-chain · "
            "red/green/blue = layer-2/3/4 edges"
        )
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    ax.legend(
        handles=[
            Line2D([0], [0], color="#d62728", lw=3, label="layer-2 edge"),
            Line2D([0], [0], color="#2ca02c", lw=3, label="layer-3 edge"),
            Line2D([0], [0], color="#1f77b4", lw=3, label="layer-4 edge"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffd966",
                   markeredgecolor="black", markersize=12, label="sampleable"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=12, label="forced-chain"),
        ],
        loc="lower left", frameon=False, fontsize=11,
    )

    if created_fig:
        plt.tight_layout()

    # ---- optional textual legend (node id → full prefix) ----
    if show_legend_table:
        print(f"\nNode id → full prefix string  (gold = sampleable):")
        for n in sorted(G.nodes):
            if n == 0:
                continue
            marker = "★" if node_sampleable.get(n, False) else " "
            print(f"  {marker} n{n:<2d} (L{node_layer[n]}):  {node_prefix[n]!r}")

    return ax
