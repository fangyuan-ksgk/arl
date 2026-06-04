"""Optimistic Prefix Advantage (OPA) — tree-structured credit assignment for GRPO.

Idea
----
GRPO gives every rollout a single scalar advantage

    a_i = (r_i - mean(r)) / (std(r) + eps)        # group z-score

and pays it flat across all of that rollout's tokens. OPA instead reuses the
fact that rollouts in a GRPO group share token *prefixes*. Build a prefix trie
over the group's completions; each node (prefix) is scored by its single best
reachable continuation

    A*(prefix) = max_i a_i  over rollouts whose path includes that prefix

(an *optimistic* backup, the per-prefix analogue of V*). Token t of a rollout
then receives the advantage of the node reached after emitting tokens[:t+1].
Shared prefixes therefore inherit the credit of the best continuation that is
still reachable from them, rather than the rollout's own flat outcome.

Public API
----------
optimistic_prefix_advantages(token_seqs, scalar_advs)
    Pure, model-free core. Testable directly in a notebook on toy sequences.

TreeTrainer(GRPOTrainer)
    Drop-in GRPO trainer that rewrites ``inputs["advantages"]`` to the per-token
    OPA signal inside ``_compute_loss``. Builds a per-batch prefix trie (grouped
    by prompt) and, optionally, a persistent global trie across batches.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, List, Optional, Sequence


__all__ = ["optimistic_prefix_advantages", "PrefixTrie", "TreeTrainer"]


# ---------------------------------------------------------------------------
# Core: token-id prefix trie with optimistic (max) advantage backup
# ---------------------------------------------------------------------------
class PrefixTrie:
    """Lightweight prefix trie keyed on hashable tokens (e.g. token ids).

    Each node tracks ``a_max`` = the maximum scalar advantage over every
    sequence whose path passes through it (the Optimistic Prefix Advantage,
    A*). ``a_max`` is a running max, so the same trie can be updated
    incrementally across batches to act as a persistent global trie.
    """

    __slots__ = ("children", "a_max")

    def __init__(self):
        self.children: dict = {}
        self.a_max: float = float("-inf")

    def insert(self, toks: Sequence[Hashable], adv: float) -> None:
        adv = float(adv)
        node = self
        if adv > node.a_max:
            node.a_max = adv
        for t in toks:
            node = node.children.setdefault(t, PrefixTrie())
            if adv > node.a_max:
                node.a_max = adv

    def walk_amax(self, toks: Sequence[Hashable]) -> List[float]:
        """A* at each prefix: position t is A*(tokens[:t+1]).

        Stops early (truncates) if a prefix is absent from the trie. When the
        sequence was inserted into this trie, the walk is always complete.
        """
        out: List[float] = []
        node = self
        for t in toks:
            node = node.children.get(t)
            if node is None:
                break
            out.append(node.a_max)
        return out

    def token_node_breakdown(self) -> dict:
        """Classify every non-root node (each = one token in the dedup'd trie).
        """
        shared = nonshared = leaf = 0

        leaves: dict = {}
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if not node.children:
                leaves[id(node)] = 1
                continue
            if not processed:
                stack.append((node, True))
                for child in node.children.values():
                    stack.append((child, False))
                continue
            n_leaves = 0
            for child in node.children.values():
                child_leaves = leaves[id(child)]
                if not child.children:
                    leaf += 1
                elif child_leaves >= 2:
                    shared += 1
                else:
                    nonshared += 1
                n_leaves += child_leaves
            leaves[id(node)] = n_leaves

        return {
            "shared": shared,
            "nonshared": nonshared,
            "leaf": leaf,
            "total": shared + nonshared + leaf,
        }

    @property
    def shared_prefix_token_fraction(self) -> float:
        """Fraction of tokens in the trie that live in a shared prefix node
        (a node traversed by >=2 rollouts). Returns 0.0 for an empty trie."""
        b = self.token_node_breakdown()
        return b["shared"] / b["total"] if b["total"] else 0.0


def optimistic_prefix_advantages(
    token_seqs: Sequence[Sequence[Hashable]],
    scalar_advs: Sequence[float],
    return_trie: bool = False,
):
    """Per-token Optimistic Prefix Advantage for one GRPO group.

    Parameters
    ----------
    token_seqs
        Sequences (lists/tuples of hashable tokens) that **share a prompt**,
        i.e. one GRPO group. Prefix sharing is only meaningful within a group.
    scalar_advs
        Per-sequence scalar advantage ``a_i`` (e.g. the GRPO group z-score),
        one per sequence in ``token_seqs``.
    return_trie
        When True, also return the built :class:`PrefixTrie` (whose every node
        carries its A* in ``a_max``) for inspection/visualization.

    Returns
    -------
    ``per_token`` (list of per-token advantage lists, ragged-aligned with
    ``token_seqs``; position ``t`` holds ``A*(tokens[:t+1]) = max a_j`` over all
    sequences whose prefix matches ``tokens[:t+1]``). If ``return_trie`` is True,
    returns ``(per_token, trie)`` instead.
    """
    if len(token_seqs) != len(scalar_advs):
        raise ValueError("token_seqs and scalar_advs must have equal length")
    root = PrefixTrie()
    for toks, a in zip(token_seqs, scalar_advs):
        root.insert(toks, a)
    per_token = [root.walk_amax(toks) for toks in token_seqs]
    if return_trie:
        return per_token, root
    return per_token


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
try:  # keep the OPA core importable even without trl installed
    import torch
    from trl import GRPOTrainer
    _HAS_TRL = True
except Exception:  # pragma: no cover - notebook may import the core only
    GRPOTrainer = object  # type: ignore
    _HAS_TRL = False


class TreeTrainer(GRPOTrainer):  # type: ignore[misc]
    """GRPO with Optimistic Prefix Advantage credit assignment.

    Inherits the full GRPOTrainer pipeline (vllm/sglang rollouts, reward
    computation, group-normalized advantages, KL, multi-iteration) and only
    rewrites the advantage that GRPO pays per token:

        scalar GRPO advantage  a_i = (r_i - mean) / (std + eps)   [from TRL]
        per-token OPA          A*(tokens[:t+1])                   [this class]

    Rollouts are grouped by prompt (token-id tuple); a prefix trie is built per
    group and token ``t`` is credited with the best reachable continuation's
    advantage. With ``use_global_tree=True`` a persistent trie is kept across
    batches and used instead, so a prefix inherits the best continuation ever
    seen from it (memory grows with the number of distinct prefixes).

    Usage is identical to GRPOTrainer::

        trainer = TreeTrainer(model=..., reward_funcs=[...], args=cfg, ...)
        trainer.train()
    """

    def __init__(self, *args, use_global_tree: bool = False, **kwargs):
        if not _HAS_TRL:
            raise ImportError("TreeTrainer requires `trl` (and torch) to be installed")
        super().__init__(*args, **kwargs)
        self.use_global_tree = use_global_tree
        # prompt-key -> PrefixTrie, persisted across batches when enabled.
        self._global_tries: dict = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _tree_token_advantages(
        prompt_ids,
        completion_ids,
        completion_mask,
        adv_scalar,
        pad_id: int,
        *,
        use_global_tree: bool = False,
        global_tries: Optional[dict] = None,
    ):
        """The credit-assignment core of :meth:`_compute_loss`.

        Turns the scalar GRPO advantages ``a_i`` (Bp,) into the per-token OPA
        advantage tensor (Bp, T) that GRPO then pays at every completion token.
        Pure tensor-in / tensor-out — needs no model or trainer state beyond the
        optional persistent ``global_tries`` dict — so it can be exercised
        directly (e.g. in a notebook) without instantiating a Trainer.

        Steps:
          1. strip padding -> real completion token-id sequences,
          2. group rollouts by prompt (prefixes only shared within a prompt),
          3. per group, A*(prefix) via :func:`optimistic_prefix_advantages`
             (or a persistent global trie when ``use_global_tree``),
          4. scatter per-token A* back into a (Bp, T) tensor, masked.
        """
        Bp, T = completion_ids.shape
        mask_b = completion_mask.bool()
        a_list = [float(x) for x in adv_scalar.tolist()]

        # (1) Real completion tokens per rollout (padding stripped).
        seqs: List[List[int]] = [
            [int(t) for t in completion_ids[i][mask_b[i]].tolist()]
            for i in range(Bp)
        ]

        # (2) Group rollouts by prompt.
        def _pkey(row):
            return tuple(int(x) for x in row.tolist() if x != pad_id)

        groups: dict = defaultdict(list)
        for i in range(Bp):
            groups[_pkey(prompt_ids[i])].append(i)

        adv_token = torch.zeros_like(completion_mask, dtype=adv_scalar.dtype)
        for pkey, idxs in groups.items():
            g_seqs = [seqs[i] for i in idxs]
            g_advs = [a_list[i] for i in idxs]

            # (3) A* per prefix.
            if use_global_tree:
                if global_tries is None:
                    global_tries = {}
                trie = global_tries.setdefault(pkey, PrefixTrie())
                for toks, a in zip(g_seqs, g_advs):
                    trie.insert(toks, a)
                per_tok = [trie.walk_amax(toks) for toks in g_seqs]
            else:
                per_tok = optimistic_prefix_advantages(g_seqs, g_advs)

            # (4) Scatter back into the (Bp, T) tensor.
            for i, vals in zip(idxs, per_tok):
                n = len(vals)
                if n:
                    adv_token[i, :n] = torch.tensor(
                        vals, dtype=adv_scalar.dtype, device=adv_token.device
                    )
        return adv_token * completion_mask.to(adv_token.dtype)

    def _compute_loss(self, model, inputs):
        # Eval guard: prediction loops run under no_grad; don't touch the
        # global trie or rewrite advantages on throwaway losses.
        if not torch.is_grad_enabled():
            return super()._compute_loss(model, inputs)

        adv_scalar = inputs.get("advantages")
        prompt_ids = inputs.get("prompt_ids")
        completion_ids = inputs.get("completion_ids")
        mask = inputs.get("completion_mask")
        if adv_scalar is None or completion_ids is None or mask is None:
            return super()._compute_loss(model, inputs)

        # TRL advantages may be (Bp,) scalar or already (Bp, T); we only
        # rewrite the scalar case. A 2D advantage means another override ran.
        if adv_scalar.dim() != 1:
            return super()._compute_loss(model, inputs)

        tok = self.processing_class
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with torch.no_grad():
            adv_token = self._tree_token_advantages(
                prompt_ids, completion_ids, mask, adv_scalar, pad_id,
                use_global_tree=self.use_global_tree,
                global_tries=self._global_tries,
            )

        inputs = dict(inputs)
        inputs["advantages"] = adv_token
        return super()._compute_loss(model, inputs)