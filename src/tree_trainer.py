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

import copy
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np


__all__ = ["optimistic_prefix_advantages", "PrefixTrie", "TreeTrainer"]


# ---------------------------------------------------------------------------
# Core: token-id prefix trie with optimistic (max) advantage backup
# ---------------------------------------------------------------------------
class PrefixTrie:
    """Lightweight prefix trie keyed on hashable tokens (e.g. token ids).

    Each node tracks ``a_max`` and ``a_min`` = the maximum / minimum scalar
    advantage over every sequence whose path passes through it. ``a_max`` is the
    Optimistic Prefix Advantage (A*, optimistic backup); ``a_min`` is the
    pessimistic backup (worst reachable continuation). Both are running
    extrema, so the same trie can be updated incrementally across batches to act
    as a persistent global trie regardless of which credit mode is read out.

    Each node additionally records, for under-explored / high-potential prefix
    selection:

    - ``count`` : number of rollouts whose path includes this prefix
                  (the *exploration* signal — low count = under-explored).
    - ``r_max`` : ``{reward_key: max reward}`` over rollouts through this prefix
                  (the *potential* signal, e.g. ``r_max['correctness_reward']``,
                  ``r_max['format_reward']``). Optimistic (max), mirroring A*.

    All extras default to empty/zero, so callers that pass only an advantage
    (e.g. :func:`optimistic_prefix_advantages`) keep the original OPA behaviour.
    """

    __slots__ = ("children", "a_max", "a_min", "count", "r_max")

    def __init__(self):
        self.children: dict = {}
        self.a_max: float = float("-inf")
        self.a_min: float = float("inf")
        self.count: int = 0
        self.r_max: Dict[Hashable, float] = {}

    def _accumulate(self, adv: float, rewards: Optional[dict]) -> None:
        self.count += 1
        if adv > self.a_max:
            self.a_max = adv
        if adv < self.a_min:
            self.a_min = adv
        if rewards:
            rm = self.r_max
            for k, v in rewards.items():
                v = float(v)
                cur = rm.get(k)
                if cur is None or v > cur:
                    rm[k] = v

    def insert(self, toks: Sequence[Hashable], adv: float,
               rewards: Optional[dict] = None) -> None:
        """Insert one rollout. ``rewards`` is an optional ``{key: value}`` map of
        per-rollout reward components (e.g. correctness / format); each node on
        the path keeps the running max per key."""
        adv = float(adv)
        node = self
        node._accumulate(adv, rewards)
        for t in toks:
            node = node.children.setdefault(t, PrefixTrie())
            node._accumulate(adv, rewards)

    def best_reward(self, key: Hashable, default: float = float("-inf")) -> float:
        """Max reward of ``key`` over rollouts through this prefix (optimistic)."""
        return self.r_max.get(key, default)

    def walk(self, toks: Sequence[Hashable], mode: str = "max") -> List[float]:
        # This method assumes a given sequence, and it retrieves all its prefix's attributes from the Trie
        # if same prefix is cached in the Trie before. 

        """Per-prefix backup at each prefix: position t is the backup of
        ``tokens[:t+1]``. ``mode='max'`` reads the optimistic A* (``a_max``);
        ``mode='min'`` reads the pessimistic backup (``a_min``).

        Stops early (truncates) if a prefix is absent from the trie. When the
        sequence was inserted into this trie, the walk is always complete.
        """
        attr = "a_min" if mode == "min" else "a_max"
        out: List[float] = []
        node = self
        for t in toks:
            node = node.children.get(t)
            if node is None:
                break
            out.append(getattr(node, attr))
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
        b = self.token_node_breakdown()
        return b["shared"] / b["total"] if b["total"] else 0.0


def optimistic_prefix_advantages(
    token_seqs: Sequence[Sequence[Hashable]],
    scalar_advs: Sequence[float],
    return_trie: bool = False,
    mode: str = "max",
    rewards: Optional[Sequence[dict]] = None,
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
    if rewards is None:
        for toks, a in zip(token_seqs, scalar_advs):
            root.insert(toks, a)
    else:
        for toks, a, rw in zip(token_seqs, scalar_advs, rewards):
            root.insert(toks, a, rewards=rw)
    per_token = [root.walk(toks, mode=mode) for toks in token_seqs]
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

    def __init__(self, *args, use_global_tree: bool = False,
                 credit_mode: str = "max",
                 shaped_reward: bool = False,
                 shaped_kwargs: Optional[dict] = None,
                 difficulty_map: Optional[dict] = None,
                 virtual_rollout: Optional[str] = None,
                 virtual_max_reward: float = 1.2,
                 record_reward_keys: Sequence[str] = ("correctness_reward", "format_reward"),
                 **kwargs):
        if not _HAS_TRL:
            raise ImportError("TreeTrainer requires `trl` (and torch) to be installed")
        if credit_mode not in ("max", "min"):
            raise ValueError(f"credit_mode must be 'max' or 'min', got {credit_mode!r}")
        if virtual_rollout not in (None, "insert_max", "insert_min", "insert_max_min",
                                   "insert_max_all_incorrect", "insert_max_mixed"):
            raise ValueError("virtual_rollout must be None, 'insert_max', 'insert_min', "
                             "'insert_max_min', 'insert_max_all_incorrect', or "
                             f"'insert_max_mixed', got {virtual_rollout!r}")
        super().__init__(*args, **kwargs)
        self.use_global_tree = use_global_tree
        self.credit_mode = credit_mode
        # Optional no-gradient "virtual rollout" reward insertion to revive dead
        # GRPO groups. Patches the reward->advantage step: a virtual reward is
        # appended to each group's reward vector before the z-score (see
        # src/arsenal.py:virtual_rollout_advantages). None = off.
        self.virtual_rollout = virtual_rollout
        self.virtual_max_reward = float(virtual_max_reward)
        self._last_rewards_per_func = None      # stashed by _calculate_rewards
        if self.virtual_rollout:                # shape on correctness -> require it by name, fail loud now
            assert "correctness_reward" in self.reward_func_names, \
                f"--virtual-rollout expects a reward function named 'correctness_reward'; got {self.reward_func_names}"
        # Optional confident-failure / rare-success advantage shaping (arsenal).
        # When True, the scalar GRPO advantage a_i is replaced by the shaped
        # per-rollout reward BEFORE the OPA trie backup. `shaped_kwargs` may
        # carry clip_logp / pos_scale / neg_scale.
        self.shaped_reward = bool(shaped_reward)
        self.shaped_kwargs = dict(shaped_kwargs or {})
        # Optional numbers-tuple -> D_q override. When absent, D_q is taken from
        # the dataset's per-row #solutions (D_q = 1/#solutions), else 1.0.
        self.difficulty_map = dict(difficulty_map or {})
        # prompt-key -> PrefixTrie, persisted across batches when enabled.
        self._global_tries: dict = {}
        # Reward-func names recorded per trie node (r_max[key]) for under-explored
        # / high-potential prefix selection. Only names present in the actual
        # reward_funcs are stashed at generation time.
        self._opa_reward_keys = tuple(
            k for k in record_reward_keys if k in self.reward_func_names
        )

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
        credit_mode: str = "max",
        reward_components: Optional[dict] = None,
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

        # (1b) Per-rollout reward dicts ({reward_key: value}) for trie recording.
        # reward_components is {key: tensor (Bp,)} aligned with completion_ids.
        rew_rows: Optional[List[dict]] = None
        if reward_components:
            rew_rows = [
                {k: float(v[i]) for k, v in reward_components.items()}
                for i in range(Bp)
            ]

        # (2) Group rollouts by prompt. (assign )
        def _pkey(row):
            return tuple(int(x) for x in row.tolist() if x != pad_id)

        groups: dict = defaultdict(list)
        for i in range(Bp):
            groups[_pkey(prompt_ids[i])].append(i)

        adv_token = torch.zeros_like(completion_mask, dtype=adv_scalar.dtype)
        for pkey, idxs in groups.items():
            g_seqs = [seqs[i] for i in idxs]
            g_advs = [a_list[i] for i in idxs]
            g_rew = [rew_rows[i] for i in idxs] if rew_rows is not None else None

            # (3) A* per prefix. 
            if use_global_tree: # @FY, we will add (1). global mean/std buffered advantage || (2). buffer sampling with global trie (add / replace rollouts)
                if global_tries is None:
                    global_tries = {}
                trie = global_tries.setdefault(pkey, PrefixTrie())
                for j, (toks, a) in enumerate(zip(g_seqs, g_advs)):
                    trie.insert(toks, a, rewards=(g_rew[j] if g_rew is not None else None))
                per_tok = [trie.walk(toks, mode=credit_mode) for toks in g_seqs]
            else:
                per_tok = optimistic_prefix_advantages(
                    g_seqs, g_advs, mode=credit_mode, rewards=g_rew
                )

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
        # Only the scalar (Bp,) GRPO advantage path is rewritable into per-token.
        if (adv_scalar is None or completion_ids is None or mask is None
                or adv_scalar.dim() != 1):
            return super()._compute_loss(model, inputs)

        # @FY. It's possible to "intercept" and change inputs.get("completion_ids") as well as inputs.get("advantages")
        #      here, before they get sent to the model for forward computation & loss scaling
        
        # Per-rollout reward components stashed on the generation batch (see
        # _generate_and_score_completions). They were shuffled+split by TRL
        # alongside completion_ids, so they stay row-aligned here.
        inputs = copy.copy(inputs)  # shallow copy: isolate our pop/set without coercing type
        reward_components = {}
        for key in self._opa_reward_keys:
            t = inputs.pop(key, None)   # pop: keep our reward columns out of TRL's loss
            if t is not None:
                reward_components[key] = t

        tok = self.processing_class
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with torch.no_grad():
            adv_token = self._tree_token_advantages(
                prompt_ids, completion_ids, mask, adv_scalar, pad_id,
                use_global_tree=self.use_global_tree,
                global_tries=self._global_tries,
                credit_mode=self.credit_mode,
                reward_components=(reward_components or None),
            )

        inputs["advantages"] = adv_token
        return super()._compute_loss(model, inputs)

    # ------------------------------------------------------------------
    # Arsenal of Ideas
    # ------------------------------------------------------------------
    def _calculate_rewards(self, *args, **kwargs):
        # TRL computes per-function rewards (gathered across processes), we collect them
        rpf = super()._calculate_rewards(*args, **kwargs)
        self._last_rewards_per_func = rpf
        return rpf

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        self._attach_reward(out)

        if self.shaped_reward:
            try:
                shaped = self._shaped_advantages(out, inputs)
            except Exception:
                shaped = None          # shaping must never break training
            if shaped is not None:
                out["advantages"] = shaped
        if self.virtual_rollout and self.model.training:
            try:
                revived = self._virtual_rollout_advantages(out)
            except Exception:
                revived = None         # virtual-rollout must never break training
            if revived is not None:
                out["advantages"] = revived
        return out

    def _local_rewards_per_func(self, out):
        """Local-process slice of the gathered per-func reward tensor (rpf)"""
        rpf, adv = self._last_rewards_per_func, out.get("advantages")
        if rpf is None or adv is None:
            return None
        Bp = adv.shape[0]
        lo = self.accelerator.process_index * Bp     # same slice TRL applies
        return rpf[lo:lo + Bp]

    def _attach_reward(self, out):
        """Attach per-rollout reward"""
        local = self._local_rewards_per_func(out)
        if local is None:
            return
        names = self.reward_func_names
        for key in self._opa_reward_keys:
            out[key] = local[:, names.index(key)].detach().clone()

    def _virtual_rollout_advantages(self, out):
        adv = out.get("advantages")
        local = self._local_rewards_per_func(out)
        ci = self.reward_func_names.index("correctness_reward")  # validated in __init__
        rewards = local.sum(axis=1)        # total reward = sum over all reward funcs
        corrects = (local[:, ci] == 1.0)   # correctness flag from its own column

        from .arsenal import virtual_rollout_advantages
        return virtual_rollout_advantages(
            rewards, corrects, self.num_generations,
            max_reward=self.virtual_max_reward, mode=self.virtual_rollout,
        ).to(adv)

    def _shaped_advantages(self, out, inputs):
        """Encourage rare success, Penalize confident failure

        Ingredients (all process-local, 1:1 with the generation batch):
          * correct  — any extracted candidate verifies (== correctness_reward)
          * logp_o   — sum of the vLLM sampling logprobs over the completion mask
          * D_q      — difficulty: explicit map, else 1/#solutions, else 1.0
        """
        adv   = out.get("advantages")
        cmask = out.get("completion_mask")
        cids  = out.get("completion_ids")

        # Sequence logprob source. Prefer the vLLM *sampling* logprobs: the
        # server/colocate engine returns the sampled-token logprob for free at
        # generation time (TRL pads it into `sampling_per_token_logps`, right-
        # padded with 0.0 — harmless since we mask). `old_per_token_logps` is a
        # recomputed policy forward (expensive, and frequently None), so it is
        # only a fallback. No GRPOConfig flag is needed: the vLLM path emits
        # these logprobs unconditionally.
        logp, src = out.get("sampling_per_token_logps"), "vllm_sampling"
        if logp is None:
            logp, src = out.get("old_per_token_logps"), "policy_recompute"
        if adv is None or cmask is None or cids is None or logp is None:
            return None
        if getattr(self, "_logp_src", None) != src:
            self._logp_src = src
            print(f"[shaped-reward] logp_o source = {src}", flush=True)

        B = cids.shape[0]

        # log p_theta(o|q): sum sampling logprobs over real completion tokens.
        logp_o = (logp * cmask).sum(dim=1).detach().float().cpu().numpy()

        ci = self.reward_func_names.index("correctness_reward")
        lo = self.accelerator.process_index * B
        correct = self._last_rewards_per_func[lo:lo + B, ci].detach().cpu().numpy()

        D_q = [1.0] * B          # TBD. add internal difficulty calculation logic in the future

        from .arsenal import confident_failure_rare_success
        shaped = confident_failure_rare_success(
            correct, logp_o, np.asarray(D_q, dtype=float),
            **self.shaped_kwargs,
        )
        return torch.as_tensor(shaped, dtype=adv.dtype, device=adv.device)