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
import json
import os
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np


__all__ = ["optimistic_prefix_advantages", "PrefixTrie", "TreeTrainer"]

# Synthetic reward key recorded on every global-trie node: the total scalar
# reward (sum over reward funcs) of each rollout, so a buffered group baseline
# (mean/std) can be read straight off a prefix node (see _buffered_advantages).
_TOTAL_KEY = "_total_reward"


# ---------------------------------------------------------------------------
# Core: token-id prefix trie with optimistic (max) advantage backup
# ---------------------------------------------------------------------------
def _trie_deep_call(fn, _stack_bytes=512 * 1024 * 1024):
    """Run ``fn`` in a thread with a large C stack, a raised recursion limit, and the
    pure-Python json (en|de)coder forced on — so trie (de)serialization is NOT bounded
    by completion length. Trie depth == #tokens on the longest buffered completion,
    which can exceed (a) Python's default ~1000 recursion limit, (b) the 8 MiB thread
    stack, and (c) the C json coder's fixed C-recursion limit (which ignores
    setrecursionlimit on CPython >=3.12). Verified to round-trip a 20k-deep trie."""
    import sys as _sys, threading as _th
    from json import scanner as _js
    _sys.setrecursionlimit(max(_sys.getrecursionlimit(), 1 << 22))   # ~4.19M frames
    box = {}
    def _runner():
        _old_ms = _js.make_scanner
        _js.make_scanner = _js.py_make_scanner          # force pure-Python decoder
        try:
            box["r"] = fn()
        except BaseException as e:                       # propagate to caller
            box["e"] = e
        finally:
            _js.make_scanner = _old_ms
    prev = None
    try:
        prev = _th.stack_size(_stack_bytes)              # large per-thread C stack
    except (ValueError, RuntimeError):
        pass                                             # platform refused -> default stack
    t = _th.Thread(target=_runner); t.start(); t.join()
    if prev is not None:
        try: _th.stack_size(prev)
        except (ValueError, RuntimeError): pass
    if "e" in box:
        raise box["e"]
    return box.get("r")


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
    - ``stats`` : ``{reward_key: [n, mean, var, max]}`` over rollouts through this
                  prefix. ``mean`` / ``var`` are **EMA** estimates (exponentially
                  recency-weighted, decay ``reward_ema``) so the baseline tracks the
                  *current* policy as it improves rather than all of history. Read
                  via ``best_reward`` (optimistic max, mirroring A*), ``reward_mean``
                  / ``reward_std``. E.g. ``best_reward('correctness_reward')``.

    All extras default to empty/zero, so callers that pass only an advantage
    (e.g. :func:`optimistic_prefix_advantages`) keep the original OPA behaviour.
    """

    __slots__ = ("children", "a_max", "a_min", "count", "stats", "parent", "token")

    def __init__(self, parent: Optional["PrefixTrie"] = None,
                 token: Optional[Hashable] = None):
        self.children: dict = {}
        # Back-reference to the path: ``token`` is the edge from ``parent`` to this
        # node. Root has both None. Lets ``prefix()`` reconstruct a node's tokens.
        self.parent = parent
        self.token = token
        self.a_max: float = float("-inf")
        self.a_min: float = float("inf")
        self.count: int = 0
        # Per reward key: ``[n, mean, var, max]`` — EMA running mean/variance
        # (recency-weighted; see _accumulate) plus the optimistic max. A key may
        # appear on only a subset of the rollouts through this node.
        self.stats: Dict[Hashable, list] = {}

    def _accumulate(self, adv: float, rewards: Optional[dict],
                    reward_ema: float = 0.9) -> None:
        self.count += 1
        if adv > self.a_max:
            self.a_max = adv
        if adv < self.a_min:
            self.a_min = adv
        if rewards:
            st = self.stats
            for k, r in rewards.items():
                r = float(r)
                acc = st.get(k)
                if acc is None:
                    st[k] = [1, r, 0.0, r]      # [n, mean, var, max]; var=0 on first
                    continue
                n, m, v, mx = acc
                d = r - m
                m = m + (1 - reward_ema) * d                    # EMA mean
                v = reward_ema * (v + (1 - reward_ema) * d * d)  # EMA var (West)
                acc[0], acc[1], acc[2] = n + 1, m, v
                if r > mx:
                    acc[3] = r

    def insert(self, toks: Sequence[Hashable], adv: float,
               rewards: Optional[dict] = None, reward_ema: float = 0.9) -> None:
        """Insert one rollout. ``rewards`` is an optional ``{key: value}`` map of
        per-rollout reward components (e.g. correctness / format); each node on the
        path keeps the running max plus an EMA mean/var per key. ``reward_ema`` is
        the EMA retention factor (higher = slower adaptation; 0.9 default)."""
        adv = float(adv)
        node = self
        node._accumulate(adv, rewards, reward_ema)
        for t in toks:
            child = node.children.get(t)
            if child is None:
                child = PrefixTrie(parent=node, token=t)
                node.children[t] = child
            node = child
            node._accumulate(adv, rewards, reward_ema)

    def best_reward(self, key: Hashable, default: float = float("-inf")) -> float:
        """Max reward of ``key`` over rollouts through this prefix (optimistic)."""
        acc = self.stats.get(key)
        return acc[3] if acc is not None else default

    def reward_mean(self, key: Hashable, default: float = float("nan")) -> float:
        """Running mean reward of ``key`` over rollouts through this prefix."""
        acc = self.stats.get(key)
        return acc[1] if acc is not None else default

    def reward_std(self, key: Hashable, default: float = float("nan")) -> float:
        """EMA running std of ``key`` reward over rollouts through this prefix
        (square root of the EMA variance). Returns ``default`` if unseen."""
        acc = self.stats.get(key)
        if acc is None:
            return default
        return acc[2] ** 0.5

    # --- (de)serialization: JSON-safe, drops parent/token (rebuilt on load) ---
    @staticmethod
    def _enc_f(x: float):
        if x == float("inf"):
            return "inf"
        if x == float("-inf"):
            return "-inf"
        return x

    @staticmethod
    def _dec_f(x):
        if x == "inf":
            return float("inf")
        if x == "-inf":
            return float("-inf")
        return float(x)

    def to_dict(self) -> dict:
        """Recursively serialize into a JSON-safe dict. ``parent``/``token`` are
        dropped (reconstructed by :meth:`from_dict`); ``±inf`` extrema are encoded
        as strings. ``children`` is a list of ``[token, child_dict]`` pairs so
        non-string token types (e.g. int token-ids) survive a JSON round-trip."""
        return {
            "a_max": self._enc_f(self.a_max),
            "a_min": self._enc_f(self.a_min),
            "count": self.count,
            "stats": {k: list(v) for k, v in self.stats.items()},
            "children": [[k, c.to_dict()] for k, c in self.children.items()],
        }

    @classmethod
    def from_dict(cls, d: dict, parent: Optional["PrefixTrie"] = None,
                  token: Optional[Hashable] = None) -> "PrefixTrie":
        """Inverse of :meth:`to_dict`; rebuilds ``parent``/``token`` back-links."""
        node = cls(parent=parent, token=token)
        node.a_max = cls._dec_f(d["a_max"])
        node.a_min = cls._dec_f(d["a_min"])
        node.count = int(d["count"])
        node.stats = {k: list(v) for k, v in d.get("stats", {}).items()}
        for tok, cd in d.get("children", []):
            node.children[tok] = cls.from_dict(cd, parent=node, token=tok)
        return node

    def prefix(self) -> List[Hashable]:
        """Tokens from the root to this node (the prefix it represents). Root -> ``[]``.
        Reconstructed by climbing ``parent`` links, so it is O(depth)."""
        toks: List[Hashable] = []
        node = self
        while node.parent is not None:
            toks.append(node.token)
            node = node.parent
        toks.reverse()
        return toks

    def get_node(self, prefix: Sequence[Hashable]) -> Optional["PrefixTrie"]:
        """Return the node reached by following ``prefix`` from this node, or
        ``None`` if the prefix is absent. ``get_node([])`` returns ``self``."""
        node = self
        for t in prefix:
            node = node.children.get(t)
            if node is None:
                return None
        return node

    def leaves(self):
        """Yield every leaf node (no children). A leaf is the final prefix of a
        completed rollout, so the SET of leaves is the set of distinct rollouts
        stored in the trie (regardless of how many times each was inserted)."""
        stack = [self]
        while stack:
            node = stack.pop()
            if not node.children:
                yield node
            else:
                stack.extend(node.children.values())

    def sample_rollout(self, correct: bool = True,
                       key: Hashable = "correctness_reward",
                       rng=None) -> Optional[List[Hashable]]:
        """Uniformly sample one complete rollout (root -> leaf token sequence)
        whose terminal reward matches ``correct``: ``best_reward(key) > 0`` for a
        correct rollout, ``== 0`` (or unseen) for a wrong one. Sampling is over
        the SET of distinct rollouts (leaves), ignoring visit counts ``n`` so a
        frequently-resampled rollout is no likelier than a rare one. ``rng`` may
        be a ``random.Random``; defaults to the module RNG. Returns the token
        list, or ``None`` if no leaf matches."""
        import random
        rng = rng or random
        want = bool(correct)
        pool = [lf for lf in self.leaves()
                if (lf.best_reward(key) > 0) == want]
        if not pool:
            return None
        return rng.choice(pool).prefix()

    def iter_nodes(self):
        stack: List[tuple] = [([], self)]
        while stack:
            prefix, node = stack.pop()
            for tok, child in node.children.items():
                p = prefix + [tok]
                yield p, child
                stack.append((p, child))

    def prefix_score(self, key: Hashable = "correctness_reward") -> Optional[float]:
        """Most difficult, Achievable Prefix recieves highest score"""
        if not self.children:
            return None
        return self.best_reward(key) * (1.0 - self.reward_mean(key)) / self.count

    def sample_prefix(self, key: Hashable = "correctness_reward",
                      rng=None) -> Optional[List[Hashable]]:
        """Sample one under-explored prefix (token list, root -> node), weighted
        by :meth:`prefix_score` over prefixes with a reachable success
        (``best_reward(key) > 0``). Returns ``None`` if no prefix qualifies
        (e.g. every success is already reliable -> all scores 0)."""
        rng = rng or random
        cands: List[tuple] = []
        for p, n in self.iter_nodes():
            if n.best_reward(key) <= 0:
                continue
            sc = n.prefix_score(key)
            if sc is not None and sc > 0:
                cands.append((p, sc))
        if not cands:
            return None
        x = rng.random() * sum(sc for _, sc in cands)
        for p, sc in cands:
            x -= sc
            if x <= 0:
                return p
        return cands[-1][0]

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
    mode: str = "base",
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
    if mode == "base":   # vanilla GRPO: same scalar advantage on every token
        per_token = [[a] * len(toks) for toks, a in zip(token_seqs, scalar_advs)]
    else:
        per_token = [root.walk(toks, mode=mode) for toks in token_seqs]
    if return_trie:
        return per_token, root
    return per_token


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
try:  # keep the OPA core importable even without trl installed
    import contextlib
    import io

    # `import trl` pulls in transformers/protobuf, which prints benign
    # "MessageFactory object has no attribute 'GetPrototype'" tracebacks to stderr
    # on some protobuf versions. Mute that import-time noise (real failures still
    # raise and are handled below); the permanent fix is pinning protobuf.
    with contextlib.redirect_stderr(io.StringIO()):
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
                 credit_mode: str = "base",
                 shaped_reward: bool = False,
                 shaped_kwargs: Optional[dict] = None,
                 difficulty_map: Optional[dict] = None,
                 virtual_rollout: Optional[str] = None,
                 virtual_max_reward: float = 1.2,
                 record_reward_keys: Sequence[str] = ("correctness_reward", "format_reward"),
                 tree_persist_path: Optional[str] = None,
                 buffered_baseline: bool = False,
                 buffered_eps: float = 1e-4,
                 inject_rollout: bool = False,
                 inject_incorrect: bool = False,
                 resample_prefix: bool = False,
                 resample_train_prefix: bool = False,
                 resample_inject: bool = False,
                 **kwargs):
        if not _HAS_TRL:
            raise ImportError("TreeTrainer requires `trl` (and torch) to be installed")
        if credit_mode not in ("base", "max", "min"):
            raise ValueError(f"credit_mode must be 'base', 'max', or 'min', got {credit_mode!r}")
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

        # --- Idea 1: cross-run persistence of the global prefix trie ----------
        self.tree_persist_path = tree_persist_path
        if tree_persist_path and os.path.exists(tree_persist_path):
            self.load_tries(tree_persist_path)
        # --- Idea 3: buffered (cross-batch) group baseline for advantages -----
        self.buffered_baseline = bool(buffered_baseline)
        self.buffered_eps = float(buffered_eps)
        # --- repair degenerate groups with a REAL buffered rollout -------------
        # inject_rollout: degenerate group -> inject buffered CORRECT rollout
        # (buffer's best total reward). inject_incorrect additionally repairs
        # all-correct groups with a buffered INCORRECT rollout (zero reward).
        self.inject_rollout = bool(inject_rollout)
        self.inject_incorrect = bool(inject_incorrect)
        # --- Idea 4: resample all-correct groups from a buffered prefix --------
        self.resample_prefix = bool(resample_prefix)
        # True  -> forced prefix tokens get gradient (advantage-weighted BC of
        #          the prefix; immediate lift of p(prefix), off-policy).
        # False -> prefix is context only: attended (completion_mask=1) but
        #          excluded from the loss via TRL's tool_mask (loss_mask =
        #          completion_mask * tool_mask); only the on-policy
        #          continuation is trained.
        self.resample_train_prefix = bool(resample_train_prefix)
        # resample_inject: reserve ONE random slot of each resampled group for
        # a buffered CORRECT rollout (full rollout from the prompt, no forced
        # prefix) — guarantees contrast even if all g forced continuations
        # come back wrong. Scored by the same reward funcs as the rest.
        self.resample_inject = bool(resample_inject)
        # ``use_global_tree`` ONLY controls the credit source in _compute_loss
        # (per-token A* read from the persistent trie instead of a per-batch
        # trie). Buffer maintenance is a separate, internal concern: the trie is
        # ingested whenever ANY consumer needs it, without changing the credit
        # assignment mode.
        self._use_buffer = (self.use_global_tree or self.buffered_baseline
                            or self.inject_rollout or self.resample_prefix)

    # ------------------------------------------------------------------
    # Idea 1: persistence helpers
    # ------------------------------------------------------------------
    def save_tries(self, path: Optional[str] = None) -> None:
        """Serialize every per-prompt global trie to JSON. Only the main process
        writes. Prompt keys (int token-id tuples) are stored as lists."""
        path = path or self.tree_persist_path
        if not path:
            return
        if getattr(self, "accelerator", None) is not None and not self.accelerator.is_main_process:
            return
        def _do_save():
            payload = {
                "version": 1,
                "tries": [[list(pkey), trie.to_dict()] for pkey, trie in self._global_tries.items()],
            }
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)            # json.dump -> pure-Python iterencode (respects limit)
            os.replace(tmp, path)   # atomic
        _trie_deep_call(_do_save)   # unbounded by completion length (trie depth ~ #tokens)

    def load_tries(self, path: Optional[str] = None) -> None:
        """Load per-prompt global tries written by :meth:`save_tries`."""
        path = path or self.tree_persist_path
        if not path or not os.path.exists(path):
            return
        def _do_load():
            with open(path) as f:
                payload = json.JSONDecoder().decode(f.read())   # fresh decoder -> pure-Python scanner
            return {tuple(pkey): PrefixTrie.from_dict(d) for pkey, d in payload.get("tries", [])}
        self._global_tries = _trie_deep_call(_do_load)

    def _save_checkpoint(self, *args, **kwargs):
        # Persist the trie alongside every model checkpoint.
        out = super()._save_checkpoint(*args, **kwargs)
        try:
            self.save_tries()
        except Exception:
            pass   # persistence must never break checkpointing
        return out

    def _pad_id(self) -> int:
        tok = self.processing_class
        return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @staticmethod
    def _row_pkey(prompt_row, pad_id: int) -> tuple:
        return tuple(int(x) for x in prompt_row.tolist() if x != pad_id)

    # ------------------------------------------------------------------
    # Absorb phase: replay stitched healthy groups BEFORE trainer.train()
    # ------------------------------------------------------------------
    def stitch_healthy_groups(self, groups_per_query: int = 1, rng=None,
                              n_pos: int = 1, n_neg: int = 1):
        """Stitch healthy GRPO groups from the buffered trie dict.
           取出全部 healthy groups 每个 query 有固定数目的 group, 计算 advantage
           配比：n_pos 胜 + n_neg 败 + 其余随机
        """
        rng = rng or random
        g = self.num_generations
        assert 1 <= n_pos and 1 <= n_neg and n_pos + n_neg <= g, \
            f"need n_pos>=1, n_neg>=1, n_pos+n_neg<=num_generations ({g}); " \
            f"got n_pos={n_pos}, n_neg={n_neg}"
        groups = []
        for pkey, trie in self._global_tries.items():
            leaves = list(trie.leaves())
            pos = [lf for lf in leaves if lf.best_reward("correctness_reward") > 0]
            neg = [lf for lf in leaves if not (lf.best_reward("correctness_reward") > 0)]
            if not pos or not neg:
                continue
            for _ in range(groups_per_query):
                picks = [rng.choice(pos) for _ in range(n_pos)]
                picks += [rng.choice(neg) for _ in range(n_neg)]
                picks += [rng.choice(leaves) for _ in range(g - n_pos - n_neg)]
                rng.shuffle(picks)
                r = torch.tensor([lf.best_reward(_TOTAL_KEY) for lf in picks],
                                 dtype=torch.float32)
                assert torch.isfinite(r).all(), \
                    f"buffered leaf missing _TOTAL_KEY reward for a healthy prompt"
                adv = (r - r.mean()) / (r.std() + self.buffered_eps)
                groups.append((pkey, [lf.prefix() for lf in picks], adv.tolist()))
        return groups

    def absorb_buffer(self, steps: int = 10, groups_per_query: int = 1,
                      n_pos: int = 1, n_neg: int = 1) -> None:
        """Train on ALL stitched healthy groups in exactly ``steps`` gradient
        updates, before ``trainer.train()`` starts.

        The groups are packed into ``steps`` chunks (effective gradient
        accumulation = ceil(n_groups / steps) groups per update, derived — not
        configured). One forward per group (prompt is uniform within a group,
        so no prompt padding); loss is the ratio-1 GRPO objective, i.e. plain
        advantage-weighted log-likelihood over completion tokens.

        Buffered rollouts are off-policy tokens with no stored logps, so this
        must run BEFORE any on-policy update (ratio == 1 only on the first
        pass). Uses the trainer's own optimizer (created here, reused by
        ``train()``). Multi-rank: the stitching rng is seeded with
        ``args.seed``, so every rank performs identical replicated updates and
        stays in sync without gradient reduction. vLLM picks up the absorbed
        weights at the first generation (global_step != _last_loaded_step).
        """
        groups = self.stitch_healthy_groups(
            groups_per_query, rng=random.Random(self.args.seed),
            n_pos=n_pos, n_neg=n_neg)
        if not groups:
            print("[absorb] no healthy groups in buffer — skipped", flush=True)
            return
        random.Random(self.args.seed).shuffle(groups) # -> redundant but fine
        steps = min(int(steps), len(groups))
        chunks = [groups[i::steps] for i in range(steps)]
        print(f"[absorb] {len(groups)} healthy groups "
              f"({groups_per_query}/query, group size {self.num_generations}) "
              f"-> {steps} updates ({len(chunks[0])} groups/update)", flush=True)
        self.create_optimizer()
        opt = self.optimizer
        device = self.accelerator.device
        pad_id = self._pad_id()
        width_cap = int(self.args.max_completion_length)
        model = self.model
        model.train()
        # micro-batch the per-group forward+backward to per_device_train_batch_size, so the
        # BACKWARD batch is mb (not the whole 8-seq group). The old code backprop'd the entire
        # group in ONE backward -> it retained the full (group x T x vocab) logits graph and
        # OOM'd at long T (normal training keeps its backward batch at per_device_train_batch_size
        # with grad accumulation). Mathematically identical: the group loss is a token-mean, so
        # normalize each slice by the group's total completion tokens.
        mb = max(1, int(self.args.per_device_train_batch_size))
        for chunk in chunks:
            opt.zero_grad(set_to_none=True)
            for pkey, seqs, advs in chunk:
                seqs = [list(s)[:width_cap] for s in seqs]
                group_tokens = max(sum(len(s) for s in seqs), 1)
                for i in range(0, len(seqs), mb):
                    sub, sub_adv = seqs[i:i + mb], advs[i:i + mb]
                    T = max(len(s) for s in sub)
                    B = len(sub)
                    prompt = torch.tensor(pkey, dtype=torch.long,
                                          device=device).unsqueeze(0).expand(B, -1)
                    cids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
                    cmask = torch.zeros((B, T), dtype=torch.long, device=device)
                    for j, s in enumerate(sub):
                        cids[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                        cmask[j, :len(s)] = 1
                    ids = torch.cat([prompt, cids], dim=1)
                    attn = torch.cat([torch.ones_like(prompt), cmask], dim=1)
                    with self.accelerator.autocast():
                        logps, _ = self._get_per_token_logps_and_entropies(
                            model, ids, attn, T)
                        adv_t = torch.tensor(sub_adv, device=device,
                                             dtype=logps.dtype).unsqueeze(1)
                        loss = -(adv_t * logps * cmask).sum() / group_tokens / len(chunk)
                        if self.beta != 0.0:
                            assert self.ref_model is not None, \
                                "beta != 0 needs self.ref_model (PEFT ref path not wired here)"
                            with torch.no_grad():
                                ref_logps, _ = self._get_per_token_logps_and_entropies(
                                    self.ref_model, ids, attn, T)
                            kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
                            loss = loss + self.beta * (kl * cmask).sum() / group_tokens / len(chunk)
                    loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)

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
        update_tree: bool = True,
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

        # (2) Group rollouts by prompt.
        groups: dict = defaultdict(list)
        for i in range(Bp):
            groups[TreeTrainer._row_pkey(prompt_ids[i], pad_id)].append(i)

        adv_token = torch.zeros_like(completion_mask, dtype=adv_scalar.dtype)
        for pkey, idxs in groups.items():
            g_seqs = [seqs[i] for i in idxs]
            g_advs = [a_list[i] for i in idxs]
            g_rew = [rew_rows[i] for i in idxs] if rew_rows is not None else None

            # (3) A* per prefix. 'base' never reads the trie (no redistribution).
            if use_global_tree and credit_mode != "base":
                if global_tries is None:
                    global_tries = {}
                trie = global_tries.setdefault(pkey, PrefixTrie())
                # ``update_tree`` is False when the trie was already populated at
                # generation time (see _update_global_tree): walking again here
                # without re-inserting avoids double-counting across the
                # num_iterations inner GRPO updates that reuse one batch.
                if update_tree:
                    for j, (toks, a) in enumerate(zip(g_seqs, g_advs)):
                        trie.insert(toks, a, rewards=(g_rew[j] if g_rew is not None else None))
                if trie.count == 0:
                    # Buffer empty (e.g. ingestion skipped): fall back to a fresh
                    # per-group OPA so credit is never silently zeroed.
                    per_tok = optimistic_prefix_advantages(
                        g_seqs, g_advs, mode=credit_mode, rewards=g_rew
                    )
                else:
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

        # Per-rollout reward components stashed on the generation batch (see
        # _generate_and_score_completions). They were shuffled+split by TRL
        # alongside completion_ids, so they stay row-aligned here.
        inputs = copy.copy(inputs)  # shallow copy: isolate our pop/set without coercing type
        reward_components = {}
        for key in self._opa_reward_keys:
            t = inputs.pop(key, None)   # pop: keep our reward columns out of TRL's loss
            if t is not None:
                reward_components[key] = t

        pad_id = self._pad_id()
        with torch.no_grad():
            adv_token = self._tree_token_advantages(
                prompt_ids, completion_ids, mask, adv_scalar, pad_id,
                use_global_tree=self.use_global_tree,
                global_tries=self._global_tries,
                credit_mode=self.credit_mode,
                reward_components=(reward_components or None),
                update_tree=not self.use_global_tree,
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

    def _hook(self, out, fn, *args):
        """Run one advantage hook; a non-None return replaces out['advantages'].
        Exceptions are swallowed: hooks must never break training."""
        try:
            new = fn(out, *args)
        except Exception:
            return
        if new is not None:
            out["advantages"] = new

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        local = self._local_rewards_per_func(out)
        self._attach_reward(out, local)

        if self.shaped_reward:
            self._hook(out, self._shaped_advantages, inputs, local)
        if self.virtual_rollout and self.model.training:
            self._hook(out, self._virtual_rollout_advantages, local)

        if not self.model.training:
            return out

        pkeys = None
        if self._use_buffer:
            pids, pad_id = out["prompt_ids"], self._pad_id()
            pkeys = [self._row_pkey(pids[i], pad_id) for i in range(pids.shape[0])]
            self._update_global_tree(out, local, pkeys) # update global tree

        # Group repair first: both rewrite tokens + rewards (``local`` and the
        # out columns) and leave a plain LOCAL group z-score advantage.
        # treatment for degenerate group using a REAL rollout from the buffer
        # (as opposed to virtual_rollout's reward-only insertion)
        if self.inject_rollout:
            self._inject_buffer_rollouts(out, local, pkeys)
        # Idea 4: same-step resample — every all-correct group (zero learning
        # signal) is REPLACED by g continuations of an under-explored buffered
        # prefix, re-scored; the original group is discarded.
        if self.resample_prefix:
            self._resample_all_correct_groups(out, inputs, local, pkeys)

        # Buffered baseline LAST: the single advantage interceptor. Recomputes
        # every row (repaired ones included, via the updated rewards) against
        # the buffer mean/std — composition by ordering, not by embedding.
        if self.buffered_baseline:
            self._hook(out, self._buffered_advantages, local, pkeys)
        return out

    # ------------------------------------------------------------------
    # Idea 1: ingest a generation batch into the global trie (once per batch)
    # ------------------------------------------------------------------
    def _update_global_tree(self, out, local=None, pkeys=None) -> None:
        adv = out.get("advantages")
        cids = out.get("completion_ids")
        mask = out.get("completion_mask")
        if adv is None or cids is None or mask is None or pkeys is None or adv.dim() != 1:
            return
        mb = mask.bool()
        # Per-rollout reward components for node stats, plus the synthetic total
        # reward used by the buffered baseline (Idea 3).
        rew_cols = {k: out[k] for k in self._opa_reward_keys if k in out}
        total = local.sum(dim=1) if local is not None else None
        for i, pkey in enumerate(pkeys):
            toks = [int(t) for t in cids[i][mb[i]].tolist()]
            rdict = {k: float(v[i]) for k, v in rew_cols.items()}
            if total is not None:
                rdict[_TOTAL_KEY] = float(total[i])
            trie = self._global_tries.setdefault(pkey, PrefixTrie())
            trie.insert(toks, float(adv[i]), rewards=(rdict or None))

    # ------------------------------------------------------------------
    # Idea 3: re-baseline advantages with the buffered group mean/std
    # ------------------------------------------------------------------
    def _buffered_advantages(self, out, local=None, pkeys=None):
        """Rescale rewards by the BUFFERED std (stable across batches, never
        the degenerate within-group 0/eps), then re-center per group so each
        group's advantages sum to ZERO — the update stays a within-group
        contrast, never a uniform imitate-everything push."""
        adv = out.get("advantages")
        if adv is None or pkeys is None or local is None or adv.dim() != 1:
            return None
        r = local.sum(dim=1).float()
        new = adv.clone().float()
        for i, pkey in enumerate(pkeys):
            trie = self._global_tries[pkey]
            s = trie.reward_std(_TOTAL_KEY)
            assert s == s, f"_TOTAL_KEY stats missing for pkey (trie count={trie.count})"
            new[i] = r[i] / (s + self.buffered_eps)
        g = self.num_generations
        for s0 in range(0, new.shape[0], g):
            new[s0:s0 + g] -= new[s0:s0 + g].mean()
        return new.to(adv)

    # ------------------------------------------------------------------
    # repair degenerate groups with a REAL buffered rollout (token swap)
    # ------------------------------------------------------------------
    def _inject_buffer_rollouts(self, out, local=None, pkeys=None) -> None:
        """Swap ONE slot per broken group for a real rollout from the buffer:
          * degenerate (all_wrong / format_only / reward_hacking)
              -> buffered CORRECT rollout
          * all-correct (only when ``inject_incorrect``)
              -> buffered INCORRECT rollout
        The injected slot carries the per-func rewards RECORDED at the
        rollout's leaf node when it was ingested (written into ``local`` and
        the out columns so every downstream consumer sees them), and the WHOLE
        group's advantages are recomputed as the LOCAL group z-score over the
        substituted rewards. Re-baselining against the buffer is NOT done here
        — _buffered_advantages runs after this hook when Idea 1 is enabled."""
        adv = out.get("advantages")
        cids = out.get("completion_ids")
        cmask = out.get("completion_mask")
        if adv is None or cids is None or cmask is None or local is None or pkeys is None:
            return
        # The injected tokens are off-policy and carry no stored logps: they
        # only train correctly when GRPO's importance ratio is exactly 1.
        if (getattr(self, "num_iterations", 1) != 1
                or "old_per_token_logps" in out
                or "importance_sampling_ratio" in out):
            return

        correct, fmt, total = self._split_rewards(local)
        pad_id, g = self._pad_id(), self.num_generations
        r = total.clone().float()
        for s in range(0, adv.shape[0], g):
            label = self._classify_group(correct[s:s + g], fmt[s:s + g], total[s:s + g])
            if label in ("all_wrong", "format_only", "reward_hacking"):
                want_correct = True
            elif label == "all_correct" and self.inject_incorrect:
                want_correct = False
            else:
                continue                                   # mixed: nothing to repair
            trie = self._global_tries[pkeys[s]]            # one prompt per group
            toks = trie.sample_rollout(correct=want_correct, key="correctness_reward",
                                       rng=random)
            if not toks:
                continue                                   # buffer lacks that kind
            slot = s + random.randrange(g)
            self._overwrite_completion(cids, cmask, slot, toks, pad_id)
            # Per-func rewards RECORDED at the rollout's leaf node, written
            # into ``local`` + out columns so downstream consumers (incl.
            # _buffered_advantages, which runs after) see the substitution.
            node = trie.get_node(toks)
            names = self.reward_func_names
            for key in self._opa_reward_keys:
                v = float(node.best_reward(key))
                local[slot, names.index(key)] = v
                if key in out:
                    out[key][slot] = v
            r[slot] = float(node.best_reward(_TOTAL_KEY))
            # Tool-use training: the row's tool_mask described the DISCARDED
            # completion; the injected rollout is plain text, so reset to 1s.
            if "tool_mask" in out:
                out["tool_mask"][slot] = 1
            # LOCAL group z-score over the substituted rewards — nothing else.
            grp = r[s:s + g]
            adv[s:s + g] = ((grp - grp.mean()) / (grp.std() + self.buffered_eps)).to(adv)

    @staticmethod
    def _overwrite_completion(cids, cmask, i: int, toks, pad_id: int) -> None:
        """Overwrite row ``i`` of the completion tensors with token-ids ``toks``
        (truncated to the batch's completion width); mask = 1 on the new tokens,
        0 on the trailing pad."""
        width = cids.shape[1]
        toks = list(toks)[:width]
        n = len(toks)
        cids[i] = cids.new_full((width,), pad_id)
        cids[i, :n] = torch.tensor(toks, dtype=cids.dtype, device=cids.device)
        cmask[i] = cmask.new_zeros(width)
        cmask[i, :n] = 1

    def _split_rewards(self, local):
        names = self.reward_func_names
        correct = local[:, names.index("correctness_reward")]
        fmt = (local[:, names.index("format_reward")]
               if "format_reward" in names else torch.zeros_like(correct))
        return correct, fmt, local.sum(dim=1)

    def _classify_group(self, correct, fmt, total):
        """mixed, all_correct, reward_hacking, format_only, all_wrong"""
        n = len(correct)
        n_correct = int((correct > 0).sum())
        n_hack = int((fmt>0).sum())
        if 0 < n_correct < n:
            return "mixed"
        if n_correct == n:
            return "all_correct"
        if n_hack == n: 
            return "reward_hacking"
        if n_correct == 0 and n_hack > 0:
            return "format_only"
        return "all_wrong"

    # ------------------------------------------------------------------
    # Idea 4: same-step resample of all-correct groups from a buffered prefix
    # ------------------------------------------------------------------
    def _resample_all_correct_groups(self, out, inputs, local=None, pkeys=None) -> None:
        """Replace each ALL-CORRECT group (zero variance, nothing to learn) with
        fresh continuations of an under-explored buffered prefix.
        ONLY WORK for training runs without Tool Use (Need to fix, if this works)
        """
        
        adv = out.get("advantages")
        cids = out.get("completion_ids")
        cmask = out.get("completion_mask")
        if adv is None or cids is None or cmask is None or local is None or pkeys is None:
            return
        if (getattr(self, "num_iterations", 1) != 1
                or "old_per_token_logps" in out
                or "importance_sampling_ratio" in out):
            return

        correct, fmt, total = self._split_rewards(local)
        pad_id, g, tok = self._pad_id(), self.num_generations, self.processing_class
        names = self.reward_func_names

        # Collect this rank's forced groups: (row offset, trie, prefix, text).
        forced = []
        for s in range(0, adv.shape[0], g):
            label = self._classify_group(correct[s:s + g], fmt[s:s + g], total[s:s + g])
            if label != "all_correct":
                continue
            trie = self._global_tries[pkeys[s]]            # one prompt per group
            prefix = trie.sample_prefix(rng=random)
            if not prefix:
                continue                                   # no under-explored prefix
            prefix = [int(t) for t in prefix]
            text = tok.decode(list(pkeys[s]) + prefix, skip_special_tokens=False)
            forced.append((s, trie, prefix, text))

        # (1) Regenerate in ONE rank-uniform call. vLLM-server generation is a
        # collective (gather_object + broadcast, sliced by process_index *
        # len(prompts)), so EVERY rank must send an identically-shaped prompt
        # list: pad to the max forced-group count across ranks with a dummy
        # group (this rank's first prompt) and discard its output.
        n = torch.tensor([len(forced)], device=self.accelerator.device)
        M = int(self.accelerator.gather(n).max().item())
        if M == 0:
            return
        dummy_text = tok.decode(list(pkeys[0]), skip_special_tokens=False)
        texts = []
        for k in range(M):
            texts.extend([forced[k][3] if k < len(forced) else dummy_text] * g)
        # Tool-use training: go through _generate (single-turn generation +
        # _tool_call_loop), so tool calls in the forced continuation are
        # EXECUTED and their output tokens come back masked in f_tmask.
        # Without tools, _generate_single_turn is the same path minus the loop.
        # TRL 1.5.x: _generate_single_turn(prompt_ids, images, multimodal_fields)
        #   -> (completion_ids, logprobs);  _generate(prompts) -> 9-tuple.
        # Build per-row prompt TOKEN-IDS (prompt + forced prefix), mirroring `texts`.
        _pad_prompt = list(pkeys[0])
        prompt_ids_list = []
        for _k in range(M):
            _pid = (list(pkeys[forced[_k][0]]) + forced[_k][2]) if _k < len(forced) else _pad_prompt
            prompt_ids_list.extend([_pid] * g)
        if getattr(self, "tools", None):
            # tool path untested here (no tools in Game-24); 9-tuple unpack per TRL 1.5.x
            (_, f_cids, f_tmask, _, _, _, _, _, _) = super()._generate(texts)
        else:
            f_cids, _ = super()._generate_single_turn(prompt_ids_list, None, None)
            f_tmask = None

        # Gradient-free prefix: loss_mask = completion_mask * tool_mask in TRL,
        # so zeroing tool_mask on the prefix removes its gradient WITHOUT
        # touching attention (unlike zeroing completion_mask).
        tmask = out.get("tool_mask")
        if tmask is None and (f_tmask is not None or not self.resample_train_prefix):
            tmask = out["tool_mask"] = torch.ones_like(cmask)

        width = cids.shape[1]
        for k, (s, trie, prefix, _) in enumerate(forced):
            new_completions = [(prefix + [int(t) for t in c])[:width]
                               for c in f_cids[k * g:(k + 1) * g]]
            # [resample_inject] reserve one slot for a buffered CORRECT
            # rollout THROUGH THIS PREFIX: the prefix was sampled because its
            # subtree contains a success (best_reward > 0), so sample the
            # rollout from the prefix NODE — the group then contrasts "the
            # success through this prefix" against g-1 fresh attempts from the
            # same prefix. Full rollout row: fully trained (tool_mask stays 1).
            j_inj = -1
            if self.resample_inject:
                pnode = trie.get_node(prefix)
                inj = (pnode.sample_rollout(correct=True, key="correctness_reward",
                                            rng=random) if pnode is not None else None)
                if not inj:                            # robustness fallback
                    inj = trie.sample_rollout(correct=True, key="correctness_reward",
                                              rng=random)
                if inj:
                    j_inj = random.randrange(g)
                    new_completions[j_inj] = [int(t) for t in inj][:width]
            # (2) Re-score locally; write the new rewards into ``local`` + out
            # columns so downstream consumers (incl. _buffered_advantages,
            # which runs after) see the substituted group.
            rpf = self._score_completions(inputs[s:s + g], new_completions)  # (g, n_funcs)
            r_new = rpf.sum(dim=1)
            local[s:s + g] = rpf.to(local)
            for key in self._opa_reward_keys:
                if key in out:
                    out[key][s:s + g] = rpf[:, names.index(key)].to(out[key])
            # (3) LOCAL group z-score over the new rewards — nothing else;
            # buffered re-baselining is the hook's job, not this method's.
            adv[s:s + g] = ((r_new - r_new.mean())
                            / (r_new.std() + self.buffered_eps)).to(adv)
            # (4) Swap tokens in place. Any pre-existing tool_mask row
            # described the DISCARDED completion — rebuild it for the new one:
            # 1s, continuation tool-output zeros from f_tmask (tool path),
            # then prefix zeros unless the prefix should be trained.
            for j in range(g):
                self._overwrite_completion(cids, cmask, s + j, new_completions[j], pad_id)
                if tmask is not None:
                    tmask[s + j] = 1
                    if j == j_inj:
                        continue                    # full rollout: train it all
                    if f_tmask is not None:
                        fm = torch.as_tensor(f_tmask[k * g + j])
                        cont = min(len(fm), width - len(prefix))
                        tmask[s + j, len(prefix):len(prefix) + cont] = \
                            fm[:cont].to(tmask)
                    if not self.resample_train_prefix:
                        tmask[s + j, :len(prefix)] = 0
            # Forced rows have no valid sampling logps; zero them so any
            # downstream consumer (e.g. shaped_reward) sees mask-consistent 0s.
            slp = out.get("sampling_per_token_logps")
            if slp is not None:
                slp[s:s + g] = 0.0

        # KL regularization (beta != 0): ref_per_token_logps was computed by
        # TRL BEFORE this swap, so on forced rows it describes the discarded
        # completions. Recompute it for those rows against the new tokens.
        rlp = out.get("ref_per_token_logps")
        if rlp is not None and forced:
            rows = [s + j for s, *_ in forced for j in range(g)]
            pc_ids = torch.cat([out["prompt_ids"][rows], cids[rows]], dim=1)
            attn = torch.cat([out["prompt_mask"][rows], cmask[rows]], dim=1)
            with torch.no_grad():
                if self.ref_model is not None:
                    new_rlp, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, pc_ids, attn, width)
                else:
                    from trl.trainer.utils import use_adapter
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        new_rlp, _ = self._get_per_token_logps_and_entropies(
                            self.model, pc_ids, attn, width)
            rlp[rows] = new_rlp.to(rlp)

    def _score_completions(self, rows, completion_ids_list):
        """Run the raw reward funcs on decoded completions, process-locally
        (no gather — unlike TRL's _calculate_rewards). ``rows`` are the raw
        dataset rows for the group (carrying e.g. gold_answer / test_list);
        the completion text is wrapped to match the dataset's format so the
        same reward funcs work unchanged. Returns a (len(rows), n_funcs)
        tensor."""
        from trl.data_utils import is_conversational
        tok = self.processing_class
        texts = [tok.decode(c, skip_special_tokens=True) for c in completion_ids_list]
        conversational = bool(rows) and isinstance(rows[0], dict) and is_conversational(rows[0])
        completions = ([[{"role": "assistant", "content": t}] for t in texts]
                       if conversational else texts)
        prompts = [r.get("prompt") if isinstance(r, dict) else None for r in rows]
        keys = ([k for k in rows[0] if k not in ("prompt", "completion", "completion_ids")]
                if rows and isinstance(rows[0], dict) else [])
        kwargs = {k: [r[k] for r in rows] for k in keys}
        kwargs["trainer_state"] = self.state
        rpf = torch.zeros(len(rows), len(self.reward_funcs))
        for i, func in enumerate(self.reward_funcs):
            vals = func(prompts=prompts, completions=completions,
                        completion_ids=completion_ids_list, **kwargs)
            rpf[:, i] = torch.tensor([float("nan") if v is None else float(v) for v in vals])
        return rpf.nan_to_num(0.0)

    def _local_rewards_per_func(self, out):
        """Local-process slice of the gathered per-func reward tensor (rpf)"""
        rpf, adv = self._last_rewards_per_func, out.get("advantages")
        if rpf is None or adv is None:
            return None
        Bp = adv.shape[0]
        lo = self.accelerator.process_index * Bp     # same slice TRL applies
        return rpf[lo:lo + Bp]

    def _attach_reward(self, out, local):
        """Attach per-rollout reward columns ({key: (Bp,)}) onto the batch."""
        if local is None:
            return
        names = self.reward_func_names
        for key in self._opa_reward_keys:
            out[key] = local[:, names.index(key)].detach().clone()

    def _virtual_rollout_advantages(self, out, local):
        adv = out.get("advantages")
        ci = self.reward_func_names.index("correctness_reward")  # validated in __init__
        rewards = local.sum(axis=1)        # total reward = sum over all reward funcs
        corrects = (local[:, ci] == 1.0)   # correctness flag from its own column

        from .arsenal import virtual_rollout_advantages
        return virtual_rollout_advantages(
            rewards, corrects, self.num_generations,
            max_reward=self.virtual_max_reward, mode=self.virtual_rollout,
        ).to(adv)

    def _shaped_advantages(self, out, inputs, local=None):
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
        if adv is None or cmask is None or cids is None or logp is None or local is None:
            return None
        if getattr(self, "_logp_src", None) != src:
            self._logp_src = src
            print(f"[shaped-reward] logp_o source = {src}", flush=True)

        B = cids.shape[0]

        # log p_theta(o|q): sum sampling logprobs over real completion tokens.
        logp_o = (logp * cmask).sum(dim=1).detach().float().cpu().numpy()

        ci = self.reward_func_names.index("correctness_reward")
        correct = local[:, ci].detach().cpu().numpy()

        D_q = [1.0] * B          # TBD. add internal difficulty calculation logic in the future

        from .arsenal import confident_failure_rare_success
        shaped = confident_failure_rare_success(
            correct, logp_o, np.asarray(D_q, dtype=float),
            **self.shaped_kwargs,
        )
        return torch.as_tensor(shaped, dtype=adv.dtype, device=adv.device)