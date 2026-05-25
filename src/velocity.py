"""Predictive-velocity (PV) reward — batched, shared-prefix KV cache.

Definitions
-----------
Given a prompt ``q``, a chain-of-thought ``o = o_0 … o_{T-1}`` and a reference
answer ``a``, the predictive velocity at chunk-end ``t_g`` is::

    v_g  =  log p(a | q + o[:t_g])  −  log p(a | q + o[:t_{g-1}])

``v_g`` is the gain in the model's belief in ``a`` contributed by CoT tokens
inside chunk ``g``. The total PV reward is
``R_T = Σ_g v_g = log p(a | q + o) − log p(a | q)``. Per-token rewards are
obtained by spreading each chunk's velocity uniformly across the tokens in
that chunk.

Public API
----------
- :func:`compute_pv_reward` — training-time, takes padded id tensors, returns
  per-token reward ``r_t`` of shape ``(B, o_max)``.
- :func:`evaluate_pv_reward` — evaluation-time, takes raw strings, returns a
  list of per-rollout dicts including ``vt``, ``logps``, ``R_T`` etc. Groups
  rollouts by ``(q, o)`` so a rollout scored against N references shares its
  prefix forward.
- :func:`build_t_grid` — uniform or random chunk-end grid.
- :class:`VelocityRewardComputer` — trainer-side wrapper with online answer
  buffer fallback for incorrect rollouts.

Naming convention (apply throughout)
-----------------------------------
- ``_raw`` · ragged Python list of token ids (per-rollout, pre-tensor)
- ``_len`` · per-rollout length, tensor ``(B,)``
- ``_max`` · batch-wide max length, Python ``int``
- ``_eff`` · per-rollout effective length AT a given chunk-end, tensor ``(B,)``
- no-suffix tensors are padded batched tensors of shape ``(B, *_max)``
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Hashable, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache


__all__ = [
    "build_t_grid",
    "pack_pv_inputs",
    "PVInputs",
    "compute_pv_reward",
    "evaluate_pv_reward",
    "VelocityRewardComputer",
]


# ---------------------------------------------------------------------------
# String → padded tensor packing for compute_pv_reward
# ---------------------------------------------------------------------------

class PVInputs(NamedTuple):
    """Padded batched inputs accepted by :func:`compute_pv_reward`.

    Splat the first five fields directly: ``compute_pv_reward(*pv[:5], ends, m)``.
    ``o_len`` / ``o_max`` are exposed so the caller can build the chunk grid.
    """
    qo_ids:  torch.Tensor   # (B, qo_max)
    qo_mask: torch.Tensor   # (B, qo_max)
    a_ids:   torch.Tensor   # (B, a_max)
    a_mask:  torch.Tensor   # (B, a_max)
    q_len:   torch.Tensor   # (B,)
    o_len:   torch.Tensor   # (B,)
    o_max:   int


def pack_pv_inputs(prompts, completions, references, tokenizer, device) -> PVInputs:
    """Tokenize ragged ``(q, o, a)`` strings, right-pad, return :class:`PVInputs`.

    Intended for tests / micro-benchmarks that want to call
    :func:`compute_pv_reward` directly. Production paths that already have
    token tensors should construct :class:`PVInputs` themselves; eval paths
    that want prefix-cache sharing across references should use
    :func:`evaluate_pv_reward` instead.
    """
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    q_raw = [tokenizer(q, add_special_tokens=False).input_ids for q in prompts]
    o_raw = [tokenizer(o, add_special_tokens=False).input_ids for o in completions]
    a_raw = [tokenizer(r, add_special_tokens=False).input_ids for r in references]
    B = len(q_raw)

    q_len = torch.tensor([len(x) for x in q_raw], device=device)
    o_len = torch.tensor([len(x) for x in o_raw], device=device)
    a_len = torch.tensor([len(x) for x in a_raw], device=device)

    qo_max = int((q_len + o_len).max())
    o_max  = int(o_len.max())
    a_max  = int(a_len.max()) if B else 0

    qo_ids  = torch.full ((B, qo_max), pad, device=device, dtype=torch.long)
    qo_mask = torch.zeros((B, qo_max),      device=device, dtype=torch.long)
    a_ids   = torch.full ((B, a_max),  pad, device=device, dtype=torch.long)
    a_mask  = torch.zeros((B, a_max),       device=device, dtype=torch.long)
    for b in range(B):
        lq, lo, la = len(q_raw[b]), len(o_raw[b]), len(a_raw[b])
        qo_ids [b, :lq]        = torch.tensor(q_raw[b], device=device)
        qo_ids [b, lq:lq + lo] = torch.tensor(o_raw[b], device=device)
        qo_mask[b, :lq + lo]   = 1
        a_ids  [b, :la]        = torch.tensor(a_raw[b], device=device)
        a_mask [b, :la]        = 1

    return PVInputs(qo_ids, qo_mask, a_ids, a_mask, q_len, o_len, o_max)


# ---------------------------------------------------------------------------
# KV-cache helpers (HF version compatibility)
# ---------------------------------------------------------------------------

def _kv_lists(past):
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(past.key_cache), list(past.value_cache)
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    return [l[0] for l in past], [l[1] for l in past]


def _make_cache(keys, vals):
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(zip(keys, vals)))
    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(keys, vals)):
        cache.update(k, v, i)
    return cache


# ---------------------------------------------------------------------------
# Chunk-end grid
# ---------------------------------------------------------------------------

def build_t_grid(
    o_max: int,
    *,
    strategy: str = "uniform",
    n_chunks: int | None = None,
    chunk_size: int | None = None,
    rng: "np.random.Generator | None" = None,
) -> List[int]:
    """Build the chunk-end grid for PV reward.

    The grid always starts at 0 and ends at ``o_max`` so that
    ``logps[-1] - logps[0] = R_T`` exactly, regardless of how the interior is
    sampled.

    Strategy
    --------
    ``"uniform"`` — equally-spaced cuts. Use ``chunk_size`` (preferred) or
    ``n_chunks``.
    ``"random"`` — fixed number of chunks, interior cut points sampled
    uniformly without replacement from ``{1, …, o_max-1}``. Requires
    ``n_chunks``.

    Edge cases
    ----------
    ``o_max == 0`` → ``[0]``. ``o_max < n_chunks`` for ``"random"`` →
    per-token grid ``list(range(o_max + 1))``.
    """
    if o_max == 0:
        return [0]

    if strategy == "uniform":
        if chunk_size is not None and chunk_size >= 1:
            grid = list(range(0, o_max + 1, chunk_size))
        elif n_chunks is not None and n_chunks >= 1:
            cs = max(1, -(-o_max // n_chunks))   # ceil-div
            grid = list(range(0, o_max + 1, cs))
        else:
            grid = list(range(o_max + 1))
        if grid[-1] != o_max:
            grid.append(o_max)
        return grid

    if strategy == "random":
        if n_chunks is None or n_chunks < 1:
            raise ValueError("strategy='random' requires n_chunks >= 1")
        if o_max <= n_chunks:
            return list(range(o_max + 1))
        if rng is None:
            rng = np.random.default_rng()
        cuts = rng.choice(np.arange(1, o_max), size=n_chunks - 1, replace=False)
        return [0] + sorted(int(x) for x in cuts) + [o_max]

    raise ValueError(f"unknown chunk_strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Core: logp(a | q + o[:o_eff]) given a precomputed prefix forward
# ---------------------------------------------------------------------------

def _pv_answer_logp(
    first: torch.Tensor,              # (B,) pre-gathered logp(a_0 | prefix)
    keys: List[torch.Tensor],         # each (B, H, qo_max, D)
    vals: List[torch.Tensor],
    qo_eff: torch.Tensor,             # (B,) per-row effective prefix length
    a_ids: torch.Tensor,              # (B, a_max)
    a_mask: torch.Tensor,             # (B, a_max)
    model,
) -> torch.Tensor:
    """logp(a | prefix[:qo_eff_b]) for each row b, shape ``(B,)``.

    ``first`` is the first-token logp gathered upstream from prefix logits
    (it's just a ``(B,)`` read; cheaper to gather it once outside than to
    pass the full ``logp_qo`` through here, which would force us to copy
    a ``(B, qo_max, V)`` tensor whenever the caller wants to subset rows).
    ``keys``/``vals`` may be ``.expand``-views over a B=1 cache when all
    rows share the same prefix.
    """
    dev   = a_ids.device
    B, a_max = a_ids.shape
    if a_max == 1:
        return first

    # Slice prefix cache to per-row length qo_eff_b, common length qo_eff_max.
    qo_eff_max = int(qo_eff.max())
    valid = torch.arange(qo_eff_max, device=dev)[None, :] < qo_eff[:, None]  # (B, qo_eff_max)
    ks = [k[:, :, :qo_eff_max] * valid[:, None, :, None].to(k.dtype) for k in keys]
    vs = [v[:, :, :qo_eff_max] * valid[:, None, :, None].to(v.dtype) for v in vals]
    cache = _make_cache(ks, vs)

    attn = torch.cat([valid.long(), a_mask[:, :-1]], dim=1)
    pos  = qo_eff[:, None] + torch.arange(a_max - 1, device=dev)[None, :]
    out  = model(input_ids=a_ids[:, :-1], attention_mask=attn,
                 position_ids=pos, past_key_values=cache, use_cache=False)
    lp   = F.log_softmax(out.logits.float(), dim=-1)
    rest = lp.gather(2, a_ids[:, 1:, None]).squeeze(-1) * a_mask[:, 1:]      # (B, a_max-1)
    return first + rest.sum(1)


def _scatter_chunks_to_tokens(
    logps: torch.Tensor,              # (B, n_ends)
    o_len: torch.Tensor,              # (B,)
    o_max: int,
    chunk_ends: List[int],
) -> torch.Tensor:
    """Spread per-chunk velocities uniformly across the tokens in each chunk.

    Returns ``r_t`` of shape ``(B, o_max)``, zero past each row's ``o_len_b``.
    Sum-invariant: ``Σ_t r_t[b] = logps[b, -1] - logps[b, 0] = R_T(b)``.
    """
    dev   = logps.device
    B, n_ends = logps.shape
    vt  = logps[:, 1:] - logps[:, :-1]                                       # (B, n_ends-1)
    idx = torch.arange(o_max, device=dev)[None, :]                           # (1, o_max)
    r_t = torch.zeros(B, o_max, device=dev)
    for g in range(n_ends - 1):
        sta = torch.clamp(torch.full_like(o_len, chunk_ends[g]),     max=o_len)
        end = torch.clamp(torch.full_like(o_len, chunk_ends[g + 1]), max=o_len)
        chunk_len = (end - sta).clamp(min=1).float()                         # (B,)
        in_chunk  = (idx >= sta[:, None]) & (idx < end[:, None])             # (B, o_max)
        r_t += (vt[:, g:g + 1] / chunk_len[:, None]) * in_chunk.float()
    return r_t


# ---------------------------------------------------------------------------
# Training-time: id tensors in, per-token reward out
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pv_reward(
    qo_ids: torch.Tensor,             # (B, qo_max) right-padded q+o ids
    qo_mask: torch.Tensor,            # (B, qo_max) 1 on real (q+o) tokens
    a_ids: torch.Tensor,              # (B, a_max)  right-padded answer ids
    a_mask: torch.Tensor,             # (B, a_max)  1 on real answer tokens
    q_len: torch.Tensor,              # (B,)        per-rollout prompt length
    chunk_ends: List[int],            # cuts in [0, o_max], shared across batch
    model,
) -> torch.Tensor:
    """Per-token PV reward, fully batched. Returns ``r_t`` of shape ``(B, o_max)``.

    Sum-invariant: ``Σ_t r_t[b]  =  logp(a | q+o[:o_len_b]) − logp(a | q)``.
    Pad columns (past each rollout's ``o_len_b``) are zero.

    One batched prefix forward over ``qo_ids`` → for each ``chunk_end``,
    slice the cache to per-row ``q_len + o_eff_b`` and forward ``a[:-1]`` to
    get the remaining answer logp. ``log p(a_0 | prefix)`` is read for free
    from the prefix logits.
    """
    dev   = qo_ids.device
    o_len = qo_mask.sum(1) - q_len                                           # (B,)
    o_max = int(o_len.max())

    # ── 1) ONE batched prefix forward over q+o ─────────────────────────────
    res        = model(qo_ids, attention_mask=qo_mask, use_cache=True)
    logp_qo    = F.log_softmax(res.logits.float(), dim=-1)                   # (B, qo_max, V)
    keys, vals = _kv_lists(res.past_key_values)

    # ── 2) For each chunk_end, compute logp(a | q+o[:o]) — batch-shrunk.
    #
    # A row's logp(a | q+o[:o_eff_b]) only changes when o_eff_b changes
    # iteration-over-iteration. For ragged o_len, short rows hit o_eff_b =
    # o_len_b early and produce identical logps for all remaining chunk_ends.
    # We skip them in the answer-forward and inherit logps[b, g] = logps[b, g-1].
    B      = qo_ids.shape[0]
    n_ends = len(chunk_ends)
    brange = torch.arange(B, device=dev)
    logps  = torch.empty(B, n_ends, device=dev)
    prev_o_eff = torch.full_like(o_len, -1)
    for g, o in enumerate(chunk_ends):
        o_eff  = torch.clamp(torch.full_like(o_len, o), max=o_len)           # (B,)
        qo_eff = q_len + o_eff                                               # (B,)
        first  = logp_qo[brange, qo_eff - 1, a_ids[:, 0]]                    # (B,) free gather

        if g == 0:
            active = torch.ones(B, dtype=torch.bool, device=dev)
        else:
            active = o_eff != prev_o_eff
            logps[:, g] = logps[:, g - 1]                                    # inactive inherit
        if active.any():
            idx = active.nonzero(as_tuple=True)[0]
            logps[idx, g] = _pv_answer_logp(
                first[idx],
                [k[idx] for k in keys], [v[idx] for v in vals],
                qo_eff[idx], a_ids[idx], a_mask[idx], model,
            )
        prev_o_eff = o_eff

    # ── 3) chunk velocities → per-token scatter (per-row effective widths) ─
    return _scatter_chunks_to_tokens(logps, o_len, o_max, chunk_ends)


# ---------------------------------------------------------------------------
# Evaluation-time: strings in, per-rollout dicts out
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_pv_reward(
    prompts: List[str],
    completions: List[str],
    references: List[str],
    model,
    tokenizer,
    *,
    chunk_strategy: str = "uniform",
    chunk_size: int | None = 1,
    n_chunks: int | None = None,
    rng: "np.random.Generator | None" = None,
    strip_answer_marker: bool = True,
) -> List[Dict[str, Any]]:
    """Score rollouts against reference answers; returns per-rollout dicts.

    For each ``(prompt, completion, reference)`` triple returns::

        {"toks":        list[str],         # CoT tokens (length o_len)
         "vt":          np.ndarray,        # (n_chunks,) per-chunk velocity
         "logps":       np.ndarray,        # (n_ends,)   logp(a | q + o[:t_g])
         "t_grid":      list[int],         # chunk-end positions
         "R_T":         float,             # logps[-1] - logps[0]
         "R_per_token": float}             # R_T / o_len

    Grouping
    --------
    Rollouts sharing the same ``(prompt, completion)`` are grouped so the
    prefix forward runs once per group; refs in the group are batched along
    the answer-forward dim via expanded views over the B=1 prefix cache.
    This is the "all-solutions" path: one rollout scored against N
    references costs 1 prefix forward + N×(chunk-by-chunk answer forwards),
    not N prefix forwards.
    """
    dev = next(model.parameters()).device
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # ── tokenize once per rollout ──────────────────────────────────────────
    per: List[Tuple[List[int], List[int], List[int]]] = []
    for q, cot, a in zip(prompts, completions, references):
        if strip_answer_marker:
            i = cot.find("####")
            cot = cot[:i] if i >= 0 else cot
        a_text = a if a.lstrip().startswith("####") else f"#### {a}"
        q_raw = tokenizer(q,      add_special_tokens=False).input_ids
        o_raw = tokenizer(cot,    add_special_tokens=False).input_ids
        a_raw = tokenizer(a_text, add_special_tokens=False).input_ids
        per.append((q_raw, o_raw, a_raw))

    # ── group by (q, o) to share the prefix forward across refs ────────────
    groups: Dict[Tuple[tuple, tuple], List[Tuple[int, List[int]]]] = defaultdict(list)
    for idx, (q_raw, o_raw, a_raw) in enumerate(per):
        groups[(tuple(q_raw), tuple(o_raw))].append((idx, a_raw))

    out: List[Dict[str, Any]] = [None] * len(per)  # type: ignore

    for (q_tup, o_tup), members in groups.items():
        q_raw_list = list(q_tup); o_raw_list = list(o_tup)
        q_len_int  = len(q_raw_list); o_len_int = len(o_raw_list)
        toks = tokenizer.convert_ids_to_tokens(o_raw_list)

        if o_len_int == 0:
            for idx, _ in members:
                out[idx] = _empty_result(toks, [0])
            continue

        chunk_ends = build_t_grid(
            o_len_int, strategy=chunk_strategy, n_chunks=n_chunks,
            chunk_size=chunk_size if chunk_strategy == "uniform" else None,
            rng=rng,
        )
        n_ends = len(chunk_ends)

        # ── single-row prefix forward; will be expanded across refs ──
        qo_ids_one = torch.tensor([q_raw_list + o_raw_list], device=dev)
        res = model(qo_ids_one, use_cache=True)
        logp_qo_one    = F.log_softmax(res.logits.float(), dim=-1)           # (1, Q+T, V)
        keys_one, vals_one = _kv_lists(res.past_key_values)                  # each (1, H, Q+T, D)
        del res

        # ── pack refs into a (B_g, a_max) batch ──
        a_raws = [m[1] for m in members]
        a_lens = [len(a) for a in a_raws]
        B_g    = len(members)
        a_max  = max(max(a_lens), 1)

        a_ids  = torch.full((B_g, a_max), pad, device=dev, dtype=torch.long)
        a_mask = torch.zeros((B_g, a_max),     device=dev, dtype=torch.long)
        for b, a in enumerate(a_raws):
            if len(a):
                a_ids [b, :len(a)] = torch.tensor(a, device=dev)
                a_mask[b, :len(a)] = 1

        # Expand the prefix to B_g rows (views over the B=1 prefix).
        logp_qo = logp_qo_one.expand(B_g, -1, -1)
        keys    = [k.expand(B_g, -1, -1, -1) for k in keys_one]
        vals    = [v.expand(B_g, -1, -1, -1) for v in vals_one]

        q_len = torch.full((B_g,), q_len_int, device=dev, dtype=torch.long)
        o_len = torch.full((B_g,), o_len_int, device=dev, dtype=torch.long)

        # ── per-chunk logps ──
        # Within a group, all rows share (q, o), so o_eff is a single scalar
        # for the whole group at each g. The "batch-shrink" check from
        # compute_pv_reward collapses to: if o_eff didn't change, reuse the
        # previous logps column for free.
        logps  = torch.empty(B_g, n_ends, device=dev)
        brange = torch.arange(B_g, device=dev)
        prev_o = -1
        for g, o in enumerate(chunk_ends):
            o_eff  = torch.clamp(torch.full_like(o_len, o), max=o_len)
            qo_eff = q_len + o_eff
            if g > 0 and o_eff[0].item() == prev_o:
                logps[:, g] = logps[:, g - 1]
                continue
            first = logp_qo[brange, qo_eff - 1, a_ids[:, 0]]
            logps[:, g] = _pv_answer_logp(first, keys, vals, qo_eff,
                                          a_ids, a_mask, model)
            prev_o = int(o_eff[0].item())

        # ── per-rollout output dicts ──
        logps_np = logps.cpu().numpy().astype(np.float64)
        for b, (idx, _) in enumerate(members):
            if a_lens[b] == 0:
                out[idx] = _empty_result(toks, list(chunk_ends))
                continue
            row   = logps_np[b]
            vt    = row[1:] - row[:-1]
            R_T   = float(row[-1] - row[0])
            out[idx] = {
                "toks":        toks,
                "vt":          vt,
                "logps":       row,
                "t_grid":      list(chunk_ends),
                "R_T":         R_T,
                "R_per_token": R_T / o_len_int,
            }

        del logp_qo_one, keys_one, vals_one
    return out


def _empty_result(toks: List[str], t_grid: List[int]) -> Dict[str, Any]:
    return {"toks": toks, "vt": np.array([]), "logps": np.array([]),
            "t_grid": t_grid, "R_T": float("nan"),
            "R_per_token": float("nan")}


# ---------------------------------------------------------------------------
# Trainer-side wrapper: pick reference answer, scatter to per-token reward
# ---------------------------------------------------------------------------

import re as _re

_DEFAULT_ANSWER_RE = _re.compile(r"####\s*(.+?)\s*$")


def _default_extract_answer(text: str) -> str | None:
    """Pull '#### <expr>' from the end of a completion. Returns None on miss."""
    m = _DEFAULT_ANSWER_RE.search(text.strip())
    return m.group(1).strip() if m else None


class VelocityRewardComputer:

    def __init__(
        self,
        buffer,
        *,
        chunk_strategy: str = "uniform",
        n_chunks: int | None = None,
        chunk_size: int | None = None,
        extract_answer=None,
        normalize_by_chunk: bool = True,
        strip_answer_marker: bool = True,
    ):
        if chunk_strategy == "uniform":
            if chunk_size is None and n_chunks is None:
                raise ValueError("uniform strategy: provide chunk_size or n_chunks")
        elif chunk_strategy == "random":
            if n_chunks is None:
                raise ValueError("random strategy: n_chunks is required")
        else:
            raise ValueError(f"unknown chunk_strategy: {chunk_strategy!r}")

        self.buffer              = buffer
        self.chunk_strategy      = chunk_strategy
        self.n_chunks            = n_chunks
        self.chunk_size          = chunk_size
        self.extract_answer      = extract_answer or _default_extract_answer
        self.normalize_by_chunk  = normalize_by_chunk
        self.strip_answer_marker = strip_answer_marker

    @torch.no_grad()
    def compute_per_token_reward(
        self,
        prompt_ids: torch.Tensor,        # (Bp, P) left- or right-padded with pad_id
        completion_ids: torch.Tensor,    # (Bp, T) right-padded with pad_id
        completion_mask: torch.Tensor,   # (Bp, T) 1 on real tokens
        *,
        model,
        tokenizer,
        query_keys: List[Hashable],
        correctness: List[bool],
        rng: "np.random.Generator | None" = None,
        record_sink: list | None = None,
    ) -> torch.Tensor:
        """Return ``r_t`` of shape ``(Bp, T)`` (already mask-applied).

        Pipeline (no string round-trip on the CoT):

        1. Per rollout: read real prompt ids (strip pad), real completion ids
           (via mask). Decide the reference answer string — own ``#### …`` if
           correct, else buffer sample. Tokenize the reference once.
        2. Optionally id-space-cut the completion before ``####`` so the
           CoT used for scoring doesn't already contain the answer.
        3. Pack into right-padded ``(qo_ids, qo_mask, a_ids, a_mask, q_len)``
           and call :func:`compute_pv_reward` once. It already returns a
           ``(B_valid, o_max)`` per-token reward (sum-invariant, chunk-
           normalized). Scatter that block back into the ``(Bp, T)`` slot.

        The chunk grid is shared across the batch (sized to ``o_max``). For
        uniform strategy this is equivalent to a per-row grid; for random it
        means rows share random cuts. See :func:`compute_pv_reward` for the
        clamping semantics that handle ragged ``o_len``.
        """
        if rng is None:
            rng = np.random.default_rng()

        Bp, T  = completion_ids.shape
        device = completion_ids.device
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        q_raws: List[List[int]] = []
        c_raws: List[List[int]] = []   # already cut to o_eff (marker-stripped)
        a_raws: List[List[int]] = []
        a_strs: List[str]       = []   # reference answer string used per rollout
        valid_idx: List[int]    = []

        for b in range(Bp):
            p_ids = [int(x) for x in prompt_ids[b].tolist() if x != pad_id]
            if not p_ids:
                continue

            m = completion_mask[b].bool()
            c_ids_full = completion_ids[b][m].tolist()
            if not c_ids_full:
                continue
            completion_str = tokenizer.decode(c_ids_full, skip_special_tokens=False)

            if correctness[b]:
                ref = self.extract_answer(completion_str)
            else:
                ref = self.buffer.sample(query_keys[b], rng=rng)
            if not ref or not ref.strip():
                continue

            # id-space marker cut: decode → find "####" → re-tokenize prefix.
            # Used only for its length; no byte-perfect round-trip required.
            if self.strip_answer_marker:
                i = completion_str.find("####")
                if i >= 0:
                    prefix_ids = tokenizer(
                        completion_str[:i], add_special_tokens=False
                    ).input_ids
                    o_eff = min(len(prefix_ids), len(c_ids_full))
                else:
                    o_eff = len(c_ids_full)
            else:
                o_eff = len(c_ids_full)
            if o_eff <= 0:
                continue

            ref_text = ref if ref.lstrip().startswith("####") else f"#### {ref}"
            a_ids_b  = tokenizer(ref_text, add_special_tokens=False).input_ids
            if not a_ids_b:
                continue

            q_raws.append(p_ids)
            c_raws.append(c_ids_full[:o_eff])
            a_raws.append(a_ids_b)
            a_strs.append(ref_text)
            valid_idx.append(b)

        r_t = torch.zeros(Bp, T, device=device, dtype=torch.float32)
        cot_mask = torch.zeros(Bp, T, device=device, dtype=torch.long)
        if not valid_idx:
            return r_t, cot_mask

        # ── pack into PV tensors directly in id space ─────────────────────
        B = len(valid_idx)
        q_len = torch.tensor([len(q) for q in q_raws], device=device)
        o_len = torch.tensor([len(c) for c in c_raws], device=device)
        a_len = torch.tensor([len(a) for a in a_raws], device=device)
        qo_max = int((q_len + o_len).max())
        o_max  = int(o_len.max())
        a_max  = int(a_len.max())

        qo_ids  = torch.full ((B, qo_max), pad_id, device=device, dtype=torch.long)
        qo_mask = torch.zeros((B, qo_max),         device=device, dtype=torch.long)
        a_ids   = torch.full ((B, a_max),  pad_id, device=device, dtype=torch.long)
        a_mask  = torch.zeros((B, a_max),          device=device, dtype=torch.long)
        for j in range(B):
            lq, lo, la = len(q_raws[j]), len(c_raws[j]), len(a_raws[j])
            qo_ids [j, :lq]        = torch.tensor(q_raws[j], device=device)
            qo_ids [j, lq:lq + lo] = torch.tensor(c_raws[j], device=device)
            qo_mask[j, :lq + lo]   = 1
            a_ids  [j, :la]        = torch.tensor(a_raws[j], device=device)
            a_mask [j, :la]        = 1

        chunk_ends = build_t_grid(
            o_max,
            strategy=self.chunk_strategy,
            n_chunks=self.n_chunks,
            chunk_size=self.chunk_size if self.chunk_strategy == "uniform" else None,
            rng=rng,
        )

        r_sub = compute_pv_reward(
            qo_ids, qo_mask, a_ids, a_mask, q_len, chunk_ends, model,
        )                                                            # (B, o_max)

        # R_T per rollout = Σ_t velocity = logp(a | q+o) − logp(a | q).
        # Captured BEFORE the normalize_by_chunk widening so it always means
        # "predictive log-prob of the reference answer given the full CoT".
        R_T_per_row = r_sub.sum(dim=1)                               # (B,)

        if not self.normalize_by_chunk:
            # compute_pv_reward always divides by chunk width to get token-level
            # rewards summing to R_T. Recover raw per-chunk values by undoing
            # that division: multiply each token by its chunk width.
            idx = torch.arange(o_max, device=device)[None, :]        # (1, o_max)
            for g in range(len(chunk_ends) - 1):
                sta = torch.clamp(torch.full_like(o_len, chunk_ends[g]),     max=o_len)
                end = torch.clamp(torch.full_like(o_len, chunk_ends[g + 1]), max=o_len)
                width    = (end - sta).clamp(min=1).float()
                in_chunk = (idx >= sta[:, None]) & (idx < end[:, None])
                r_sub = r_sub + (width[:, None] - 1.0) * in_chunk.float() * r_sub

        # ── place CoT rewards into (Bp, T) slot ────────────────────────────
        idx_t = torch.tensor(valid_idx, device=device)
        r_t[idx_t, :o_max] = r_sub.to(r_t.dtype)

        # ── answer-marker positions get the rollout's R_T ──────────────────
        # Archery analogy: per-CoT-token velocity tells the policy *which
        # moves* were good; the answer tokens carry R_T = "how confident
        # the model is in the reference answer given the full CoT", i.e.
        # how close the bow was pointing to the target. Together with a
        # separate advantage pool for these tokens (see trainer), this
        # prevents the policy from being told "your own #### marker was
        # below-mean" when its CoT was actually good.
        cot_mask     = torch.zeros(Bp, T, device=device, dtype=torch.long)
        comp_lens    = completion_mask.sum(dim=1)                    # (Bp,)
        o_len_cpu    = o_len.tolist()
        comp_len_cpu = comp_lens.tolist()
        R_T_cpu      = R_T_per_row.tolist()
        for j, b in enumerate(valid_idx):
            o_eff_j    = int(o_len_cpu[j])
            comp_len_b = int(comp_len_cpu[b])
            cot_mask[b, :o_eff_j] = 1
            if comp_len_b > o_eff_j:
                r_t[b, o_eff_j:comp_len_b] = float(R_T_cpu[j])

        # ── optional per-rollout reward dump ───────────────────────────────
        # The trainer wires this when `velocity_log_path` is set on it. Each
        # record carries enough state to reconstruct the velocity reward
        # offline: tokens, per-token reward, per-chunk velocity, R_T, ref.
        if record_sink is not None:
            r_sub_cpu = r_sub.detach().cpu().tolist()
            for j, b in enumerate(valid_idx):
                o_eff_j    = int(o_len_cpu[j])
                cot_ids    = c_raws[j]
                cot_tokens = tokenizer.convert_ids_to_tokens(cot_ids)
                # per-token CoT reward (length o_eff_j); answer-marker tail
                # positions are constant R_T (omitted to keep records small).
                r_cot   = r_sub_cpu[j][:o_eff_j]
                # per-chunk velocity v_g = sum of token rewards inside chunk.
                # Sum-invariant by construction of compute_pv_reward.
                vt = []
                for g in range(len(chunk_ends) - 1):
                    sta = min(chunk_ends[g],     o_eff_j)
                    end = min(chunk_ends[g + 1], o_eff_j)
                    vt.append(float(sum(r_cot[sta:end])) if end > sta else 0.0)
                qk = query_keys[b]
                record_sink.append({
                    "row":         int(b),
                    "query_key":   list(qk) if isinstance(qk, tuple) else qk,
                    "correct":     bool(correctness[b]),
                    "ref":         a_strs[j],
                    "q_len":       int(q_len[j].item()),
                    "o_len":       int(o_eff_j),
                    "a_len":       int(a_len[j].item()),
                    "comp_len":    int(comp_len_cpu[b]),
                    "t_grid":      [int(x) for x in chunk_ends],
                    "vt":          vt,
                    "R_T":         float(R_T_cpu[j]),
                    "R_per_token": float(R_T_cpu[j] / max(1, o_eff_j)),
                    "tokens":      cot_tokens,
                    "r_cot":       [float(x) for x in r_cot],
                })

        return r_t * completion_mask.float(), cot_mask

    def update_buffer(
        self,
        query_keys: List[Hashable],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        correctness: List[bool],
        *,
        tokenizer,
    ) -> int:
        """Record accepted answers from this batch into the buffer.

        Returns the number of *newly unique* ``(query_key, answer)`` pairs
        added; existing entries just bump their count.
        """
        n_new = 0
        for b, ok in enumerate(correctness):
            if not ok:
                continue
            m = completion_mask[b].bool()
            c_ids = completion_ids[b][m].tolist()
            if not c_ids:
                continue
            completion_str = tokenizer.decode(c_ids, skip_special_tokens=False)
            ans = self.extract_answer(completion_str)
            if ans:
                n_new += int(self.buffer.add(query_keys[b], ans))
        return n_new
