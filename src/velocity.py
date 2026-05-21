"""Decoding-velocity reward (vectorized).

Definitions
-----------
Given a prompt ``q``, a chain-of-thought ``cot = o_0 … o_{T-1}`` produced by
the policy, and a reference answer ``a``, the *decoding velocity* at CoT
position ``t`` (1 ≤ t ≤ T) is::

    v_t  =  log p_ref(a | q, o_{<t})  -  log p_ref(a | q, o_{<t-1})

where ``p_ref`` is the (frozen) reference model (typically the pre-RL base
checkpoint). ``v_t`` measures how much CoT token ``o_{t-1}`` raised the
reference model's belief in the correct answer. The total decoding reward
is ``R_T = Σ_t v_t = log p_ref(a | q, o) - log p_ref(a | q)``.

Implementation
--------------
A naive computation requires ``T + 1`` serial forward passes per rollout
(one per ``t``). Two layers of vectorization make this tractable:

1. **Shared-prefix KV cache.** Forward ``q + o`` once per rollout, capture
   the full cache, and reuse it for every ``t`` by slicing to length
   ``Q + t``. ``log p(a_0 | q + o[:t])`` is then a free read from the
   already-computed logits at position ``Q + t - 1``; only ``a[1:]`` needs
   a second forward, batched across ``t``.
2. **fp32 reduction.** Log-softmax in fp32 even when weights are bf16,
   removing the dominant source of micro-batch round-off (~1e-1 → ~1e-5).

Forward FLOPs drop from O((Q+T)³) to O((Q+T)² · A), i.e. ~(Q+T)/A× fewer
than the rebuild-prefix-each-t baseline. ~2× wall clock at Q+T ≈ 500,
A ≈ 20 on Qwen3-1.7B.

Public API
----------
- ``compute_vt_batched(prompts, completions, references, model, tokenizer, *,
                       micro_batch_size=32, strip_answer_marker=True)``
- ``compute_cot_perplexity(prompts, completions, model, tokenizer, *,
                           micro_batch_size=4)``  — cheaper side-quest:
  one forward per rollout, returns per-CoT-token log-probability.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache


__all__ = [
    "compute_vt_batched",
    "compute_vt_prefix_cache",
    "compute_cot_perplexity",
    "compute_vt_vllm_remote",
]


@torch.no_grad()
def compute_vt_batched(
    prompts: List[str],
    completions: List[str],
    references: List[str],
    model,
    tokenizer,
    *,
    micro_batch_size: int = 64,
    chunk_size: int = 1,
    strip_answer_marker: bool = True,
) -> List[Dict[str, Any]]:
    """Vectorized decoding-velocity computation (shared-prefix KV cache).

    For each rollout, forwards ``q + o`` once to obtain the full KV cache,
    then for every ``t = 0..T`` slices the cache to length ``Q + t`` and
    forwards only ``a[1:]`` on top. ``log p(a_0 | q + o[:t])`` is read
    directly from the first-pass logits at position ``Q + t - 1`` (free).
    Forward FLOPs: O((Q+T)² · A) vs the rebuild-prefix baseline's
    O((Q+T)³) — ~2× wall clock at Q+T ≈ 500, A ≈ 20 on Qwen3-1.7B.

    Parameters
    ----------
    prompts
        Already chat-templated query strings (use
        ``tokenizer.apply_chat_template(..., tokenize=False,
        add_generation_prompt=True)`` upstream).
    completions
        Raw CoT strings produced by the policy. If ``strip_answer_marker``
        is True, anything from the first ``####`` onward is stripped — this
        matches the convention used in ``src.game24diagnostics``.
    references
        Reference answer strings. A ``"#### "`` prefix is prepended if
        absent so the conditioning text matches what the policy would emit
        at the end of a successful rollout.
    model, tokenizer
        HuggingFace ``AutoModelForCausalLM`` and matching tokenizer. The
        model must already be on its target device and in ``.eval()`` mode.
    micro_batch_size
        How many ``t`` slices to forward at once **within a rollout**.
        Tune to saturate GPU memory; lower if you hit OOM on long CoTs.
    chunk_size
        Stride at which ``t`` is sampled along the CoT (1 = every position,
        the original O(T) behavior). With ``chunk_size=k`` we evaluate
        ``v_t`` only at t ∈ {0, k, 2k, ..., T}, reducing the number of
        cache-sliced forwards by ~k. ``R_T`` (endpoint difference) is
        unchanged because t=0 and t=T are always included. The returned
        ``vt`` array has length ceil(T/k); each entry covers a window of
        ``k`` CoT tokens. For downstream resampling to a fixed grid, the
        coarser per-chunk velocities suffice. Choose k so that
        ``T/chunk_size`` ≥ ``vt_resample_pts`` to avoid loss of resolution
        in ``cumR_resampled``.

    Returns
    -------
    list of dicts, one per input rollout, each with::

        {
          "toks":         list[str]      # CoT tokens (length T)
          "vt":           np.ndarray(T,) # per-token decoding velocity
          "logps":        np.ndarray(T+1,)  # log p_ref(a | q, o_{<t})
          "R_T":          float          # logps[-1] - logps[0]
          "R_per_token":  float          # R_T / T
        }
    """
    device = next(model.parameters()).device

    # --- tokenize once per rollout -----------------------------------------
    per = []
    for q, cot, a in zip(prompts, completions, references):
        if strip_answer_marker:
            i = cot.find("####")
            cot = cot[:i].rstrip() if i >= 0 else cot
        a_text = a if a.lstrip().startswith("####") else f"#### {a}"
        q_ids = tokenizer(q,      add_special_tokens=False).input_ids
        o_ids = tokenizer(cot,    add_special_tokens=False).input_ids
        a_ids = tokenizer(a_text, add_special_tokens=False).input_ids
        per.append((q_ids, o_ids, a_ids))

    # Group rollouts by shared (q_ids, o_ids) so the (Q+T)-prefix forward
    # is computed once per unique rollout, not once per (rollout, ref) pair.
    # Callers like run_game24_deepspeed.py score one rollout against N refs
    # by repeating (prompt, completion) N times — without grouping we'd
    # redo the prefix N times. Grouping shaves ~(N-1)/N off the prefix work.
    groups: Dict[Tuple[tuple, tuple], List[Tuple[int, List[int]]]] = defaultdict(list)
    for idx, (q_ids, o_ids, a_ids) in enumerate(per):
        groups[(tuple(q_ids), tuple(o_ids))].append((idx, a_ids))

    out: List[Dict[str, Any]] = [None] * len(per)  # type: ignore

    for (q_tup, o_tup), members in groups.items():
        q_ids = list(q_tup); o_ids = list(o_tup)
        Q, T = len(q_ids), len(o_ids)
        toks = tokenizer.convert_ids_to_tokens(o_ids)

        if T == 0:
            for idx, a_ids in members:
                out[idx] = {"toks": toks, "vt": np.array([]),
                            "logps": np.array([]),
                            "R_T": float("nan"),
                            "R_per_token": float("nan")}
            continue

        # Sample t at strides of `chunk_size`; always include endpoints 0 and T
        # so R_T = logps[-1] - logps[0] is exact regardless of chunk_size.
        if chunk_size <= 1:
            t_grid_list = list(range(T + 1))
        else:
            t_grid_list = list(range(0, T + 1, chunk_size))
            if t_grid_list[-1] != T:
                t_grid_list.append(T)
        G = len(t_grid_list)
        t_grid = torch.tensor(t_grid_list, device=device, dtype=torch.long)

        # --- (1) one forward over q+o; capture logits + full cache ---------
        qo_ids = torch.tensor([q_ids + o_ids], device=device)
        res = model(qo_ids, use_cache=True)
        logp_qo = F.log_softmax(res.logits.float(), dim=-1)[0]   # (Q+T, V)
        past = res.past_key_values
        if hasattr(past, "to_legacy_cache"):                     # DynamicCache → tuple
            past = past.to_legacy_cache()
        del res

        for idx, a_ids in members:
            La = len(a_ids)
            if La == 0:
                out[idx] = {"toks": toks, "vt": np.array([]),
                            "logps": np.array([]),
                            "R_T": float("nan"),
                            "R_per_token": float("nan")}
                continue

            # log p(a_0 | q + o[:t]) for t in t_grid (positions Q+t-1).
            positions = (Q - 1) + t_grid                          # (G,)
            logp_a0 = logp_qo[positions, a_ids[0]]                # (G,)

            if La == 1:
                logps_t = logp_a0
            else:
                # --- (2) for each t in grid, forward a[:-1] with cache sliced to Q+t
                a_input  = torch.tensor(a_ids[:-1], device=device)
                a_target = torch.tensor(a_ids[1:],  device=device)
                logp_rest = torch.zeros(G, La - 1,
                                        device=device, dtype=torch.float32)

                for s in range(0, G, micro_batch_size):
                    e = min(s + micro_batch_size, G)
                    ts_tensor = t_grid[s:e]                       # (B,)
                    B = ts_tensor.numel()
                    t_max = int(ts_tensor[-1].item())
                    Cmax  = Q + t_max

                    # Per-batch valid cache length L_b = Q + t_b; mask of shape
                    # (B, Cmax) marks positions < L_b. We use it to (a) zero
                    # padding in K/V and (b) build the attention mask in one shot.
                    L_per_b = Q + ts_tensor                       # (B,)
                    pos_idx = torch.arange(Cmax, device=device)   # (Cmax,)
                    valid_mask = pos_idx.unsqueeze(0) < L_per_b.unsqueeze(1)  # (B, Cmax)

                    # Vectorized padded-cache build: one masked broadcast per
                    # layer per K/V, replacing the previous nested Python loop
                    # over (layers × batch). Each layer's k/v slice is broadcast
                    # to (B, H, Cmax, D) and gated by mask4d in a single fused
                    # multiply that materializes the result tensor — same memory
                    # traffic, ~B× fewer kernel launches.
                    padded: List[Tuple[torch.Tensor, torch.Tensor]] = []
                    for k, v in past:
                        k_slice = k[0, :, :Cmax, :]               # (H, Cmax, D) view
                        v_slice = v[0, :, :Cmax, :]
                        mask4d_k = valid_mask.unsqueeze(1).unsqueeze(-1).to(k.dtype)
                        mask4d_v = mask4d_k if v.dtype == k.dtype else \
                                   valid_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                        K = k_slice.unsqueeze(0) * mask4d_k        # (B, H, Cmax, D)
                        V = v_slice.unsqueeze(0) * mask4d_v
                        padded.append((K, V))
                    padded_cache = DynamicCache.from_legacy_cache(tuple(padded))

                    input_ids = a_input.unsqueeze(0).expand(B, -1).contiguous()
                    total = Cmax + (La - 1)
                    attn = torch.zeros((B, total), dtype=torch.long, device=device)
                    attn[:, :Cmax] = valid_mask.long()
                    attn[:, Cmax:] = 1
                    # Position ids: row b is [Q+t_b, ..., Q+t_b+La-2]
                    base = (Q + ts_tensor).unsqueeze(1)            # (B, 1)
                    offsets = torch.arange(La - 1, device=device).unsqueeze(0)
                    pos_ids = base + offsets                       # (B, La-1)

                    res2 = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        position_ids=pos_ids,
                        past_key_values=padded_cache,
                        use_cache=False,
                    )
                    lp = F.log_softmax(res2.logits.float(), dim=-1)
                    gathered = lp.gather(
                        2, a_target.view(1, -1, 1).expand(B, -1, 1)
                    ).squeeze(-1)                                   # (B, La-1)
                    logp_rest[s:e] = gathered

                logps_t = logp_a0 + logp_rest.sum(dim=1)            # (G,)

            logps = logps_t.cpu().numpy().astype(np.float64)
            vt    = logps[1:] - logps[:-1]                          # (G-1,) per-chunk velocity
            R_T   = float(logps[-1] - logps[0])
            # R_per_token is normalized by full CoT length T (not G-1), so the
            # scalar is comparable across chunk_size settings.
            R_per_token = R_T / T if T > 0 else float("nan")
            out[idx] = {"toks": toks, "vt": vt, "logps": logps,
                        "R_T": R_T, "R_per_token": R_per_token}

        # Release prefix cache before the next group; it can be sizeable
        # (Q+T tokens × n_layers × n_heads × head_dim × 2 bytes × 2 K/V).
        del past, logp_qo
    return out


# Back-compat alias: prior code called the KV-cached path explicitly.
compute_vt_prefix_cache = compute_vt_batched


@torch.no_grad()
def compute_cot_perplexity(
    prompts: List[str],
    completions: List[str],
    model,
    tokenizer,
    *,
    micro_batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Per-token log p(o_t | q, o_{<t}) for each rollout.

    Cheap relative to :func:`compute_vt_batched`: one forward pass per
    rollout instead of T+1. Useful for studying CoT-perplexity dynamics
    (does training drive the CoT toward higher or lower self-perplexity?).

    Returns
    -------
    list of dicts, one per rollout::

        {
          "toks":            list[str]
          "logp_per_token":  np.ndarray(T,)
          "mean_logp":       float
          "ppl":             float                # exp(-mean_logp)
          "delta_logp":      np.ndarray(T-1,)     # velocity of CoT-token logp
        }
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    items = []
    for q, cot in zip(prompts, completions):
        q_ids = tokenizer(q,   add_special_tokens=False).input_ids
        o_ids = tokenizer(cot, add_special_tokens=False).input_ids
        items.append((q_ids, o_ids))

    out: List[Dict[str, Any]] = [None] * len(items)
    for s in range(0, len(items), micro_batch_size):
        chunk = items[s:s + micro_batch_size]
        seqs = [q + o for q, o in chunk]
        max_len = max(len(x) for x in seqs)
        B = len(chunk)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attn      = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for b, x in enumerate(seqs):
            L = len(x)
            input_ids[b, :L] = torch.tensor(x, device=device)
            attn[b, :L] = 1
        log_probs = F.log_softmax(
            model(input_ids=input_ids, attention_mask=attn).logits.float(),
            dim=-1,
        )
        for b, (q_ids, o_ids) in enumerate(chunk):
            Lq, T = len(q_ids), len(o_ids)
            i_global = s + b
            if T == 0:
                out[i_global] = {
                    "toks": [],
                    "logp_per_token": np.array([]),
                    "mean_logp": float("nan"),
                    "ppl": float("nan"),
                    "delta_logp": np.array([]),
                }
                continue
            pos  = torch.arange(Lq - 1, Lq + T - 1, device=device)
            toks = torch.tensor(o_ids, device=device)
            lp = log_probs[b, pos, toks].cpu().numpy().astype(np.float64)
            out[i_global] = {
                "toks": tokenizer.convert_ids_to_tokens(o_ids),
                "logp_per_token": lp,
                "mean_logp": float(lp.mean()),
                "ppl": float(np.exp(-lp.mean())),
                "delta_logp": lp[1:] - lp[:-1],
            }
    return out


# ---------------------------------------------------------------------------
# Remote (vLLM-served) variant
# ---------------------------------------------------------------------------

def compute_vt_vllm_remote(
    prompts: List[str],
    completions: List[str],
    references: List[str],
    tokenizer,
    *,
    server_url: str = "http://localhost:8000",
    chunk_size: int = 1,
    strip_answer_marker: bool = True,
    max_sequences_per_request: int = 256,
    timeout: float = 600.0,
) -> List[Dict[str, Any]]:
    """Decoding-velocity reward against a remote ``trl vllm-serve`` instance.

    Functionally equivalent to :func:`compute_vt_batched` but offloads every
    forward to a running vLLM server via the ``/get_sequence_logprobs/``
    endpoint (TRL >= 1.4). No GPU memory is consumed in this process; the
    server's prefix cache automatically deduplicates the heavy
    ``q + o[:t]`` shared prefixes across all (rollout, t) requests.

    Parameters
    ----------
    prompts, completions, references
        Same semantics as :func:`compute_vt_batched`.
    tokenizer
        Same tokenizer used by the served model. Needed locally to construct
        ``prompt_token_ids`` and to compute ``prompt_length`` for each
        ``(q + o[:t]) + a`` sequence.
    server_url
        Base URL of the ``trl vllm-serve`` process (no trailing slash needed).
    chunk_size
        Stride for the ``t`` grid; identical to :func:`compute_vt_batched`.
    strip_answer_marker
        If True, truncate ``cot`` at the first ``####`` (matches the local
        implementation).
    max_sequences_per_request
        Cap on how many sequences are POSTed in a single HTTP call. The
        server batches concurrent requests internally, so this is purely a
        client-side payload-size guard. Adjust upward if your CoTs are short.
    timeout
        Request timeout in seconds.

    Returns
    -------
    list of dicts with the same keys as :func:`compute_vt_batched`:
    ``toks``, ``vt``, ``logps``, ``R_T``, ``R_per_token``.
    """
    import base64
    import requests

    server_url = server_url.rstrip("/")

    # --- tokenize once per rollout (mirrors compute_vt_batched) ----------
    per: List[Tuple[List[int], List[int], List[int]]] = []
    for q, cot, a in zip(prompts, completions, references):
        if strip_answer_marker:
            i = cot.find("####")
            cot = cot[:i].rstrip() if i >= 0 else cot
        a_text = a if a.lstrip().startswith("####") else f"#### {a}"
        q_ids = tokenizer(q,      add_special_tokens=False).input_ids
        o_ids = tokenizer(cot,    add_special_tokens=False).input_ids
        a_ids = tokenizer(a_text, add_special_tokens=False).input_ids
        per.append((q_ids, o_ids, a_ids))

    # Group by (q_ids, o_ids) so a rollout scored against multiple refs only
    # builds the (q+o[:t]) prefixes once. With prefix caching on the server
    # this matters less than for the local KV-cache path, but it still
    # reduces tokenization + HTTP payload size.
    groups: Dict[Tuple[tuple, tuple], List[Tuple[int, List[int]]]] = defaultdict(list)
    for idx, (q_ids, o_ids, a_ids) in enumerate(per):
        groups[(tuple(q_ids), tuple(o_ids))].append((idx, a_ids))

    out: List[Dict[str, Any]] = [None] * len(per)  # type: ignore

    # Build the full request manifest: one entry per (rollout, t-grid point).
    # We accumulate across groups, then flush in chunks of
    # `max_sequences_per_request` so a single huge POST doesn't stall.
    manifest: List[Dict[str, Any]] = []  # one entry = one sequence to score
    grid_lookup: Dict[Tuple[tuple, tuple], List[int]] = {}

    for (q_tup, o_tup), members in groups.items():
        q_ids = list(q_tup); o_ids = list(o_tup)
        Q, T = len(q_ids), len(o_ids)
        toks = tokenizer.convert_ids_to_tokens(o_ids)

        if T == 0:
            for idx, a_ids in members:
                out[idx] = {"toks": toks, "vt": np.array([]),
                            "logps": np.array([]),
                            "R_T": float("nan"),
                            "R_per_token": float("nan")}
            continue

        if chunk_size <= 1:
            t_grid = list(range(T + 1))
        else:
            t_grid = list(range(0, T + 1, chunk_size))
            if t_grid[-1] != T:
                t_grid.append(T)
        grid_lookup[(q_tup, o_tup)] = t_grid

        for idx, a_ids in members:
            La = len(a_ids)
            if La == 0:
                out[idx] = {"toks": toks, "vt": np.array([]),
                            "logps": np.array([]),
                            "R_T": float("nan"),
                            "R_per_token": float("nan")}
                continue
            for g, t in enumerate(t_grid):
                seq = q_ids + o_ids[:t] + a_ids
                manifest.append({
                    "rollout_idx": idx,
                    "grid_idx":    g,
                    "n_grid":      len(t_grid),
                    "toks":        toks,
                    "T":           T,
                    "sequence":    seq,
                    "prompt_length": Q + t,   # everything before `a`
                })

    if not manifest:
        return out

    # Per-rollout accumulator: list of (grid_idx, logp(a|prefix))
    rollout_logps: Dict[int, Dict[int, float]] = defaultdict(dict)
    rollout_meta:  Dict[int, Dict[str, Any]] = {}
    for entry in manifest:
        rollout_meta[entry["rollout_idx"]] = {
            "toks": entry["toks"], "T": entry["T"], "n_grid": entry["n_grid"],
        }

    # --- POST in chunks --------------------------------------------------
    for s in range(0, len(manifest), max_sequences_per_request):
        chunk = manifest[s:s + max_sequences_per_request]
        payload = {
            "sequences":      [e["sequence"]      for e in chunk],
            "prompt_lengths": [e["prompt_length"] for e in chunk],
            "top_logprobs":   1,           # we only need the actual-token track
            "temperature":    1.0,
            "response_format": "binary",
        }
        r = requests.post(f"{server_url}/get_sequence_logprobs/",
                          json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()

        shape = body["shape"]                      # [B, max_comp_len, top_k]
        B, max_comp_len, _ = shape
        comp_lengths = body["completion_lengths"]
        actual_lp = np.frombuffer(
            base64.b64decode(body["actual_logprobs_b64"]), dtype=np.float32,
        ).reshape(B, max_comp_len, 1)

        for i, e in enumerate(chunk):
            cl = comp_lengths[i]
            # log p(a | q + o[:t]) = sum of per-token actual logprobs over
            # the |a| completion positions. Padded slots beyond cl are -inf;
            # slice to cl.
            logp = float(actual_lp[i, :cl, 0].sum()) if cl > 0 else float("nan")
            rollout_logps[e["rollout_idx"]][e["grid_idx"]] = logp

    # --- assemble per-rollout outputs ------------------------------------
    for idx, grid_dict in rollout_logps.items():
        meta = rollout_meta[idx]
        n_grid = meta["n_grid"]; T = meta["T"]
        logps = np.array([grid_dict[g] for g in range(n_grid)], dtype=np.float64)
        vt = logps[1:] - logps[:-1]
        R_T = float(logps[-1] - logps[0])
        out[idx] = {
            "toks":        meta["toks"],
            "vt":          vt,
            "logps":       logps,
            "R_T":         R_T,
            "R_per_token": R_T / T if T > 0 else float("nan"),
        }
    return out
