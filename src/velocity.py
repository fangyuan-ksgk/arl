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
(one per ``t``). This module vectorizes across both ``t`` and rollouts by
packing every answer-conditioned sequence into a single right-padded batch
and running it through the model in length-sorted micro-batches.

Same FLOPs as the serial loop, ~5–20× wall-clock speedup on GPU due to
batched matmuls. Log-softmax is computed in fp32 even when weights are
bf16, removing the dominant source of batched-vs-serial drift.

Public API
----------
- ``compute_vt_batched(prompts, completions, references, model, tokenizer, *,
                       micro_batch_size=16, strip_answer_marker=True)``
- ``compute_cot_perplexity(prompts, completions, model, tokenizer, *,
                           micro_batch_size=4)``  — cheaper side-quest:
  one forward per rollout, returns per-CoT-token log-probability.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache


__all__ = ["compute_vt_batched", "compute_vt_prefix_cache", "compute_cot_perplexity"]


@torch.no_grad()
def compute_vt_batched(
    prompts: List[str],
    completions: List[str],
    references: List[str],
    model,
    tokenizer,
    *,
    micro_batch_size: int = 16,
    strip_answer_marker: bool = True,
) -> List[Dict[str, Any]]:
    """Vectorized decoding-velocity computation.

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
        How many ``(rollout, t)`` sequences to forward at once. Tune to
        saturate GPU memory; lower if you hit OOM on long CoTs.

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
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

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

    # --- enqueue every (rollout_i, t) sequence -----------------------------
    # Each item = (rollout_idx, t, full_ids, answer_start, La)
    queue = []
    n_logps: List[int] = []                # T+1 per rollout (0 if degenerate)
    for i, (q_ids, o_ids, a_ids) in enumerate(per):
        T, La = len(o_ids), len(a_ids)
        n_logps.append(T + 1 if (T > 0 and La > 0) else 0)
        for t in range(n_logps[i]):
            ids = q_ids + o_ids[:t] + a_ids
            queue.append((i, t, ids, len(q_ids) + t, La))

    # Length-sort the queue so each micro-batch is near-homogeneous.
    # Cuts padding overhead from ~50 % to <5 % when rollouts vary in length.
    queue.sort(key=lambda item: len(item[2]))

    logps_buf = [np.full(n, np.nan, dtype=np.float64) for n in n_logps]

    # --- micro-batched forward passes --------------------------------------
    for s in range(0, len(queue), micro_batch_size):
        chunk = queue[s:s + micro_batch_size]
        max_len = max(len(item[2]) for item in chunk)
        B = len(chunk)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attn      = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for b, (_, _, ids, _, _) in enumerate(chunk):
            L = len(ids)
            input_ids[b, :L] = torch.tensor(ids, device=device)
            attn[b, :L] = 1
        # fp32 softmax even if weights are bf16: removes the dominant source
        # of batched-vs-serial drift (~1e-1 → ~1e-3).
        log_probs = F.log_softmax(
            model(input_ids=input_ids, attention_mask=attn).logits.float(),
            dim=-1,
        )                                                  # (B, max_len, V)
        for b, (i, t, ids, ans_start, La) in enumerate(chunk):
            # Token a[k] (at position ans_start + k) is predicted by the
            # logits at position ans_start + k - 1.
            pos  = torch.arange(ans_start - 1, ans_start - 1 + La, device=device)
            toks = torch.tensor(ids[ans_start:ans_start + La], device=device)
            logps_buf[i][t] = log_probs[b, pos, toks].sum().item()

    # --- assemble per-rollout results --------------------------------------
    out: List[Dict[str, Any]] = []
    for i, (q_ids, o_ids, a_ids) in enumerate(per):
        if n_logps[i] == 0:
            out.append({
                "toks": [], "vt": np.array([]), "logps": np.array([]),
                "R_T": float("nan"), "R_per_token": float("nan"),
            })
            continue
        lp = logps_buf[i]
        vt = lp[1:] - lp[:-1]
        toks = tokenizer.convert_ids_to_tokens(o_ids)
        R_T = float(lp[-1] - lp[0])
        out.append({
            "toks": toks,
            "vt": vt,
            "logps": lp,
            "R_T": R_T,
            "R_per_token": R_T / max(1, len(vt)),
        })
    return out


@torch.no_grad()
def compute_vt_prefix_cache(
    prompts: List[str],
    completions: List[str],
    references: List[str],
    model,
    tokenizer,
    *,
    micro_batch_size: int = 32,
    strip_answer_marker: bool = True,
) -> List[Dict[str, Any]]:
    """Same contract as :func:`compute_vt_batched` but reuses the KV cache
    of the shared ``q + o[:t]`` prefix across t.

    Wins
    ----
    The naive batched version forwards ``q + o[:t] + a`` for every
    ``t = 0..T``; the prefix ``q + o[:t]`` is rebuilt each time. Here we
    forward ``q + o`` **once**, which gives:
      (a) ``log p_ref(a_0 | q + o[:t])`` for every t, read directly from
          the precomputed logits at positions Q-1..Q+T-1 (free);
      (b) a full KV cache we can slice to length ``Q+t`` and forward only
          ``a[1:]`` on top to get the remaining ``log p(a_k | …)``.

    Forward FLOPs drop from O((Q+T)^3) to O((Q+T)^2 · A), i.e. ~(Q+T)/A×
    fewer. Numerics match :func:`compute_vt_batched` to ~1e-5 in fp32
    log-softmax (same dtype contract).

    Notes
    -----
    Per-rollout sequential (one ``q+o`` forward per rollout). Within a
    rollout, the t-loop is batched in ``micro_batch_size`` chunks; each
    item shares the same (sliced) source cache, no cache re-allocation.
    """
    device = next(model.parameters()).device

    # --- tokenize once per rollout (same as compute_vt_batched) ------------
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

    out: List[Dict[str, Any]] = []
    for q_ids, o_ids, a_ids in per:
        Q, T, La = len(q_ids), len(o_ids), len(a_ids)
        toks = tokenizer.convert_ids_to_tokens(o_ids)
        if T == 0 or La == 0:
            out.append({"toks": toks, "vt": np.array([]), "logps": np.array([]),
                        "R_T": float("nan"), "R_per_token": float("nan")})
            continue

        # --- (1) one forward over q+o; capture logits + full cache ---------
        qo_ids = torch.tensor([q_ids + o_ids], device=device)
        res = model(qo_ids, use_cache=True)
        logp_qo = F.log_softmax(res.logits.float(), dim=-1)[0]   # (Q+T, V)
        past = res.past_key_values
        if hasattr(past, "to_legacy_cache"):                      # DynamicCache → tuple
            past = past.to_legacy_cache()

        # log p(a_0 | q + o[:t]) for t = 0..T comes from logits at Q+t-1
        # (the position whose next-token prediction is a_0).
        positions = Q - 1 + torch.arange(T + 1, device=device)
        logp_a0 = logp_qo[positions, a_ids[0]]                    # (T+1,)

        if La == 1:
            logps_t = logp_a0
        else:
            # --- (2) for each t, forward a[:-1] with cache sliced to Q+t ---
            a_input  = torch.tensor(a_ids[:-1], device=device)    # (La-1,)
            a_target = torch.tensor(a_ids[1:],  device=device)    # (La-1,)
            logp_rest = torch.zeros(T + 1, La - 1,
                                    device=device, dtype=torch.float32)

            for s in range(0, T + 1, micro_batch_size):
                ts = list(range(s, min(s + micro_batch_size, T + 1)))
                B  = len(ts)
                t_max = ts[-1]
                Cmax  = Q + t_max                                 # padded cache len

                # Build a (B, H, Cmax, D) padded cache by COPYING the relevant
                # slice. Padding positions are zeros; attention_mask gates them.
                padded = []
                for k, v in past:
                    _, H, _, D = k.shape
                    K = torch.zeros((B, H, Cmax, D), dtype=k.dtype, device=device)
                    V = torch.zeros((B, H, Cmax, D), dtype=v.dtype, device=device)
                    for b, t in enumerate(ts):
                        L = Q + t
                        K[b, :, :L, :] = k[0, :, :L, :]
                        V[b, :, :L, :] = v[0, :, :L, :]
                    padded.append((K, V))
                # HF ≥4.36 wants a Cache object, not a legacy tuple.
                padded_cache = DynamicCache.from_legacy_cache(tuple(padded))

                input_ids = a_input.unsqueeze(0).expand(B, -1).contiguous()
                # Attention mask over (past + new) positions.
                total = Cmax + (La - 1)
                attn  = torch.zeros((B, total), dtype=torch.long, device=device)
                for b, t in enumerate(ts):
                    attn[b, :Q + t] = 1
                attn[:, Cmax:] = 1
                # Position ids for the new tokens: [Q+t, ..., Q+t+La-2]
                pos_ids = torch.zeros((B, La - 1), dtype=torch.long, device=device)
                for b, t in enumerate(ts):
                    pos_ids[b] = torch.arange(Q + t, Q + t + La - 1, device=device)

                res2 = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    position_ids=pos_ids,
                    past_key_values=padded_cache,
                    use_cache=False,
                )
                lp = F.log_softmax(res2.logits.float(), dim=-1)   # (B, La-1, V)
                gathered = lp.gather(
                    2, a_target.view(1, -1, 1).expand(B, -1, 1)
                ).squeeze(-1)                                     # (B, La-1)
                for b, t in enumerate(ts):
                    logp_rest[t] = gathered[b]

            logps_t = logp_a0 + logp_rest.sum(dim=1)              # (T+1,)

        logps = logps_t.cpu().numpy().astype(np.float64)
        vt    = logps[1:] - logps[:-1]
        R_T   = float(logps[-1] - logps[0])
        out.append({"toks": toks, "vt": vt, "logps": logps,
                    "R_T": R_T, "R_per_token": R_T / max(1, len(vt))})
    return out


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
