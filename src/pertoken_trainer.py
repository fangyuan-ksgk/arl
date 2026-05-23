"""Per-token-advantage GRPO trainer.

Extends :class:`trl.GRPOTrainer` with a per-token reward signal ``r_t`` and
three interchangeable advantage baselines:

- ``"token"``    — z-score the whole microbatch as one pool.
- ``"position"`` — group z-score per time-step ``t`` with an absorbing-state
                   tail filling padding (finished rollouts keep paying their
                   terminal reward so ``mu/sd`` stay smooth).
- ``"progress"`` — bucket by CoT progress ``p = t / T_eff`` into ``K`` chunks,
                   group-z-score the per-chunk mean reward, scatter back.

Two reward routes:

1. **Callable**: pass ``per_token_reward_fn(prompt_ids, completion_ids,
   completion_mask, *, tokenizer) -> (Bp, T)``. Simple, stateless.

2. **VelocityRewardComputer**: pass ``velocity_computer`` (see
   :class:`src.velocity.VelocityRewardComputer`). The trainer builds
   per-rollout ``query_keys`` (token-tuple hash of the prompt with padding
   stripped) and ``correctness`` (via the injected ``is_correct`` callable),
   then asks the computer for ``r_t`` and feeds accepted rollouts back into
   the answer buffer.

   When ``velocity_computer`` is provided, the model used to score
   ``log p_θ(a | q + o[:t])`` is the current policy (no frozen reference) —
   ``self.accelerator.unwrap_model(model).eval()`` is used inside a
   ``torch.no_grad()`` block.

If neither hook is set the trainer falls back to vanilla GRPO behavior.
"""

from __future__ import annotations

import re
from typing import Callable, Hashable, List, Optional

import torch
from trl import GRPOTrainer


__all__ = ["PerTokenAdvantageTrainer", "prompt_token_hash"]


# ---------------------------------------------------------------------------- utilities

def prompt_token_hash(prompt_row: torch.Tensor, pad_id: int) -> Hashable:
    """Stable hash of a single prompt row (padding stripped).

    Works with either left- or right-padded prompt tensors.
    """
    ids = [int(x) for x in prompt_row.tolist() if x != pad_id]
    return hash(tuple(ids))


# ---------------------------------------------------------------------------- trainer

class PerTokenAdvantageTrainer(GRPOTrainer):
    """GRPO with a per-token reward and one of three advantage baselines.

    Parameters
    ----------
    per_token_reward_fn
        Callable ``(prompt_ids, completion_ids, completion_mask, *, tokenizer)
        -> (Bp, T)``. Mutually exclusive with ``velocity_computer``.
    velocity_computer
        :class:`src.velocity.VelocityRewardComputer`. Mutually exclusive with
        ``per_token_reward_fn``. Requires ``is_correct``.
    is_correct
        Callable ``(completion_str, prompt_str) -> bool`` used to decide
        whether a rollout's own answer can be used as the velocity reference
        (and whether it gets added to the answer buffer). Required when
        ``velocity_computer`` is set.
    query_key_fn
        Callable ``(prompt_str) -> Hashable`` mapping a decoded prompt to
        the key used in the answer buffer. Default: a stable token-tuple
        hash of the prompt with padding stripped. Override with a task-
        specific identity (e.g., the puzzle numbers tuple) to make the
        offline seed and the online lookups agree even if tokenization is
        not pinned.
    adv_mode
        ``"token"`` | ``"position"`` | ``"progress"``.
    adv_n_chunks
        ``K`` for the ``"progress"`` baseline.
    adv_stride
        Tail width for the ``"position"`` baseline's absorbing-state fill.
    """

    def __init__(
        self,
        *args,
        per_token_reward_fn: Optional[Callable] = None,
        velocity_computer=None,
        is_correct: Optional[Callable[[str, str], bool]] = None,
        query_key_fn: Optional[Callable[[str], Hashable]] = None,
        prefix_buffer=None,
        adv_mode: str = "token",
        adv_n_chunks: int = 8,
        adv_stride: int = 5,
        **kw,
    ):
        assert adv_mode in ("token", "position", "progress"), adv_mode
        if per_token_reward_fn is not None and velocity_computer is not None:
            raise ValueError("pass per_token_reward_fn OR velocity_computer, not both")
        if velocity_computer is not None and is_correct is None:
            raise ValueError("velocity_computer requires is_correct")

        super().__init__(*args, **kw)
        self.per_token_reward_fn = per_token_reward_fn
        self.velocity_computer = velocity_computer
        self.is_correct = is_correct
        self.query_key_fn = query_key_fn
        self.prefix_buffer = prefix_buffer
        self.adv_mode = adv_mode
        self.adv_n_chunks = adv_n_chunks
        self.adv_stride = adv_stride

    # ------------------------------------------------------------------ reward

    def _per_token_reward(self, prompt_ids, completion_ids, mask):
        """Dispatch to whichever reward route was configured."""
        tok = self.processing_class

        if self.velocity_computer is not None:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

            prompt_strs = [
                tok.decode(
                    [int(x) for x in prompt_ids[b].tolist() if x != pad_id],
                    skip_special_tokens=False,
                )
                for b in range(prompt_ids.size(0))
            ]
            if self.query_key_fn is not None:
                query_keys: List[Hashable] = [self.query_key_fn(s) for s in prompt_strs]
            else:
                query_keys = [
                    prompt_token_hash(prompt_ids[b], pad_id)
                    for b in range(prompt_ids.size(0))
                ]
            completion_strs = [
                tok.decode(
                    completion_ids[b][mask[b].bool()].tolist(),
                    skip_special_tokens=False,
                )
                for b in range(completion_ids.size(0))
            ]
            correctness = [
                bool(self.is_correct(completion_strs[b], prompt_strs[b]))
                for b in range(completion_ids.size(0))
            ]

            # Score with the live policy (unwrapped for raw forward).
            scoring_model = self.accelerator.unwrap_model(self.model)
            was_training = scoring_model.training
            scoring_model.eval()
            try:
                r_t, cot_mask = self.velocity_computer.compute_per_token_reward(
                    prompt_ids, completion_ids, mask,
                    model=scoring_model,
                    tokenizer=tok,
                    query_keys=query_keys,
                    correctness=correctness,
                )
            finally:
                if was_training:
                    scoring_model.train()

            # Feed accepted answers back into the answer buffer.
            self.velocity_computer.update_buffer(
                query_keys, completion_ids, mask, correctness, tokenizer=tok,
            )
            # Feed accepted full CoTs into the prefix buffer (if any).
            if self.prefix_buffer is not None:
                for b, ok in enumerate(correctness):
                    if ok:
                        self.prefix_buffer.add(query_keys[b], completion_strs[b])
            return r_t, cot_mask

        if self.per_token_reward_fn is not None:
            r_t = self.per_token_reward_fn(
                prompt_ids, completion_ids, mask, tokenizer=tok,
            )
            # Callable route has no CoT/answer split — treat all real tokens
            # as CoT so advantage pooling is the existing single-pool behavior.
            return r_t, mask.long()

        return None

    # ------------------------------------------------------------------ loss

    def _compute_loss(self, model, inputs):
        prompt_ids     = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]    # (Bp, T)
        mask           = inputs["completion_mask"]   # (Bp, T)

        with torch.no_grad():
            out = self._per_token_reward(prompt_ids, completion_ids, mask)
            if out is None:
                return super()._compute_loss(model, inputs)
            r_t, cot_mask = out

            assert r_t.shape == completion_ids.shape, (
                f"per-token reward must be (Bp, T)={tuple(completion_ids.shape)}, "
                f"got {tuple(r_t.shape)}"
            )

            m       = mask.bool()
            cot_m   = cot_mask.bool() & m                   # CoT tokens
            ans_m   = (~cot_mask.bool()) & m                # answer-marker tokens

            if self.adv_mode == "token":
                # Two pools: per-chunk velocity (CoT) and R_T (answer).
                # The split matters for correctness — answer tokens carry
                # *cumulative* signal on a different scale than per-token
                # velocity, so they need their own baseline.
                adv = torch.zeros_like(r_t)
                if cot_m.any():
                    p_c   = r_t[cot_m]
                    mu_c, sd_c = p_c.mean(), p_c.std()
                    adv[cot_m] = (r_t[cot_m] - mu_c) / (sd_c + 1e-6)
                if ans_m.any():
                    p_a   = r_t[ans_m]
                    mu_a, sd_a = p_a.mean(), p_a.std()
                    adv[ans_m] = (r_t[ans_m] - mu_a) / (sd_a + 1e-6)
            else: 
                raise ValueError(f"Unknown adv_mode: {self.adv_mode} | Need to add support for CoT / Answer token separate handling")
            # elif self.adv_mode == "position":
            #     # group z-score per t with absorbing-state tail fill.
            #     G, (Bp, T) = self.num_generations, r_t.shape
            #     S = self.adv_stride
            #     seq_len  = m.sum(dim=1)
            #     t_idx    = torch.arange(T, device=r_t.device).unsqueeze(0)
            #     in_tail  = (t_idx >= (seq_len - S).unsqueeze(1)) & (t_idx < seq_len.unsqueeze(1))
            #     term_r   = (r_t * in_tail).sum(dim=1) / in_tail.sum(dim=1).clamp(min=1.0)
            #     r_filled = torch.where(m, r_t, term_r.unsqueeze(1))
            #     r_grp    = r_filled.view(Bp // G, G, T)
            #     mu, sd   = r_grp.mean(dim=1), r_grp.std(dim=1, unbiased=False)
            #     adv      = ((r_t.view(Bp // G, G, T) - mu[:, None]) /
            #                 (sd[:, None] + 1e-6)).view(Bp, T) * m
            # else:  # "progress"
            #     K, G, (Bp, T) = self.adv_n_chunks, self.num_generations, r_t.shape
            #     T_eff = m.float().sum(dim=1).clamp(min=1.0)
            #     t_idx = torch.arange(T, device=r_t.device).float().unsqueeze(0)
            #     chunk = (t_idx / T_eff[:, None] * K).clamp(max=K - 1).long() * m.long()
            #     cnt   = torch.zeros(Bp, K, device=r_t.device).scatter_add_(1, chunk, m.float())
            #     cs    = torch.zeros(Bp, K, device=r_t.device).scatter_add_(1, chunk, r_t * m)
            #     ck    = (cs / cnt.clamp(min=1.0)).view(Bp // G, G, K)
            #     mu, sd = ck.mean(dim=1), ck.std(dim=1, unbiased=False)
            #     ck_adv = ((ck - mu[:, None]) / (sd[:, None] + 1e-6)).view(Bp, K)
            #     adv    = ck_adv.gather(1, chunk) * m

        inputs = dict(inputs)
        inputs["advantages"] = adv
        return super()._compute_loss(model, inputs)
