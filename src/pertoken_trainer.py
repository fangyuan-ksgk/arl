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

import json
import re
import warnings
from pathlib import Path
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
    task_extras_fn
        Optional ``(completion_str, query_key) -> dict`` callback. Whatever
        dict it returns is merged into each rollout record written to
        ``rollouts.jsonl`` / ``eval_rollouts.jsonl``. For Game-of-24 we use
        ``lambda c, qk: {"numbers": list(qk), "expr": extract_expr(c)}``.
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
        task_extras_fn: Optional[Callable[[str, Hashable], dict]] = None,
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
        # Set by an external EvalFlagCallback during prediction loops so
        # `_compute_loss` routes rollouts to ``eval_rollouts.jsonl`` and
        # skips buffer updates. The autograd-state check is the robust
        # fallback (see `_compute_loss`).
        self.in_eval: bool = False
        self.per_token_reward_fn = per_token_reward_fn
        self.velocity_computer = velocity_computer
        self.is_correct = is_correct
        self.query_key_fn = query_key_fn
        self.task_extras_fn = task_extras_fn
        self.prefix_buffer = prefix_buffer
        self.adv_mode = adv_mode
        self.adv_n_chunks = adv_n_chunks
        self.adv_stride = adv_stride
        # Optional frozen scorer for log p(a | q + o[:t]). Default (None)
        # uses the live policy — same as before. Setting this to a frozen
        # copy of the base model decouples the velocity reward from the
        # parameters being updated. Set externally before training:
        #     trainer.velocity_ref_model = ref_model.eval().to(device)
        self.velocity_ref_model = None
        # Unified per-rollout logging — exactly two JSONL sinks, train + eval.
        # Lazy-resolved against ``args.output_dir`` on first write; override
        # by setting these attributes before training.
        self.rollouts_path: Optional[Path] = None
        self.eval_rollouts_path: Optional[Path] = None
        # `inner_step` = which microbatch within the current global step
        # (advantages z-score within a `_compute_loss` call → they sum to ~0
        # over a single inner_step). Resets when global_step advances.
        self._inner_step = 0
        self._last_global_step = -1

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

            # Pick the scorer for log p(a | q + o[:t]):
            #   • velocity_ref_model is None  → live policy (default).
            #   • velocity_ref_model set       → frozen ref model. Reward then
            #     measures progress under a fixed scorer, not the policy
            #     under update — useful for isolating reward shape from
            #     policy drift.
            if self.velocity_ref_model is not None:
                scoring_model = self.velocity_ref_model
                was_training  = False  # frozen — never flip back
            else:
                scoring_model = self.accelerator.unwrap_model(self.model)
                was_training  = scoring_model.training
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
            # Stash decoded strings + query_keys + correctness for the
            # unified rollout logger so we don't decode twice.
            self._last_decoded = {
                "prompt_strs":     prompt_strs,
                "completion_strs": completion_strs,
                "query_keys":      query_keys,
                "correctness":     correctness,
            }
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
        # Eval path: skip per-token reward (no buffer contamination, no
        # velocity-scorer forwards on losses we're about to throw away), but
        # still log eval rollouts with r_t/advantage = None.
        #
        # Eval guard: prefer the autograd-state signal over the callback-set
        # `in_eval` flag. `eval_on_start=True` triggers eval BEFORE
        # `EvalFlagCallback.on_evaluate` fires; `prediction_step` always wraps
        # this in `torch.no_grad()`, so `not torch.is_grad_enabled()` is the
        # robust signal for "this is eval, do not touch buffers".
        in_eval = self.in_eval or not torch.is_grad_enabled()
        if in_eval:
            try:
                self._log_rollouts(
                    inputs["prompt_ids"],
                    inputs["completion_ids"],
                    inputs["completion_mask"],
                    r_t=None, adv=None, loss=None, in_eval=True,
                )
            except Exception:
                pass  # logging must never break training
            return super()._compute_loss(model, inputs)

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
                # NaN-safe pool z-score: a single-element pool gives std==nan
                # under Bessel correction, which propagates to adv -> loss ->
                # gradient -> optimizer writes nan into the policy. After the
                # vLLM weight sync, the server returns None logprobs and TRL's
                # `torch.tensor(None)` crashes _generate_and_score_completions.
                # `unbiased=False` + numel>1 guard kills the cascade at source.
                if cot_m.any():
                    p_c = r_t[cot_m]
                    mu_c = p_c.mean()
                    sd_c = p_c.std(unbiased=False) if p_c.numel() > 1 else p_c.new_zeros(())
                    adv[cot_m] = (p_c - mu_c) / (sd_c + 1e-6)
                if ans_m.any():
                    p_a = r_t[ans_m]
                    mu_a = p_a.mean()
                    sd_a = p_a.std(unbiased=False) if p_a.numel() > 1 else p_a.new_zeros(())
                    adv[ans_m] = (p_a - mu_a) / (sd_a + 1e-6)
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

            # Per-token loss for logging: TRL's GRPO loss is
            #   L_t = -min(ratio_t * A_t, clip(ratio_t) * A_t) * mask_t  (+ beta * KL_t)
            # At the first inner iteration ratio_t == 1 exactly (the default
            # since TRL uses num_iterations=1), so L_t = -A_t * mask_t. We use
            # this here for the rollouts log; it's a diagnostic, not the value
            # backpropagated through super()._compute_loss.
            loss_per_tok = -adv * mask.float()

            # ── unified per-rollout logging (train) ──────────────────────
            try:
                self._log_rollouts(
                    prompt_ids, completion_ids, mask,
                    r_t=r_t, adv=adv, loss=loss_per_tok, in_eval=False,
                )
            except Exception:
                pass  # logging must never break training

        inputs = dict(inputs)
        inputs["advantages"] = adv
        return super()._compute_loss(model, inputs)

    # ------------------------------------------------------------------ logging

    def _bump_inner_step(self) -> int:
        """Inner-step counter: 0..gas-1 within each optimizer step.

        Advantages z-score within a single `_compute_loss` call, so all
        records sharing `(global_step, inner_step)` form a single advantage
        pool whose values sum to ~0 (over real tokens).
        """
        cur = int(getattr(self.state, "global_step", 0))
        if cur != self._last_global_step:
            self._inner_step = 0
            self._last_global_step = cur
            return 0
        self._inner_step += 1
        return self._inner_step

    def _resolve_log_path(self, in_eval: bool) -> Optional[Path]:
        attr = "eval_rollouts_path" if in_eval else "rollouts_path"
        p = getattr(self, attr, None)
        if p is not None:
            return Path(p)
        out_dir = getattr(getattr(self, "args", None), "output_dir", None)
        if out_dir is None:
            return None
        p = Path(out_dir) / ("eval_rollouts.jsonl" if in_eval else "rollouts.jsonl")
        setattr(self, attr, p)
        return p

    def _log_rollouts(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        mask: torch.Tensor,
        *,
        r_t: Optional[torch.Tensor],
        adv: Optional[torch.Tensor],
        loss: Optional[torch.Tensor],
        in_eval: bool,
    ) -> None:
        """Append one JSONL line per rollout in this microbatch.

        Schema (per record):
            global_step, inner_step, idx, split, completion, n_tokens,
            r_t, advantage, loss, correct, + task_extras_fn(completion, qk).

        On the train path `r_t`/`adv`/`loss` are the per-token tensors; only
        the real (non-padding) positions are dumped. On the eval path all
        three are None (velocity + advantage aren't computed during eval)
        and the record stores empty lists.

        The logged `loss` is the per-token GRPO PG term under the ratio==1
        assumption (exact when `num_iterations == 1`, which is TRL's
        default). It does not include the KL term super()._compute_loss may
        add, so per-token sums won't exactly equal the trainer's reported
        scalar loss — close enough for diagnostics.
        """
        path = self._resolve_log_path(in_eval)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)

        tok    = self.processing_class
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # Reuse decoded strings from `_per_token_reward` if available; on the
        # eval path nothing has been decoded yet so we do it here.
        cached = getattr(self, "_last_decoded", None) if not in_eval else None
        if cached is not None and len(cached["completion_strs"]) == completion_ids.size(0):
            prompt_strs     = cached["prompt_strs"]
            completion_strs = cached["completion_strs"]
            query_keys      = cached["query_keys"]
            correctness     = cached["correctness"]
        else:
            prompt_strs = [
                tok.decode(
                    [int(x) for x in prompt_ids[b].tolist() if x != pad_id],
                    skip_special_tokens=False,
                )
                for b in range(prompt_ids.size(0))
            ]
            completion_strs = [
                tok.decode(
                    completion_ids[b][mask[b].bool()].tolist(),
                    skip_special_tokens=False,
                )
                for b in range(completion_ids.size(0))
            ]
            if self.query_key_fn is not None:
                query_keys = [self.query_key_fn(s) for s in prompt_strs]
            else:
                query_keys = [
                    prompt_token_hash(prompt_ids[b], pad_id)
                    for b in range(prompt_ids.size(0))
                ]
            if self.is_correct is not None:
                correctness = [
                    bool(self.is_correct(completion_strs[b], prompt_strs[b]))
                    for b in range(completion_ids.size(0))
                ]
            else:
                correctness = [False] * completion_ids.size(0)
        # Consume the cache exactly once so subsequent calls don't reuse it.
        self._last_decoded = None

        gs    = int(getattr(self.state, "global_step", 0))
        inner = 0 if in_eval else self._bump_inner_step()

        # OSError-tolerant write (research telemetry, never crash training).
        try:
            with path.open("a") as f:
                for b in range(completion_ids.size(0)):
                    m_b = mask[b].bool()
                    n_tok = int(m_b.sum().item())
                    if r_t is not None:
                        r_b = [float(x) for x in r_t[b][m_b].tolist()]
                    else:
                        r_b = []
                    if adv is not None:
                        a_b = [float(x) for x in adv[b][m_b].tolist()]
                    else:
                        a_b = []
                    if loss is not None:
                        l_b = [float(x) for x in loss[b][m_b].tolist()]
                    else:
                        l_b = []
                    rec = {
                        "global_step": gs,
                        "inner_step":  inner,
                        "idx":         b,
                        "split":       "eval" if in_eval else "train",
                        "completion":  completion_strs[b],
                        "n_tokens":    n_tok,
                        "r_t":         r_b,
                        "advantage":   a_b,
                        "loss":        l_b,
                        "correct":     bool(correctness[b]),
                    }
                    if self.task_extras_fn is not None:
                        try:
                            rec.update(self.task_extras_fn(completion_strs[b], query_keys[b]))
                        except Exception:
                            pass
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            if not getattr(self, "_rollouts_log_warn_emitted", False):
                warnings.warn(
                    f"rollouts log write failed at step {gs} "
                    f"({type(e).__name__}: {e}); continuing without "
                    f"this step's entries. Further failures silenced."
                )
                self._rollouts_log_warn_emitted = True
