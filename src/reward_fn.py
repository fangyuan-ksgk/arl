"""Composable per-token reward factories.

Each factory takes a *base* reward function and returns a new reward function
that modulates the base by some per-token signal (entropy / NLL / grad mag),
by a per-rollout scalar derived from a signal (e.g. ``p(answer | q, o)``),
or by structural quantities (length).

Contract — all reward fns (base or modulated) implement::

    fn(prompt_ids, completion_ids, completion_mask, *,
       tokenizer, traj_reward=None, **signals) -> (B, T) tensor

and carry a ``.needs`` attribute (set of strings drawn from
``{"nll", "entropy", "grad_mag"}``) declaring which signals the trainer
should compute on their behalf. Factories propagate ``.needs`` correctly:
``modulated.needs = base.needs | {new_signal}``.

The Vk numbering follows the design notes at the top of
``script/ablate_game24.py``:

  =====  =================================================  ============
  Vk     reward                                              ``.needs``
  =====  =================================================  ============
  V0/V1  base                                                ``∅``
  V3     base / cot_len                                      ``∅``
  V4     base × clip(p(a | q, o))                            ``{"nll"}``
  V5     base × clip(1 / p(a | q, o))                        ``{"nll"}``
  V6     base × entropy                                      ``{"entropy"}``
  V7     base / (entropy + ε)                                ``{"entropy"}``
  V8     base × 1 / perplexity = base × exp(-NLL)            ``{"nll"}``
  V9     base × clip(perplexity) = base × clip(exp(NLL))     ``{"nll"}``
  V10    base / (grad_mag + ε)                               ``{"grad_mag"}``
  =====  =================================================  ============

For V10 the trainer must additionally be configured with an
``answer_mask_fn`` so that ``grad_mag`` is computed over the
answer-token NLL only (see :class:`PerTokenAdvantageTrainer`).

For V4/V5 the factory takes its own ``answer_mask_fn`` because the answer
log-prob is computed *inside* the reward fn from the ``nll`` signal.

Example
-------
>>> from script.ablate_game24 import progress_per_token_reward, game24_answer_mask
>>> from src.reward_fn import scale_by_inv_perplexity, scale_by_inv_grad_mag
>>> v8 = scale_by_inv_perplexity(progress_per_token_reward)
>>> v10 = scale_by_inv_grad_mag(progress_per_token_reward)
>>> # When wiring the trainer for V10, also set:
>>> #     trainer.answer_mask_fn = game24_answer_mask
"""
from __future__ import annotations

from typing import Callable, Optional, Set

import torch


# Numerical guards. Picked to be small enough not to wash out real variation
# but large enough to avoid 1/0 explosions when a signal is sharply zero.
_EPS_ENTROPY  = 1e-3
_EPS_GRAD_MAG = 1e-3
# Cap on raw perplexity scaling so a single ultra-low-prob token can't dominate
# the reward. Tunable; this default lets ~6 nats (≈400× perplexity) through.
_PPL_CLIP     = 400.0
# Cap on p(answer)^{-1} for V5. Caps reward inflation for rare-but-correct cases.
_P_INV_CLIP   = 100.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _needs_of(fn) -> Set[str]:
    """Return ``fn.needs`` as a set, treating missing/None as empty."""
    return set(getattr(fn, "needs", None) or ())


def _call_base(base, prompt_ids, completion_ids, completion_mask,
               *, tokenizer, traj_reward, signals: dict) -> torch.Tensor:
    """Invoke ``base`` while only passing kwargs it actually requested.

    The wrapped factories declare a ``.needs`` superset (base ∪ modulator-need);
    if a base reward fn doesn't accept ``nll`` as a kwarg, splatting it would
    raise ``TypeError``. So we restrict the splat to what the base declared.
    """
    base_needs = _needs_of(base)
    kw = {k: v for k, v in signals.items() if k in base_needs}
    return base(prompt_ids, completion_ids, completion_mask,
                tokenizer=tokenizer, traj_reward=traj_reward, **kw)


# ---------------------------------------------------------------------------
# Generic factories
# ---------------------------------------------------------------------------

def scale_by_token_signal(signal_name: str,
                          transform: Callable[[torch.Tensor], torch.Tensor],
                          ) -> Callable[[Callable], Callable]:
    """Wrap a base reward fn so the result is ``base * transform(signal)``.

    ``signal_name`` ∈ ``{"nll", "entropy", "grad_mag"}``. ``transform`` is
    applied elementwise to the (B, T) signal tensor. The result is multiplied
    elementwise with the base reward.

    The returned decorator can be applied to any base reward fn::

        v6 = scale_by_token_signal("entropy", lambda x: x)(base)
    """

    def deco(base):
        def reward_fn(prompt_ids, completion_ids, completion_mask, *,
                      tokenizer, traj_reward=None, **signals):
            b = _call_base(base, prompt_ids, completion_ids, completion_mask,
                           tokenizer=tokenizer, traj_reward=traj_reward,
                           signals=signals)
            s = signals.get(signal_name)
            if s is None:
                # Signal not provided (e.g. called outside trainer for unit tests).
                # Degrade gracefully to the unscaled base.
                return b
            return b * transform(s)
        reward_fn.needs = _needs_of(base) | {signal_name}
        reward_fn.__name__ = f"scale_{signal_name}({getattr(base, '__name__', 'base')})"
        return reward_fn
    return deco


# ---------------------------------------------------------------------------
# Public, named modulators
# ---------------------------------------------------------------------------

#: V6 — boost high-entropy tokens (explore where the model is uncertain).
scale_by_entropy = scale_by_token_signal(
    "entropy", lambda h: h,
)

#: V7 — boost low-entropy (confident) tokens.
scale_by_inv_entropy = scale_by_token_signal(
    "entropy", lambda h: 1.0 / (h + _EPS_ENTROPY),
)

#: V8 — boost tokens by next-token probability p = exp(-NLL).
#: Encourages "the model's own confident choices" without rewarding template
#: tokens explicitly (they get the same boost as any other low-entropy token).
scale_by_inv_perplexity = scale_by_token_signal(
    "nll", lambda nll: torch.exp(-nll),
)

#: V9 — boost surprising tokens (high NLL → high perplexity).
#: Clipped to avoid runaway scaling on out-of-distribution tokens.
scale_by_perplexity = scale_by_token_signal(
    "nll", lambda nll: torch.exp(nll).clamp(max=_PPL_CLIP),
)

#: V10 — boost tokens whose embedding is *not* load-bearing for the answer.
#: Requires ``answer_mask_fn`` to be set on the trainer so ``grad_mag`` is
#: computed against answer-token NLL only (see PerTokenAdvantageTrainer).
scale_by_inv_grad_mag = scale_by_token_signal(
    "grad_mag", lambda g: 1.0 / (g + _EPS_GRAD_MAG),
)


# ---------------------------------------------------------------------------
# Structural (no signal) modulator
# ---------------------------------------------------------------------------

def scale_by_inv_length(base):
    """V3 — divide every per-token reward by the rollout's effective length.

    Encourages shorter, denser responses without changing the trajectory-level
    integral much (∑ r_t becomes O(1) instead of O(T)).
    """
    def reward_fn(prompt_ids, completion_ids, completion_mask, *,
                  tokenizer, traj_reward=None, **signals):
        b = _call_base(base, prompt_ids, completion_ids, completion_mask,
                       tokenizer=tokenizer, traj_reward=traj_reward,
                       signals=signals)
        T_eff = completion_mask.sum(1, keepdim=True).clamp(min=1).float()
        return b / T_eff
    reward_fn.needs = _needs_of(base)
    reward_fn.__name__ = f"scale_inv_length({getattr(base, '__name__', 'base')})"
    return reward_fn


# ---------------------------------------------------------------------------
# Answer-conditional scaling (V4/V5) — uses NLL + a task-supplied answer mask
# ---------------------------------------------------------------------------

def scale_by_answer_prob(
    base,
    *,
    answer_mask_fn: Callable,
    invert: bool = False,
    clip: Optional[float] = None,
) -> Callable:
    """Scale every per-token reward by a per-rollout function of ``p(a | q, o)``.

    The answer log-probability is computed *inside* the reward fn from the
    ``nll`` signal::

        log p(a | q, o)  =  − Σ_{t ∈ answer}  nll_t
        p(a | q, o)      =  exp(log p(a | q, o))

    Args:
        base:            base reward fn (any contract-compliant callable).
        answer_mask_fn:  ``(completion_ids, completion_mask, tokenizer)
                         -> (B, T) bool`` identifying answer tokens.
        invert:          if False (V4) multiply by ``p(a)``;
                         if True  (V5) multiply by ``1 / p(a)``.
        clip:            optional ceiling on the scalar. Default ``None``
                         when ``invert=False`` (p is already in [0,1]) and
                         ``_P_INV_CLIP`` when ``invert=True`` (avoid blowup).
    """
    if clip is None and invert:
        clip = _P_INV_CLIP

    def reward_fn(prompt_ids, completion_ids, completion_mask, *,
                  tokenizer, traj_reward=None, **signals):
        b = _call_base(base, prompt_ids, completion_ids, completion_mask,
                       tokenizer=tokenizer, traj_reward=traj_reward,
                       signals=signals)
        nll = signals.get("nll")
        if nll is None:
            return b
        ans = answer_mask_fn(completion_ids, completion_mask, tokenizer).float()
        logp_a = -(nll * ans).sum(dim=1)                       # (B,)
        scale  = torch.exp(-logp_a) if invert else torch.exp(logp_a)
        if clip is not None:
            scale = scale.clamp(max=clip)
        return b * scale.unsqueeze(1)
    reward_fn.needs = _needs_of(base) | {"nll"}
    reward_fn.__name__ = (
        f"scale_{'inv_' if invert else ''}answer_prob"
        f"({getattr(base, '__name__', 'base')})"
    )
    return reward_fn
