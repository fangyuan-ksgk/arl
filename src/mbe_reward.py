# MBE-based reward functions for TRL's GRPOTrainer
# --------------------------------------------------

import math
import torch
import re
from src.mbe import mbe_reverse_gram


def extract_answer_from_completion(text: str) -> str:
    """Parse the final numeric answer from a model completion."""
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


@torch.no_grad()
def full_forward(model, input_ids):
    """Single forward pass returning logits and all hidden states."""
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    return outputs.logits, outputs.hidden_states


def compute_mbe_trace(hidden_states, prompt_len, patch_size=8, layer=-1):
    h = hidden_states[layer][0, prompt_len:, :]  # (T_comp, D)
    T, D = h.shape
    usable = (T // patch_size) * patch_size
    if usable == 0:
        return torch.tensor([0.0])
    h = h[:usable].reshape(-1, patch_size, D)
    mbe_vals = mbe_reverse_gram(h)
    return mbe_vals


def compute_single_completion_mbe(hidden_states_layer, prompt_len):
    """Compute MBE on the full completion hidden states for a single sequence (no patching)."""
    h = hidden_states_layer[0, prompt_len:, :]  # (T_comp, D)
    if h.shape[0] < 2:
        return 0.0
    mbe_val = mbe_reverse_gram(h.unsqueeze(0))  # (1,)
    return mbe_val.item()


def _compute_mbe_for_completion(model, tokenizer, prompt, completion_text, layers=None,
                                 use_patch_mbe=False, patch_size=8):
    """Shared MBE computation logic for a single prompt-completion pair.
    Returns the mean MBE across selected layers, or 0.0 if completion is too short."""
    device = next(model.parameters()).device

    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + completion_text

    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    comp_len = full_ids.shape[1] - prompt_len
    min_len = patch_size if use_patch_mbe else 2
    if comp_len < min_len:
        return 0.0

    _, hidden_states = full_forward(model, full_ids)

    n_layers = len(hidden_states)
    layer_indices = layers if layers is not None else range(1, n_layers)

    if use_patch_mbe:
        per_layer = [
            compute_mbe_trace(hidden_states, prompt_len,
                              patch_size=patch_size, layer=li).mean().item()
            for li in layer_indices
        ]
    else:
        per_layer = [
            compute_single_completion_mbe(hidden_states[li], prompt_len)
            for li in layer_indices
        ]

    return sum(per_layer) / len(per_layer) if per_layer else 0.0


class MBEReward:
    """
    MBE-based reward with clipping and scaling: min(mbe, clip) / scale.

    Default: min(mbe, 2.0) / 40.0 → max reward ≈ 0.05, so correctness (1.0)
    is ~20x larger than MBE reward.

    Usage:
        mbe_reward = MBEReward(tokenizer)
        trainer = GRPOTrainer(model=..., reward_funcs=[..., mbe_reward], ...)
        mbe_reward.set_model(trainer.model)
        trainer.train()
    """

    def __init__(self, tokenizer, layers=None, use_patch_mbe=False, patch_size=8,
                 scale=40.0, clip=2.0):
        self.__name__ = "mbe_reward"
        self.model = None
        self.tokenizer = tokenizer
        self.layers = layers
        self.use_patch_mbe = use_patch_mbe
        self.patch_size = patch_size
        self.scale = scale
        self.clip = clip

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if self.model is None:
            return [0.0] * len(completions)

        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion_text = completion[0]["content"]
            mbe_val = _compute_mbe_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                layers=self.layers, use_patch_mbe=self.use_patch_mbe,
                patch_size=self.patch_size,
            )
            reward = min(mbe_val, self.clip) / self.scale
            rewards.append(reward)

        return rewards


def compute_mbe_running_trace(hidden_states_layer, prompt_len, stride=8, eps=1e-5):
    """Running MBE over growing prefixes of the completion (kernel-trick form).

    Identity used:
        G_k = Σ_{t=1..k} h_t h_tᵀ                   (D×D)
        tr(G_k)     = Σ_{t=1..k} ‖h_t‖²              (scalar cumsum)
        ‖G_k‖²_F   = Σ_{s,t ≤ k} ⟨h_s, h_t⟩²        (block sum of K² where K = h hᵀ)

    So we never materialise any D×D matrix; we compute the T×T kernel K once,
    square it, take a 2D cumulative sum, and read off the diagonal at the
    desired trace positions. Numerically equivalent to the streaming
    `OnlineMBE` update — but vectorised and (for T < D) much cheaper.

    For k = stride, 2*stride, ..., T_comp, returns MBE of hidden_states[:k].
    """
    h = hidden_states_layer[0, prompt_len:, :].float()           # (T_comp, D)
    T = h.shape[0]
    if T < 2:
        return torch.tensor([0.0], device=h.device)
    Kmat   = h @ h.T                                              # (T, T) inner-product kernel
    sq_cum = Kmat.pow(2).cumsum(0).cumsum(1)                      # upper-left block sums of K²
    tr_cum = (h * h).sum(-1).cumsum(0)                            # (T,)   cumulative trace
    idx    = torch.arange(stride - 1, T, stride, device=h.device)
    if idx.numel() == 0:                                          # completion shorter than `stride`
        idx = torch.tensor([T - 1], device=h.device)
    tr_G   = tr_cum[idx]                                          # (K,)
    sq_G   = sq_cum[idx, idx]                                     # (K,)  fancy diag-pick
    return (2 * torch.log(tr_G.abs() + eps) - torch.log(sq_G + eps)).clamp_min(0.0)


def _mbe_at_k(hidden_states_layer, k, eps=1e-5):
    """MBE of the first `k` tokens, kernel-trick form (no D×D materialisation)."""
    h = hidden_states_layer[0, :k, :].float()                       # (k, D)
    if h.shape[0] < 2:
        return 0.0
    Kmat = h @ h.T                                                  # (k, k)
    tr_G = (h * h).sum()                                            # scalar = tr(Σ h hᵀ)
    sq_G = Kmat.pow(2).sum()                                        # scalar = ‖Σ h hᵀ‖²_F
    return (2 * torch.log(tr_G.abs() + eps) - torch.log(sq_G + eps)).clamp_min(0.0).item()


def _mbe_growth_trace(hidden_states_layer, prompt_len, T_total, stride=8, eps=1e-5):
    """MBE of hidden_states[:k] for k in {prompt_len, prompt_len+stride, ..., T_total}.

    Same kernel-trick form as `compute_mbe_running_trace` but with two differences:
      1. MBE is computed over the FULL sequence prefix (query always included).
      2. Sample positions start at k = prompt_len (so trace[0] = MBE(query)) and
         step by `stride` through the response tokens; the final position is T_total.

    Returns a 1-D tensor of length ≈ T_comp/stride + 1.
    """
    h = hidden_states_layer[0, :T_total, :].float()                # (T_total, D)
    if h.shape[0] < 2:
        return torch.tensor([0.0], device=h.device)
    Kmat   = h @ h.T                                                # (T_total, T_total)
    sq_cum = Kmat.pow(2).cumsum(0).cumsum(1)
    tr_cum = (h * h).sum(-1).cumsum(0)
    ks = list(range(max(prompt_len, 2), T_total + 1, stride))
    if not ks:
        return torch.tensor([0.0], device=h.device)
    if ks[-1] != T_total:
        ks.append(T_total)
    idx  = torch.tensor([k - 1 for k in ks], device=h.device)
    tr_G = tr_cum[idx]
    sq_G = sq_cum[idx, idx]
    return (2 * torch.log(tr_G.abs() + eps) - torch.log(sq_G + eps)).clamp_min(0.0)


def _compute_mbe_velocity_for_completion(model, tokenizer, prompt, completion_text,
                                          layers=None, stride=8, mode="trajectory"):
    """Length-normalised MBE velocity, in one of two modes.

    Both modes share:
      • forward pass on (query + response)
      • MBE always computed over hidden_states[:k] including the query
      • length-norm = log(min(T_comp, D))
      • per-layer mean across `layers` (default: last layer only)

    mode="trajectory" (default):
        raw_velocity = MBE(query+response) − MBE(query)
        Single endpoint diff — measures *net* representation expansion.

    mode="rollercoaster":
        trace[i]      = MBE(hidden_states[:prompt_len + i*stride])
        deltas[i]     = trace[i+1] − trace[i]
        raw_velocity  = sum( max(0, deltas[i]) )    # only positive jumps counted
        Encourages the model to keep adding diversity step-by-step, ignoring
        drawdowns. A monotonically climbing trace and a heavily-oscillating
        "roller coaster" both score well; a flat or shrinking trace scores ~0.
    """
    assert mode in ("trajectory", "rollercoaster"), f"unknown mode: {mode}"
    device = next(model.parameters()).device
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + completion_text
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    T_total = full_ids.shape[1]
    T_comp = T_total - prompt_len
    if T_comp < 2 * stride or prompt_len < 2:
        return 0.0

    _, hidden_states = full_forward(model, full_ids)
    layer_indices = layers if layers is not None else [-1]   # last layer only by default

    D = hidden_states[-1].shape[-1]
    length_norm = math.log(min(T_comp, D))                   # >0 since T_comp ≥ 2*stride

    per_layer = []
    for li in layer_indices:
        h_layer = hidden_states[li]
        if mode == "trajectory":
            baseline = _mbe_at_k(h_layer, prompt_len)            # MBE(query)
            endpoint = _mbe_at_k(h_layer, T_total)               # MBE(query + response)
            raw_v = endpoint - baseline
        else:  # rollercoaster
            trace = _mbe_growth_trace(h_layer, prompt_len, T_total, stride=stride)
            if trace.numel() < 2:
                raw_v = 0.0
            else:
                deltas = trace[1:] - trace[:-1]
                raw_v  = deltas.clamp_min(0.0).sum().item()
        per_layer.append(raw_v / length_norm)
    return sum(per_layer) / len(per_layer) if per_layer else 0.0


class MBEVeloReward:
    """
    Length-normalised MBE *velocity* reward. Two modes:

    mode="trajectory" (default):
        raw_v = MBE(query+response) − MBE(query)
        Rewards net representation expansion.

    mode="rollercoaster":
        trace[i] = MBE(hidden_states[:prompt_len + i*stride])
        raw_v    = Σ max(0, trace[i+1] − trace[i])
        Rewards accumulated *positive* per-step diversity growth, ignoring
        drawdowns. A monotonic climb and an oscillating roller-coaster both
        score high; a flat or collapsing trace scores ~0.

    Common pipeline:
        norm_v = raw_v / log(min(T_comp, D))
        reward = clip(norm_v, ±clip) / scale

    Trajectory default (scale=4.0, clip=1.0) → max |reward| ≈ 0.25.
    Rollercoaster typically has much larger raw_v (positive deltas sum up
    across many stride points), so consider scale ≥ 8.0 if you switch modes.

    Default `layers=[-1]` (last layer only): late-layer MBE captures
    task-relevant representation diversity; averaging over all layers
    conflates lexical and semantic diversity.

    Usage:
        velo = MBEVeloReward(tokenizer, mode="rollercoaster", scale=8.0)
        trainer = GRPOTrainer(model=..., reward_funcs=[..., velo], ...)
        velo.set_model(trainer.model)
        trainer.train()
    """

    def __init__(self, tokenizer, layers=None, stride=8,
                 scale=4.0, clip=1.0, mode="trajectory"):
        assert mode in ("trajectory", "rollercoaster"), f"unknown mode: {mode}"
        # name encodes mode so TRL logs trajectory vs rollercoaster separately
        self.__name__ = f"mbe_velocity_{mode}"
        self.model = None
        self.tokenizer = tokenizer
        self.layers = layers
        self.stride = stride
        self.scale = scale
        self.clip = clip
        self.mode = mode

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if self.model is None:
            return [0.0] * len(completions)
        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion_text = completion[0]["content"]
            v = _compute_mbe_velocity_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                layers=self.layers, stride=self.stride, mode=self.mode,
            )
            v = max(-self.clip, min(v, self.clip))   # two-sided clip
            rewards.append(v / self.scale)
        return rewards


class InvLogLengthReward:
    """Pure length-normalisation baseline for the MBE velocity reward.

    Reward = clip(1 / log(min(T_comp, D)), ±clip) / scale

    This is exactly the denominator of :class:`MBEVeloReward` with the
    numerator (``raw_v``) ablated to a constant 1. It isolates the
    *length-pressure* component of MBE velocity:
      - if MBE-velocity-w10's length reduction is fully explained by the
        log-T denominator, this baseline should reproduce the same
        length-vs-accuracy curve.
      - if the diversity numerator contributes real signal, MBE velocity
        should dominate this baseline on the Pareto frontier.

    All gating logic (T_comp ≥ 2·stride, prompt_len ≥ 2) is mirrored
    from :func:`_compute_mbe_velocity_for_completion` for parity. The
    ``D`` term is taken from the model's hidden size so the cap matches
    MBE velocity exactly; falls back to a constant if no model is bound.

    Usage::

        inv_len = InvLogLengthReward(tokenizer, scale=4.0, clip=1.0)
        trainer = GRPOTrainer(model=..., reward_funcs=[..., inv_len], ...)
        inv_len.set_model(trainer.model)
        trainer.train()
    """

    def __init__(self, tokenizer, stride: int = 8,
                 scale: float = 4.0, clip: float = 1.0):
        self.__name__ = "inv_log_length"
        self.model = None
        self.tokenizer = tokenizer
        self.stride = stride
        self.scale = scale
        self.clip = clip

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        device = (
            next(self.model.parameters()).device
            if self.model is not None else torch.device("cpu")
        )
        # Hidden dim D mirrors the MBE velocity cap; default 2048 is a
        # reasonable fallback for the model class we sweep over (Qwen3 0.6/1.7/4B).
        D = getattr(getattr(self.model, "config", None), "hidden_size", 2048)

        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion_text = completion[0]["content"]
            # Mirror MBE-velocity tokenization so T_comp is computed
            # identically — fair apples-to-apples.
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            full_text  = prompt_text + completion_text
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
            full_ids   = self.tokenizer(full_text,   return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]
            T_total    = full_ids.shape[1]
            T_comp     = T_total - prompt_len

            # Same guard as MBE velocity → 0 reward on too-short completions
            # so the model can't game the length-norm via 1-token outputs.
            if T_comp < 2 * self.stride or prompt_len < 2:
                rewards.append(0.0)
                continue

            length_norm = math.log(min(T_comp, D))            # > 0
            v = 1.0 / length_norm
            v = max(-self.clip, min(v, self.clip))            # parity w/ MBEVeloReward
            rewards.append(v / self.scale)
        return rewards


# ---------------------------------------------------------------------------
# Rationale-internal velocity rewards (entropy + perplexity).
#
# Per-token velocity over the rationale o:
#       Δ_t = X(o_{t+1}) − X(o_t)        X ∈ {entropy, NLL}
#
# Two aggregation modes:
#   "trajectory":   raw_v = X(o_last) − X(o_first)
#                          = Σ_t Δ_t                   (telescoping sum)
#                   Endpoint diff. Insensitive to the path between endpoints.
#
#   "rollercoaster" (default): raw_v = Σ_t max(0, Δ_t)
#                   Sum of positive jumps. Captures sustained upward motion
#                   along the rationale; oscillation contributes through the
#                   positive half-swings only. Parallel to MBEVeloReward's
#                   rollercoaster mode.
#
# Both forms then divided by log(min(T_comp, D))   (MBE-velocity convention).
#
# Sign conventions ("entropy" mode):
#   positive raw_v ⇒ model is *exploring* through the rationale (entropy is
#                    rising or oscillating upward).
#   negative raw_v ⇒ model is *converging* through the rationale (only
#                    possible in trajectory mode; rollercoaster floors at 0).
# "nll" mode: same shape on the realised-token-NLL signal.
#
# Distinct from :class:`EntropyDensityReward`, which contrasts rationale-mean
# vs answer-mean (phase contrast). These rewards measure trends *within* the
# rationale, with no answer-side dependence.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_rationale_velocity_for_completion(model, tokenizer, prompt,
                                                completion_text,
                                                mode: str = "entropy",
                                                aggregation: str = "rollercoaster",
                                                marker: str = "####",
                                                stride: int = 8):
    """Length-normalised per-token velocity over the rationale.

    Per-token velocity Δ_t = X(o_{t+1}) − X(o_t),   X ∈ {entropy, NLL}.

    aggregation="trajectory":     raw_v = X(o_last) − X(o_first)
    aggregation="rollercoaster":  raw_v = Σ_t max(0, Δ_t)

    return raw_v / log(min(T_comp, D))   (MBE-velocity-style log normalisation)

    Returns 0.0 if marker absent, rationale too short, or guard fails.
    """
    assert mode in ("entropy", "nll"), f"unknown mode: {mode}"
    assert aggregation in ("trajectory", "rollercoaster"), \
        f"unknown aggregation: {aggregation}"
    if marker not in completion_text:
        return 0.0
    pre, _ = completion_text.split(marker, 1)
    rationale_text = pre
    if not rationale_text.strip():
        return 0.0

    device = next(model.parameters()).device
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    full_text         = prompt_text + completion_text
    full_ids          = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    prompt_ids        = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    rationale_end_ids = tokenizer(prompt_text + rationale_text, return_tensors="pt")["input_ids"]

    prompt_len    = prompt_ids.shape[1]
    rationale_end = rationale_end_ids.shape[1]
    T_total       = full_ids.shape[1]
    T_rationale   = rationale_end - prompt_len
    T_comp        = T_total - prompt_len

    # Need at least 2 rationale tokens to define an endpoint difference.
    if T_rationale < max(2, 2 * stride) or prompt_len < 2:
        return 0.0

    logits = model(full_ids, use_cache=False).logits[0]      # (T_total, V)
    log_p  = torch.log_softmax(logits[:-1].float(), dim=-1)  # (T_total-1, V)

    if mode == "entropy":
        X_all = -(log_p.exp() * log_p).sum(dim=-1)
    else:  # "nll"
        targets = full_ids[0, 1:]
        X_all   = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)

    # X[i] is the per-position quantity for the distribution that *generates*
    # token i+1. So rationale tokens (full_ids[prompt_len .. rationale_end))
    # are predicted at positions [prompt_len-1 .. rationale_end-1).
    X_rationale = X_all[prompt_len - 1 : rationale_end - 1]
    if X_rationale.numel() < 2:
        return 0.0

    # Δ_t = X(o_{t+1}) − X(o_t) over rationale tokens
    deltas = X_rationale[1:] - X_rationale[:-1]
    if aggregation == "trajectory":
        raw_v = deltas.sum().item()                          # = X[-1] − X[0]
    else:  # "rollercoaster"
        raw_v = torch.clamp(deltas, min=0.0).sum().item()
    D           = logits.shape[-1]
    length_norm = math.log(min(T_comp, D))
    return raw_v / length_norm


class EntropyVeloReward:
    """Rationale-internal entropy velocity reward.

    Per-token velocity Δ_t = H(o_{t+1}) − H(o_t) over rationale, aggregated
    by `aggregation`:

        "rollercoaster" (default):  raw_v = Σ_t max(0, Δ_t)
        "trajectory":               raw_v = Σ_t Δ_t = H(o_last) − H(o_first)

        reward = clip(raw_v / log(min(T_comp, D)), ±clip) / scale

    Rollercoaster default rationale: it captures sustained upward motion in
    rationale entropy and floors at 0 (no negative half-swings), which we
    found stabilises training relative to the trajectory endpoint diff.

    Sign at positive scale (rollercoaster mode is non-negative):
      large reward ⇒ rationale entropy has many upward jumps (exploration).
      ~0 reward    ⇒ entropy is monotone non-increasing across the trace.
    For trajectory mode, negative reward indicates convergent reasoning.
    Pass negative scale to flip the preferred direction.

    Splits completion on `marker` (default "####") to identify the rationale.
    Returns 0.0 reward for format-violating completions without the marker.

    Distinct from :class:`EntropyDensityReward` (phase contrast: rationale vs
    answer) — this one measures dynamics *within* the rationale.

    Cost: one forward pass per rollout.
    """

    def __init__(self, tokenizer, stride: int = 8, scale: float = 4.0,
                 clip: float = 1.0, marker: str = "####",
                 aggregation: str = "rollercoaster"):
        self.__name__ = "entropy_velocity"
        self.model = None
        self.tokenizer = tokenizer
        self.stride = stride
        self.scale = scale
        self.clip = clip
        self.marker = marker
        self.aggregation = aggregation

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if self.model is None:
            return [0.0] * len(completions)
        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion_text = completion[0]["content"]
            v = _compute_rationale_velocity_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                mode="entropy", aggregation=self.aggregation,
                marker=self.marker, stride=self.stride,
            )
            v = max(-self.clip, min(v, self.clip))
            rewards.append(v / self.scale)
        return rewards


class PerplexityVeloReward:
    """Rationale-internal perplexity (NLL) velocity reward.

    Per-token velocity Δ_t = NLL(o_{t+1}) − NLL(o_t) over rationale,
    aggregated by `aggregation`:

        "rollercoaster" (default):  raw_v = Σ_t max(0, Δ_t)
        "trajectory":               raw_v = Σ_t Δ_t = NLL(o_last) − NLL(o_first)

        reward = clip(raw_v / log(min(T_comp, D)), ±clip) / scale

    Rollercoaster default rationale: it captures sustained upward motion in
    rationale NLL and floors at 0 (no negative half-swings), which we found
    stabilises training relative to the trajectory endpoint diff.

    Sign at positive scale (rollercoaster mode is non-negative):
      large reward ⇒ rationale tokens repeatedly become *more surprising*
                     under the model's own distribution (exploring less-
                     predictable territory).
      ~0 reward    ⇒ NLL is monotone non-increasing across the trace.
    For trajectory mode, negative reward indicates the model finds its
    rationale more predictable as it goes.
    Pass negative scale to flip the preferred direction.

    Splits completion on `marker` (default "####") to identify the rationale.
    Returns 0.0 reward for format-violating completions without the marker.

    Distinct from any phase-contrast reward — this measures dynamics *within*
    the rationale, not across the rationale/answer boundary.

    Cost: one forward pass per rollout.
    """

    def __init__(self, tokenizer, stride: int = 8, scale: float = 4.0,
                 clip: float = 1.0, marker: str = "####",
                 aggregation: str = "rollercoaster"):
        self.__name__ = "perplexity_velocity"
        self.model = None
        self.tokenizer = tokenizer
        self.stride = stride
        self.scale = scale
        self.clip = clip
        self.marker = marker
        self.aggregation = aggregation

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if self.model is None:
            return [0.0] * len(completions)
        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion_text = completion[0]["content"]
            v = _compute_rationale_velocity_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                mode="nll", aggregation=self.aggregation,
                marker=self.marker, stride=self.stride,
            )
            v = max(-self.clip, min(v, self.clip))
            rewards.append(v / self.scale)
        return rewards


# ---------------------------------------------------------------------------
# Phase-contrast rewards (entropy density + perplexity density).
#
# Both rewards share the same skeleton: split the completion on `marker`
# (default "####"), compute a per-position quantity X (entropy or NLL),
# contrast its mean over the rationale tokens vs the answer tokens, and
# normalise by log(min(T_comp, D)) — the MBE-velocity convention.
#
#       raw_v  = mean(X over rationale)  −  mean(X over answer)
#       reward = clip(raw_v / log(min(T_comp, D)), ±clip) / scale
#
# X="entropy" → high reward when rationale is uncertain, answer decisive
#               (explore-then-commit).
# X="nll"     → high reward when rationale tokens are surprising (high NLL,
#               model didn't predict its own choices well = exploratory)
#               and answer tokens are predictable (low NLL = decisive).
# Both modes encode the "reason hard, commit confidently" intuition.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_phase_contrast_for_completion(model, tokenizer, prompt,
                                            completion_text,
                                            mode: str = "entropy",
                                            marker: str = "####",
                                            stride: int = 8):
    """Length-normalised rationale-vs-answer contrast for a single rollout.

    mode="entropy":  X = Shannon entropy of softmax(logits) at each position.
    mode="nll":      X = -log p(realised next token) at each position.

    raw_v  = mean X over rationale predictions − mean X over answer predictions
    return raw_v / log(min(T_comp, D))   (MBE-velocity-style log normalisation)

    Returns 0.0 if marker absent, rationale/answer too short, or guard fails.
    """
    assert mode in ("entropy", "nll"), f"unknown mode: {mode}"
    if marker not in completion_text:
        return 0.0
    pre, post = completion_text.split(marker, 1)
    rationale_text = pre
    answer_text    = marker + post
    if not rationale_text.strip() or not answer_text.strip():
        return 0.0

    device = next(model.parameters()).device
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    # Tokenisation boundaries:
    #   prompt_len    = where rationale starts (in full_ids)
    #   rationale_end = where answer starts   (in full_ids)
    #   T_total       = full sequence length
    full_text         = prompt_text + completion_text
    full_ids          = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    prompt_ids        = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    rationale_end_ids = tokenizer(prompt_text + rationale_text, return_tensors="pt")["input_ids"]

    prompt_len    = prompt_ids.shape[1]
    rationale_end = rationale_end_ids.shape[1]
    T_total       = full_ids.shape[1]
    T_rationale   = rationale_end - prompt_len
    T_answer      = T_total - rationale_end
    T_comp        = T_total - prompt_len

    if T_rationale < 2 * stride or T_answer < 1 or prompt_len < 2:
        return 0.0

    logits = model(full_ids, use_cache=False).logits[0]      # (T_total, V)

    # X[i] is the per-position quantity for the distribution that *generates*
    # token i+1. So rationale tokens (full_ids[prompt_len .. rationale_end))
    # are predicted at positions [prompt_len-1 .. rationale_end-1).
    log_p = torch.log_softmax(logits[:-1].float(), dim=-1)   # (T_total-1, V)
    if mode == "entropy":
        X_all = -(log_p.exp() * log_p).sum(dim=-1)           # (T_total-1,)
    else:  # "nll"
        targets = full_ids[0, 1:]                            # (T_total-1,)
        X_all   = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)

    X_rationale = X_all[prompt_len - 1 : rationale_end - 1]
    X_answer    = X_all[rationale_end - 1 : T_total - 1]
    if X_rationale.numel() < 1 or X_answer.numel() < 1:
        return 0.0

    raw_v       = X_rationale.mean().item() - X_answer.mean().item()
    D           = logits.shape[-1]
    length_norm = math.log(min(T_comp, D))
    return raw_v / length_norm


class EntropyDensityReward:
    """Length-normalised entropy contrast reward.

    raw_v   = mean(H over rationale)  −  mean(H over answer)
    reward  = clip(raw_v / log(min(T_comp, D)), ±clip) / scale

    Encodes: "reason uncertainly, commit confidently". Same log-length
    normalisation as MBE / entropy / perplexity velocity for direct
    comparability; gentle length pressure rather than aggressive linear.

    Splits completion on `marker` (default "####", GSM8K convention). Returns
    0.0 reward for completions without the marker, so format-violators are
    silently skipped (consistent with PredictiveVeloReward).

    Magnitude note: raw_v ~ O(1) in nats; log(min(T_comp, D)) ~ 5-7 → raw
    reward typically ~0.1-0.5 before scaling. Default scale=4.0 (matching
    MBE velocity) gives reward magnitudes ~0.025-0.125 after clip. Bump
    scale smaller (e.g. 0.4 ≡ w=10) for stronger weighting.

    Cost: one forward pass per rollout.
    """

    def __init__(self, tokenizer, stride: int = 8, scale: float = 4.0,
                 clip: float = 1.0, marker: str = "####"):
        self.__name__ = "entropy_density"
        self.model = None
        self.tokenizer = tokenizer
        self.stride = stride
        self.scale = scale
        self.clip = clip
        self.marker = marker

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if self.model is None:
            return [0.0] * len(completions)
        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion_text = completion[0]["content"]
            v = _compute_phase_contrast_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                mode="entropy", marker=self.marker, stride=self.stride,
            )
            v = max(-self.clip, min(v, self.clip))
            rewards.append(v / self.scale)
        return rewards


# ---------------------------------------------------------------------------
# Predictive velocity — measures the information value of the rationale.
#
#   raw_v = log p(a | q, o) − log p(a | q)
#
# Two forward passes per rollout: one with rationale, one without. The
# rationale o is everything in the completion *before* the "####" marker;
# the answer a is everything after. For GSM8K specifically, this isolates
# whether the chain-of-thought is actually contributing to answer-likelihood.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_predictive_velocity_for_completion(model, tokenizer, prompt,
                                                 completion_text,
                                                 stride: int = 8,
                                                 marker: str = "####"):
    """Length-normalised predictive velocity for a single rollout.

    Splits completion on `marker` (default "####", GSM8K convention) into
    rationale `o` and answer-region `a`. Computes log p(a | q, o) and
    log p(a | q) via two forward passes, then returns
        ( log p(a|q,o) − log p(a|q) ) / log(min(T_comp, D)).

    Returns 0.0 if:
      - completion contains no marker (unparseable)
      - rationale or answer is empty after split
      - T_comp < 2·stride or prompt_len < 2 (parity with MBE velocity guard)
      - either tokenisation produces zero answer tokens
    """
    if marker not in completion_text:
        return 0.0
    pre, post = completion_text.split(marker, 1)
    rationale_text = pre
    # Keep marker on the answer side so both sequences see "####" before `a`.
    # This isolates the rationale's contribution rather than confounding it
    # with the model's prior on the marker token itself.
    answer_text = marker + post
    if not rationale_text.strip() or not answer_text.strip():
        return 0.0

    device = next(model.parameters()).device
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    # Sequence A: q + o + a   (rationale present)
    seq_a_text = prompt_text + rationale_text + answer_text
    # Sequence B: q + a       (rationale ablated)
    seq_b_text = prompt_text + answer_text

    qoa_ids   = tokenizer(seq_a_text, return_tensors="pt")["input_ids"].to(device)
    qa_ids    = tokenizer(seq_b_text, return_tensors="pt")["input_ids"].to(device)
    q_ids     = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    qo_ids    = tokenizer(prompt_text + rationale_text, return_tensors="pt")["input_ids"]

    prompt_len = q_ids.shape[1]
    T_total    = qoa_ids.shape[1]
    T_comp     = T_total - prompt_len
    if T_comp < 2 * stride or prompt_len < 2:
        return 0.0

    # Answer span boundaries — the answer tokens are everything after the
    # rationale in each sequence. We compute log p only over those positions.
    a_start_in_qoa = qo_ids.shape[1]
    a_start_in_qa  = q_ids.shape[1]
    n_ans_qoa      = qoa_ids.shape[1] - a_start_in_qoa
    n_ans_qa       = qa_ids.shape[1]  - a_start_in_qa
    if n_ans_qoa < 1 or n_ans_qa < 1:
        return 0.0

    def _seq_answer_logp(input_ids, a_start, n_ans):
        # logits[t] predicts token t+1 → answer token at idx i in input_ids
        # is predicted by logits[i-1]. We want positions [a_start-1 .. T-2]
        # (length n_ans) and target tokens [a_start .. T-1].
        logits = model(input_ids, use_cache=False).logits[0]   # (T, V)
        log_p  = torch.log_softmax(logits[a_start - 1 : a_start - 1 + n_ans].float(), dim=-1)
        targets = input_ids[0, a_start : a_start + n_ans]
        return log_p.gather(1, targets.unsqueeze(1)).squeeze(1).sum().item()

    logp_a_given_qo = _seq_answer_logp(qoa_ids, a_start_in_qoa, n_ans_qoa)
    logp_a_given_q  = _seq_answer_logp(qa_ids,  a_start_in_qa,  n_ans_qa)

    # Per-token normalisation across the two sequences keeps raw_v on a
    # comparable scale even when retokenisation makes n_ans differ slightly
    # between (q + o + a) and (q + a) — common when leading whitespace
    # gets re-merged at the rationale/answer boundary.
    raw_v = (logp_a_given_qo / max(n_ans_qoa, 1)) - (logp_a_given_q / max(n_ans_qa, 1))

    D = model.config.hidden_size
    length_norm = math.log(min(T_comp, D))
    return raw_v / length_norm


class PredictiveVeloReward:
    """Length-normalised *predictive velocity* reward.

    raw_v   = mean log p(a | q, o) − mean log p(a | q)
    reward  = clip(raw_v / log(min(T_comp, D)), ±clip) / scale

    Splits the completion on `marker` (default "####") into rationale (`o`)
    and answer (`a`). Two forward passes per rollout. Positive reward at
    positive scale rewards rationales that *increase* the model's
    confidence in the answer; negative scale rewards rationales that
    *decrease* it (a sanity-check direction — should hurt accuracy).

    Cost: 2× the forward passes of EntropyVelo / PerplexityVelo. Skips
    rollouts without a `####` marker (returns 0.0), so format-violating
    rollouts contribute neither signal nor noise.
    """

    def __init__(self, tokenizer, stride: int = 8, scale: float = 4.0,
                 clip: float = 1.0, marker: str = "####"):
        self.__name__ = "predictive_velocity"
        self.model = None
        self.tokenizer = tokenizer
        self.stride = stride
        self.scale = scale
        self.clip = clip
        self.marker = marker

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if self.model is None:
            return [0.0] * len(completions)
        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion_text = completion[0]["content"]
            v = _compute_predictive_velocity_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                stride=self.stride, marker=self.marker,
            )
            v = max(-self.clip, min(v, self.clip))
            rewards.append(v / self.scale)
        return rewards


class CorrectnessGatedMBEReward:
    """
    Correctness-gated MBE reward with clipping and scaling.

    For each completion:
        - If incorrect → 0.0
        - If correct   → min(mbe, clip) / scale

    Usage:
        gated_mbe = CorrectnessGatedMBEReward(tokenizer)
        trainer = GRPOTrainer(model=..., reward_funcs=[..., gated_mbe], ...)
        gated_mbe.set_model(trainer.model)
        trainer.train()
    """

    def __init__(self, tokenizer, layers=None, scale=40.0, clip=2.0):
        self.__name__ = "correctness_gated_mbe"
        self.model = None
        self.tokenizer = tokenizer
        self.layers = layers
        self.scale = scale
        self.clip = clip

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, prompts, completions, gold_answer=None, **kwargs) -> list[float]:
        if self.model is None or gold_answer is None:
            return [0.0] * len(completions)

        rewards = []
        for prompt, completion, gold in zip(prompts, completions, gold_answer):
            completion_text = completion[0]["content"]

            # Gate: check correctness first
            predicted = extract_answer_from_completion(completion_text)
            try:
                is_correct = float(predicted) == float(gold)
            except (ValueError, TypeError):
                is_correct = False

            if not is_correct:
                rewards.append(0.0)
                continue

            # Correct answer — compute scaled MBE
            mbe_val = _compute_mbe_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                layers=self.layers,
            )
            reward = min(mbe_val, self.clip) / self.scale
            rewards.append(reward)

        return rewards