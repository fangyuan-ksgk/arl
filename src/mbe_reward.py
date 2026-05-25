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


def _compute_mbe_velocity_for_completion(model, tokenizer, prompt, completion_text,
                                          layers=None, stride=8):
    """Length-normalised MBE velocity over the running-prefix trace.

        raw_velocity  = trace[-1] - trace[0]                     # endpoint diff
        norm_velocity = raw_velocity / log(min(T_comp, D))       # ∈ ~[-1, +1]

    Why endpoint diff (not mean-diff): saturating traces like log(k) have a
    naturally diminishing per-step slope, so mean-diff conflates "rank
    growth direction" with "how saturated the trace is". Endpoint diff
    asks the cleaner question: did the cloud end with higher rank than it
    started with?

    Why divide by log(min(T_comp, D)): MBE is bounded by log(min(T, D)),
    so this expresses velocity as a *fraction of available rank growth*
    — making it comparable across rollouts of different lengths.

    Averaged across the selected hidden layers (default: last layer only).
    Returns 0.0 if completion is too short for a 2-point trace.
    """
    device = next(model.parameters()).device
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + completion_text
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    T_comp = full_ids.shape[1] - prompt_len
    if T_comp < 2 * stride:
        return 0.0

    _, hidden_states = full_forward(model, full_ids)
    n_layers = len(hidden_states)
    layer_indices = layers if layers is not None else [-1]   # last layer only by default

    D = hidden_states[-1].shape[-1]
    length_norm = math.log(min(T_comp, D))                   # >0 since T_comp ≥ 2*stride

    per_layer = []
    for li in layer_indices:
        trace = compute_mbe_running_trace(hidden_states[li], prompt_len, stride=stride)
        if trace.numel() < 2:
            per_layer.append(0.0)
        else:
            per_layer.append((trace[-1] - trace[0]).item() / length_norm)
    return sum(per_layer) / len(per_layer) if per_layer else 0.0


class MBEVeloReward:
    """
    Length-normalised MBE *velocity* reward.

    Per completion:
        trace[k]      = MBE(hidden_states[:k])  for k=stride..T_comp  (kernel-trick)
        raw_velocity  = trace[-1] - trace[0]
        norm_velocity = raw_velocity / log(min(T_comp, D))   # ~[-1, +1]
        reward        = clip(norm_velocity, ±clip) / scale

    Defaults (scale=4.0, clip=1.0) → max |reward| ≈ 0.25, deliberately
    smaller than correctness (1.0) so velocity acts as a tiebreaker, not
    a primary objective. Length normalisation makes the signal comparable
    across rollouts of different lengths.

    Default `layers=[-1]` (last layer only): late-layer MBE captures
    task-relevant representation diversity; averaging over all layers
    conflates lexical and semantic diversity.

    Usage:
        velo = MBEVeloReward(tokenizer)
        trainer = GRPOTrainer(model=..., reward_funcs=[..., velo], ...)
        velo.set_model(trainer.model)
        trainer.train()
    """

    def __init__(self, tokenizer, layers=None, stride=8,
                 scale=4.0, clip=1.0):
        self.__name__ = "mbe_velocity_reward"
        self.model = None
        self.tokenizer = tokenizer
        self.layers = layers
        self.stride = stride
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
            v = _compute_mbe_velocity_for_completion(
                self.model, self.tokenizer, prompt, completion_text,
                layers=self.layers, stride=self.stride,
            )
            v = max(-self.clip, min(v, self.clip))   # two-sided clip
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