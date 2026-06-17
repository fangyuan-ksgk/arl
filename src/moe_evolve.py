"""Evolve a dense Qwen MLP into an MoE for multi-seed GRPO training.

Phase 1: at seed s, hard-route everything (prompt+response) to expert s and
         train only that expert.
Phase 2: freeze experts, top-1 self-route, and train only the per-layer routers.
"""

import copy
import torch
import torch.nn as nn


class MoEMLP(nn.Module):
    """Drop-in replacement for a Qwen `*MLP`: S copies of it + a per-token router.

    Two modes (set on every MoEMLP at once via `set_mode`):
      "hard"   : all positions -> active_expert            (Phase 1 / fixed expert)
      "router" : per-token top-1 self-routing              (Phase 2 / eval)
    Set `collect_logits=True` to stash this layer's router logits in `last_logits`
    (and the top-1 pick in `last_choice`) for the router loss.
    forward(x) keeps the original (..., hidden) -> (..., hidden) shape.
    """

    def __init__(self, base_mlp: nn.Module, num_experts: int):
        super().__init__()
        assert num_experts >= 1
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [copy.deepcopy(base_mlp) for _ in range(num_experts)]
        )
        hidden = base_mlp.gate_proj.in_features
        self.router = nn.Linear(hidden, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)  # start uniform over experts
        self.mode = "hard"            # "hard" -> active_expert | "router" -> top-1 self-route
        self.active_expert = 0
        self.collect_logits = False   # stash per-layer router logits for the loss
        self.last_logits = None       # (..., S)
        self.last_choice = None       # (...,) top-1 expert id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.collect_logits:
            self.last_logits = self.router(x)              # (..., S)
        if self.mode == "hard":
            return self.experts[self.active_expert](x)
        # router mode: per-token top-1 self-routing
        logits = self.last_logits if self.collect_logits else self.router(x)
        choice = logits.argmax(-1)                         # (B, T)
        self.last_choice = choice
        out = torch.empty_like(x)
        for s in range(self.num_experts):
            sel = choice == s
            if sel.any():
                out[sel] = self.experts[s](x[sel])
        return out


# --------------------------------------------------------------------------
# Conversion + control helpers (operate over the whole model)
# --------------------------------------------------------------------------
def iter_moe(model):
    for m in model.modules():
        if isinstance(m, MoEMLP):
            yield m


_iter_moe = iter_moe  # internal alias


def set_mode(model, mode, *, active_expert=None):
    """Broadcast (mode, active_expert) to every MoEMLP."""
    for m in iter_moe(model):
        m.mode = mode
        if active_expert is not None:
            m.active_expert = active_expert


def set_collect_logits(model, flag: bool):
    for m in iter_moe(model):
        m.collect_logits = flag


def _decoder_layers(model):
    # Qwen2/Qwen3 CausalLM: model.model.layers ; fall back to a search.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and hasattr(m[0], "mlp"):
            return m
    raise AttributeError("Could not locate decoder layers on model.")


def convert_model_to_moe(model, num_experts, layers=None):
    """Replace decoder-layer `.mlp`s with `MoEMLP`. Returns the replaced idxs.

    layers: iterable of layer indices to convert (None = all layers). Lets you
    control how many / which positions become MoE.
    """
    decoder = _decoder_layers(model)
    keep = set(range(len(decoder))) if layers is None else set(layers)
    replaced = []
    for i, layer in enumerate(decoder):
        if i not in keep or isinstance(layer.mlp, MoEMLP):
            continue
        p = next(layer.mlp.parameters())
        layer.mlp = MoEMLP(layer.mlp, num_experts).to(p.device, dtype=p.dtype)
        replaced.append(i)
    return replaced


def train_expert(model, expert_idx):
    """Phase 1: hard-route to `expert_idx` (prompt+response) and make it the only
    trainable params."""
    model.requires_grad_(False)
    for m in _iter_moe(model):
        m.mode, m.active_expert = "hard", expert_idx
        m.experts[expert_idx].requires_grad_(True)


def train_router(model):
    """Phase 2: top-1 self-routing; train only the per-layer routers."""
    model.requires_grad_(False)
    for m in _iter_moe(model):
        m.mode = "router"
        m.router.requires_grad_(True)


def checkout_expert(model, src_idx, dst_idx):
    """Copy expert src_idx -> dst_idx across all layers (evolutionary seeding)."""
    for m in _iter_moe(model):
        m.experts[dst_idx].load_state_dict(m.experts[src_idx].state_dict())