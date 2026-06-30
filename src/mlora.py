# mixture of Lora on a Qwen3-0.6B model (wrapper)
# with extra "router" to select which lora to use
# Game Plan:
# - (1). ensure single lora expert can be trained (via GRPO) to achieve avg. acc
# - (2). ensure union lora expert acc. > avg. acc.
# - (3). learn a proper router to select the right lora to use -- based on query representations (optionally, we can freeze the first layer, and use first layer representations on last query token for it)

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _layer_idx(name):
    """Decoder-layer index a submodule belongs to, or None if outside the stack."""
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _append(t, clone_from, init):
    """Grow a stacked tensor by one row along dim 0: clone row `clone_from`, else `init`."""
    row = t[clone_from:clone_from + 1].clone() if clone_from is not None else init
    return nn.Parameter(torch.cat([t.data, row], 0))


class LoRAExperts(nn.Module):
    """Weighted sum of multiple LoRA adaptors"""
    def __init__(self, base: nn.Linear, n_lora=4, r=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(n_lora, r, base.in_features) / r)
        self.B = nn.Parameter(torch.zeros(n_lora, base.out_features, r))
        self.scale, self.w = alpha / r, None

    def forward(self, x):                                       # x: [B, T, in]
        out = self.base(x)
        if self.w is None:
            return out
        h = torch.einsum("btd,nrd->btnr", x, self.A)            # [B,T,N,r]
        d = torch.einsum("btnr,nor->btno", h, self.B) * self.scale
        return out + torch.einsum("btno,bn->bto", d, self.w)    # weighted LoRA

    @property
    def n_lora(self):
        return self.A.shape[0]

    @torch.no_grad()
    def grow(self, clone_from=None):
        """Append one expert. Fresh -> B=0 (inert) and random A; else clone an expert."""
        self.A = _append(self.A, clone_from, torch.randn(1, *self.A.shape[1:], device=self.A.device) / self.A.shape[1])
        self.B = _append(self.B, clone_from, torch.zeros(1, *self.B.shape[1:], device=self.B.device))
        return self.n_lora - 1

# I want to use MixtureofLoRA to train multiple seeds, then learn to route over them to 
# approach the union accuracy performance
# [Dumb Idea]. well, first we need to train multiple loras (each with different seeds)
#              then we train a router to select the best lora (based on verified greedy rollout)
#              then we test it on validation set
# [Better Idea]. involves "growing" the LoRA experts durining training, so that we naturally
#                distribute different queries into a different expert, avoiding the explicit 
#                "train for multiple seed, each time training a separate LoRAs" mechnism.
# [Issue]. avg. input embedding based routing certainly won't work well. 
#          we need sth better than that. 



class MixtureOfLoRA(nn.Module):
    """Qwen3-0.6B with a mixture of LoRA experts on chosen Linear layers. A router
    picks (soft, or hard top-1) which LoRA to use per input sequence."""
    def __init__(self, model_name="Qwen/Qwen3-0.6B", n_lora=4, r=8, alpha=16,
                 targets=("q_proj", "v_proj", "up_proj", "down_proj"), route_layer=0):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name).requires_grad_(False)
        self.route_layer = route_layer          # route on this decoder layer's output
        self.loras = []
        for name, mod in list(self.model.named_modules()):
            # LoRA only on layers ABOVE route_layer; layers <= route_layer stay base,
            # so route_layer's output is a frozen, route-independent routing feature.
            idx = _layer_idx(name)
            if (isinstance(mod, nn.Linear) and name.split(".")[-1] in targets
                    and idx is not None and idx > route_layer):
                exp = LoRAExperts(mod, n_lora, r, alpha)
                self.model.get_submodule(name.rsplit(".", 1)[0]).add_module(
                    name.rsplit(".", 1)[-1], exp)
                self.loras.append(exp)
        self.router = nn.Linear(self.model.config.hidden_size, n_lora)
        self._last, self._hard, self._last_logits, self._w = None, False, None, None
        self.model.model.layers[route_layer].register_forward_hook(self._routing_hook)

    def _routing_hook(self, _m, _in, out):
        # fires mid-forward AT route_layer: route on its last-token output (detached
        # -> trains the router head only), set w on every expert for layers above.
        h = out[0] if isinstance(out, tuple) else out                # [B, T, d]
        if self._w is not None and h.size(1) == 1:                   # decode: reuse prompt route
            for e in self.loras:
                e.w = self._w
            return
        if self._last is None or int(self._last.max()) >= h.size(1):  # prefill/decode-safe
            idx = torch.full((h.size(0),), h.size(1) - 1, device=h.device)
        else:
            idx = self._last
        ar = torch.arange(h.size(0), device=h.device)
        logits = self.router(h[ar, idx].detach())                    # [B, N]
        self._last_logits = logits
        w = F.softmax(logits, dim=-1)
        if self._hard:
            w = F.one_hot(w.argmax(-1), w.size(-1)).to(w.dtype)
        self._w = w                                                  # cache prompt route
        for e in self.loras:
            e.w = w

    def _set_route_ctx(self, attention_mask, hard):
        # last non-pad token index per sequence (works for either padding side)
        self._last = (None if attention_mask is None else
                      (attention_mask.size(1) - 1
                       - attention_mask.flip(1).float().argmax(1)).long())
        self._hard = hard

    def router_logits(self, input_ids, attention_mask=None, hard=False):
        # single forward; the route_layer hook stashes the logits it computed
        self(input_ids, attention_mask=attention_mask, hard=hard)
        return self._last_logits

    def router_loss(self, input_ids, target_ids, attention_mask=None):
        """Supervised router CE toward externally-chosen target expert ids
        (per-sequence, [B]) -- e.g. the LoRA that passed verified greedy rollout."""
        return F.cross_entropy(self.router_logits(input_ids, attention_mask), target_ids)

    @torch.no_grad()
    def add_expert(self, clone_from=None):
        """Grow every LoRA layer by one expert and widen the router by one logit."""
        for e in self.loras:
            e.grow(clone_from)
        R, dev = self.router, self.router.weight.device
        R.weight = _append(R.weight, clone_from, torch.zeros(1, R.in_features, device=dev))
        R.bias = _append(R.bias, clone_from, torch.zeros(1, device=dev))
        R.out_features += 1
        return self.loras[0].n_lora - 1

    def forward(self, input_ids, attention_mask=None, labels=None, hard=False, **kw):
        self._set_route_ctx(attention_mask, hard)                   # hook sets w mid-pass
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          labels=labels, **kw)

    def trainable_parameters(self):                              # router + all LoRA A/B
        return [p for p in self.parameters() if p.requires_grad]