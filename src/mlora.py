# mixture of Lora on a Qwen3-0.6B model (wrapper)
# with extra "router" to select which lora to use
# Game Plan:
# - (1). ensure single lora expert can be trained (via GRPO) to achieve avg. acc
# - (2). ensure union lora expert acc. > avg. acc.
# - (3). learn a proper router to select the right lora to use -- based on query representations (optionally, we can freeze the first layer, and use first layer representations on last query token for it)

import re
from pathlib import Path

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
        dt = base.weight.dtype
        self.A = nn.Parameter((torch.randn(n_lora, r, base.in_features) / r).to(dt))
        self.B = nn.Parameter(torch.zeros(n_lora, base.out_features, r, dtype=dt))
        self.scale, self.w = alpha / r, None

    def forward(self, x):                                       # x: [B, T, in]
        out = self.base(x)
        if self.w is None:
            return out
        h = torch.einsum("btd,nrd->btnr", x, self.A)            # [B,T,N,r]
        d = torch.einsum("btnr,nor->btno", h, self.B) * self.scale
        if self.w.dim() == 3:                                    # per-token [B,T,N]
            return out + torch.einsum("btno,btn->bto", d, self.w)
        return out + torch.einsum("btno,bn->bto", d, self.w)    # per-seq [B,N]

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
                 targets=("q_proj", "v_proj", "up_proj", "down_proj"), route_layer=0,
                 per_token=False, torch_dtype=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype).requires_grad_(False)
        self.per_token = per_token              # route each position independently
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
        self.router = nn.Linear(self.model.config.hidden_size, n_lora,
                                dtype=next(self.model.parameters()).dtype)
        self._last, self._hard, self._last_logits, self._w = None, False, None, None
        self.model.model.layers[route_layer].register_forward_hook(self._routing_hook)

    def _routing_hook(self, _m, _in, out):
        # fires mid-forward AT route_layer: route on its last-token output (detached
        # -> trains the router head only), set w on every expert for layers above.
        h = out[0] if isinstance(out, tuple) else out                # [B, T, d]
        rdt = self.router.weight.dtype           # router may be fp32 while h is bf16
        if self.per_token:
            # Idea 1.a: every position routes independently on its own route_layer
            # state (causal, so decode steps just route their single position).
            logits = self.router(h.detach().to(rdt))                 # [B, T, N]
            self._last_logits = logits
            w = F.softmax(logits, dim=-1)
            if self._hard:
                w = F.one_hot(w.argmax(-1), w.size(-1))
            for e in self.loras:
                e.w = w.to(h.dtype)
            return
        if self._w is not None and h.size(1) == 1:                   # decode: reuse prompt route
            for e in self.loras:
                e.w = self._w
            return
        if self._last is None or int(self._last.max()) >= h.size(1):  # prefill/decode-safe
            idx = torch.full((h.size(0),), h.size(1) - 1, device=h.device)
        else:
            idx = self._last
        ar = torch.arange(h.size(0), device=h.device)
        logits = self.router(h[ar, idx].detach().to(rdt))            # [B, N]
        self._last_logits = logits
        w = F.softmax(logits, dim=-1)
        if self._hard:
            w = F.one_hot(w.argmax(-1), w.size(-1))
        w = w.to(h.dtype)
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

    @torch.no_grad()
    def load_peft_experts(self, adapter_dirs):
        """Fill expert slot n from the n-th trained PEFT LoRA adapter dir.

        Adapter keys look like
            base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight
        and map onto the LoRAExperts wrapping model.layers.{i}...{target}.
        Layers <= route_layer are not wrapped (they stay base) — adapter
        weights for those layers are skipped (reported in the return value).
        Construct the mixture with the SAME r/alpha/targets the adapters used.
        """
        from safetensors.torch import load_file
        assert len(adapter_dirs) <= self.loras[0].n_lora if self.loras else True
        named = dict(self.model.named_modules())
        n_loaded, n_skipped = 0, 0
        for n, d in enumerate(adapter_dirs):
            sd = load_file(str(Path(d) / "adapter_model.safetensors"))
            for k, v in sd.items():
                if ".lora_A." not in k and ".lora_B." not in k:
                    continue
                # strip peft prefix/suffix -> module path inside self.model
                mod_path = (k.replace("base_model.model.", "")
                             .replace(".lora_A.weight", "")
                             .replace(".lora_B.weight", ""))
                mod = named.get(mod_path)
                if not isinstance(mod, LoRAExperts):
                    n_skipped += 1
                    continue
                if ".lora_A." in k:
                    assert mod.A[n].shape == v.shape, (k, mod.A[n].shape, v.shape)
                    mod.A[n].copy_(v.to(mod.A.dtype))
                else:
                    assert mod.B[n].shape == v.shape, (k, mod.B[n].shape, v.shape)
                    mod.B[n].copy_(v.to(mod.B.dtype))
                n_loaded += 1
        return {"loaded": n_loaded, "skipped_unwrapped": n_skipped}


class MixtureOfLoRAForCausalLM(nn.Module):
    """Adapter that makes a `MixtureOfLoRA` quack like a transformers CausalLM,
    enough for TRL's `GRPOTrainer` with `use_vllm=False`.

    TRL never serves the routed mixture through vLLM (the router picks a
    per-sequence combination of experts that vLLM can't reproduce), so all
    generation goes through `transformers`' `.generate` and all log-probs come
    from `forward` logits. This wrapper exposes exactly that surface:

      * `.config` / `.generation_config` — proxied to the frozen base model.
      * `.generate(input_ids, attention_mask, ...)` — sets the routing context
        from `attention_mask`, resets the decode-route cache, then delegates to
        the base model's `.generate` (the routing hook fires inside it).
      * `forward(input_ids, attention_mask, labels=None, logits_to_keep=None,
        **kw)` — sets the routing context on EVERY call, delegates to the
        mixture (soft routing, `hard=False`, during training). Supports
        `logits_to_keep` (int) so TRL can slice completion logits without
        materialising full fp32 vocab logits at `bs>1`.

    Trainable params are the router head + every expert's A/B (base frozen);
    this is already enforced by `MixtureOfLoRA` — `verify_trainable()` asserts it.

    Hard (top-1) routing is used only at explicit eval via `eval_hard()`; the
    default training/generation path is soft so the router receives gradient.
    """

    # Tag names touched by transformers.Trainer / TRL bookkeeping.
    _tag_names = ["trl", "grpo", "mixture-of-lora"]

    def __init__(self, mixture: MixtureOfLoRA):
        super().__init__()
        self.mixture = mixture
        # transformers.Trainer / modeling utils read/write this dict.
        self.warnings_issued = {}
        # Some transformers utilities probe these attributes with getattr.
        self._eval_hard = False

    # ---- transformers-model-like attribute surface -----------------------
    @property
    def config(self):
        return self.mixture.model.config

    @property
    def generation_config(self):
        return self.mixture.model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.mixture.model.generation_config = value

    @property
    def base_model(self):
        # frozen HF CausalLM under the mixture
        return self.mixture.model

    @property
    def model(self):
        # Backbone (decoder stack) — TRL's liger path reads `unwrapped_model.model`.
        return self.mixture.model.model

    @property
    def lm_head(self):
        return self.mixture.model.lm_head

    @property
    def device(self):
        return next(self.mixture.parameters()).device

    @property
    def dtype(self):
        return next(self.mixture.model.parameters()).dtype

    @property
    def name_or_path(self):
        return getattr(self.config, "_name_or_path", "mixture-of-lora")

    def can_generate(self):
        return True

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.mixture.model, "is_gradient_checkpointing", False)

    def add_model_tags(self, *args, **kwargs):
        # No-op: mirrors transformers.PreTrainedModel.add_model_tags.
        return None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        m = self.mixture.model
        if hasattr(m, "gradient_checkpointing_enable"):
            m.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        m = self.mixture.model
        if hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable(**kwargs)

    def enable_input_require_grads(self):
        m = self.mixture.model
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()

    def tie_weights(self):
        m = self.mixture.model
        if hasattr(m, "tie_weights"):
            m.tie_weights()

    # ---- routing helpers -------------------------------------------------
    def eval_hard(self, hard: bool = True):
        """Toggle hard (top-1) routing for explicit evaluation passes."""
        self._eval_hard = hard

    # ---- forward / generate ---------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                logits_to_keep=None, hard=None, **kw):
        # Set the routing context on EVERY forward from the attention mask.
        # Soft routing during training (hard=False) so the router head keeps
        # gradient; hard only when explicitly requested (eval).
        use_hard = self._eval_hard if hard is None else hard
        if logits_to_keep is not None:
            kw["logits_to_keep"] = logits_to_keep
        return self.mixture(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, hard=use_hard, **kw)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, hard=None, **kw):
        use_hard = self._eval_hard if hard is None else hard
        # Reset the prompt-route cache so each generation recomputes its route
        # at prefill, then reuses it across decode steps (route-once mode).
        self.mixture._w = None
        self.mixture._set_route_ctx(attention_mask, hard=use_hard)
        return self.mixture.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kw)

    # ---- trainable-param bookkeeping ------------------------------------
    def trainable_parameters(self):
        return self.mixture.trainable_parameters()

    def verify_trainable(self):
        """Assert base frozen, router + expert A/B trainable. Returns a summary.

        The experts live *inside* `self.mixture.model` (they replace target
        Linears), so 'base frozen' means every trainable param is either the
        router or an expert A/B — no original pretrained weight learns.
        """
        router_trainable = self.mixture.router.weight.requires_grad
        expert_trainable = all(
            e.A.requires_grad and e.B.requires_grad for e in self.mixture.loras)
        expert_param_ids = set()
        for e in self.mixture.loras:
            expert_param_ids.add(id(e.A))
            expert_param_ids.add(id(e.B))
        base_frozen = not any(
            p.requires_grad for p in self.mixture.model.parameters()
            if id(p) not in expert_param_ids)
        assert router_trainable, "router head must be trainable"
        assert expert_trainable, "expert A/B must be trainable"
        assert base_frozen, "base model must be frozen"
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return {"router_trainable": router_trainable,
                "expert_trainable": expert_trainable,
                "base_frozen": base_frozen,
                "n_trainable": n_train, "n_total": n_total}

    def trainable_state_dict(self):
        """State dict of just the trainable tensors (router + expert A/B)."""
        return {n: p.detach().cpu()
                for n, p in self.named_parameters() if p.requires_grad}