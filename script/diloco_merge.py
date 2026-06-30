"""Outer-optimizer merge for branch-train-merge GRPO (generalizes uniform soup).

DiLoCo (Douillard et al.) view of the merge step. Each round, K workers train H=P inner AdamW
steps from the shared global params theta_g, producing theta_i. The merge is an OUTER optimizer step:

    outer_grad = theta_g - mean_i(theta_i)        # "pseudo-gradient" (points away from the avg)
    buf        = mu * buf + outer_grad            # outer momentum (persists across rounds = MEMORY)
    update     = outer_grad + mu * buf  (nesterov) | buf  (heavy-ball)
    theta_g    = theta_g - outer_lr * update

Special cases:
    outer_lr=1, mu=0           -> theta_g = mean_i(theta_i)            == uniform model soup (MEMORYLESS)
    outer_lr<1, mu>0, nesterov -> DiLoCo (conservative, momentum-accelerated)                (MEMORY)

"decoupled" variant (our interpretation of decoupled-momentum DiLoCo, since the cited paper is not
fetchable here): take the soup as the base step and ADD a decoupled momentum correction, so momentum
does not feed back into the averaging:
    theta_g = mean_i(theta_i) - outer_lr * mu * buf

The outer momentum buffer is a full fp32 state dict (~model size) kept on disk between rounds.
"""
import os
import shutil
from pathlib import Path


def _load_fp32_state(ref):
    """Return an fp32 CPU state dict for a local checkpoint dir OR an HF model id."""
    import torch
    from safetensors.torch import load_file
    st = os.path.join(ref, "model.safetensors")
    if os.path.isdir(ref) and os.path.exists(st):
        return {k: v.float() for k, v in load_file(st).items()}
    bin_ = os.path.join(ref, "pytorch_model.bin")
    if os.path.isdir(ref) and os.path.exists(bin_):
        return {k: v.float() for k, v in torch.load(bin_, map_location="cpu").items()}
    # HF id (round-0 base): load via transformers (handles download + sharding)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(ref, torch_dtype=torch.float32)
    return {k: v.detach().float().cpu() for k, v in m.state_dict().items()}


def outer_merge(prev_global, ckpt_dirs, out_dir, momentum_path,
                outer_lr=0.7, momentum=0.9, nesterov=True, decoupled=False):
    """One outer-optimizer merge step. Writes merged bf16 weights to out_dir, updates momentum_path.

    prev_global   : theta_g this round started from (local dir or HF id).
    ckpt_dirs     : the K worker checkpoints (theta_i).
    momentum_path : .pt holding the persistent outer momentum buffer (created if absent).
    """
    import torch
    from safetensors.torch import load_file, save_file

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    theta_g = _load_fp32_state(prev_global)

    # Canonical key set = tensors actually SAVED in the branch checkpoints. Qwen3-0.6B ties
    # lm_head.weight to embed_tokens.weight, so the full-model state dict has an extra
    # 'lm_head.weight' that the checkpoints don't — keying off the checkpoint avoids a KeyError
    # (and averaging the tied embed_tokens already covers lm_head). Matches merge_soup.py.
    sd0 = load_file(os.path.join(ckpt_dirs[0], "model.safetensors"))
    keys = list(sd0.keys())
    missing = [k for k in keys if k not in theta_g]
    assert not missing, f"prev_global missing checkpoint keys: {missing[:5]}"

    # mean_i(theta_i)
    K = len(ckpt_dirs)
    avg = {k: sd0[k].float() for k in keys}
    del sd0
    for d in ckpt_dirs[1:]:
        sd = load_file(os.path.join(d, "model.safetensors"))
        for k in keys:
            avg[k] += sd[k].float()
        del sd
    for k in keys:
        avg[k] /= K

    buf = torch.load(momentum_path, map_location="cpu") if os.path.exists(momentum_path) else \
        {k: torch.zeros_like(theta_g[k]) for k in keys}

    new_global = {}
    for k in keys:
        outer_grad = theta_g[k] - avg[k]
        buf[k] = momentum * buf[k] + outer_grad
        if decoupled:
            new_global[k] = avg[k] - outer_lr * momentum * buf[k]
        else:
            upd = outer_grad + momentum * buf[k] if nesterov else buf[k]
            new_global[k] = theta_g[k] - outer_lr * upd

    torch.save(buf, momentum_path)

    # copy config/tokenizer from a worker checkpoint; write merged weights as bf16
    src = ckpt_dirs[0]
    for fn in os.listdir(src):
        if fn.startswith("model") and (fn.endswith(".safetensors") or fn.endswith(".bin")):
            continue
        if fn.endswith(".index.json"):
            continue
        sp = os.path.join(src, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, out / fn)
    save_file({k: new_global[k].to(torch.bfloat16).contiguous() for k in keys},
              str(out / "model.safetensors"), metadata={"format": "pt"})
    mode = "decoupled-diloco" if decoupled else ("nesterov" if nesterov else "heavyball")
    print(f"[diloco] {mode} outer step: lr={outer_lr} mu={momentum} K={K} -> {out_dir}", flush=True)
    return str(out)
