"""LoRA plumbing for the OPD family — the 8B unblocker.

FINDINGS.md sec 5: at 8B, fp32 master weights + Adam alone are ~96 GB, so
full fine-tuning is dead on one 80 GB card and the memory table simply stops.
LoRA moves the optimizer state from "every parameter" to "the adapters", which
is what makes 8B (and comfortable 4B) reachable.

Two traps, both already paid for elsewhere in this repo:

  * claude.md: **LoRA lr is not full-FT lr.** 5e-6 silently undertrains an
    adapter; ~1e-4 is the working range. `--lora 1` therefore RAISES the
    default lr unless you set --lr yourself.
  * a PEFT-wrapped policy is not the bare transformer any more, so anything
    that reaches into `model.model.layers` (bridge_attn's attention probe)
    must unwrap first — see `base_transformer`.
"""


def add_lora_args(ap):
    ap.add_argument("--lora", type=int, default=0,
                    help="1 = train LoRA adapters instead of full FT "
                         "(required at 8B; see FINDINGS.md sec 5)")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_target", default="q_proj,k_proj,v_proj,o_proj,"
                                             "gate_proj,up_proj,down_proj",
                    help="comma list, or 'all-linear'")
    return ap


def lora_config(a):
    """LoraConfig for TRL's peft_config=, or None when --lora 0."""
    if not getattr(a, "lora", 0):
        return None
    from peft import LoraConfig
    tgt = a.lora_target
    return LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        target_modules=(tgt if tgt == "all-linear"
                        else [t for t in tgt.split(",") if t]),
        bias="none", task_type="CAUSAL_LM")


def lora_lr(a, default):
    """Adapters need ~1e-4, not the full-FT 1e-6/5e-6 (claude.md)."""
    if getattr(a, "lora", 0) and a.lr == default:
        print(f"[lora] lr {a.lr} is a full-FT rate and undertrains adapters "
              f"-> using 1e-4 (pass --lr to override)", flush=True)
        return 1e-4
    return a.lr


def base_transformer(model):
    """Walk a (possibly PEFT/DDP/compile-wrapped) model to the module that
    owns `.layers` — what an attention probe needs to hook."""
    m = model
    for _ in range(6):
        if hasattr(m, "layers"):
            return m
        for attr in ("base_model", "model", "module", "transformer",
                     "_orig_mod"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    if hasattr(m, "h"):                      # gpt-2 style naming
        return m
    raise AttributeError(f"cannot find .layers under {type(model).__name__}")
