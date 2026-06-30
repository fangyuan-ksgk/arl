"""Uniform "model soup" merge of K checkpoints into one model dir.

Used by the branch-train-merge (Local SGD / DiLoCo) driver `local_sgd_grpo.py`.
Because every branch in a round starts from the SAME init, the uniform soup is just
the elementwise mean of the K weight tensors:

    W_merged = (1/K) * sum_i W_i
             = W_init + (1/K) * sum_i (W_i - W_init)      # == mean of task vectors

so we can average the saved weights directly without re-reading the init. Averaging is
done in fp32 (bf16 sums lose precision across K tensors) then cast back to bf16.

Config + tokenizer files are copied verbatim from the first checkpoint so the merged dir
is a self-contained, loadable model directory.

Usage:
    python script/merge_soup.py --out merged_dir ckpt0 ckpt1 ckpt2
"""
import argparse
import os
import shutil
from pathlib import Path


def load_full_state(d: str):
    """Load a full state dict from a model dir — single OR sharded safetensors, or .bin.
    Bigger models (4B/8B) shard into model-0000N-of-*.safetensors via save_pretrained."""
    import glob, json
    from safetensors.torch import load_file
    single = os.path.join(d, "model.safetensors")
    if os.path.exists(single):
        return load_file(single), True
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
        sd = {}
        for sh in shards:
            sd.update(load_file(os.path.join(d, sh)))
        return sd, True
    shs = sorted(glob.glob(os.path.join(d, "model-*-of-*.safetensors")))
    if shs:
        sd = {}
        for sh in shs:
            sd.update(load_file(sh))
        return sd, True
    b = os.path.join(d, "pytorch_model.bin")
    if os.path.exists(b):
        import torch
        return torch.load(b, map_location="cpu"), False
    raise FileNotFoundError(f"No model weights (single/sharded safetensors or .bin) in {d}")


def merge(ckpt_dirs, out_dir):
    import torch
    from safetensors.torch import load_file, save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    K = len(ckpt_dirs)
    print(f"[merge] soup over K={K} checkpoints -> {out_dir}", flush=True)

    acc = None
    keys = None
    use_st = True
    for i, d in enumerate(ckpt_dirs):
        sd, use_st = load_full_state(d)
        if acc is None:
            keys = list(sd.keys())
            acc = {k: sd[k].float() for k in keys}
        else:
            assert set(sd.keys()) == set(keys), f"key mismatch in {ckpt_dirs[i]}"
            for k in keys:
                acc[k] += sd[k].float()
        del sd
    merged = {k: (acc[k] / K).to(torch.bfloat16).contiguous() for k in keys}

    # Copy config / tokenizer / generation files from the first checkpoint.
    src = ckpt_dirs[0]
    for fn in os.listdir(src):
        if fn.startswith("model") and (fn.endswith(".safetensors") or fn.endswith(".bin")):
            continue  # we write merged weights ourselves
        if fn.endswith(".index.json"):
            continue
        sp = os.path.join(src, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, out / fn)

    if use_st:
        save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(merged, str(out / "pytorch_model.bin"))
    print(f"[merge] wrote {len(keys)} tensors -> {out / ('model.safetensors' if use_st else 'pytorch_model.bin')}",
          flush=True)
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Uniform model-soup merge of K checkpoints")
    ap.add_argument("--out", required=True, help="Output merged model dir")
    ap.add_argument("ckpts", nargs="+", help="K checkpoint dirs to average")
    args = ap.parse_args()
    merge(args.ckpts, args.out)


if __name__ == "__main__":
    main()
