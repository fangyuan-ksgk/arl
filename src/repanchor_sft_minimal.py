"""RepAnchor+SFT, minimal reproduction — the winning config (λ=100, rank_mult=4)
on the hard-GT subset, switchable to pure SFT (--anchor none).

Reproduces the matrix rows (pure hard-set training, 1193 GT rows, 300 steps):
    --anchor none : install 22.0% / test collateral −27.2
    --anchor rep  : install 16.8% / test collateral −0.9

Mechanism (--anchor rep): per-Linear protected input directions M from the SVD
of retention-set activations (effective-rank sized × rank_mult), importance
Ω_j = ||W₀ u_j||², snapshot W*; penalty λ·Σ Ω⊙((W−W*)M)² added to CE each
optimizer step. Weights move freely orthogonal to the retention subspace.
NOTE the two-axis law before reusing: the anchor composes with STYLE-ALIEN
targets (GT); on style-natural targets (student-like text) it inverts to harm.

Usage:
  python repanchor_sft_minimal.py --ckpt <policy_ckpt> \
      --data hard_gt.jsonl --retention broad_t1.jsonl \
      --out out_dir --anchor rep --lam 100 --rank_mult 4
  data/retention rows: {"question": ..., "completion": ...}
"""
import argparse, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--retention", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor", choices=["rep", "none"], default="rep")
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--rank_mult", type=float, default=4.0)
    ap.add_argument("--anchor_tokens", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=1400)
    ap.add_argument("--save_every", type=int, default=15)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16,
                                                 device_map={"": device})

    def encode(path):
        exs = []
        for line in open(path):
            r = json.loads(line)
            prompt = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                             tokenize=False, add_generation_prompt=True)
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            c = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p + c)[:a.max_len]
            exs.append((ids, min(len(p), len(ids))))
        return exs

    hard, ret = encode(a.data), encode(a.retention)

    def batch(exs):
        pad = tok.pad_token_id or tok.eos_token_id
        L = max(len(i) for i, _ in exs)
        inp = torch.tensor([i + [pad] * (L - len(i)) for i, _ in exs], device=device)
        att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in exs], device=device)
        lab = inp.clone()
        for j, (ids, npr) in enumerate(exs):
            lab[j, :npr] = -100
            lab[j, len(ids):] = -100
        return inp, att, lab

    rep = None
    if a.anchor == "rep":
        lins = [(n, m) for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
        acts = {n: [] for n, _ in lins}
        hooks = [m.register_forward_hook(
            (lambda name: lambda mod, i, o: acts[name].append(
                i[0].detach().reshape(-1, i[0].shape[-1])
                [torch.randperm(i[0].reshape(-1, i[0].shape[-1]).shape[0])[:256]]
                .float().cpu()))(n)) for n, m in lins]
        model.eval()
        rng0, seen = random.Random(1), 0
        with torch.no_grad():
            while seen < a.anchor_tokens:
                inp, att, _ = batch(rng0.sample(ret, a.bs))
                model(input_ids=inp, attention_mask=att)
                seen += 256
        for h in hooks:
            h.remove()

        def eff_rank(S):
            return float((S**2).sum()**2 / (S**4).sum().clamp_min(1e-12))

        rep = {}
        for n, m in lins:
            R = torch.cat(acts[n], 0)
            U, S, _ = torch.linalg.svd(R.t() @ R)
            r = max(1, min(int(round(a.rank_mult * eff_rank(S.sqrt()))), S.numel()))
            M = U[:, :r].to(device=device, dtype=torch.float32)
            Om = (m.weight.detach().float() @ M).pow(2).sum(0)
            rep[n] = {"M": M, "Om": Om / Om.mean().clamp_min(1e-12),
                      "Ws": m.weight.detach().clone().float()}
            acts[n] = None
        lin_by_name = dict(lins)
        print(f"[rep] anchored {len(rep)} Linears, "
              f"{sum(v['M'].shape[1] for v in rep.values())} dirs", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = random.Random(0)
    model.train()
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        ce_sum = 0.0
        for _ in range(a.ga):
            inp, att, lab = batch(rng.sample(hard, a.bs))
            ce = model(input_ids=inp, attention_mask=att, labels=lab).loss
            (ce / a.ga).backward()
            ce_sum += float(ce) / a.ga
        pen = 0.0
        if rep is not None:
            loss = torch.zeros((), device=device, dtype=torch.float32)
            for n, v in rep.items():
                D = (lin_by_name[n].weight.float() - v["Ws"]) @ v["M"]
                loss = loss + (v["Om"].unsqueeze(0) * D**2).sum()
            (a.lam * loss).backward()
            pen = float(loss)
        opt.step()
        if step % 10 == 0:
            print(f"step {step}: ce {ce_sum:.3f} anchor {pen:.4f}", flush=True)
        if step % a.save_every == 0:
            d = Path(a.out) / f"checkpoint-{step}"
            model.save_pretrained(d, safe_serialization=True)
            tok.save_pretrained(d)
    print(f"done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
