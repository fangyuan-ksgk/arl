"""Deliverable 2 — on-policy distillation that reproduces the 78% GSM8K result.

[TBD]. We need to ablate out forward / backward KL loss
[TBD]. We need to ablate out offpolicy distillation & onpolicy distillation
[TBD]. I want to test out "on-policy context distillation" --> here we need to verify conditioned acc >> unconditioned acc to motivate the design
[TBD]. I want to test out "mix-policy distillation" (pick a point of divergence, have the teacher rollout from it (conditioned on its own correct rollout, optionally), then teach it to the student)


The soup (uniform average of 8 GRPO seeds) sits at ~77.3%. Naive offline SFT distillation
FROM the soup actually loses ~7 points. The win comes from ON-POLICY distillation against a
per-query master teacher:

  * Per query, the MASTER is the (strongest) seed that solves it greedily. The union of
    masters covers every solvable query.
  * The STUDENT (init = soup) samples its OWN rollout for the query.
  * We minimize a per-token KL between student and master on the student's rollout:
        L = E_{y~student} sum_t KL( pi_master(.|y<t) || pi_student(.|y<t) )    (forward KL, default)
    On-policy (student-sampled) + dense (full-vocab) signal avoids the off-policy SFT collapse.
  * --contested: train only on queries solved by <= max_solvers seeds (the hard ones),
    concentrating the useful signal instead of diluting it ~5:1 with easy queries.

The validated 78% recipe (this script's defaults):
    --contested --max_solvers 5 --epochs 2 --lr 1e-6 --bs 8 --max_new 512 --limit 2000  (forward KL)
  -> greedy@1 = 0.7801 (round 1). NOTE: do NOT iterate -- a 2nd round regressed to 0.762.

(Issue). Then it is unclear whether on policy distillation even help here, since we start from the soup merge (77%)
         and are only able to reach 78%, that is well within noise level. 

KL direction: forward KL(teacher||student) is mass-covering and stable; reverse
KL(student||teacher) is mode-seeking and collapse-prone here -- keep the default unless
ablating.

Prereq: per-seed greedy masters in --greedy_dir (run repro/gen_masters.py once).

Usage (reproduce 78%):
    python repro/onpolicy_distill.py --out output/repro_onpolicy_contested
Iterate / ablate:
    python repro/onpolicy_distill.py --kl reverse ...
    python repro/onpolicy_distill.py --anchor 0.5 ...          # GKD CE anchor on master's trace
    python repro/onpolicy_distill.py --init <prev_round>/model  # WARP/ReST-style iteration
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = "Ksgk-fy/arl-gsm8k-multiseed"
BASE = "Qwen/Qwen3-0.6B"
STEP = 200
SEEDS = list(range(8))


def build_soup(out_dir):
    """Uniform average of the 8 seed checkpoints (the student init). Cached on disk."""
    import torch
    torch.set_num_threads(16)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    out_dir = Path(out_dir)
    if (out_dir / "model.safetensors").exists():
        print("[cache] soup init at", out_dir, flush=True)
        return str(out_dir)
    base = {k: v.float() for k, v in load_file(hf_hub_download(BASE, "model.safetensors")).items()}
    snap = snapshot_download(REPO, allow_patterns=[f"seed{s}/checkpoint-{STEP}/*" for s in SEEDS])
    seeds = [load_file(os.path.join(snap, f"seed{s}", f"checkpoint-{STEP}", "model.safetensors")) for s in SEEDS]
    names = [k for k in base if k in seeds[0]]
    mu = {k: torch.stack([seeds[i][k].float() for i in range(len(SEEDS))], 0).mean(0) for k in names}
    ref = os.path.join(snap, f"seed{SEEDS[0]}", f"checkpoint-{STEP}")
    m = AutoModelForCausalLM.from_pretrained(ref, dtype=torch.bfloat16)
    m.load_state_dict({k: mu[k].to(torch.bfloat16) for k in names}, strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(ref).save_pretrained(out_dir)
    print("soup init saved ->", out_dir, flush=True)
    return str(out_dir)


def select_masters(greedy_dir, contested, max_solvers):
    """Per query, pick the strongest seed that solves it greedily.

    Returns (pairs, mtrace, n_kept): pairs = [(master_seed, query_idx), ...];
    mtrace[q] = master's correct greedy completion text (for the optional GKD anchor).
    """
    fs = sorted(glob.glob(os.path.join(greedy_dir, "seed*.json")))
    if not fs:
        sys.exit(f"no greedy masters in {greedy_dir} -- run: python repro/gen_masters.py")
    seeds = [int(f.split("seed")[-1].split(".")[0]) for f in fs]
    data = [json.load(open(f)) for f in fs]
    C = np.array([[r["solutions"][0]["correct"] for r in d["records"]] for d in data])  # (n_seeds, n_q)
    acc = C.mean(1)
    by_master, n_kept = {}, 0
    for q in range(C.shape[1]):
        solvers = np.where(C[:, q])[0]
        if not len(solvers):
            continue                                  # unsolvable by any seed -> skip
        if contested and len(solvers) > max_solvers:
            continue                                  # easy (many seeds solve) -> skip
        n_kept += 1
        master = seeds[max(solvers, key=lambda i: acc[i])]
        by_master.setdefault(master, []).append(q)
    pairs = [(ms, q) for ms, qs in by_master.items() for q in qs]
    mtrace = {q: data[seeds.index(ms)]["records"][q]["solutions"][0]["text"]
              for ms, qs in by_master.items() for q in qs}
    questions = [r["question"] for r in data[0]["records"]]
    return pairs, mtrace, questions, n_kept


def main():
    p = argparse.ArgumentParser(description="On-policy distillation (reproduces 78% on GSM8K)")
    p.add_argument("--out", default=os.path.join(PROJECT, "output/repro_onpolicy_contested"))
    p.add_argument("--greedy_dir", default=os.path.join(PROJECT, "output/gsm8k_distill/greedy"))
    p.add_argument("--init", default=None, help="Student init dir (default: build the soup). Pass a prior round to iterate.")
    p.add_argument("--limit", type=int, default=2000, help="Max master queries to use.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--temp", type=float, default=1.0, help="Student rollout sampling temperature.")
    p.add_argument("--kl_temp", type=float, default=1.0, help="Softmax temperature for the KL.")
    p.add_argument("--kl", choices=["forward", "reverse"], default="forward",
                   help="forward=KL(teacher||student) mass-covering/stable; reverse=mode-seeking/collapse-prone")
    p.add_argument("--contested", dest="contested", action="store_true", default=True,
                   help="Train only on contested queries (default on; this is the 78% recipe).")
    p.add_argument("--all_queries", dest="contested", action="store_false",
                   help="Disable contested focus; train on every solvable query.")
    p.add_argument("--max_solvers", type=int, default=5, help="(contested) keep queries solved by <= this many seeds.")
    p.add_argument("--anchor", type=float, default=0.0,
                   help="GKD-style: add anchor*CE on the master's correct greedy trace.")
    p.add_argument("--shuffle_seed", type=int, default=-1, help=">=0 shuffles training-pair order (WARP diversity).")
    p.add_argument("--no_eval", action="store_true", help="Skip the final greedy@1 eval.")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    from eval_gsm8k import build_prompts

    # --- master teachers per query ---
    pairs, mtrace, questions, n_kept = select_masters(args.greedy_dir, args.contested, args.max_solvers)
    if args.contested:
        print(f"contested-focus: {n_kept} queries (solvers 1..{args.max_solvers})", flush=True)
    if args.shuffle_seed >= 0:
        import random
        random.Random(args.shuffle_seed).shuffle(pairs)
    pairs = pairs[:args.limit]
    masters_used = sorted(set(m for m, _ in pairs))
    print(f"master queries: {len(pairs)}; teachers used: {masters_used}", flush=True)

    snap = snapshot_download(REPO, allow_patterns=[f"seed{s}/checkpoint-{STEP}/*" for s in masters_used])
    soup = args.init if args.init else build_soup(out / "soup_init")
    print(f"student init: {soup}", flush=True)

    tok = AutoTokenizer.from_pretrained(soup)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    student = AutoModelForCausalLM.from_pretrained(soup, dtype=torch.bfloat16).cuda()
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    t0 = time.time(); step = 0
    for ep in range(args.epochs):
        order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])   # group by master -> load each teacher once
        cur_master, teacher = None, None
        i = 0
        while i < len(order):
            ms = pairs[order[i]][0]
            if ms != cur_master:
                del teacher; torch.cuda.empty_cache()
                tdir = os.path.join(snap, f"seed{ms}", f"checkpoint-{STEP}")
                teacher = AutoModelForCausalLM.from_pretrained(tdir, dtype=torch.bfloat16).cuda().eval()
                cur_master = ms
            # batch of same-master queries
            batch_q = []
            while i < len(order) and pairs[order[i]][0] == ms and len(batch_q) < args.bs:
                batch_q.append(pairs[order[i]][1]); i += 1

            prompts = build_prompts(tok, [questions[q] for q in batch_q])
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            plen = enc.input_ids.shape[1]

            # on-policy rollout from the student (kv-cache ON to sample fast)
            student.eval(); student.config.use_cache = True
            with torch.no_grad():
                full = student.generate(**enc, do_sample=True, temperature=args.temp, top_p=1.0,
                                        max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id)
            student.train(); student.config.use_cache = False

            mask = (full != tok.pad_token_id).long()
            comp_mask = torch.zeros_like(full); comp_mask[:, plen:] = 1
            comp_mask = comp_mask * mask

            # logits predict token t from <t: align positions plen-1 .. end-1
            slog = student(input_ids=full, attention_mask=mask).logits
            with torch.no_grad():
                tlog = teacher(input_ids=full, attention_mask=mask).logits
            sl = slog[:, plen - 1:-1, :] / args.kl_temp
            tl = tlog[:, plen - 1:-1, :] / args.kl_temp
            m = comp_mask[:, plen:].float()
            logp_s = F.log_softmax(sl.float(), -1)
            logp_t = F.log_softmax(tl.float(), -1)
            if args.kl == "reverse":
                kl = (logp_s.exp() * (logp_s - logp_t)).sum(-1)      # D(student||teacher) mode-seeking
            else:
                kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)      # D(teacher||student) mass-covering (stable)
            loss = (kl * m).sum() / m.sum().clamp(min=1)

            # optional GKD anchor: CE on the master's correct greedy trace (right-padded for clean masking)
            if args.anchor > 0:
                tok.padding_side = "right"
                a_txt = [build_prompts(tok, [questions[q]])[0] + mtrace[q] for q in batch_q]
                a_enc = tok(a_txt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
                p_lens = [len(tok(build_prompts(tok, [questions[q]])[0]).input_ids) for q in batch_q]
                tok.padding_side = "left"
                albl = a_enc.input_ids.clone()
                for bi, pl in enumerate(p_lens):
                    albl[bi, :pl] = -100
                albl[a_enc.attention_mask == 0] = -100
                alog = student(input_ids=a_enc.input_ids, attention_mask=a_enc.attention_mask).logits
                V = alog.shape[-1]
                ce = F.cross_entropy(alog[:, :-1].reshape(-1, V).float(), albl[:, 1:].reshape(-1), ignore_index=-100)
                loss = loss + args.anchor * ce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"  ep{ep} step{step} master=seed{ms} kl_loss={loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        del teacher; torch.cuda.empty_cache()

    sft = out / "model"
    student.config.use_cache = True
    student.save_pretrained(sft)
    tok.padding_side = "right"; tok.save_pretrained(sft)
    print("saved ->", sft, flush=True)
    teacher = None; del student      # teacher already freed at end of the epoch loop
    import gc; gc.collect(); torch.cuda.empty_cache()

    if args.no_eval:
        return
    # eval in a fresh process (parent still holds a CUDA context -> low util)
    eo = out / "eval.json"
    subprocess.run([sys.executable, os.path.join(HERE, "eval_gsm8k.py"), "--model_path", str(sft),
                    "--label", "onpolicy_distill", "--out", str(eo), "--max_tokens", "2048",
                    "--max_model_len", "3072", "--gpu_memory_utilization", "0.5"],
                   env=dict(os.environ, PYTHONPATH=f"{PROJECT}:{HERE}"), check=True)
    print(f"onpolicy_distill greedy@1 = {json.loads(eo.read_text())['accuracy']:.4f} "
          f"(soup 0.773, target ~0.780)", flush=True)


if __name__ == "__main__":
    main()
