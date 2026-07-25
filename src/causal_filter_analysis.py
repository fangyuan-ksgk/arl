"""Causal-filter analysis (user, Jul 22): before trusting the H3/H4 gates,
measure what the solver-vs-non-solver contrast actually does.

On champion-model greedy rollouts (train queries having BOTH solvers and
non-solvers), on the student's top-64 support:
  A. NEGATION RATES — among coordinates with a substantial delta
     (|logp_ckpt - logp_base| > tau), what fraction is MATCHED by the
     non-solver mean delta (within rho, relative)? Reported per individual
     solver ckpt and for the solver-average delta. That is the fraction of
     apparent "learning signal" that the causal control disqualifies.
  B. SUPPORT STORYLINE EXTENDED — overlap@64 and top-1 agreement of the
     student vs solvers AND vs non-solvers on the same states: do failing
     ckpts even look different?
  C. BEHAVIORAL DIVERGENCE — positions where solver argmax != non-solver
     argmax (the sparse set a contrast-designed target would live on).

    CUDA_VISIBLE_DEVICES=1 python script/opd/causal_filter_analysis.py --n 400
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script"))

# ============= Wake Up! here is the obvious issues =============
# Issue 1. With OPD we train from the base model, we do NOT train from ckpt300
#          therefore collecting logits from ckpt-300 rollout does not match with 
#          our OPD pipeline, at all. Fix it and re-run experiment
# Issue 2. We want pairwise pos & neg shift similarity to serve as causal ablation 
#          NOT the avg. neg shift here, I've modified it, pls re-run the experiment. 

RUN = ROOT / "output/forgetting_drgrpo/qwen3-0.6b"
# Fix the below line! use the base Qwen3-0.6B model instead!
OPD = ROOT / "output/opd_min_check/checkpoint-300" # -> Issue 1. with OPD, we train from the base model, NOT the ckpt-300, don't be ridiculous. 
STEPS = list(range(25, 301, 25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--m_sol", type=int, default=3)
    ap.add_argument("--n_ns", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="substantial-delta threshold (nats)")
    ap.add_argument("--rho", type=float, default=0.3,
                    help="matched if |d_ns - d_sol| < rho*|d_sol|")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--gpu_util", type=float, default=0.10)
    ap.add_argument("--out", default="output/causal_filter")
    a = ap.parse_args()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from night_merge_eval import INSTR

    train = load_dataset("openai/gsm8k", "main")["train"]
    sol_map, ns_map = {}, {}
    for line in (RUN / "train_forgetting_annotations.jsonl").open():
        r = json.loads(line)
        s = [st for st in STEPS if r["correct_at"].get(str(st))]
        ns = [st for st in STEPS if not r["correct_at"].get(str(st))]
        if s and ns:
            sol_map[r["train_idx"]] = s
            ns_map[r["train_idx"]] = ns
    rng = random.Random(0)
    idxs = rng.sample(sorted(sol_map), min(a.n, len(sol_map)))
    print(f"{len(idxs)} queries with both solvers and non-solvers "
          f"(of {len(sol_map)} eligible)", flush=True)

    tok = AutoTokenizer.from_pretrained(str(OPD))
    prompts = [tok.apply_chat_template(
        [{"role": "user",
          "content": f"{train['question'][i]}\n\n{INSTR} /think"}],
        tokenize=False, add_generation_prompt=True) for i in idxs]
    llm = LLM(model=str(OPD), dtype="bfloat16",
              gpu_memory_utilization=a.gpu_util, disable_log_stats=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                max_tokens=768))
    rolls = []
    for qi, o in zip(idxs, outs):
        rolls.append(dict(p=list(o.prompt_token_ids),
                          c=list(o.outputs[0].token_ids)[:768],
                          sol=sol_map[qi][-a.m_sol:],
                          ns=rng.sample(ns_map[qi],
                                        min(a.n_ns, len(ns_map[qi])))))
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # student support + per-model logp on that support
    @torch.no_grad()
    def logp_on(model_path, rows, topi_map=None):
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16).to("cuda").eval()
        res = {}
        pad = tok.pad_token_id or tok.eos_token_id
        for b0 in range(0, len(rows), a.bs):
            sub = rows[b0:b0 + a.bs]
            seqs = [rolls[i]["p"] + rolls[i]["c"] for i in sub]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(sub), L), pad, dtype=torch.long)
            attn = torch.zeros((len(sub), L), dtype=torch.long)
            for bi, x in enumerate(seqs):
                ids[bi, :len(x)] = torch.tensor(x)
                attn[bi, :len(x)] = 1
            lg = m(input_ids=ids.to("cuda"),
                   attention_mask=attn.to("cuda")).logits
            for bi, i in enumerate(sub):
                s0 = len(rolls[i]["p"])
                s1 = s0 + len(rolls[i]["c"])
                sel = lg[bi, s0 - 1:s1 - 1].float()
                lz = torch.logsumexp(sel, -1)
                if topi_map is None:
                    tk = sel.topk(a.k, -1).indices
                    res[i] = (tk.cpu(),
                              (sel.gather(-1, tk) - lz[:, None]).cpu())
                else:
                    tk = topi_map[i].to(sel.device)
                    res[i] = (sel.gather(-1, tk) - lz[:, None]).cpu()
            del lg
        del m
        torch.cuda.empty_cache()
        return res

    all_rows = list(range(len(rolls)))
    stud = logp_on(str(OPD), all_rows)                 # support + logp
    topi = {i: v[0] for i, v in stud.items()}
    print("scored student", flush=True)
    base = logp_on("Qwen/Qwen3-0.6B", all_rows, topi)
    print("scored base", flush=True)
    need = collections.defaultdict(list)
    for i, r in enumerate(rolls):
        for c in set(r["sol"]) | set(r["ns"]):
            need[c].append(i)
    CK = {}
    for c, rows in sorted(need.items()):
        CK[c] = logp_on(str(RUN / f"checkpoint-{c}"), rows, topi)
        print(f"scored ckpt-{c} ({len(rows)})", flush=True)

    import numpy as np
    neg_per_solver, neg_avg, consensus_survival = [], [], []
    ov_sol, ov_ns, top1_sol, top1_ns, behav = [], [], [], [], []
    for i, r in enumerate(rolls):
        d_sol = np.stack([(CK[c][i] - base[i]).numpy() for c in r["sol"]])
        d_ns = np.stack([(CK[c][i] - base[i]).numpy() for c in r["ns"]])
        # retain every bad checkpoint: ANY matching bad shift invalidates the coordinate
        # What coordinate change 
        big = np.abs(d_sol) > a.tau                    # per solver
        pair_match = (
            np.abs(d_sol[:, None] - d_ns[None])
            < a.rho * np.abs(d_sol[:, None])
        )
        match_any = pair_match.any(axis=1)
        consensus = ((d_sol > a.tau).all(axis=0)
                     | (d_sol < -a.tau).all(axis=0))
        if consensus.any():
            negated_by_any_ns = pair_match.any(axis=(0, 1))
            consensus_survival.append(float(
                (consensus & ~negated_by_any_ns).sum() / consensus.sum()))
        for k_ in range(d_sol.shape[0]):
            if big[k_].any():
                neg_per_solver.append(
                    float((big[k_] & match_any[k_]).sum() / big[k_].sum()))
        d_avg = d_sol.mean(0)
        biga = np.abs(d_avg) > a.tau
        if biga.any():
            avg_match_any = (
                np.abs(d_ns - d_avg[None])
                < a.rho * np.abs(d_avg[None])
            ).any(axis=0)
            neg_avg.append(float(
                (biga & avg_match_any).sum() / biga.sum()))
        # B: support/top1 comparisons (on student support ordering)
        s_lp = stud[i][1].numpy()
        for c in r["sol"]:
            lp = CK[c][i].numpy()
            top1_sol.append(float(
                (lp.argmax(-1) == s_lp.argmax(-1)).mean()))
        for c in r["ns"]:
            lp = CK[c][i].numpy()
            top1_ns.append(float(
                (lp.argmax(-1) == s_lp.argmax(-1)).mean()))
        # C: behavioral divergence solver vs non-solver argmax
        sol_arg = np.stack([CK[c][i].numpy().argmax(-1)
                            for c in r["sol"]])
        ns_arg = np.stack([CK[c][i].numpy().argmax(-1) for c in r["ns"]])
        behav.append(float(
            (sol_arg[0] != ns_arg[0]).mean()))
    rep = dict(
        n_queries=len(rolls), tau=a.tau, rho=a.rho,
        negation_rate_per_solver=float(np.mean(neg_per_solver)),
        negation_rate_avg_delta=float(np.mean(neg_avg)),
        solver_consensus_survival=(float(np.mean(consensus_survival))
                                   if consensus_survival else None),
        top1_agree_student_vs_solver=float(np.mean(top1_sol)),
        top1_agree_student_vs_nonsolver=float(np.mean(top1_ns)),
        behavioral_divergence_sol_vs_ns=float(np.mean(behav)))
    outd = ROOT / a.out
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "report.json").write_text(json.dumps(rep, indent=1))
    for k_, v in rep.items():
        print(f"{k_}: {v}")
    print("CAUSAL_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
