"""E4 — decodability-constrained chain thinning (user mental model, Jul 17).

CoT = "123456789", q = "1", a = "9": T_CoT = "9" is invalid because the 1->9
transition is undecodable; T_CoT = "124789" is valid iff every kept step sits
at good logit rank given the PREVIOUS KEPT steps. Criterion: per-token rank
of the truncated chain teacher-forced on itself (query + kept prefix), not
p(answer | kept) alone — answer_only fails at its first seam automatically.

Greedy thinning: repeatedly remove the segment whose removal keeps the seam
ranks acceptable (score = mean log-rank of the 8 tokens after each new seam);
stop when no segment can be removed under --tau (max allowed seam rank,
geometric-mean over the window). The answer text is appended as the final
"segment" and must stay decodable, so sufficiency is folded into the same
criterion.

Outputs: kept fraction vs tau per chain, thinned texts, then a vLLM pass
verifying answers still decode greedily from the thinned closed block.

    CUDA_VISIBLE_DEVICES=0 python script/cot_thin.py --taus 5,20,100 --n 60
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script.cot_geometry_v2 import (segments_of, messages_of,  # noqa: E402
                                    make_ablated_prefix_ids, extract, eq)

RUN = Path("output/forgetting_drgrpo/qwen3-0.6b")
OUT = RUN / "cot_geometry_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint-300")
    ap.add_argument("--taus", default="5,20,100")
    ap.add_argument("--n", type=int, default=60, help="chains to thin")
    ap.add_argument("--seam_win", type=int, default=8)
    ap.add_argument("--gpu_util", type=float, default=0.15)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in (OUT / "responses.jsonl").open()]
    random.Random(0).shuffle(rows)
    rows = [r for r in rows if r["correct"] and r.get("answer")][:a.n]

    tok = AutoTokenizer.from_pretrained(str(RUN / a.ckpt))
    model = AutoModelForCausalLM.from_pretrained(
        str(RUN / a.ckpt), dtype=torch.bfloat16).to("cuda").eval()

    def ids_of(t):
        return tok.encode(t, add_special_tokens=False)

    @torch.no_grad()
    def chain_cost(p_ids, seg_tok, kept_idx, PRE, TAIL, n_scaffold):
        """User criterion: geometric-mean logit rank over ALL content tokens
        of the truncated chain (kept segments + answer text), teacher-forced
        on itself. Scaffold tokens (<think> prefix, closing tag) excluded —
        they are trivially predictable and dilute the gate. Returns
        (grank_content, worst_seam_grank)."""
        block = PRE + sum((seg_tok[i] for i in kept_idx), []) + TAIL
        ids = torch.tensor([p_ids + block], device="cuda")
        logits = model(input_ids=ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        ranks = (logits > logits.gather(
            -1, tgt[:, None])).sum(-1).float() + 1.0     # 1-indexed rank
        b0 = len(p_ids) - 1
        blk_r = ranks[b0:b0 + len(block)]
        content = torch.ones(len(block), dtype=torch.bool)
        content[:len(PRE)] = False                        # <think>\n
        a_start = len(block) - len(TAIL)
        content[a_start:a_start + n_scaffold] = False     # \n</think>\n\n
        g_all = float(blk_r[content].log().mean().exp())
        # diagnostic: worst seam window (content-aligned)
        seams, cur = [], len(PRE)
        prev = None
        for i in kept_idx:
            if prev is not None and i != prev + 1:
                seams.append(cur)
            cur += len(seg_tok[i])
            prev = i
        seams.append(cur + n_scaffold)                    # answer start
        worst = 0.0
        for s in seams:
            win = blk_r[s:s + a.seam_win]
            win = win[content[s:s + a.seam_win]]
            if len(win):
                worst = max(worst, float(win.log().mean().exp()))
        return g_all, worst

    taus = [float(x) for x in a.taus.split(",")]
    dump = []
    for qi, r in enumerate(rows):
        segs = segments_of(r["cot"])
        if len(segs) < 3:
            continue
        p_ids = tok.apply_chat_template(
            messages_of(r["question"]), tokenize=True,
            add_generation_prompt=True, enable_thinking=True)
        if hasattr(p_ids, "keys"):
            p_ids = p_ids["input_ids"]
        if p_ids and isinstance(p_ids[0], list):
            p_ids = p_ids[0]
        p_ids = list(p_ids)
        seg_tok = [ids_of(s) for s in segs]
        # fold the answer in as a mandatory final segment
        pre = ids_of("<think>\n")
        n_scaffold = len(ids_of("\n</think>\n\n"))
        tail = ids_of("\n</think>\n\n" + r["answer"][:64]) # Q1. why do we only slice a preifx of the 'tail'? i thi
        n_all = sum(len(t) for t in seg_tok)
        for tau in taus:
            kept = list(range(len(segs)))
            while len(kept) > 1:
                cands = []
                for i in kept:
                    trial = [k for k in kept if k != i]
                    g, w = chain_cost(p_ids, seg_tok, trial, pre, tail,
                                      n_scaffold)
                    cands.append((g, w, i))
                g, w, i = min(cands)
                if g > tau:
                    break
                kept = [k for k in kept if k != i]
            frac = sum(len(seg_tok[i]) for i in kept) / max(n_all, 1)
            dump.append(dict(
                q=qi, tau=tau, kept_idx=kept, kept_frac=frac,
                kept="".join(segs[i] for i in kept),
                question=r["question"], gold=r["gold"]))
            print(f"[{qi}] tau={tau} kept {len(kept)}/{len(segs)} segs "
                  f"({frac:.0%} tokens)", flush=True)
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    from vllm import LLM, SamplingParams
    llm = LLM(model=str(RUN / a.ckpt), dtype="bfloat16",
              gpu_memory_utilization=a.gpu_util, disable_log_stats=True)
    outs = llm.generate(
        [{"prompt_token_ids": make_ablated_prefix_ids(
            tok, messages_of(d["question"]), d["kept"])} for d in dump],
        SamplingParams(temperature=0.0, max_tokens=128))
    for d, o in zip(dump, outs):
        d["S"] = bool(eq(extract(o.outputs[0].text), d["gold"])) # exact match?
        d["continuation"] = o.outputs[0].text
    import numpy as np
    print("\nDECODABILITY-CONSTRAINED THINNING (answer folded into chain):")
    print(f"{'tau':>6s} {'kept_frac':>10s} {'S':>6s} {'n':>4s}")
    for tau in taus:
        v = [d for d in dump if d["tau"] == tau]
        print(f"{tau:6.0f} {np.mean([d['kept_frac'] for d in v]):10.2%} "
              f"{np.mean([d['S'] for d in v]):6.3f} {len(v):4d}")
    with (OUT / "thinning_e4.jsonl").open("w") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")
    print("E4_DONE")


if __name__ == "__main__":
    main()
