"""Minimal single-file reference for the FOUR exp39 Group-OPD losses — and
nothing else (no hgate/blend/bok/ensemble; for the champion imitate loss
itself see script/opd/opd_min.py, whose skeleton this file shares).

Shared signal, computed on the student's own rollouts:
    r_t = log q_T(y_t) - log p_S(y_t)   per-token teacher/student log-ratio
                                        (both full-softmax renormalized at
                                        the sampled token y_t)
    ok  = rollout correctness in {0,1}  (####-else-last-number vs gold)
    lam = 1.0

The four arms (results Jul 22, main trainer, seed 0, champion rr/K64 base):
    a  A = center(mean_t r_t)          scales the imitate DIVERGENCE
       -> 66.3 / 19.8   harmful
    b  A = center(mean_t r_t + lam*ok) scales the divergence
       -> 18.6 / 59.8   catastrophic (neg-weighted KL = anti-distillation
                        on ~half of each group; correctness dominates A)
    c  loss = -sum_t (r_t + lam*ok) * log p(y_t)      REINFORCE, additive
       -> 74.8 / 15.2   first arm >= champion 74.5+-0.2; seeds queued
    d  loss = -sum_t (r_t * ok) * log p(y_t)          REINFORCE, gated
       -> 71.3 / 17.6   baseline-level
Champion geometry assumption (a/b centering): per_device_train_batch_size
== num_generations, so each micro-batch is ONE query's G rollouts and
A = R - R.mean() is the within-group advantage.

Train (same prerequisites and yaml caveat as opd_min.py):
    accelerate launch --config_file output/night_20260713/zero2_ga16_np1.yaml \
        --num_processes 1 script/groupopd_min.py --group c \
        --out output/groupopd_min_c
"""
import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSTR = ("Solve the problem step by step inside <think>...</think>. "
         "After </think>, give a brief final explanation and end your "
         "response with a line of the exact form:\n#### <number>\n"
         "where <number> is the final numeric answer with no units, no "
         "commas, and no extra text.")


def extract(text):
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"-?[\d,]*\.?\d+", text)
    return m[-1].replace(",", "") if m else None


def equal(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def build_teacher_table(ann_path, ckpt_steps):
    table = {}
    for line in open(ann_path):
        r = json.loads(line)
        solvers = [s for s in ckpt_steps if r["correct_at"].get(str(s))]
        table[r["train_idx"]] = solvers[-1] if solvers else None
    return table


def main(a):
    import torch
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    steps = list(range(25, 301, 25))
    ds = load_dataset("openai/gsm8k", "main")["train"]
    table = build_teacher_table(
        Path(a.teacher_run) / "train_forgetting_annotations.jsonl", steps)
    ds = ds.map(lambda ex, i: {
        "prompt": [{"role": "user",
                    "content": f"{ex['question']}\n\n{INSTR} /think"}],
        "gold_answer": extract(ex["answer"]),
        "best_ckpt": table.get(i) or -1}, with_indices=True)
    ds = ds.filter(lambda ex: ex["best_ckpt"] > 0)

    def correctness(completions, gold_answer, **kw):   # logging only
        return [float(equal(extract(c[0]["content"] if isinstance(c, list)
                                    else c), g))
                for c, g in zip(completions, gold_answer)]

    class GroupOPDTrainer(GRPOTrainer):
        def __init__(self, *args, teacher_run, topk, alpha, group, lam,
                     **kw):
            self.topk, self.alpha = topk, alpha
            self.group, self.lam = group, lam
            self.teacher_run = Path(teacher_run)
            super().__init__(*args, **kw)
            from transformers import AutoModelForCausalLM
            self._cache = OrderedDict()
            self._AutoM = AutoModelForCausalLM
            self.anchor = AutoModelForCausalLM.from_pretrained(
                a.model, dtype=torch.bfloat16).to(self.accelerator.device)
            self.anchor.eval()
            [p.requires_grad_(False) for p in self.anchor.parameters()]

        def teacher(self, step):
            if step not in self._cache:
                while len(self._cache) >= 12:
                    _, m = self._cache.popitem(last=False)
                    del m
                    torch.cuda.empty_cache()
                m = self._AutoM.from_pretrained(
                    str(self.teacher_run / f"checkpoint-{step}"),
                    dtype=torch.bfloat16).to(self.accelerator.device)
                m.eval()
                [p.requires_grad_(False) for p in m.parameters()]
                self._cache[step] = m
            return self._cache[step]

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            dev = out["completion_ids"].device
            out["best_ckpt"] = torch.tensor(
                [int(r["best_ckpt"]) for r in inputs], device=dev)
            # per-rollout correctness for the b/c/d coefficients
            texts = self.processing_class.batch_decode(
                out["completion_ids"], skip_special_tokens=True)
            out["roll_ok"] = torch.tensor(
                [float(equal(extract(t), r["gold_answer"]))
                 for t, r in zip(texts, inputs)], device=dev)
            return out

        def _compute_loss(self, model, inputs):
            p_ids, p_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            c_ids, c_mask = (inputs["completion_ids"],
                             inputs["completion_mask"])
            ids = torch.cat([p_ids, c_ids], 1)
            attn = torch.cat([p_mask, c_mask], 1)
            T, K = c_ids.size(1), self.topk

            logits = model(input_ids=ids, attention_mask=attn).logits
            logits = logits[:, -T - 1:-1]
            logZ = torch.logsumexp(logits.float(), -1)
            topv_raw, topi = logits.topk(K, -1)          # student top-K
            topv = topv_raw.float() - logZ.unsqueeze(-1)
            pbar = torch.softmax(topv.detach(), -1)

            def at(m, rows):
                ok_, oy_ = [], []
                with torch.no_grad():
                    for c in range(0, len(rows), 2):
                        rr = rows[c:c + 2]
                        tl = m(input_ids=ids[rr],
                               attention_mask=attn[rr]).logits[:, -T - 1:-1]
                        lz = torch.logsumexp(tl.float(), -1)
                        ok_.append(tl.float().gather(-1, topi[rr])
                                   - lz.unsqueeze(-1))
                        oy_.append(tl.float().gather(
                            -1, c_ids[rr].unsqueeze(-1)).squeeze(-1) - lz)
                return torch.cat(ok_), torch.cat(oy_)

            rows_all = torch.arange(ids.size(0), device=ids.device)
            _, ref_y = at(self.anchor, rows_all)
            t_k = torch.empty_like(topv)                 # teacher top-K logp
            t_y = torch.empty_like(ref_y)                # teacher logp @ y_t
            for s_ in inputs["best_ckpt"].unique().tolist():
                rows = (inputs["best_ckpt"] == s_).nonzero(
                    as_tuple=True)[0]
                t_k[rows], t_y[rows] = at(self.teacher(int(s_)), rows)

            # student logp at the sampled tokens (WITH grad — this is the
            # only term REINFORCE arms c/d push)
            lp = logits.float().gather(
                -1, c_ids.unsqueeze(-1)).squeeze(-1) - logZ
            m_ = c_mask.float()
            r_tok = (t_y - lp).detach()                  # r_t  (B,T)
            ok = inputs["roll_ok"].float()               # (B,)

            # ============ the four exp39 losses — nothing else ============
            if self.group in ("a", "b"):
                # advantage-weighted DIVERGENCE (both failed).
                # dtok = per-token reverse-KL of student top-K vs teacher
                # (identical to the champion imitate term in opd_min.py).
                qbar = torch.softmax(t_k, -1)
                dtok = ((pbar.log() - qbar.log())
                        * torch.softmax(topv, -1)).sum(-1)          # (B,T)
                R = (r_tok * m_).sum(-1) / m_.sum(-1).clamp(min=1)  # (B,)
                if self.group == "b":
                    R = R + self.lam * ok
                A = (R - R.mean()).detach()   # within-group centering —
                # the failure-by-design: A<0 on ~half the group, and
                # negative-weighted KL actively pushes the student AWAY
                # from the teacher there (anti-distillation).
                dense = (dtok * A[:, None] * m_).sum() \
                    / m_.sum().clamp(min=1)
            elif self.group == "c":
                # REINFORCE on the sampled token, ADDITIVE correctness
                # bonus (the 74.8 arm): every token is reinforced by how
                # much the teacher prefers it, plus a flat +lam if the
                # rollout reached the right answer.
                coef = r_tok + self.lam * ok[:, None]
                dense = -(coef * lp * m_).sum() / m_.sum().clamp(min=1)
            else:                                        # "d"
                # REINFORCE, MULTIPLICATIVE gate: only correct rollouts
                # learn — discards the teacher's signal on wrong rollouts,
                # which is where repair happens (71.3, baseline-level).
                coef = r_tok * ok[:, None]
                dense = -(coef * lp * m_).sum() / m_.sum().clamp(min=1)
            # ==============================================================

            # k3 KL anchor to base on sampled tokens (same as opd_min.py)
            d = ref_y - lp
            kl = (torch.exp(d) - d - 1).mul(c_mask).sum() \
                / c_mask.sum().clamp(min=1)
            return dense + self.alpha * kl

    cfg = GRPOConfig(
        output_dir=a.out, seed=a.seed, max_steps=300,
        learning_rate=1e-6, per_device_train_batch_size=4,
        gradient_accumulation_steps=16, num_generations=4,
        max_completion_length=1024, temperature=1.0,
        loss_type="dr_grpo", scale_rewards="none",
        use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15,
        save_strategy="steps", save_steps=25, save_only_model=True,
        logging_steps=10, report_to=[], bf16=True)
    trainer = GroupOPDTrainer(
        model=a.model, reward_funcs=[correctness], args=cfg,
        train_dataset=ds, teacher_run=a.teacher_run, topk=a.topk,
        alpha=1.0, group=a.group, lam=a.lam)
    trainer.train()
    trainer.save_model(a.out)
    print("Training complete (groupopd_min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list("abcd"))
    ap.add_argument("--out", default="output/groupopd_min")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--teacher_run",
                    default="output/forgetting_drgrpo/qwen3-0.6b")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
