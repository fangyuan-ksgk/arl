"""Minimal single-file — attn RTG-additive: the COMPRESSION operating
point. 73.5 +/- 0.4 at ~360 mean CoT tokens (~30% shorter than the
74.5-champion family's ~440-500) — +2.6 over the RL baseline WITH the
compression, no teacher checkpoints.

Idea lineage (Jul 23): per-token attention credit applied MULTIPLICATIVELY
is the 74.5-parity champion (bridgemix_min.py); applied as IMMEDIATE
additive reward it collapses (20.5 — the always-earnable bonus is
reward-hackable); applied as RETURN-TO-GO additive reward it lands here.
"Reward tokens that lead INTO attended regions", not "reward this token".

Per step, after generation:
  1. probe forward (eager clone of policy), layer 21 head-avg; v1
     ####-only answer anchor (no #### => plain Dr.GRPO row).
  2. raw score a_t = attention from answer-predicting rows to token t,
     normalized by the GROUP mean (norm must NOT be per-rollout: that
     pins the rtg bridge to ~0 and deletes trajectory-level signal),
     clamped to [0.2, 5].
  3. s_t = lam * clamp(W_t - 1, -cmax, +cmax)        bounded shaping
  4. s_t <- MEAN-to-go: average of s over the remaining suffix
     (a raw SUM-to-go grows ~sqrt(T) and saturates any clamp)
  5. s_t <- s_t - mean_GROUP(s)                      reward-level baseline
  6. A_t = A_i + s_t                                 added, never multiplied
Ordering guarantee: max differential 2*lam*cmax = 0.5 < the 1.0
correctness gap — shaping can never flip a group's ranking (verified over
7.5k simulated instances, additive_shaping_sim.ipynb).

Knobs and their measured curve: lam .5/cmax .5 = the operating point;
lam .25 -> 73.2; +echo-cap -> 71.4 (do NOT stack hygiene tricks).
Stability: see rtg_compression.ipynb (length trajectory across seeds).

Train:
    accelerate launch --config_file output/night_20260713/zero2_ga16_np1.yaml \
        --num_processes 1 script/opd/attnrtg_min.py train --out output/rtg_min
Eval: python script/opd/attnrtg_min.py eval --out output/rtg_min
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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


def train(a):
    import torch
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset("openai/gsm8k", "main")["train"]
    ds = ds.map(lambda ex: {
        "prompt": [{"role": "user",
                    "content": f"{ex['question']}\n\n{INSTR} /think"}],
        "gold_answer": extract(ex["answer"])})

    def correctness(completions, gold_answer, **kw):
        return [float(equal(extract(c[0]["content"] if isinstance(c, list)
                                    else c), g))
                for c, g in zip(completions, gold_answer)]

    def format_reward(completions, **kw):
        return [0.5 * bool(re.search(
            r"####\s*[\d,\.\-]+",
            c[0]["content"] if isinstance(c, list) else c))
            for c in completions]

    class AttnRTGTrainer(GRPOTrainer):
        def __init__(self, *args, attn_layer=21, w_min=0.2, w_max=5.0,
                     lam=0.5, cmax=0.5, **kw):
            self.attn_layer = attn_layer
            self.w_min, self.w_max = w_min, w_max
            self.lam, self.cmax = lam, cmax
            self._probe = None
            self._probe_step = -1
            self._fails = 0
            super().__init__(*args, **kw)

        def _get_probe(self):
            import torch
            from transformers import AutoModelForCausalLM
            base = self.accelerator.unwrap_model(self.model)
            if self._probe is None:
                self._probe = AutoModelForCausalLM.from_config(
                    base.config, attn_implementation="eager")
                self._probe = self._probe.to(torch.bfloat16).to(
                    self.accelerator.device)
                self._probe.eval()
                [p.requires_grad_(False)
                 for p in self._probe.parameters()]
            if self._probe_step != self.state.global_step:
                sd = {k: v.detach().to(torch.bfloat16)
                      for k, v in base.state_dict().items()}
                self._probe.load_state_dict(sd, strict=True)
                self._probe_step = self.state.global_step
            return self._probe

        def _char_to_tok(self, toks, target):
            lo, hi = 0, len(toks)
            while lo < hi:
                mid = (lo + hi) // 2
                if len(self.processing_class.decode(
                        toks[:mid], skip_special_tokens=True)) < target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        def _answer_span(self, comp_ids_row, mask_row):
            toks = comp_ids_row[mask_row.bool()].tolist()
            text = self.processing_class.decode(
                toks, skip_special_tokens=True)
            m = re.search(r"####\s*(-?[\d,\.]+)", text)
            if m is None:
                return None
            a0 = self._char_to_tok(toks, m.start(1))
            a1 = max(self._char_to_tok(toks, m.end(1)), a0 + 1)
            return a0, a1

        def _attn_W(self, output):
            import torch
            with torch.no_grad():
                probe = self._get_probe()
                p_ids, p_mask = (output["prompt_ids"],
                                 output["prompt_mask"])
                c_ids, c_mask = (output["completion_ids"],
                                 output["completion_mask"])
                ids = torch.cat([p_ids, c_ids], 1)
                attn = torch.cat([p_mask, c_mask], 1)
                B, T = c_ids.shape
                P = p_ids.size(1)
                W = torch.ones((B, T), device=c_ids.device,
                               dtype=torch.float32)
                raw = {}
                for c0 in range(0, B, 2):
                    rr = slice(c0, min(c0 + 2, B))
                    out = probe(input_ids=ids[rr],
                                attention_mask=attn[rr],
                                output_attentions=True)
                    Ah = out.attentions[self.attn_layer].mean(1).float()
                    del out
                    for bl, bg in enumerate(range(rr.start, rr.stop)):
                        span = self._answer_span(c_ids[bg], c_mask[bg])
                        if span is None:
                            continue
                        n_valid = int(c_mask[bg].sum())
                        a0 = min(span[0], n_valid)
                        a1 = min(span[1], n_valid)
                        if a0 < 8:
                            continue
                        raw[bg] = (Ah[bl, P + a0 - 1:P + a1 - 1,
                                      P:P + a0].mean(0), a0)
                    del Ah
                # GROUP-mean normalization (per-rollout would pin the
                # rtg bridge to ~0 — trajectory signal must survive)
                G = self.num_generations
                sums = {}
                for bg, (sc, _) in raw.items():
                    b = bg // G
                    s_, n_ = sums.get(b, (0.0, 0))
                    sums[b] = (s_ + float(sc.sum()), n_ + sc.numel())
                for bg, (sc, a0) in raw.items():
                    s_, n_ = sums[bg // G]
                    mu = max(s_ / max(n_, 1), 1e-8)
                    W[bg, :a0] = (sc / mu).clamp(self.w_min, self.w_max)
                return W

        def _generate(self, prompts):
            out = super()._generate(prompts)         # None-logps guard
            logps = out[5]
            if logps is not None:
                fixed, bad = [], 0
                for lp, c in zip(logps, out[1]):
                    if lp is None:
                        bad += 1
                        fixed.append([0.0] * len(c))
                    elif any(x is None for x in lp):
                        bad += 1
                        fixed.append([0.0 if x is None else x
                                      for x in lp])
                    else:
                        fixed.append(lp)
                if bad:
                    print(f"[rtg] sanitized {bad} rollout(s)", flush=True)
                    out = out[:5] + (fixed,) + out[6:]
            return out

        def _generate_and_score_completions(self, inputs):
            import torch
            output = super()._generate_and_score_completions(inputs)
            try:
                W = self._attn_W(output)
                adv = output["advantages"]
                if adv.dim() == 1:
                    adv = adv.unsqueeze(-1).expand(
                        -1, W.size(1)).contiguous()
                m = output["completion_mask"].to(adv.dtype)
                # ==== the RTG shaping (steps 3-6 of the docstring) =====
                s = (W.to(adv.dtype) - 1.0).clamp(
                    -self.cmax, self.cmax) * m
                num = torch.flip(torch.cumsum(
                    torch.flip(s, [-1]), -1), [-1])
                den = torch.flip(torch.cumsum(
                    torch.flip(m, [-1]), -1), [-1])
                s = (num / den.clamp(min=1)) * m       # mean-to-go
                G = self.num_generations
                for g0 in range(0, s.size(0), G):
                    sl = slice(g0, g0 + G)
                    mu = (s[sl] * m[sl]).sum() \
                        / m[sl].sum().clamp(min=1)
                    s[sl] = (s[sl] - mu) * m[sl]       # group baseline
                output["advantages"] = adv + self.lam * s
                # =======================================================
                if self.state.global_step % 10 == 0:
                    cw = s[output["completion_mask"].bool()]
                    print(f"[rtg] |s| mean {cw.abs().mean():.3f} "
                          f"max {cw.abs().max():.3f}", flush=True)
            except Exception as e:
                self._fails += 1
                print(f"[rtg] FAILED ({type(e).__name__}: {e}) "
                      f"x{self._fails}", flush=True)
                if self._fails >= 5:
                    raise
                return output
            self._fails = 0
            return output

    cfg = GRPOConfig(
        output_dir=a.out, seed=a.seed, max_steps=300,
        learning_rate=5e-6, per_device_train_batch_size=4,
        gradient_accumulation_steps=16, num_generations=8,
        max_completion_length=1024, temperature=1.0,
        loss_type="dr_grpo", scale_rewards="none", beta=0.01,
        use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15,
        save_strategy="steps", save_steps=25, save_only_model=True,
        logging_steps=10, report_to=[], bf16=True)
    trainer = AttnRTGTrainer(
        model=a.model, reward_funcs=[correctness, format_reward],
        args=cfg, lam=a.lam, cmax=a.cmax)
    trainer.train()
    trainer.save_model(a.out)
    print("Training complete (attnrtg_min)")


def evaluate(a):
    import gc
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    test = load_dataset("openai/gsm8k", "main")["test"]
    golds = [extract(r["answer"]) for r in test]
    out = Path(a.out)
    dirs = sorted(out.glob("checkpoint-*"),
                  key=lambda p: int(p.name.split("-")[1]))
    M, steps, lens = [], [], []
    for d in dirs:
        tok = AutoTokenizer.from_pretrained(str(d))
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": f"{q}\n\n{INSTR} /think"}],
            tokenize=False, add_generation_prompt=True)
            for q in test["question"]]
        llm = LLM(model=str(d), dtype="bfloat16",
                  gpu_memory_utilization=0.6, disable_log_stats=True)
        outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                    max_tokens=1024))
        ok = [equal(extract(o.outputs[0].text), g)
              for o, g in zip(outs, golds)]
        ln = float(np.mean([len(o.outputs[0].token_ids) for o in outs]))
        M.append(ok)
        steps.append(int(d.name.split("-")[1]))
        lens.append(ln)
        print(f"[{d.name}] acc {np.mean(ok):.4f}  len {ln:.0f}",
              flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    M = np.array(M)
    final, union = M[-1].mean(), M.any(0).mean()
    print(f"\nfinal {100 * final:.1f} | union {100 * union:.1f} | "
          f"GAP {100 * (union - final):.1f} | final len {lens[-1]:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--out", default="output/rtg_min")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--cmax", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"train": train, "eval": evaluate}[args.cmd](args)
