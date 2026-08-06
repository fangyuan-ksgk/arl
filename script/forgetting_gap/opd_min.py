"""Minimal single-file — the CHAMPION: imitate-mode OPD (top-K logit
distillation from each query's own best checkpoint) + k3 KL anchor to the base
model at weight 1.0, over round-robin teacher-homogeneous batches.
Historic record: 74.5 +/- 0.2 GSM8K greedy at ~440-500 mean CoT tokens, vs a
~70.5 Dr.GRPO baseline.

Rebuilt (Jul 25) from script/opd/directopd_trainer.py (mode="imitate", the
top-k OPD control of Eq. 4 + CkptGroupedSampler) and script/groupopd_min.py
(whose skeleton this file shares).

MEASURED ON THE REBUILD (Jul 25, seed 0, teacher = output/forgetting_drgrpo/
seed0, trl 1.7 / transformers 5.12 / vllm 0.23): final 71.4, best ckpt 73.2
(step 200), union-over-ckpts 88.2, 608 CoT tokens; base model 65.1. So the
distillation works (+6 to +8 over base) and the best ckpt reaches the recorded
74.5-class number, but it does NOT beat the Dr.GRPO baseline as re-measured on
this stack (74.5 +/- 0.7 over 4 seeds, script/opd/README.md). The recorded
ordering was established against a 70.5 baseline that this stack does not
reproduce — read the README before quoting a delta.

Method, per optimizer step:
  1. sample G rollouts per prompt from the STUDENT (on-policy, T=1.0); the
     batch is teacher-HOMOGENEOUS — every prompt in it is solved by the same
     checkpoint (--interleave rr cycles the ckpts round-robin, one teacher per
     step; "blocked" walks them in ascending order = curriculum).
  2. support S_t = student's own top-K (K=64) at every visited prefix;
     student log-probs renormalized on S_t -> pbar (Eq. 11).
  3. teacher pi_T = that QUERY's best ckpt (latest snapshot that greedily
     solves it, from train_forgetting_annotations.jsonl), queried only on S_t
     and renormalized there -> qbar.
  4. imitate loss (Eq. 4 top-k control): reverse-KL of pbar against qbar,
     weighted by the student's own (grad-carrying) renormalized probs:
         dense = sum_t sum_{v in S_t} softmax(topv)_v * (log pbar_v - log qbar_v)
     i.e. match pi_T ABSOLUTELY on the visited support, not the pi_T/pi_ref
     logit SHIFT of mode="shift".
  5. + alpha * k3 KL(pi_theta || pi_S) on the sampled tokens, alpha = 1.0
     FIXED (the adaptive-alpha controller of Eq. 16 only exists for the
     directional "shift" mode). pi_S = the student's init = the base model =
     the same frozen copy used as the shift anchor.

Why teacher-homogeneous batches matter (exp6, Jul 16): different checkpoints
solve different queries and therefore carry different logit-shift directions;
mixing per-query teachers inside one optimizer step makes their distillation
gradients interfere. Round-robin keeps one teacher per step while still
covering every ckpt each epoch.

Prerequisites: a Dr.GRPO teacher run with per-step checkpoints AND its
train_forgetting_annotations.jsonl (script/opd/drgrpo_min.py train + annotate).
The student starts from the BASE model, never from a teacher ckpt.

Train (single process; teachers + policy + vLLM share the GPU):
    CUDA_VISIBLE_DEVICES=0 python script/opd/opd_min.py train \
        --teacher_run output/forgetting_drgrpo/seed0 --out output/opd_min
Eval:  python script/opd/opd_min.py eval --out output/opd_min
"""
import argparse
import json
import re
import sys
import time as _time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import domains  # noqa: E402
import peftcfg  # noqa: E402

INSTR = ("Solve the problem step by step inside <think>...</think>. "
         "After </think>, give a brief final explanation and end your "
         "response with a line of the exact form:\n#### <number>\n"
         "where <number> is the final numeric answer with no units, no "
         "commas, and no extra text.")


DATASET = ["gsm8k"]          # set from --dataset before make_dataset()


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


def ckpt_steps_of(run):
    return sorted(int(p.name.split("-")[1])
                  for p in Path(run).glob("checkpoint-*"))


def ckpt_accuracy(ann_path, ckpt_steps):
    """step -> greedy accuracy over the annotated train queries."""
    tot = {s: 0 for s in ckpt_steps}
    n = 0
    for line in open(ann_path):
        r = json.loads(line)
        n += 1
        for s in ckpt_steps:
            tot[s] += bool(r["correct_at"].get(str(s)))
    return {s: tot[s] / max(n, 1) for s in ckpt_steps}


def ckpt_lengths(run_dir, ckpt_steps):
    """(train_idx, step) -> token length of that ckpt's greedy train answer.

    Read from the train_step<S>.jsonl the annotate pass already writes, so a
    length-aware teacher rule costs nothing extra.
    """
    out = {}
    for s in ckpt_steps:
        f = Path(run_dir) / f"train_step{s}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            out[(r["idx"], s)] = r.get("n_tokens")
    return out


def build_teacher_table(ann_path, ckpt_steps, pick="latest", seed=0):
    """train_idx -> teacher ckpt step (None if no ckpt solves the query).

    Queries are typically solved by MANY ckpts, so "latest" piles most of the
    table onto the last snapshot and "earliest" onto the first — both are rule
    artifacts. "latest" is the champion rule (exp1-6); "random" maximizes
    teacher diversity.

    "best" (Jul 30) exists because `latest` INVERTS with scale. At 0.6B the
    teacher run improves monotonically, so the latest solver is also a good
    solver. At 1.7B the teacher peaks at step 40 and decays; `latest` then
    sends ~99% of queries to post-peak checkpoints, which are also the most
    overfit (train-test gap 6.4 vs 3.2 at the peak), and the OPD arms
    faithfully distil a damaged teacher (FINDINGS.md sec 2.2). `best` keeps the
    per-query solver constraint but breaks the tie toward the globally
    strongest checkpoint instead of the newest one.
    """
    import random as _random
    rng = _random.Random(seed)
    acc = ckpt_accuracy(ann_path, ckpt_steps) if pick == "best" else {}
    lens = (ckpt_lengths(Path(ann_path).parent, ckpt_steps)
            if pick == "shortest" else {})
    if pick == "shortest" and not lens:
        raise SystemExit("--pick shortest needs train_step<S>.jsonl next to "
                         f"{ann_path} (run `drgrpo_min.py annotate` first)")
    if pick == "best":
        print(f"[opd] per-ckpt train acc "
              f"{ {s: round(100 * a_, 1) for s, a_ in acc.items()} } "
              f"-> global best ckpt-{max(acc, key=lambda s: (acc[s], s))}",
              flush=True)
    table = {}
    for line in open(ann_path):
        r = json.loads(line)
        solvers = [s for s in ckpt_steps if r["correct_at"].get(str(s))]
        if not solvers:
            table[r["train_idx"]] = None
        elif pick == "latest":
            table[r["train_idx"]] = solvers[-1]
        elif pick == "earliest":
            table[r["train_idx"]] = solvers[0]
        elif pick == "random":
            table[r["train_idx"]] = rng.choice(solvers)
        elif pick == "best":
            # highest-accuracy solver; ties -> the later checkpoint
            table[r["train_idx"]] = max(solvers, key=lambda s: (acc[s], s))
        elif pick == "shortest":
            # among ckpts that SOLVE this query, the one that did it in the
            # fewest tokens; missing lengths sort last
            i = r["train_idx"]
            table[i] = min(solvers,
                           key=lambda s: (lens.get((i, s)) or 10 ** 9, s))
        else:
            raise ValueError(f"unknown pick={pick}")
    return table


def forgotten_set(ann_path, ckpt_steps):
    """train_idx solved by SOME ckpt but not by the last one — the actual
    forgetting set, the only queries where an intermediate teacher knows
    something the final checkpoint has lost."""
    out = set()
    last = str(ckpt_steps[-1])
    for line in open(ann_path):
        r = json.loads(line)
        ca = r["correct_at"]
        if any(ca.get(str(s)) for s in ckpt_steps) and not ca.get(last):
            out.add(r["train_idx"])
    return out


def make_dataset(teacher_run, ckpt_steps, pick, seed, subset="all",
                 keep_untaught=False):
    from datasets import load_dataset
    ann = Path(teacher_run) / "train_forgetting_annotations.jsonl"
    table = build_teacher_table(ann, ckpt_steps, pick=pick, seed=seed)
    if subset != "all":
        forg = forgotten_set(ann, ckpt_steps)
        keep = forg if subset == "forgotten" else set(table) - forg
        table = {k: (v if k in keep else None) for k, v in table.items()}
        print(f"[opd] subset={subset}: {len(keep & set(table))} queries "
              f"({len(forg)} forgotten of {len(table)})", flush=True)
    ds, _ = domains.get(DATASET[0])["load"]()
    ds = ds.map(lambda ex, i: {"best_ckpt": table.get(i) or -1},
                with_indices=True)
    if not keep_untaught:
        ds = ds.filter(lambda ex: ex["best_ckpt"] > 0)
    sizes = {}
    for c in ds["best_ckpt"]:
        sizes[c] = sizes.get(c, 0) + 1
    print(f"[opd] {len(ds)} teachable queries "
          f"(of {len(table)} annotated); per-ckpt {dict(sorted(sizes.items()))}",
          flush=True)
    return ds


class CkptGroupedSampler:
    """RepeatSampler variant with teacher-homogeneous generation batches.

    Indices are grouped by best_ckpt; each batch of `batch_size` unique
    prompts shares a single teacher. "rr" cycles ckpts round-robin (one
    teacher per optimizer step), "blocked" visits them in ascending order.
    Per-group tail remainders (< batch_size) are dropped, matching TRL's
    RepeatSampler; the yield structure (mini_repeat_count / repeat_count) is
    identical so _prepare_inputs buffering still works.
    """

    def __init__(self, ckpts, mini_repeat_count, batch_size=1,
                 repeat_count=1, mode="rr", seed=None, upsample=False):
        import torch
        self.upsample = upsample
        self.groups = {}
        for i, c in enumerate(ckpts):
            self.groups.setdefault(int(c), []).append(i)
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.mode = mode
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.num_batches = sum(len(v) // batch_size
                               for v in self.groups.values())

    def _batches(self):
        import torch
        per_ckpt = []
        for c in sorted(self.groups):
            idx = self.groups[c]
            perm = torch.randperm(len(idx), generator=self.generator).tolist()
            shuf = [idx[j] for j in perm]
            if self.upsample and 0 < len(shuf) < self.batch_size:
                # Without this, every ckpt group smaller than one generation
                # batch is DROPPED — on the Jul 26 table that silently deleted
                # 11 of 15 teachers and left 95/110 batches on ckpt-300, i.e.
                # single-teacher distillation. Upsampling with replacement
                # gives each teacher at least one batch per epoch.
                need = self.batch_size - len(shuf)
                extra = [shuf[int(torch.randint(len(shuf), (1,),
                                                generator=self.generator))]
                         for _ in range(need)]
                shuf = shuf + extra
            n = len(shuf) // self.batch_size
            per_ckpt.append([shuf[k * self.batch_size:(k + 1) * self.batch_size]
                             for k in range(n)])
        if self.mode == "blocked":
            return [b for bl in per_ckpt for b in bl]
        batches = []
        for k in range(max(len(bl) for bl in per_ckpt)):
            for bl in per_ckpt:
                if k < len(bl):
                    batches.append(bl[k])
        return batches

    def __iter__(self):
        for batch in self._batches():
            for _ in range(self.repeat_count):
                for i in batch:
                    for _ in range(self.mini_repeat_count):
                        yield i

    def __len__(self):
        return (self.num_batches * self.batch_size
                * self.repeat_count * self.mini_repeat_count)


def train(a):
    import torch
    TD = torch.float32 if a.teacher_dtype == "fp32" else torch.bfloat16
    # anchor precision is separable from teacher precision. At 4B an fp32
    # teacher is 16 GB and an fp32 anchor another 16 GB, which does not fit
    # beside a bf16 policy + 8-bit Adam + a 3072-token activation graph. The
    # TEACHER is the one that must stay fp32: precision_probe.py measures only
    # 4-6x SNR on the teacher/student log-ratio in bf16. The anchor only
    # supplies a k3 KL penalty, a far coarser quantity, so it takes the hit.
    AD = (TD if a.anchor_dtype == "same"
          else (torch.float32 if a.anchor_dtype == "fp32" else torch.bfloat16))
    # fp32 weights + TF32 matmul: precision_probe.py measures SNR 221 (sampled
    # token) / 498 (top-K support) on the teacher/student log-ratio, vs 4-6 for
    # bf16 — i.e. TF32 is numerically ample here, at ~bf16 speed. `--tf32 0`
    # forces strict fp32 matmul (~3x slower) if you need the reference.
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    from trl import GRPOConfig, GRPOTrainer

    steps = ckpt_steps_of(a.teacher_run)
    if not steps:
        raise SystemExit(f"no checkpoint-* in {a.teacher_run}")
    if a.max_teacher_step:
        steps = [s for s in steps if s <= a.max_teacher_step]
        if not steps:
            raise SystemExit(f"--max_teacher_step {a.max_teacher_step} "
                             f"excludes every checkpoint")
        print(f"[opd] teacher pool capped at step {a.max_teacher_step}: "
              f"{steps}", flush=True)
    ds = make_dataset(a.teacher_run, steps, a.pick, a.seed,
                      subset=a.subset, keep_untaught=bool(a.hybrid))

    D = domains.get(a.dataset)

    def correctness(completions, **kw):                 # logging only
        n = len(completions)
        cols = {k: v for k, v in kw.items()
                if isinstance(v, (list, tuple)) and len(v) == n}
        rows = ([dict(zip(cols, vals)) for vals in zip(*cols.values())]
                if cols else [{} for _ in range(n)])
        return [float(D["correct"](c[0]["content"] if isinstance(c, list)
                                   else c, r))
                for c, r in zip(completions, rows)]

    class OPDTrainer(GRPOTrainer):
        def __init__(self, *args, teacher_run, anchor_path, topk=64,
                     alpha=1.0, interleave="rr", upsample=False,
                     teacher_cache=12, hybrid=False, profile=False, **kw):
            self.teacher_run = Path(teacher_run)
            self.anchor_path = anchor_path
            self.topk, self.alpha = topk, alpha
            self.interleave = interleave
            self.upsample = upsample
            self.cache_cap = teacher_cache
            self.hybrid = hybrid
            self.profile = profile
            super().__init__(*args, **kw)
            from transformers import AutoModelForCausalLM
            self._AutoM = AutoModelForCausalLM
            self._cache = OrderedDict()          # LRU of teacher ckpts
            dev = self.accelerator.device
            # pi_S (KL reference) == the student's init == the base model ==
            # the anchor: one frozen bf16 copy serves both roles.
            self.anchor = AutoModelForCausalLM.from_pretrained(
                anchor_path, dtype=AD).to(dev)
            self.anchor.eval()
            [p.requires_grad_(False) for p in self.anchor.parameters()]
            print(f"[opd] imitate topk={topk} alpha={alpha} "
                  f"interleave={interleave} anchor={anchor_path}", flush=True)

        # ------------------------------------------------------- profiling
        # `--profile 1` breaks the step down by phase. Costs a cuda sync per
        # block, so it is off by default.
        def _tick(self):
            import torch
            if self.profile:
                torch.cuda.synchronize()
            return _time.perf_counter()

        def _tock(self, name, t0):
            if not self.profile:
                return
            import torch
            torch.cuda.synchronize()
            self._metrics["train"][f"t/{name}"].append(
                _time.perf_counter() - t0)

        # -------------------------------------------------------- teachers
        def teacher(self, step):
            import torch
            if step in self._cache:
                self._cache.move_to_end(step)
                return self._cache[step]
            while len(self._cache) >= self.cache_cap:
                _, old = self._cache.popitem(last=False)
                del old
                torch.cuda.empty_cache()
            t0 = self._tick()
            m = self._AutoM.from_pretrained(
                str(self.teacher_run / f"checkpoint-{step}"),
                dtype=TD).to(self.accelerator.device)
            self._tock("teacher_load", t0)
            m.eval()
            [p.requires_grad_(False) for p in m.parameters()]
            self._cache[step] = m
            return m

        # ------------------------------------------- teacher-homogeneous rr
        def _get_train_sampler(self, dataset=None):
            if self.interleave in (None, "none", "mixed"):
                return super()._get_train_sampler(dataset)
            if dataset is None:
                dataset = self.train_dataset
            sampler = CkptGroupedSampler(
                dataset["best_ckpt"],
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size
                // self.num_generations,
                repeat_count=self.num_iterations
                * self.args.steps_per_generation,
                mode=self.interleave,
                seed=self.args.seed, upsample=self.upsample)
            print(f"[opd] interleave={self.interleave} -> "
                  f"{sampler.num_batches} homogeneous batches/epoch",
                  flush=True)
            return sampler

        def _generate_and_score_completions(self, inputs):
            import torch
            out = super()._generate_and_score_completions(inputs)
            # thread each row's teacher id through TRL's micro-batch splitter
            out["best_ckpt"] = torch.tensor(
                [int(r["best_ckpt"]) for r in inputs],
                device=out["completion_ids"].device)
            return out

        # ------------------------------------------------------------ loss
        def _compute_loss(self, model, inputs):
            p_ids, p_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            c_ids, c_mask = (inputs["completion_ids"],
                             inputs["completion_mask"])
            ids = torch.cat([p_ids, c_ids], 1)
            attn = torch.cat([p_mask, c_mask], 1)
            T, K = c_ids.size(1), self.topk

            t0 = self._tick()
            logits = model(input_ids=ids, attention_mask=attn).logits
            self._tock("student_fwd", t0)
            logits = logits[:, -T - 1:-1]                 # next-token
            # memory-lean logp: never materialize a full-vocab fp32 softmax
            logZ = torch.logsumexp(logits.float(), -1)
            topv_raw, topi = logits.topk(K, -1)           # support S_t
            topv = topv_raw.float() - logZ.unsqueeze(-1)  # top-K logp (grad)
            pbar = torch.softmax(topv.detach(), -1)       # Eq. 11

            def at(m, rows):
                """no-grad logp of m on S_t and at the sampled tokens."""
                ok_, oy_ = [], []
                with torch.no_grad():
                    for c in range(0, len(rows), 2):      # row-chunked
                        rr = rows[c:c + 2]
                        tl = m(input_ids=ids[rr],
                               attention_mask=attn[rr]
                               ).logits[:, -T - 1:-1]
                        lz = torch.logsumexp(tl.float(), -1)
                        ok_.append(tl.float().gather(-1, topi[rr])
                                   - lz.unsqueeze(-1))
                        oy_.append(tl.float().gather(
                            -1, c_ids[rr].unsqueeze(-1)).squeeze(-1) - lz)
                        del tl, lz
                return torch.cat(ok_), torch.cat(oy_)

            rows_all = torch.arange(ids.size(0), device=ids.device)
            t0 = self._tick()
            _, ref_y = at(self.anchor, rows_all)          # pi_S at y_t
            self._tock("anchor_fwd", t0)
            t_k = torch.zeros_like(topv)                  # pi_T on S_t
            taught = inputs["best_ckpt"] > 0
            for s_ in inputs["best_ckpt"].unique().tolist():
                if s_ <= 0:                  # hybrid: no teacher for this row
                    continue
                rows = (inputs["best_ckpt"] == s_).nonzero(as_tuple=True)[0]
                tch = self.teacher(int(s_))     # timed separately (may load)
                t0 = self._tick()
                t_k[rows], _ = at(tch, rows)
                self._tock("teacher_fwd", t0)

            lp = logits.float().gather(
                -1, c_ids.unsqueeze(-1)).squeeze(-1) - logZ

            # ===================== imitate (Eq. 4, top-k) =================
            qbar = torch.softmax(t_k, -1)                 # teacher on S_t
            # per-token contribution, so taught and untaught rows can share one
            # token-mean instead of being averaged separately
            opd_tok = (pbar.log() - qbar.log()).mul(
                torch.softmax(topv, -1)).sum(-1)          # (B, T)
            # ==============================================================

            # ------- hybrid: Dr.GRPO on rows no checkpoint ever solved -----
            # OPD is only defined where a teacher exists. On GSM8K that is
            # 94.5% of prompts so dropping the rest is invisible; on MBPP it is
            # 36.6%, i.e. the method silently forfeits 63% of the training
            # signal — and precisely the hard tail. --hybrid keeps those rows
            # and trains them with the plain Dr.GRPO term, so OPD becomes an
            # ADDITION to the baseline rather than a replacement of its data.
            per_tok = torch.where(taught[:, None], opd_tok,
                                  torch.zeros_like(opd_tok))
            if self.hybrid:
                adv = inputs.get("advantages")
                if adv is not None:
                    pg = -(adv.detach().float()[:, None] * lp)   # (B, T)
                    per_tok = torch.where(taught[:, None], per_tok, pg)
            dense = (per_tok * c_mask).sum() / c_mask.sum().clamp(min=1)

            # k3 KL(pi_theta || pi_S) on the sampled tokens, fixed alpha
            d = ref_y - lp
            kl = (torch.exp(d) - d - 1).mul(c_mask).sum() \
                / c_mask.sum().clamp(min=1)

            loss = dense + self.alpha * kl
            self._metrics["train"]["opd/dense"].append(float(dense.detach()))
            self._metrics["train"]["opd/kl"].append(float(kl.detach()))
            self._metrics["train"]["opd/frac_taught"].append(
                float(taught.float().mean()))
            return loss

    cfg = GRPOConfig(
        output_dir=a.out, seed=a.seed, max_steps=a.max_steps,
        learning_rate=peftcfg.lora_lr(a, 1e-6),
        per_device_train_batch_size=a.per_device_bs,
        gradient_accumulation_steps=a.ga, num_generations=a.num_generations,
        max_completion_length=(a.max_completion_length
                               or D["max_completion_length"]),
        temperature=1.0,
        loss_type="dr_grpo", scale_rewards="none",
        use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.gpu_util,
        gradient_checkpointing=bool(a.grad_ckpt), optim=a.optim,
        **({"vllm_max_model_length": a.vllm_max_len} if a.vllm_max_len else {}),
        model_init_kwargs=({"dtype": a.model_dtype}
                           if a.model_dtype != "auto" else None),
        save_strategy="steps", save_steps=a.save_steps, save_only_model=True,
        logging_steps=10, report_to=[], bf16=not a.fp32_policy)
    trainer = OPDTrainer(
        model=a.model, reward_funcs=[correctness], args=cfg, train_dataset=ds,
        teacher_run=a.teacher_run, anchor_path=a.model, topk=a.topk,
        alpha=a.alpha, interleave=a.interleave,
        upsample=bool(a.upsample), teacher_cache=a.teacher_cache,
        hybrid=bool(a.hybrid), profile=bool(a.profile),
        peft_config=peftcfg.lora_config(a))
    trainer.train()
    trainer.save_model(a.out)
    print("Training complete (opd_min)", flush=True)


def evaluate(a):
    import gc
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    D = domains.get(a.dataset)
    _, test = D["load"]()
    if getattr(a, "eval_samples", 0):
        test = test.select(range(min(a.eval_samples, len(test))))
    out = Path(a.out)
    dirs = sorted(out.glob("checkpoint-*"),
                  key=lambda p: int(p.name.split("-")[1]))
    # --only_steps re-reads a subset of checkpoints; --eval_tag keeps the
    # result in its own file so a re-eval at a DIFFERENT token budget cannot
    # overwrite the one it is meant to be compared against
    only = set(getattr(a, "only_steps", None) or [])
    if only:
        dirs = [d for d in dirs if int(d.name.split("-")[1]) in only]
        if not dirs:
            raise SystemExit(f"--only_steps {sorted(only)} matches no ckpt")
    tag = getattr(a, "eval_tag", "") or ""
    M, steps, lens, cots = [], [], [], []
    for d in dirs:
        tok = AutoTokenizer.from_pretrained(str(d))
        prompts = [tok.apply_chat_template(test[i]["prompt"], tokenize=False,
                                           add_generation_prompt=True)
                   for i in range(len(test))]
        llm = LLM(model=str(d), dtype="bfloat16",
                  gpu_memory_utilization=a.eval_util, disable_log_stats=True)
        outs = llm.generate(prompts, SamplingParams(
            temperature=0.0,
            max_tokens=(a.max_completion_length
                        or D["max_completion_length"])))
        ok, ln, ct = [], [], []
        for i, o in enumerate(outs):
            text = o.outputs[0].text
            ok.append(bool(D["correct"](text, test[i])))
            ln.append(len(o.outputs[0].token_ids))
            e = D["answer_end"](text)
            ct.append(len(tok.encode(text[:e] if e is not None else text,
                                     add_special_tokens=False)))
        M.append(ok)
        steps.append(int(d.name.split("-")[1]))
        lens.append(float(np.mean(ln)))
        cots.append(float(np.mean(ct)))
        print(f"[{d.name}] acc {np.mean(ok):.4f}  len {lens[-1]:.0f}  "
              f"cot {cots[-1]:.0f}", flush=True)
        (out / f"test{tag}_step{steps[-1]}.jsonl").write_text("".join(
            json.dumps({"idx": i, "step": steps[-1], "correct": bool(o),
                        "n_tokens": int(t), "n_cot_tokens": int(c)}) + "\n"
            for i, (o, t, c) in enumerate(zip(ok, ln, ct))))
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    M = np.array(M)
    final, union = M[-1].mean(), M.any(0).mean()
    print(f"\nfinal {100 * final:.1f} | best {100 * M.mean(1).max():.1f} | "
          f"union {100 * union:.1f} | GAP {100 * (union - final):.1f} | "
          f"final len {lens[-1]:.0f} | final cot {cots[-1]:.0f}")
    (out / f"eval_summary{tag}.json").write_text(json.dumps(
        {"dataset": a.dataset, "n_test": int(M.shape[1]),
         "max_tokens": int(a.max_completion_length
                           or D["max_completion_length"]),
         "steps": steps, "acc": [float(x) for x in M.mean(1)],
         "mean_tokens": lens, "mean_cot_tokens": cots,
         "final": float(final), "union": float(union),
         "gap": float(union - final)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--out", default="output/opd_min")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--teacher_run", default="output/forgetting_drgrpo/seed0")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="k3 KL-to-init weight (champion: 1.0, fixed)")
    ap.add_argument("--interleave", default="rr",
                    choices=["rr", "blocked", "mixed", "none"],
                    help="rr = round-robin homogeneous batches (champion); blocked = ascending curriculum; mixed/none = TRL's default shuffled order, teachers mixed inside a step")
    ap.add_argument("--subset", default="all",
                    choices=["all", "forgotten", "kept"],
                    help="forgotten = only queries an intermediate ckpt solves "
                         "and the FINAL ckpt gets wrong (the churn the method "
                         "exists to harvest)")
    ap.add_argument("--upsample", type=int, default=0,
                    help="1 = upsample ckpt groups smaller than one batch "
                         "instead of dropping them (keeps all teachers)")
    ap.add_argument("--pick", default="latest",
                    choices=["latest", "earliest", "random", "best",
                             "shortest"],
                    help="which solver ckpt teaches a query. best = the "
                         "globally strongest solver (use at >=1.7B, where the "
                         "teacher peaks early and `latest` distils a decayed, "
                         "overfit checkpoint)")
    ap.add_argument("--hybrid", type=int, default=0,
                    help="1 = keep prompts NO checkpoint ever solved and "
                         "train them with plain Dr.GRPO, so OPD adds to the "
                         "baseline instead of replacing its dataset. Matters "
                         "where the ever-solved rate is low: GSM8K 94.5%%, "
                         "MATH 60.9%%, MBPP 36.6%%")
    ap.add_argument("--max_teacher_step", type=int, default=0,
                    help="ignore teacher checkpoints after this step (0 = "
                         "all). A blunter version of --pick best: caps the "
                         "pool at the teacher's peak.")
    ap.add_argument("--dataset", default="gsm8k",
                    choices=["gsm8k", "math", "mbpp"])
    ap.add_argument("--max_completion_length", type=int, default=0)
    ap.add_argument("--anchor_dtype", default="same",
                    choices=["same", "fp32", "bf16"],
                    help="precision of the k3 anchor copy. bf16 at >=4B: two "
                         "fp32 4B copies (teacher + anchor) do not fit")
    ap.add_argument("--teacher_dtype", default="fp32",
                    choices=["fp32", "bf16"],
                    help="fp32 (default): with bf16 teachers only ~4-6x of the "
                         "teacher/student log-ratio survives quantization "
                         "(script/opd/precision_probe.py)")
    ap.add_argument("--tf32", type=int, default=1,
                    help="TF32 matmul for the fp32 paths (default on)")
    ap.add_argument("--fp32_policy", action="store_true",
                    help="disable bf16 autocast on the policy too")
    ap.add_argument("--teacher_cache", type=int, default=4,
                    help="LRU teacher slots; rr uses one teacher per step, so "
                         "4 is plenty and keeps fp32 teachers affordable")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--per_device_bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=16,
                    help="raise for a bigger batch; prompts per generation "
                         "batch = per_device_bs*ga/num_generations")
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--model_dtype", default="auto",
                    help="bfloat16 for >=4B: TRL otherwise "
                         "upcasts to fp32, doubling weights "
                         "and grads (32GB at 4B)")
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--vllm_max_len", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--profile", type=int, default=0,
                    help="log per-phase step timings as t/* metrics")
    ap.add_argument("--eval_samples", type=int, default=0)
    ap.add_argument("--eval_tag", default="",
                    help="suffix for eval_summary{tag}.json / test{tag}_*.jsonl"
                         " — use when re-evaluating at a different budget")
    ap.add_argument("--only_steps", type=int, nargs="*",
                    help="evaluate only these checkpoint steps")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--gpu_util", type=float, default=0.15)
    ap.add_argument("--eval_util", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    peftcfg.add_lora_args(ap)
    args = ap.parse_args()
    DATASET[0] = args.dataset
    {"train": train, "eval": evaluate}[args.cmd](args)
