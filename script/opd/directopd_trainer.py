"""Direct-OPD (arXiv 2607.05394) as a GRPOTrainer subclass — faithful to the

We have some lost files, too, including: 
first we train Dr.GRPO, tracking intermediate ckpts every 20 steps
in all intermediate ckpt, we pick the one that solves each query (pick the last ckpt that solve any query)
opd_min.py | Performs imitate mode OPD with per-token logit distillation + KL (K3) regularization with weight of 1
             best config require round-roubin style data order: each batch consistn of queries one ckpt solve, and 
             we do round-boubin style batch assignment, first batch for ckpt0, second batch for ckpt1, ... 
reinforce_opd.py | perform token level reinforce loss | on-par to champion

distill_rl.py | seems to work, inferior to champion 
bridge_attn.py | track attn from answer tokens backward, normalize them by 1/mean, call it 'b', track attention from query to the cot tokens, noramlize it with 1/mean, call it 'f', each token receives extra reward of (b+f)/2 besides its trajectory broadcasted reward
                 equivalently we operate on the advantage values, we found that with on-policy attn scrore, this gives on-par 
                 performance to the champion config

attnrtg_min.py | survived script, it works inferior in accurary to champion config, but it 
#                reduces the CoT length on multi-seed reproduction


Paper details implemented:
  * On-policy: sequences sampled from the student every step (inherited from
    TRL's GRPO generation loop = verl-style; temperature 1.0, rollout n and
    batch geometry set in the launch config; paper Table 3: n=4, batch 64).
  * Top-k support: S_t = top-16 of the STUDENT distribution at each visited
    prefix; student probs renormalized on S_t (Eq. 11).
  * Dense implicit reward: r_t(v) = log pi_T(v|s_t) - log pi_Tref(v|s_t)
    (Eq. 10), teachers queried only on S_t.
  * Rao-Blackwellized surrogate with detached coefficient
    A_t(v) = stop_grad(pbar_t(v) * r_t(v)) (Eq. 13-14); loss term
    -sum_t sum_{v in S_t} A_t(v) log pi_theta(v|s_t).
  * KL anchor to the student's INITIAL policy pi_S, coefficient alpha
    (Eq. 15), token-level k3 estimator on sampled tokens.
  * Adaptive KL control (Eq. 16): alpha <- clip(alpha*(1+eps*sgn(rbar)),
    0.5, 2.5), eps=0.01, rbar = batch mean of pbar*r over visited prefixes.

Our instantiation: pi_T varies PER QUERY = that train query's best checkpoint
(latest snapshot that greedily solves it, from train_forgetting_annotations);
pi_Tref = the shift anchor p (base model or a GRPO ckpt). All teacher ckpts
are preloaded bf16. mode="imitate" gives the top-k OPD control (Eq. 4,
imitate pi_T absolutely) for exp5.

GRPO rewards/advantages still run for rollout logging but the loss is fully
replaced.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from trl import GRPOTrainer


def build_best_ckpt_table(ann_path, ckpt_steps, pick="latest", seed=0):
    """train_idx -> teacher ckpt step, or None if no ckpt solves the query.

    pick: which solving ckpt becomes the teacher. Queries are typically
    solved by MANY ckpts (median 12/12 for the easy 81%; mean 9.5/12), so
    "latest" piles 81% of the table onto ckpt-300 and "earliest" piles the
    same mass onto ckpt-25 — both are rule artifacts, not data facts.
      latest   — last solving snapshot (original rule, exp1-6)
      earliest — first solving snapshot (least drifted from base)
      random   — uniform over the query's solving ckpts (max diversity)
    """
    import random as _random
    rng = _random.Random(seed)
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
        else:
            raise ValueError(f"unknown pick={pick}")
    return table

class CkptGroupedSampler(torch.utils.data.Sampler):
    """RepeatSampler variant with teacher-homogeneous generation batches.

    Tests the hypothesis that mixing per-query teachers inside one optimizer
    step makes their logit-distillation gradients interfere. Indices are
    grouped by best_ckpt; each batch of `batch_size` unique prompts shares a
    single teacher. Batch order: "rr" cycles ckpts round-robin (interleaved
    distillation, one teacher per step), "blocked" visits ckpts in ascending
    order (curriculum). Per-group tail remainders (< batch_size) are dropped,
    matching RepeatSampler; yield structure (mini_repeat_count / repeat_count)
    is identical to RepeatSampler so _prepare_inputs buffering still works.
    """

    def __init__(self, ckpts, mini_repeat_count, batch_size=1, repeat_count=1,
                 mode="rr", seed=None):
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
        per_ckpt = []
        for c in sorted(self.groups):
            idx = self.groups[c]
            perm = torch.randperm(len(idx), generator=self.generator).tolist()
            shuf = [idx[j] for j in perm]
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


# OPD: distill logit target into student model
# DirectOPD: distill logit shift into student model
# - from a churn gap perspective, different ckpt solves different queries, therefore admiting different 
#   logit shift direction over different queries. The ckpt-specific logit shift seems to be the KEY 
#   that we should target here

class DirectOPDTrainer(GRPOTrainer):

    def __init__(self, *args, teacher_run=None, teacher_steps=None,
                 anchor_path=None, shift_ref_path=None, center_shift=False,
                 support="student", mode="shift", topk=16, teacher_cache=12,
                 alpha=1.0, alpha_eps=0.01, alpha_bounds=(0.5, 2.5),
                 adaptive_alpha=True, interleave=None, **kwargs):
        
        self.dopd_interleave = interleave  # None | "rr" | "blocked"
        self.dopd_mode = mode        # "shift" | "imitate" | "compose"
        # compose: imitate the proxy-tuning composition
        #   q ~ softmax(log pi_S + (log pi_T - log pi_shift_ref))
        # = the query-specific residual grafted onto the student init (R2b).
        self.dopd_topk = topk
        self.dopd_alpha = alpha
        self.dopd_alpha_eps = alpha_eps
        self.dopd_alpha_bounds = alpha_bounds
        self.dopd_adaptive = adaptive_alpha
        self.center_shift = center_shift
        self.dopd_support = support
        self._shift_ref_path = shift_ref_path
        self._teacher_run = Path(teacher_run)
        self._teacher_steps = list(teacher_steps)
        self._anchor_path = anchor_path
        super().__init__(*args, **kwargs)

        from transformers import AutoModelForCausalLM
        dev = self.accelerator.device
        # LRU teacher cache (Phase-0 scale-up change): preloading all 12
        # teachers is fine at 0.6B (~15GB) but impossible at 4B (~96GB).
        # capacity 12 keeps 0.6B behavior identical; set dopd_teacher_cache
        # smaller for big models (bucketed batches make misses rare).
        from collections import OrderedDict
        self._teacher_cache = OrderedDict()
        self._teacher_cache_cap = int(teacher_cache)
        self._teacher_dev = dev

        class _Teachers:
            def __init__(t, outer):
                t.o = outer
            def __getitem__(t, step):
                o = t.o
                if step in o._teacher_cache:
                    o._teacher_cache.move_to_end(step)
                    return o._teacher_cache[step]
                if len(o._teacher_cache) >= o._teacher_cache_cap:
                    _, old_m = o._teacher_cache.popitem(last=False)
                    del old_m
                    torch.cuda.empty_cache()
                m = AutoModelForCausalLM.from_pretrained(
                    str(o._teacher_run / f"checkpoint-{step}"),
                    dtype=torch.bfloat16).to(o._teacher_dev)
                m.eval()
                [q.requires_grad_(False) for q in m.parameters()]
                o._teacher_cache[step] = m
                return m
        self.teachers = _Teachers(self)
        self.anchor = AutoModelForCausalLM.from_pretrained(
            anchor_path, dtype=torch.bfloat16).to(dev) # 'anchor_path' points to another model ckpt used as 'anchor'
        self.anchor.eval()
        [p.requires_grad_(False) for p in self.anchor.parameters()]
        # pi_S (KL reference) == the student's init == anchor_path in all our
        # exps (student starts at p), so reuse the same frozen copy.
        print(f"[dopd] {len(self._teacher_steps)} teachers (lazy LRU), anchor={anchor_path}, "
              f"mode={mode}, topk={topk}, alpha={alpha} adaptive={adaptive_alpha}",
              flush=True)

    def _get_train_sampler(self, dataset=None):
        # exp6: teacher-homogeneous batches (interference hypothesis, Jul 16)
        if not self.dopd_interleave:
            return super()._get_train_sampler(dataset)
        if dataset is None:
            dataset = self.train_dataset
        sampler = CkptGroupedSampler(
            dataset["best_ckpt"],
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            mode=self.dopd_interleave,
            seed=self.args.seed,
        )
        sizes = {c: len(v) for c, v in sorted(sampler.groups.items())}
        print(f"[dopd] interleave={self.dopd_interleave} group sizes {sizes} "
              f"-> {sampler.num_batches} homogeneous batches/epoch", flush=True)
        return sampler

    def _generate_and_score_completions(self, inputs):
        # Q5. what are we doing in this method? generate from current policy, and?
        
        # Thread each row's per-query teacher id through to _compute_loss —
        # TRL's micro-batch splitter slices every tensor in this dict.
        output = super()._generate_and_score_completions(inputs)
        output["best_ckpt"] = torch.tensor(
            [int(row["best_ckpt"]) for row in inputs],
            device=output["completion_ids"].device)
        return output

    # ------------------------------------------------------------------ loss
    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        best_steps = inputs["best_ckpt"]                    # tensor [B]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attn = torch.cat([prompt_mask, completion_mask], dim=1)
        T = completion_ids.size(1)
        K = self.dopd_topk

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        logits = logits[:, -T - 1:-1]                       # [B,T,V] next-token
        # memory-lean logp: never materialize a full-vocab fp32 softmax —
        # logZ reduces immediately, gradient flows via logits + logZ.
        logZ = torch.logsumexp(logits.float(), dim=-1)      # [B,T]
        if self.dopd_support == "shift":
            # Idea 1 (R3): support = top-K tokens by |teacher - shift_ref|
            # logit shift, not the student's top-k. Needs a no-grad pre-pass.
            with torch.no_grad():
                topi_list = []
                sref_m = self.shift_ref if self.shift_ref is not None else self.anchor
                for c in range(0, input_ids.size(0), 2):
                    rr = slice(c, min(c + 2, input_ids.size(0)))
                    rows_idx = torch.arange(input_ids.size(0))[rr]
                    tl = torch.empty(rows_idx.numel(), T,
                                     logits.size(-1), device=logits.device)
                    for s_ in best_steps[rr].unique().tolist():
                        sub = (best_steps[rr] == s_).nonzero(as_tuple=True)[0]
                        tl[sub] = self.teachers[int(s_)](
                            input_ids=input_ids[rr][sub],
                            attention_mask=attn[rr][sub]
                        ).logits[:, -T - 1:-1].float()
                    sl = sref_m(input_ids=input_ids[rr],
                                attention_mask=attn[rr]
                                ).logits[:, -T - 1:-1].float()
                    topi_list.append((tl - sl).abs().topk(K, dim=-1).indices)
                    del tl, sl
                topi = torch.cat(topi_list)
            topv_raw = logits.gather(-1, topi)
        else:
            topv_raw, topi = logits.topk(K, dim=-1)         # student top-k
        topv = topv_raw.float() - logZ.unsqueeze(-1)        # top-k logp w/ grad
        pbar = torch.softmax(topv.detach(), dim=-1)         # renormalized (Eq.11) probability of top-K token

        # Concern 1. here we are truncating top-K token on each logit for the current model
        #            for distillation we need to match top-K tokens distribution between different model
        #            these K tokens will likely be different

        def _teacher_at(m, rows):
            # Here we assume access to "topi" which decides on which token we align the logit distribution on
            """no-grad logp of model m at topi + sampled tokens, row-chunked."""
            outs_k, outs_y = [], []
            for c in range(0, len(rows), 2):
                rr = rows[c:c + 2]
                tl = m(input_ids=input_ids[rr],
                       attention_mask=attn[rr]).logits[:, -T - 1:-1]
                lz = torch.logsumexp(tl.float(), dim=-1)
                outs_k.append(tl.float().gather(-1, topi[rr])
                              - lz.unsqueeze(-1))
                outs_y.append(tl.float().gather(
                    -1, completion_ids[rr].unsqueeze(-1)).squeeze(-1) - lz)
                # TO be Validated: my understanding of outs_y is a per-token single logits sequence
                #                  outs_k is instead a per-token K logits sequence, where the indices of K logist are 
                #                  stored in 'topi' --- topi is defined in line 122 (student model's top K token logit)
                del tl, lz
            return torch.cat(outs_k), torch.cat(outs_y)

        with torch.no_grad():
            all_rows = torch.arange(input_ids.size(0), device=input_ids.device)
            # Concern 2. 
            # The logit shift from reference "anchor" model to the "teacher" model on the top K tokens from the current 
            # model, is our 'reward' here.  
            
            ref_at_k, ref_at_y = _teacher_at(self.anchor, all_rows) # reference logit from anchor model
            t_at_k = torch.empty_like(ref_at_k)
            for s in best_steps.unique().tolist():
                rows = (best_steps == s).nonzero(as_tuple=True)[0]
                t_at_k[rows], _ = _teacher_at(self.teachers[int(s)], rows) # teacher logit from target model

        if self.dopd_mode == "shift":
            # Q6. what is 'pbar'? -> 'pbar' is the normalized current policy topK next token prediction distribution
            
            # Q7. what is is A here? -> element-wise dot product between 'target logit shift' and 'current (normalized)'
            #     logit, I think it is similar to an advantage term, because it has detached gradient and it scales the 
            #     log-probability, 
            
            # Q8. what is topv? -> roughly speaking it's the log-probability of the topK next token prediction 
            #     distribution for the current model 

            # Idea 1. 
            # between teacher and reference model, the top K token exhibiting the biggest log-prob shift is likely 
            # different from the top K token from the current model (on some of the top K token, there might not be
            # any shift at all), so picking the top K 'shifted' token as "topi" is worth trying. 

            # Idea 2.
            # it feels naive when we simply multiply 'desirable shift' with 'current log-prob' and asks the model to 
            # maximize the product, I wonder if there are other ways we can play this? I guess the big question is, if 
            # logits as a target is NOT as reliable as logit shift as a target, what does it implies and how do we 
            # further exploit it? A dumb idea, is to use SVD on the logit shift and supervise on those instead (so 
            # we reduce the noisy shift maximally and only focus on the 'gist' movement directions)

            # Idea 3. 
            # If multiple ckpt solve the same queries, and multiple ckpt fails the same queries, then we can obtain 
            # MANY pairwise logit shift right? we can use these pairwise logit shift to compute out a mean logit shift
            # (denoised version of it), which we then use as the directional target here instead
            
            r = (t_at_k - ref_at_k)                          # Eq. 10 on S_t
            A = (pbar * r).detach()                          # Eq. 14
            mask = completion_mask.unsqueeze(-1).float()
            dense = -(A * topv).mul(mask).sum() / mask.sum().clamp(min=1)
        else:                                               # "imitate": top-k on policy logit distillation (Eq. 4)
            qbar = torch.softmax(t_at_k, dim=-1)
            mask = completion_mask.unsqueeze(-1).float()
            dense = (pbar.log() - qbar.log()).mul(
                torch.softmax(topv, -1)).mul(mask).sum() / mask.sum().clamp(min=1)
            # this is where we set the teacher logit as the naive 'target' for the student to match verbatim

        # KL(pi_theta || pi_S) on sampled tokens, k3 estimator
        # Q9. I thought original Direct OPD method is doing 'weak to strong distillation', which means reference model
        #     used for KL regularization is the strong model's base ckpt, which is almost an EMA ckpt, but the anchor 
        #     model used for calculating the logit shift is certainly the weak model, and the teacher model is obviously
        #     also the weak model (a trained one). But here, it seems that we are flipping the role of reference & anchor 
        #     model, we basically use the anchor model as the reference model, and that is not the original design. I am 
        #     not saying our design is wrong, just that it is different, and I wonder what'd happen if we make anchor 
        #     model different from reference model. 
        
        lp_theta = logits.float().gather(
            -1, completion_ids.unsqueeze(-1)).squeeze(-1) - logZ
        lp_ref = ref_at_y
        d = lp_ref - lp_theta
        kl = (torch.exp(d) - d - 1).mul(completion_mask).sum() \
            / completion_mask.sum().clamp(min=1)

        loss = dense + self.dopd_alpha * kl
        # [Request]. We need to ablate on 'dopd_alpha' here, it is not yet clear that KL regularization is 
        #            very necessary, also the 'why it is needed' is also unclear

        if self.dopd_adaptive and self.dopd_mode == "shift":   # Eq. 16
            # This turns out to be critical for directional distillation
            # Q10. what is this logic? We computes an 'rbar' and an 'alpha' term but we did not use it 
            #      in our loss function here, so how does this affect our training? 
            
            rbar = float((pbar * r).mul(completion_mask.unsqueeze(-1))
                         .sum() / completion_mask.sum().clamp(min=1))
            lo, hi = self.dopd_alpha_bounds
            sign = 0.0 if rbar == 0 else (1.0 if rbar > 0 else -1.0)
            # Q11. For the adaptive control on alppha (weight for the KL regularization term)
            #      why do we reduce the alpha weight when rbar is negative (on position where logit movement is negative?)
            #      why do we increase alpha weight when rbar is positive (on position where logit movement is positive)
        # [A11] rbar = E_pbar[r] on the STUDENT'S OWN top-k at visited prefixes:
        #   rbar>0 -> the student's existing preferences already align with the
        #   teacher's movement; the dense gradient becomes self-amplifying
        #   (push up own top tokens -> bigger pbar -> bigger A next step), so
        #   RAISE alpha (tighten the anchor) to prevent runaway sharpening.
        #   rbar<0 -> the student's mass sits on teacher-SUPPRESSED tokens; the
        #   needed correction moves probability away from the student's own
        #   distribution, which the KL-to-init anchor directly resists, so
        #   LOWER alpha (release the brake) to let the correction through.
        #   It's a trust modulator, not a KL-target controller. Empirical: our
        #   smoke had alpha 1.0->0.67 (corrective phase at init); fixed a=0.8
        #   collapsed to 0.4% (missing raise-on-positive phase); fixed a=2.0
        #   dragged to 60.5 (brake locked during correction).

            self.dopd_alpha = float(min(hi, max(
                lo, self.dopd_alpha * (1 + self.dopd_alpha_eps * sign)))) # we update on the weight for KL regularization
            self._metrics["train"]["dopd/rbar"].append(rbar)
        self._metrics["train"]["dopd/alpha"].append(self.dopd_alpha)
        self._metrics["train"]["dopd/kl"].append(float(kl.detach()))
        self._metrics["train"]["dopd/dense"].append(float(dense.detach()))
        return loss
