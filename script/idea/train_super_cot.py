"""Amortized super-logit reasoning on GSM8K.

Teach the MODEL to do its reasoning in K soft slots (super logits) instead of text.
Every mode shares one sequence layout:
  [chat prompt ... <think>\n] [K slots] [</think>\n\nThe final answer is \boxed{] [ans}]
with CE only on the answer tokens (+ closing brace).

Modes (design notes on SuperCoT below):
  direct  : K=0. No extra compute.
  filler  : K literal '.' tokens. Pause-token control: same compute, zero content.
  bptt    : idea #2. Differentiable self-fed rollout; answer CE backprops through the
            whole K-step recurrence (K*n_layers deep -> weak credit assignment on slots).
  distill : idea #1. Per-instance search over FREE slot logits ('.'-init, model frozen),
            then imitate: CE(ans | p*) + lam * CE_slot(p_theta || p*). Parallel, SFT-like,
            but p* is off-policy w.r.t. inference-time self-feeding.
  search  : idea #3. Same search, but initialized from the on-policy rollout p0 and
            KL-anchored to it -> p* is a small, on-policy improvement step (expert
            iteration). Optionally also supervise CE(ans | p0).

Both search modes gate on improvement: an instance is only distilled if the inner
search actually lowered its answer CE (--gate).

  python train_super_cot.py --mode search --k 16 --steps 700 --out runs/search_k16
"""
import argparse, contextlib, json, os, random, re, time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"


def autocast():  # fp32 master weights, bf16 compute
    return torch.autocast(DEVICE, dtype=torch.bfloat16)


# ---------------------------------------------------------------- data --------

def load_gsm8k(tok, split, limit, max_prompt_tokens=320, seed=0):
    """-> list of dicts {prompt_ids, ans_ids, gt} (chat template + '<think>\n')."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    
    out = []
    for i in idx:
        ex = ds[i]
        gt = ex["answer"].split("####")[-1].strip().replace(",", "")
        text = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=False, add_generation_prompt=True) + "<think>\n"
        p = tok(text, add_special_tokens=False).input_ids
        if len(p) > max_prompt_tokens:
            continue
        a = tok(gt + "}", add_special_tokens=False).input_ids
        out.append({"prompt_ids": p, "ans_ids": a, "gt": gt})
        if limit and len(out) >= limit:
            break
    return out


def parse_answer(text):
    m = re.search(r"-?\d+\.?\d*", text.split("}")[0].replace(",", ""))
    return m.group(0).rstrip(".") if m else None


# ------------------------------------------------------------- batching -------

def left_pad(seqs, pad_id, device):
    """-> (ids [B,L], mask [B,L]) left-padded."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        ids[b, L - len(s):] = torch.tensor(s, device=device)
        mask[b, L - len(s):] = 1
    return ids, mask


def pos_from_mask(mask):
    return (mask.cumsum(-1) - 1).clamp(min=0)


@contextlib.contextmanager
def frozen(model):
    """No param grads inside (inner search optimizes slot logits only)."""
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)


class Cached:
    """Incremental KV-cached prefix: .feed(embs [B,n,d]) then read .last logits."""

    def __init__(self, sc, pids, pmask):
        self.sc, self.mask, self.B = sc, pmask, pids.shape[0]
        self.cur = pmask.sum(-1)  # next position id per row
        self.out = sc.model(inputs_embeds=sc.emb(pids), attention_mask=pmask,
                           position_ids=pos_from_mask(pmask), use_cache=True)
        self.cache = self.out.past_key_values

    def feed(self, e):
        n = e.shape[1]
        pos = self.cur.unsqueeze(1) + torch.arange(n, device=DEVICE).unsqueeze(0)
        self.mask = torch.cat(
            [self.mask, torch.ones((self.B, n), dtype=torch.long, device=DEVICE)], 1)
        self.out = self.sc.model(inputs_embeds=e, attention_mask=self.mask,
                                 position_ids=pos, past_key_values=self.cache, use_cache=True)
        self.cache, self.cur = self.out.past_key_values, self.cur + n

    @property
    def last(self):
        return self.out.logits[:, -1].float()


# ---------------------------------------------------------------- model -------

class SuperCoT:
    def __init__(self, args):
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.freeze_embed:  # keep the vocab geometry the soft slots mix over fixed
            model.get_input_embeddings().weight.requires_grad_(False)
        self.model = model
        self.E = model.get_input_embeddings().weight  # [V, d]
        t = self.tok
        self.CLOSE = t.encode("</think>\n\n", add_special_tokens=False)
        self.APFX = t.encode("The final answer is \\boxed{", add_special_tokens=False)
        self.TAIL = self.CLOSE + self.APFX
        self.DOT = t.encode(".", add_special_tokens=False)[0]
        self.pad_id = t.pad_token_id if t.pad_token_id is not None else t.eos_token_id

    def emb(self, ids):
        return self.E[ids]

    # -- slot construction --------------------------------------------------
    # Dumb Idea #1. 
    #      we can optimize, case by case, the best soft logits, then we train the model to mimic those 'optimal choice' 
    #      we can perform on-line update of soft logits choice (that is an analogy and parallel to the on-policy rollout in RLVR)
    #      we keep doing such 'search -> learn' loop, with the hope of teaching the model useful tricks, and that things will converge
    # Dumb Idea #2.  
    #      we optimize with loss_c, which includes 'soft logit prediction accuracy' AND 'answer prediction accuracy'
    #      if we train for long enough, this might gives us a converged model that is able to generalize to test cases
    # Reflection #1. 
    #      honestly, we probably need an augmented dataset where we train the model to do this sort of stuff,  subsampled data (600k samples) from the Dolci-Think-SFT dataset (AI2, 2025b), this dataset is used in AbstractCoT, and is bigger
    # Reflection #2. 
    #      Idea #2. sacrifices the quality of soft logit, but it doesn't have the moving target issues, whilst 
    #      Idea #1. creates a moving target, but its optimized soft logit is top-notch
    #               we likely need to stabilize the optimized soft-logits on idea #1. 
    # Current Logic: 
    #      we let the model to deterministically rollout (we loop back its logit prediction)
    #      then we have K + 1 propagation of the original model in order to learn things, that likely mean the model 
    #      will simply try to figure out the answer from whatever soft logit it produce in the first place, instead of 
    #      diligently learn what soft-logits it should produce here, the gradient got 400+ effective layers to propagate
    #      that is impossible

    # Reflection #3. 
    #      it feels like my Idea #1, is more akin to SFT, in the sense that CoT are explicitly provided
    #      whilst the current logic is more akin to RL (we can actually just rollout deterministically without requiring gradient tracking at all)
    #      and just train the model to be able to predict the correct answer from its own soft logit rollout
    #      I guess a mid-ground, is if we do one-step gradient update on the soft logit rollouts (so the shift is small)
    #      but then we'd teach the model (via teacher forcing) what soft logit to produce, as well as how to answer given the soft logit?
    
    # Idea #3. 
    #      idea #1. do not require per soft-logit rollouts and is parallelizable, this resemble SFT, however, it mismatch with on-policy rollout and inference time behavior
    #      idea #2. requires per soft-logit rollout, is not parallelizable, but lacks ability of 'search'
    #      therefore, in idea #3. we do one-time rollout to get on-policy soft-logits, then, we do a few step optimization to update ONLY the soft logits
    #      then, we supervise the model on its soft logits to match the target, AND we supervise the model to better produce the answer contidioning either on the original soft logits, or the target (updated) soft logits
    #      [optional gate] KL(p_t* || p_t) where p_t is the on-policy rollout soft-logit from the model itself, and p_t* is the updated soft logit, we can clip the update within a threshold, or we use it to anchor the p_t* update (optimization)
    #       


    def pad(self, batch):
        return left_pad([ex["prompt_ids"] for ex in batch], self.pad_id, DEVICE)

    # soft logit embedding
    def mix(self, probs):  # [B,k,V] -> [B,k,d]: convex combination of vocab embeddings
        return probs.to(self.E.dtype) @ self.E

    def dot_probs(self, B, peak=5.0):  # '.'-peaked slots: off-policy search init (idea #1)
        lg = torch.zeros(B, self.args.k, self.E.shape[0], device=DEVICE)
        lg[..., self.DOT] = peak
        return F.softmax(lg, -1)

    def filler_embs(self, B):
        return self.emb(torch.full((B, self.args.k), self.DOT, device=DEVICE))

    # -- ONE forward scores everything: answer CE and (optionally) slot CE ---
    def score(self, batch, slot_embs, slot_target=None):
        """slot_embs: [B,k,d] or None. -> (ce_ans [B], ce_slot [B] or None).
        slot_target [B,k,V]: distribution the model should predict AT each slot position;
        its logits live at sstart-1 .. sstart+k-2, i.e. the same forward, teacher-forced."""
        pids, mask = self.pad(batch)
        embs, sstart = self.emb(pids), pids.shape[1]
        if slot_embs is not None:
            embs = torch.cat([embs, slot_embs], 1)
            mask = torch.cat([mask, torch.ones((embs.shape[0], slot_embs.shape[1]),
                                               dtype=torch.long, device=DEVICE)], 1)
        amax = max(len(ex["ans_ids"]) for ex in batch)
        ta = torch.tensor([self.TAIL + ex["ans_ids"] + [self.pad_id] * (amax - len(ex["ans_ids"]))
                           for ex in batch], device=DEVICE)
        embs = torch.cat([embs, self.emb(ta)], 1)
        mask = torch.cat([mask, torch.ones_like(ta)], 1)
        out = self.model(inputs_embeds=embs, attention_mask=mask,
                         position_ids=pos_from_mask(mask))
        labels = torch.full(mask.shape, -100, dtype=torch.long, device=DEVICE)
        astart = embs.shape[1] - amax
        for b, ex in enumerate(batch):
            labels[b, astart:astart + len(ex["ans_ids"])] = torch.tensor(ex["ans_ids"],
                                                                        device=DEVICE)
        tgt = labels[:, 1:]
        ce = F.cross_entropy(out.logits[:, :-1].float().flatten(0, 1), tgt.flatten(),
                             ignore_index=-100, reduction="none").view(tgt.shape)
        m = (tgt != -100).float()
        ce_ans = (ce * m).sum(-1) / m.sum(-1).clamp_min(1)
        ce_slot = None
        if slot_target is not None:
            k = slot_target.shape[1]
            pred = out.logits[:, sstart - 1:sstart - 1 + k].float()
            ce_slot = -(slot_target * F.log_softmax(pred / self.args.tau, -1)).sum(-1).mean(-1)
        return ce_ans, ce_slot

    # -- slot producers ------------------------------------------------------
    # on-policy rollout of soft-logits (with gradient tracking)
    def self_fed(self, batch):
        """idea #2: grad-tracked self-fed rollout (no cache). -> probs [B,k,V]."""
        a = self.args
        pids, mask = self.pad(batch)
        embs, probs = self.emb(pids), []
        ones = torch.ones((embs.shape[0], 1), dtype=torch.long, device=DEVICE)
        for _ in range(a.k):
            out = self.model(inputs_embeds=embs, attention_mask=mask,
                             position_ids=pos_from_mask(mask))
            lg = out.logits[:, -1].float()
            if a.gumbel > 0:  # stochastic slots (categorical-VAE flavor)
                g = -torch.log(-torch.log(torch.rand_like(lg).clamp_min(1e-9)).clamp_min(1e-9))
                lg = lg + a.gumbel * g
            p = F.softmax(lg / a.tau, -1)
            probs.append(p)
            embs = torch.cat([embs, self.mix(p[:, None])], 1)
            mask = torch.cat([mask, ones], 1)
        return torch.stack(probs, 1)

    # no-grad version of on-policy rollout
    @torch.no_grad()
    def feed_slots(self, c):
        """Roll K slots onto an existing cache, leaving it ready for what follows.
        -> probs [B,k,V]. Same recurrence as self_fed, minus the graph."""
        probs = []
        for _ in range(self.args.k):
            p = F.softmax(c.last / self.args.tau, -1)
            probs.append(p)
            c.feed(self.mix(p[:, None]))
        return torch.stack(probs, 1)

    @torch.no_grad()
    def rollout(self, batch):
        """on-policy slots for a fresh batch, KV-cached, no grad. -> probs [B,k,V]."""
        self.model.config.use_cache = True
        probs = self.feed_slots(Cached(self, *self.pad(batch)))
        self.model.config.use_cache = False
        return probs

    def search(self, batch, p0, anchor):
        """Inner loop over FREE slot logits, model frozen. -> (p* [B,k,V], info).
        Parameterized so softmax(lg/tau) == p0 at step 0; KL(p||p0) anchors/caps the shift."""
        a = self.args
        lg = (p0.clamp_min(1e-9).log() * a.tau).detach().requires_grad_(True)
        opt = torch.optim.Adam([lg], lr=a.inner_lr)
        logp0, ce, kl, i = p0.clamp_min(1e-9).log(), 0.0, 0.0, 0
        with frozen(self.model):
            for i in range(a.inner_steps):
                p = F.softmax(lg / a.tau, -1)
                kl_t = (p * (p.clamp_min(1e-9).log() - logp0)).sum(-1).mean()
                ce_t = self.score(batch, self.mix(p))[0].mean()
                ce, kl = float(ce_t), float(kl_t)
                if a.kl_cap and kl >= a.kl_cap:  # trust region on the improvement step
                    break
                loss = ce_t + (a.beta * kl_t if anchor else 0.0)
                opt.zero_grad(); loss.backward(); opt.step()
        return F.softmax(lg / a.tau, -1).detach(), {
            "inner_steps": i, "inner_ce": round(ce, 4), "inner_kl": round(kl, 4)}

    # -- training ------------------------------------------------------------
    def loss(self, batch):
        a, B = self.args, len(batch)
        with autocast():
            # .score put CoT embedding (soft logit) inside and evaluate answer log-prob
            if a.mode in ("direct", "filler"):
                return self.score(batch, self.filler_embs(B) if a.k else None)[0].mean(), {}
            if a.mode == "bptt":
                return self.score(batch, self.mix(self.self_fed(batch)))[0].mean(), {}
            # distill (idea #1) / search (idea #3): search, gate, then imitate
            on_policy = a.mode == "search"
            p0 = self.rollout(batch) if on_policy else self.dot_probs(B)
            pstar, info = self.search(batch, p0, anchor=on_policy)
            # ce0 serves double duty: gate reference, and (with --also_p0) a loss term.
            # One forward either way; grad tracked only when it is actually a loss term.
            grad_p0 = on_policy and a.also_p0
            with contextlib.nullcontext() if grad_p0 else torch.no_grad():
                ce0 = self.score(batch, self.mix(p0))[0]
            ce_a, ce_s = self.score(batch, self.mix(pstar), slot_target=pstar)
            w = (ce0.detach() - ce_a.detach() > a.gate).float()  # imitate real improvements only
            loss = (w * (ce_a + a.lam * ce_s)).sum() / w.sum().clamp_min(1)
            if grad_p0:  # ungated: answering from on-policy slots is valid for every instance
                loss = loss + ce0.mean()
            info.update(keep=round(float(w.mean()), 3), ce0=round(float(ce0.mean()), 4),
                        ce_slot=round(float(ce_s.mean()), 4))
            return loss, info

    # -- batched cached generation for eval ---------------------------------
    # Slots always come from feed_slots (the no-grad rollout), even for the modes
    # trained by teacher forcing: that is exactly the exposure-bias check they need.
    @torch.no_grad()
    def eval_batch(self, batch, max_new=8, want_probs=False):
        a = self.args
        self.model.config.use_cache = True
        with autocast():
            c = Cached(self, *self.pad(batch))
            probs = None
            if a.k and a.mode == "filler":
                c.feed(self.filler_embs(c.B))
            elif a.k and a.mode != "direct":
                probs = self.feed_slots(c)
            c.feed(self.emb(torch.tensor([self.TAIL] * c.B, device=DEVICE)))
            B, toks = c.B, torch.full((c.B, max_new), self.pad_id, dtype=torch.long, device=DEVICE)
            done = torch.zeros(B, dtype=torch.bool, device=DEVICE)
            for j in range(max_new):
                nt = c.last.argmax(-1)
                toks[:, j] = torch.where(done, toks[:, j], nt)
                done |= (nt == self.tok.eos_token_id)
                if done.all():
                    break
                c.feed(self.emb(nt.unsqueeze(1)))
        self.model.config.use_cache = False
        preds = [parse_answer(self.tok.decode([t for t in row.tolist() if t != self.pad_id]))
                 for row in toks]
        return preds, (probs if want_probs else None)

    @torch.no_grad()
    def evaluate(self, data, bs=16, inspect_n=0):
        self.model.eval()
        hits, n, dumps = 0, 0, []
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            preds, probs = self.eval_batch(batch, want_probs=(i == 0 and inspect_n > 0))
            for ex, pr in zip(batch, preds):
                hits += (pr == ex["gt"]); n += 1
            if probs is not None:
                for b in range(min(inspect_n, len(batch))):
                    slots = []
                    for t in range(probs.shape[1]):
                        top = torch.topk(probs[b, t], 3)
                        ent = -(probs[b, t] * probs[b, t].clamp_min(1e-12).log2()).sum()
                        slots.append({"H_bits": round(float(ent), 2),
                                      "top": [(self.tok.decode([int(j)]), round(float(v), 3))
                                              for v, j in zip(top.values, top.indices)]})
                    dumps.append({"gt": batch[b]["gt"], "pred": preds[b], "slots": slots})
        self.model.train()
        return hits / max(n, 1), dumps


# ----------------------------------------------------------------- main -------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["direct", "filler", "bptt", "distill", "search"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--gumbel", type=float, default=0.0, help="bptt only")
    # inner search (distill / search)
    ap.add_argument("--inner_steps", type=int, default=4)
    ap.add_argument("--inner_lr", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.1, help="KL(p*||p0) anchor, search only")
    ap.add_argument("--kl_cap", type=float, default=1.0, help="stop inner loop at this KL; 0=off")
    ap.add_argument("--lam", type=float, default=1.0, help="weight on slot-matching CE")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="distill an instance only if ce(p0)-ce(p*) exceeds this; -1e9 = off")
    ap.add_argument("--also_p0", action="store_true",
                    help="search: also train answer CE on the un-searched on-policy slots")
    ap.add_argument("--freeze_embed", action="store_true",
                    help="freeze input embeddings (tied to lm_head on Qwen3)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--train_n", type=int, default=0, help="0 = all")
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=175)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "direct":
        args.k = 0
    out_dir = args.out or f"runs/{args.mode}_k{args.k}"
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.jsonl"), "a")

    def log(**kw):
        kw["t"] = round(time.time(), 1)
        log_f.write(json.dumps(kw) + "\n"); log_f.flush()
        print({k: v for k, v in kw.items() if k != "slots_dump"}, flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sc = SuperCoT(args)
    train = load_gsm8k(sc.tok, "train", args.train_n, seed=args.seed)
    test = load_gsm8k(sc.tok, "test", args.eval_n, seed=args.seed)
    log(event="start", **vars(args), n_train=len(train), n_test=len(test))

    opt = torch.optim.AdamW([p for p in sc.model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) * (1 - 0.9 * s / max(args.steps, 1)))

    acc0, dump0 = sc.evaluate(test, inspect_n=2)
    log(event="eval", step=0, acc=acc0, slots_dump=dump0)

    sc.model.train()
    ema, order, ptr = None, list(range(len(train))), 0
    for step in range(1, args.steps + 1):
        if ptr + args.batch > len(order):
            random.shuffle(order); ptr = 0
        batch = [train[i] for i in order[ptr:ptr + args.batch]]; ptr += args.batch
        loss, info = sc.loss(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in sc.model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        l = float(loss)
        ema = l if ema is None else 0.95 * ema + 0.05 * l
        if step % 20 == 0:
            log(event="train", step=step, loss=round(l, 4), ema=round(ema, 4), **info)
        if step % args.eval_every == 0 or step == args.steps:
            acc, dump = sc.evaluate(test, inspect_n=2)
            log(event="eval", step=step, acc=acc, slots_dump=dump)

    sc.model.save_pretrained(os.path.join(out_dir, "model"))
    log(event="done")


if __name__ == "__main__":
    main()
