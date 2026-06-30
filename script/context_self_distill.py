#!/usr/bin/env python
"""Phase B: contextualized self-distillation.

The model is its OWN teacher: conditioned on a privileged CONTEXT hint (the correct
solution found by best-of-N), it gives a confident full-logit distribution over the
target y* = correct CoT (+answer). The student, conditioned only on the plain question,
is trained to match it: loss = KL(teacher || student) over y*. Internalizes the model's
own successful reasoning so greedy@1 climbs toward pass@N. Self-distillation => full-vocab
logits both sides, exact KL, no external teacher.

Swept knobs (use the data to pick):
  --context {full_cot, answer, cot, none}   what the teacher sees (none => KL=0 control)
  --span    {completion, answer}            which target tokens the KL is over
  --direction {forward, reverse}
Pin GPU via CUDA_VISIBLE_DEVICES=1.
"""
import argparse, json, sys, re
from pathlib import Path
import torch
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from script.grpo_gsm8k import load_gsm8k, extract_answer_from_completion  # noqa: E402

MARKER = "####"


def build_pair(tok, q, cot, answer, full, context, device):
    """Return (teacher_ids, t_start, student_ids, s_start, L, ans_off) for one example.
    Teacher prompt carries the CONTEXT hint; student prompt is plain. Both are followed
    by the same target y*=full; KL is read over the y* span (or its answer suffix)."""
    plain = tok.apply_chat_template([{"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True)
    if context == "none":
        hint = ""
    elif context == "answer":
        hint = f"\n\n(Hint — the correct final answer is {answer}.)"
    elif context == "cot":
        hint = f"\n\n(Hint — correct reasoning:{cot})"
    else:  # full_cot
        hint = f"\n\n(Hint — a correct solution:{cot}{MARKER} {answer})"
    ctx = tok.apply_chat_template([{"role": "user", "content": q + hint}],
                                  tokenize=False, add_generation_prompt=True)
    y_ids = tok(full, add_special_tokens=False)["input_ids"]
    s_pre = tok(plain, add_special_tokens=False)["input_ids"]
    t_pre = tok(ctx, add_special_tokens=False)["input_ids"]
    L = len(y_ids)
    # answer-token offset within y* (tokens after the #### marker)
    m = re.search(re.escape(MARKER), full)
    ans_off = len(tok(full[:m.start()], add_special_tokens=False)["input_ids"]) if m else max(0, L - 2)
    student_full = torch.tensor([s_pre + y_ids], device=device)
    teacher_full = torch.tensor([t_pre + y_ids], device=device)
    return teacher_full, len(t_pre), student_full, len(s_pre), L, ans_off


def kl_loss_one(student, teacher, tok, ex, context, span, direction, device,
                tau=1.0, max_len=900):
    """Frozen `teacher` (base model) provides the context-conditioned target; trainable
    `student` is scored on the plain prompt. Teacher MUST be a frozen snapshot — sharing
    weights with the student makes the target drift and collapses training.
    tau: distillation temperature (1.0 = best config; tau>1 softens and hurts here)."""
    t_full, t_start, s_full, s_start, L, ans_off = build_pair(
        tok, ex["question"], ex["cot"], ex["answer"], ex["full"], context, device)
    if t_full.shape[1] > max_len or s_full.shape[1] > max_len or L < 1:
        return None
    lo = ans_off if span == "answer" else 0
    if L - lo < 1:
        return None
    with torch.no_grad():
        t_logits = teacher(t_full, use_cache=False).logits[0][t_start - 1: t_start - 1 + L][lo:].float()
    s_logits = student(s_full, use_cache=False).logits[0][s_start - 1: s_start - 1 + L][lo:].float()
    log_q = torch.log_softmax(t_logits / tau, -1)    # teacher (detached)
    log_p = torch.log_softmax(s_logits / tau, -1)    # student (grad)
    if direction == "forward":                        # KL(teacher || student)
        kl = (log_q.exp() * (log_q - log_p)).sum(-1)
    else:                                             # KL(student || teacher)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return (tau * tau) * kl.mean()


@torch.no_grad()
def quick_eval(model, tok, test_ds, n, bs=64, max_new=640):
    model.eval()
    qs = test_ds.select(range(min(n, len(test_ds))))
    prompts = [tok.apply_chat_template(e["prompt"], tokenize=False, add_generation_prompt=True) for e in qs]
    golds = [e["gold_answer"] for e in qs]
    tok.padding_side = "left"
    correct = 0
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for j, seq in enumerate(g):
            pred = extract_answer_from_completion(tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True))
            try:
                correct += int(float(pred) == float(golds[i + j]))
            except (ValueError, TypeError):
                pass
    model.train()
    return correct / len(prompts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--teacher_model", default=None,
                   help="frozen teacher checkpoint (default = same as --model = self-distill). "
                        "Set to a STRONGER same-tokenizer model (e.g. Qwen3-1.7B) for cross-model distill.")
    p.add_argument("--data", default=str(PROJ / "data" / "best_of_n" / "correct.jsonl"))
    p.add_argument("--context", choices=["full_cot", "answer", "cot", "none"], default="full_cot")
    p.add_argument("--span", choices=["completion", "answer"], default="completion")
    p.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    p.add_argument("--kl_tau", type=float, default=1.0, help="distillation temperature (1.0 = best)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--micro_bs", type=int, default=2, help="examples accumulated per optimizer step")
    p.add_argument("--eval_samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model", action="store_true", help="save the trained student for future re-eval")
    p.add_argument("--output_dir", default=str(PROJ / "output" / "ctx_self_distill"))
    args = p.parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
              attn_implementation="eager").to("cuda").train()
    # frozen teacher — the context-conditioned target must NOT drift. Defaults to a snapshot
    # of the student (self-distill); a stronger same-tokenizer model makes it cross-model KD.
    teacher_ckpt = args.teacher_model or args.model
    teacher = AutoModelForCausalLM.from_pretrained(teacher_ckpt, dtype=torch.bfloat16,
                attn_implementation="eager").to("cuda").eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    data = [json.loads(l) for l in open(args.data)]
    import random; random.Random(args.seed).shuffle(data)
    _, test = load_gsm8k()
    # weight_decay=0: AdamW's default 0.01 decays weights every step even at zero
    # gradient, silently degrading the model (the none-context control drifted 0.65->0.54).
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    tag = (f"ctx={args.context} span={args.span} dir={args.direction} tau={args.kl_tau} "
           f"teacher={teacher_ckpt.split('/')[-1]} lr={args.lr} steps={args.steps}")
    print(f"[csd] {tag} | {len(data)} examples", flush=True)

    di, step, acc_kl = 0, 0, []
    while step < args.steps:
        opt.zero_grad(); terms = []
        for _ in range(args.micro_bs):
            ex = data[di % len(data)]; di += 1
            kl = kl_loss_one(model, teacher, tok, ex, args.context, args.span, args.direction, "cuda",
                             tau=args.kl_tau)
            if kl is not None:
                terms.append(kl)
        if terms:
            loss = torch.stack(terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc_kl.append(loss.item())
        step += 1
        if step % 25 == 0:
            print(f"[csd] step {step}/{args.steps} meanKL={sum(acc_kl[-25:])/max(1,len(acc_kl[-25:])):.4f}", flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
        print(f"[csd] saved student -> {out/'model'}", flush=True)
    n_eval = min(args.eval_samples, len(test))
    acc = quick_eval(model, tok, test, n_eval)
    line = f"FINAL_EVAL acc={acc:.4f} on {n_eval} test | {tag}"
    print(line, flush=True)
    (out / "final_eval.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
