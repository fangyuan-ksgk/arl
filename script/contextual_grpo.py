#!/usr/bin/env python
"""Contextual-distillation GRPO (parameterized; editable copy of cont-self-distill/idea35).

Dr.GRPO + on-policy contextual-distillation aux loss:
  • success rollouts → CD PULL (cd_lambda): student (sees only Q) matches a frozen teacher that got
    the correct CoT as a hint → internalize correct reasoning.
  • failed rollouts → bounded CD PUSH (neg_lambda, clipped) → push off wrong reasoning.
Both terms required (success + fail). Total = Dr.GRPO loss + cd_lambda·mean(pull) − neg_lambda·mean(push).

Parameterized vs idea35 so we can run the TRUSTWORTHY config and ablate the base:
  --scale_rewards none  = full Dr.GRPO (no std norm) | group = LENGTH-FIX-ONLY (keep 1/std norm; user
                          found this better) | both use loss_type=dr_grpo (unbiased length).
  --lr_scheduler linear --max_steps 300  (trustworthy) ; --eval_max_tokens 2048 (fair eval, not 640).
  --save_model for soup/re-eval.  --beta 0 (VRAM-light Dr.GRPO; idea35 used 0.04).

RUN: CUDA_VISIBLE_DEVICES=0 python script/contextual_grpo.py --cd_lambda 0.1 --neg_lambda 0.05 \
       --scale_rewards group --lr_scheduler linear --max_steps 300 --eval_max_tokens 2048 --save_model
"""
import argparse, hashlib, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from script.grpo_gsm8k import load_gsm8k, correctness_reward, extract_answer_from_completion
from script.grpo_kl_distill import _user_from_prompt_ids, quick_eval

MARKER = "####"
_key = lambda s: hashlib.sha1(s.encode()).hexdigest()
_ids = lambda tok, s: tok(s, add_special_tokens=False)["input_ids"]


def contextual_distill_loss(student, teacher, tok, question, cot, answer, max_len=1800):
    """forward KL( teacher(y | q, HINT) ‖ student(y | q) ) over target tokens y=cot+answer."""
    target = f"{cot}\n{MARKER} {answer}"
    hint = f"\n\n(Hint — a correct solution:{cot}{MARKER} {answer})"
    teach_prompt = tok.apply_chat_template([{"role": "user", "content": question + hint}], tokenize=False, add_generation_prompt=True)
    stud_prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    y = _ids(tok, target); t_pre, s_pre = _ids(tok, teach_prompt), _ids(tok, stud_prompt)
    if not y or len(t_pre) + len(y) > max_len:
        return None
    tf = torch.tensor([t_pre + y], device="cuda"); sf = torch.tensor([s_pre + y], device="cuda")
    with torch.no_grad():
        tl = teacher(tf, use_cache=False).logits[0, len(t_pre) - 1: len(t_pre) - 1 + len(y)].float()
    sl = student(sf, use_cache=False).logits[0, len(s_pre) - 1: len(s_pre) - 1 + len(y)].float()
    lq, lp = tl.log_softmax(-1), sl.log_softmax(-1)
    return (lq.exp() * (lq - lp)).sum(-1).mean()


class ContextDistillGRPO(GRPOTrainer):
    def configure(self, teacher, tok, gold, cd_lambda, neg_lambda):
        self.teacher, self.tok, self.gold = teacher, tok, gold
        self.cd_lambda, self.neg_lambda = cd_lambda, neg_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
        if return_outputs:
            loss, rest = loss[0], loss[1:]
        pos, neg = [], []
        for p_ids, c_ids in zip(inputs["prompt_ids"], inputs["completion_ids"]):
            question = _user_from_prompt_ids(self.tok, p_ids)
            completion = self.tok.decode(c_ids, skip_special_tokens=True)
            gold = self.gold.get(_key(question))
            if gold is None:
                continue
            pred = extract_answer_from_completion(completion)
            correct = (str(pred).strip() == str(gold).strip())
            cot, ans = completion.split(MARKER)[0].rstrip(), (str(pred).strip() if pred is not None else "")
            if not cot or not ans:
                continue
            cd = contextual_distill_loss(model, self.teacher, self.tok, question, cot, ans)
            if cd is not None:
                (pos if correct else neg).append(cd)
        if pos and self.cd_lambda > 0:
            loss = loss + self.cd_lambda * torch.stack(pos).mean()
        if neg and self.neg_lambda > 0:
            loss = loss - self.neg_lambda * torch.stack(neg).clamp(max=2.0).mean()
        return (loss, *rest) if return_outputs else loss


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cd_lambda", type=float, default=0.1)
    a.add_argument("--neg_lambda", type=float, default=0.05)
    a.add_argument("--scale_rewards", default="group", choices=["none", "group", "batch"],
                   help="group = length-fix-only (keep 1/std); none = full Dr.GRPO")
    a.add_argument("--beta", type=float, default=0.0, help="KL coeff (0 = VRAM-light, no ref model)")
    a.add_argument("--lr", type=float, default=5e-6)
    a.add_argument("--lr_scheduler", default="linear")
    a.add_argument("--warmup", type=int, default=15)
    a.add_argument("--grad_accum", type=int, default=4)
    a.add_argument("--num_gen", type=int, default=8)
    a.add_argument("--max_completion_length", type=int, default=1024)
    a.add_argument("--max_steps", type=int, default=300)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--eval_samples", type=int, default=500)
    a.add_argument("--eval_max_tokens", type=int, default=2048)
    a.add_argument("--save_model", action="store_true")
    a.add_argument("--model", default="Qwen/Qwen3-0.6B")
    a.add_argument("--output_dir", default=str(PROJ / "output/contextual_grpo"))
    args = a.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                   attn_implementation="eager").to("cuda").eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    train, test = load_gsm8k()
    gold = {_key(e["prompt"][0]["content"]): e["gold_answer"] for e in train}

    cfg = GRPOConfig(
        output_dir=args.output_dir, max_steps=args.max_steps, num_generations=args.num_gen,
        per_device_train_batch_size=args.num_gen, gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length, learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler, warmup_steps=args.warmup,
        loss_type="dr_grpo", scale_rewards=(False if args.scale_rewards == "none" else args.scale_rewards),
        beta=args.beta, seed=args.seed, use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3, gradient_checkpointing=True, logging_steps=20,
        save_strategy="no", report_to="none",
    )
    trainer = ContextDistillGRPO(model=model, reward_funcs=[correctness_reward], args=cfg, train_dataset=train)
    trainer.configure(teacher, tok, gold, args.cd_lambda, args.neg_lambda)
    trainer.train()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        trainer.save_model(str(out / "model")); tok.save_pretrained(str(out / "model"))
    if args.eval_samples <= 0:
        print(f"FINAL_EVAL skipped (eval_samples=0); model saved={args.save_model}", flush=True)
        return
    acc = quick_eval(trainer.model, tok, test, args.eval_samples, max_new=args.eval_max_tokens)
    cd_on = (args.cd_lambda > 0 or args.neg_lambda > 0)
    tag = (f"CD(cd={args.cd_lambda},neg={args.neg_lambda})" if cd_on else "no-CD") + \
          f" base={'lenfix(group)' if args.scale_rewards=='group' else 'drgrpo(none)'} seed={args.seed}"
    line = f"FINAL_EVAL acc={acc:.4f} on {args.eval_samples}@{args.eval_max_tokens} | {tag} steps={args.max_steps}"
    print(line, flush=True); (out / "final_eval.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
