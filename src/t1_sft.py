"""t=1 shortest-correct SFT, minimal & self-contained (the `q9_sft_t1` recipe).

Two stages, exactly the pipeline that produced our best from-base SFT model
(GSM8K test 69.0 at step 300, Qwen3-0.6B):

  # 1. sample t=1.0 @ k=8 per train query from a source checkpoint;
  #    keep the SHORTEST CORRECT rollout per query
  python t1_sft_minimal.py build --ckpt <source_ckpt> --out t1_sft.jsonl

  # 2. completion-masked SFT on the BASE model (prompt tokens -> -100),
  #    effective batch 32, lr 1e-5 constant, 300 steps, ckpts every 15
  python t1_sft_minimal.py sft --data t1_sft.jsonl --out out_dir

Notes kept from the full pipeline:
 - correctness = strict final-answer match ('#### <number>'), not containment.
 - the SFT masking is transparent manual masking, no collator magic.
 - truncation at --max_len is reported loudly (silent truncation cuts the
   answer tail and poisons the dataset).
Deps: torch, transformers, datasets, vllm (stage 1 only).
"""
import argparse, json, re


# ---------------------------------------------------------------- shared
def extract_answer(text):
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def load_gsm8k_train():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")["train"]
    return [{"question": r["question"],
             "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
            for r in ds]


# ---------------------------------------------------------------- stage 1
def build(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    train = load_gsm8k_train()[:a.n]
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                       tokenize=False, add_generation_prompt=True)
               for r in train]
    llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=0.85)
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, n=a.k,
                                                max_tokens=a.max_tokens))
    rows = []
    for i, o in enumerate(outs):
        correct = sorted((c.text for c in o.outputs
                          if extract_answer(c.text) == train[i]["gold"]), key=len)
        if correct:
            rows.append({"idx": i, "question": train[i]["question"],
                         "completion": correct[0]})   # shortest correct
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build] {len(train)} queries x{a.k} -> {len(rows)} SFT rows "
          f"({100*len(rows)/len(train):.0f}% coverage) -> {a.out}")


# ---------------------------------------------------------------- stage 2
def sft(a):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data)]

    def tokenize_masked(question, completion):
        prompt = tok.apply_chat_template([{"role": "user", "content": question}],
                                         tokenize=False, add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(completion + tok.eos_token, add_special_tokens=False)["input_ids"]
        return {"input_ids": (p + c)[:a.max_len],
                "labels": ([-100] * len(p) + c)[:a.max_len]}

    ds = [tokenize_masked(r["question"], r["completion"]) for r in rows]
    ncut = sum(1 for d in ds if len(d["input_ids"]) >= a.max_len)
    print(f"[sft] {len(ds)} examples | truncated at max_len={a.max_len}: {ncut}"
          + ("  RAISE --max_len" if ncut / len(ds) > 0.02 else ""))

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or tok.eos_token_id
        return {"input_ids": torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch]),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch])}

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    args = TrainingArguments(
        output_dir=a.out, max_steps=a.steps, save_steps=15, save_strategy="steps",
        save_only_model=True,
        per_device_train_batch_size=4, gradient_accumulation_steps=8,   # eff. 32
        learning_rate=a.lr, lr_scheduler_type="constant", warmup_steps=0,
        bf16=True, logging_steps=10, report_to=[], seed=a.seed)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate,
            processing_class=tok).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[sft] done -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ckpt", required=True, help="checkpoint that generates the rollouts")
    b.add_argument("--out", default="t1_sft.jsonl")
    b.add_argument("--n", type=int, default=7473)
    b.add_argument("--k", type=int, default=8)
    b.add_argument("--max_tokens", type=int, default=1024)
    s = sub.add_parser("sft")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="Qwen/Qwen3-0.6B", help="init (the BASE model)")
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--lr", type=float, default=1e-5)
    s.add_argument("--max_len", type=int, default=1400)
    s.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"build": build, "sft": sft}[a.cmd](a)
