"""L11 rank-repair, minimal reproduction — V2 (token-choice) and V1 (span-rephrase).

Self-contained except for: an OpenAI-compatible chat endpoint (env LLM_BASE /
LLM_MODEL / LLM_KEY_FILE) and an answer verifier (swap `verify` for your task).

Protocol (V2, the winner: med max-rank 396->135, 70% under gate-200 on 150 targets):
  1. teacher-force the target under the student -> per-token ranks
  2. find the worst-rank token; done if max rank <= gate
  3. V2: give the teacher the student's top-8 viable token strings at that
     position; teacher regenerates the SUFFIX starting with one of them
     V1: teacher rephrases only the +-40-char span around the token
  4. verify answer; accept iff max rank strictly decreased; repeat <= rounds

Usage:
  python l11_repair_minimal.py --model <student_ckpt> --data in.jsonl \
      --out out.jsonl --variant v2 --gate 200 --rounds 3
  in.jsonl rows: {"question": ..., "completion": ..., "gold": ...}
"""
import argparse, json, os, time, urllib.request


def chat(system, user, retries=3):
    key = open(os.environ.get("LLM_KEY_FILE", os.path.expanduser("~/.openrouter_key"))).read().strip()
    base = os.environ.get("LLM_BASE", "https://openrouter.ai/api/v1")
    model = os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
    body = json.dumps({"model": model, "temperature": 0.3, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for t in range(retries):
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except Exception:
            if t == retries - 1:
                raise
            time.sleep(5 * (t + 1))


class Student:
    def __init__(self, path, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map={"": device}).eval()

    def prompt_ids(self, question):
        p = self.tok.apply_chat_template([{"role": "user", "content": question}],
                                         tokenize=False, add_generation_prompt=True)
        return self.tok(p, add_special_tokens=False)["input_ids"]

    def ranks(self, question, text):
        """Per-token rank of `text` under the student (0 = argmax).
        Returns (ranks, token_ids, per-position logits)."""
        torch = self.torch
        p_ids = self.prompt_ids(question)
        t_ids = self.tok(text, add_special_tokens=False)["input_ids"][:700]
        full = torch.tensor([p_ids + t_ids], device=self.device)
        with torch.no_grad():
            lg = self.model(input_ids=full).logits[0].float()
        lp = lg[len(p_ids) - 1:-1]
        tgt = torch.tensor(t_ids, device=self.device)
        tlp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        rk = (lp > tlp.unsqueeze(-1)).sum(-1).tolist()
        return rk, t_ids, lp


def worst(student, question, text, pad=40):
    rk, t_ids, lp = student.ranks(question, text)
    w = max(range(len(rk)), key=lambda i: rk[i])
    pre = student.tok.decode(t_ids[:w])
    tokstr = student.tok.decode(t_ids[w:w + 1])
    span = text[max(0, len(pre) - pad):min(len(text), len(pre) + len(tokstr) + pad)]
    viable = [student.tok.decode([t]) for t in lp[w].topk(8).indices.tolist()]
    return max(rk), w, pre, tokstr, span, viable


def verify(text, gold):
    """GSM8K-style: final '#### <answer>'. Swap for your task."""
    import re
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    return bool(m) and m[-1].replace(",", "") == str(gold).replace(",", "")


def repair(student, question, gold, text, variant="v2", gate=200, rounds=3):
    cur = text
    mr0 = worst(student, question, cur)[0]
    mr = mr0
    for _ in range(rounds):
        if mr <= gate:
            break
        _, w, pre, tokstr, span, viable = worst(student, question, cur)
        try:
            if variant == "v2":
                cont = chat("You repair a solution so a small model finds it natural.",
                    f"Problem: {question}\n\nSolution so far (KEEP EXACTLY):\n{pre}\n\n"
                    f"Continue from there. The continuation MUST START with one of: {viable}\n"
                    f"Keep it correct, end with '#### {gold}'. Return ONLY the continuation.")
                cand = pre + cont
            else:  # v1
                cand = chat("You rewrite one phrase of a math solution so a small "
                            "language model finds it more natural and predictable.",
                    f"Problem: {question}\n\nCurrent solution:\n{cur}\n\n"
                    f"The phrase «{span}» (around \"{tokstr}\") is unnatural for the "
                    f"student model. Rewrite the SOLUTION with only that part rephrased "
                    f"into simpler wording. Keep every calculation and the final line "
                    f"'#### {gold}'. Return the full solution only.")
        except Exception:
            break
        if not verify(cand, gold):
            continue
        mr_new = worst(student, question, cand)[0]
        if mr_new < mr:
            cur, mr = cand, mr_new
    return cur, mr0, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["v1", "v2"], default="v2")
    ap.add_argument("--gate", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    st = Student(a.model)
    with open(a.out, "w") as f:
        for line in open(a.data):
            r = json.loads(line)
            cur, mr0, mr = repair(st, r["question"], r["gold"], r["completion"],
                                  a.variant, a.gate, a.rounds)
            f.write(json.dumps({**r, "completion": cur, "max_rank_before": mr0,
                                "max_rank_after": mr}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
