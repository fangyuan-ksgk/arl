"""Minimal super-positioned logit training (idea 8).

The residual stream target is fine, but I prefer the answer token CE loss
eventually, it's worth putting the 'super-positioned CoT' prediction loss into the whole objective
this then resembles the framework we have for SoRL (without the discrete searching steps, that is)

There are 2 ways to play this: SFT / RL
With SFT the objective is just CE on answer + weighted CE on each super logits
With RL we preform rollout with fixed number of super logits, do similar stuff as RLVR (I thought here we can use our dense, decodability reward instead?)

K soft CoT tokens are trainable LOGIT vectors; input embedding = softmax(l/tau) @ E.
Model frozen; init = filler dots; optimize with Adam against either
  A: cross-entropy of the answer tokens          (answer planted in the loss)
  B: MSE to the grafting residuals (layer 14)    (answer never in the loss)
then greedy-decode through "</think> The final answer is \\boxed{".

    python idea8_minimal.py
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, K, LAYER, STEPS = "Qwen/Qwen3-1.7B", 16, 14, 250
ALPHA = 0.1                                    # weight of the super-logit decodability CE in loss_C

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="eager").eval()
for p in model.parameters(): # freeze any parameter in the model entirely
    p.requires_grad_(False)
E = model.get_input_embeddings().weight
CLOSE = tok.encode("</think>\n\n", add_special_tokens=False) # we assume the <think> as well as the user end, assistant start indicator is contained in the prompt & cot but NOT the </think> symbol, a strange assumption
APFX = tok.encode("The final answer is \\boxed{", add_special_tokens=False)
DOT = tok.encode(".", add_special_tokens=False)[0]
emb = lambda ids: E[torch.tensor(ids, device="cuda")]
soft = lambda l: torch.softmax(l, -1).to(E.dtype) @ E

# Q1. what is K, what is E? 
#     is K the number of super logits? and E the LLM embedding matrix? therefore E.shape[0] is the vocabualry size for the tokenizer? 
# Q2. why do we start with 6.0 as raw logit for filler token? I thought the maximal logit 
#     should be computed using vocabulary size here? isn't 6 but an approximation instead here? (but it doesn't really matter to be honest)
#     but after softmax, no matter what the initial raw logit is they lead to tbe same probability & initial embedding here
#     is that a problem for us? that logit scale doesn't matter during optimization? 



def optimize(loss_fn, steps=STEPS, lr=0.15):
    logits = torch.zeros(K, E.shape[0], device="cuda")
    logits[:, DOT] = 6.0                       # start AT the filler baseline
    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        loss = loss_fn(soft(logits), logits)
        opt.zero_grad(); loss.backward(); opt.step()
    return logits.detach()


@torch.no_grad()
def decode(prefix_embeds, n=12): # just greedy decoding for n+1 more tokens
    o = model(inputs_embeds=torch.cat([prefix_embeds, emb(CLOSE + APFX)])[None], use_cache=True)
    ca, nt, out = o.past_key_values, int(o.logits[0, -1].argmax()), []
    for _ in range(n):
        if nt == tok.eos_token_id:
            break
        out.append(nt)
        o = model(input_ids=torch.tensor([[nt]], device="cuda"), past_key_values=ca, use_cache=True)
        ca, nt = o.past_key_values, int(o.logits[0, -1].argmax())
    m = re.findall(r"-?\d+\.?\d*|[A-J]\b|yes|no", tok.decode(out).replace(",", ""), re.I)
    return m[0].lower().rstrip(".") if m else None


def run(prompt, cot, gt):
    p = tok(prompt, add_special_tokens=False).input_ids
    pe = emb(p) # prompt embedding 
    ans = tok(str(gt), add_special_tokens=False).input_ids
    tail = torch.tensor(ans, device="cuda") 

    def loss_A(se, _l=None):                               # answer CE (planted) given super-positioned logit embedding
        o = model(inputs_embeds=torch.cat([pe, se, emb(CLOSE + APFX + ans)])[None])
        return torch.nn.functional.cross_entropy(o.logits[0, -(len(ans)+1):-1].float(), tail)

    def loss_C(se, l):                                     # answer CE + alpha * decodability CE
        o = model(inputs_embeds=torch.cat([pe, se, emb(CLOSE + APFX + ans)])[None])
        ce_ans = torch.nn.functional.cross_entropy(o.logits[0, -(len(ans)+1):-1].float(), tail)
        # CE(super-logits | query): the model's next-token distribution at positions
        # len(p)-1 .. len(p)+K-2 predicts soft tokens 0..K-1; target = softmax of the raw logits
        pred = torch.log_softmax(o.logits[0, len(p)-1:len(p)+K-1].float(), -1)
        q = torch.softmax(l.float(), -1)
        ce_dec = -(q * pred).sum(-1).mean()
        return ce_ans + ALPHA * ce_dec

    with torch.no_grad():                                   # graft targets, ONLY needed for loss_B
        c = tok(cot, add_special_tokens=False).input_ids    # Concern #1. tokenize separately and concatenate might not be equivalent to tokenize on the entire sequence
        o = model(torch.tensor([p + c + CLOSE + APFX], device="cuda"),
                  output_hidden_states=True, output_attentions=True)
        lo, hi = len(p), len(p) + len(c)
        sc = sum(A[0, :, hi:, lo:hi].float().mean(0).mean(0) for A in o.attentions)
        sel = (torch.topk(sc, K).indices.sort().values + lo).tolist()
        target = o.hidden_states[LAYER][0, sel].float()
        del o

    def loss_B(se, _l=None):                               # residual MSE (answer-free)
        o = model(inputs_embeds=torch.cat([pe, se, emb(CLOSE)])[None], output_hidden_states=True)
        return torch.nn.functional.mse_loss(o.hidden_states[LAYER][0, len(p):len(p)+K].float(), target)

    a = decode(torch.cat([pe, soft(optimize(loss_A))]))
    b = decode(torch.cat([pe, soft(optimize(loss_B))]))
    c_ = decode(torch.cat([pe, soft(optimize(loss_C))]))
    f = decode(torch.cat([pe, emb([DOT] * K)]))
    return a, b, c_, f


if __name__ == "__main__":
    import random, collections
    random.seed(0)
    pool = collections.defaultdict(list)
    for r in torch.load("/home/claudeuser/cot_lab/rollouts_v2.pt", weights_only=False):
        if r["kind"] != "agentic" and r.get("correct") and r.get("complete") and r["gt"] is not None:
            pool[r["bench"]].append(r)
    hits = {"A": 0, "B": 0, "C": 0, "filler": 0, "n": 0}
    for bench, rs in sorted(pool.items()):
        r = random.choice(rs)
        gt = str(r["gt"]).replace(",", "").lower()
        a, b, c_, f = run(r["prompt"], r["cot"], gt)
        hits["A"] += a == gt; hits["B"] += b == gt; hits["C"] += c_ == gt
        hits["filler"] += f == gt; hits["n"] += 1
        print(f"{bench:12s} gt={gt:8s} A={a} B={b} C={c_} filler={f}", flush=True)
    print(f"\nA {hits['A']}/{hits['n']}   B {hits['B']}/{hits['n']}   "
          f"C {hits['C']}/{hits['n']}   filler {hits['filler']}/{hits['n']}")
