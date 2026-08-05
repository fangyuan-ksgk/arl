"""Minimal CoT grafting.

Extract: run prompt+cot+answer once; keep the residual streams (layer L) of the K
         CoT tokens most attended by the answer span.
Inject:  run prompt + K filler dots (at the ORIGINAL position ids) and overwrite the
         dots' layer-L states with the saved vectors; layers above L recompute.
The full CoT is replaced by K vectors.

    python graft_minimal.py            # demo over a few ensemble traces
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Q1. what is K? 
# Q2. what does _hook do? 

MODEL, K, LAYER = "Qwen/Qwen3-1.7B", 16, 14

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="eager").eval()
CLOSE = tok.encode("</think>\n\n", add_special_tokens=False)
APFX = tok.encode("The final answer is \\boxed{", add_special_tokens=False)
DOT = tok.encode(".", add_special_tokens=False)[0]

# --- the graft hook: overwrite layer L-1's output at chosen positions during prefill
G = {"on": False, "pos": None, "vec": None}
def _hook(mod, inp, out):
    # out[0] is the hidden state output from current layer, we replace it with G["vec"] and go to the next layer
    # only graft on dot positions, that is why G["pos"] is also necessary here
    h = out[0] if isinstance(out, tuple) else out
    if not G["on"] or h.shape[1] == 1:
        return out
    h = h.clone()
    h[0, G["pos"], :] = G["vec"].to(h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h
model.model.layers[LAYER - 1].register_forward_hook(_hook)


# This function requires prompt & cot to contain proper formatting (but CoT need not contain the </think> sign)
# Then it selects the indices of CoT that is mostly attened backward from answer tokens, then return the layer L
# residual stream of these CoT tokens

# The indices goes into 'pos' and residual stream extracted goes into 'vec' and with the hook above
# we graft the residual stream to the same layer to have a model to re-process them on-the-fly 

@torch.no_grad()
def extract(prompt, cot, pool="mean"):
    """-> (positions, vectors): top-K answer-attended CoT tokens' layer-L states.

    pool: "mean" -- average over heads (dilutes single-head retrieval signal)
          "max"  -- per-head max after query-averaging (one specialized head suffices)
    """
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(cot, add_special_tokens=False).input_ids
    ids = p + c + CLOSE + APFX
    o = model(torch.tensor([ids], device="cuda"),
              output_hidden_states=True, output_attentions=True)
    lo, hi = len(p), len(p) + len(c)
    # avg. backward attention score from answer tokens back to a specific CoT token
    # A has shape: (B, H, seq_len, seq_len) 
    if pool == "mean":
        score = sum(A[0, :, hi:, lo:hi].float().mean(0).mean(0) for A in o.attentions)
    elif pool == "max":
        score = sum(A[0, :, hi:, lo:hi].float().mean(1).amax(0) for A in o.attentions)
    else:
        raise ValueError(f"unknown pool: {pool!r} (use 'mean' or 'max')")
    sel = (torch.topk(score, min(K, len(c))).indices.sort().values + lo).tolist()
    return sel, o.hidden_states[LAYER][0, sel].clone(), len(p), hi


@torch.no_grad()
def generate(prompt, graft=None, n=12):
    """Greedy generation from prompt + K dots (+ optional graft at original positions).

    Returns the decoded continuation text.
    """
    p = tok(prompt, add_special_tokens=False).input_ids
    if graft is None:
        ids, pos = p + [DOT] * K + CLOSE + APFX, None
    else:
        # arm the graft hook: G is read by _hook at forward time
        sel, vec, lo, hi = graft
        ids = p + [DOT] * len(sel) + CLOSE + APFX
        # RoPE ids: dots impersonate the ORIGINAL CoT positions (sel)
        pos = list(range(lo)) + sel + list(range(hi, hi + len(CLOSE) + len(APFX)))
        G.update(on=True, pos=list(range(lo, lo + len(sel))), vec=vec)
    kw = {} if pos is None else {"position_ids": torch.tensor([pos], device="cuda")}
    o = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True, **kw)
    G["on"] = False  # disarm so the next (ungrafted) prefill is not contaminated
    ca, nt, out = o.past_key_values, int(o.logits[0, -1].argmax()), []
    start = (pos[-1] if pos else len(ids) - 1) + 1
    for j in range(n):
        if nt == tok.eos_token_id:
            break
        out.append(nt)
        o = model(input_ids=torch.tensor([[nt]], device="cuda"), past_key_values=ca,
                  use_cache=True, position_ids=torch.tensor([[start + j]], device="cuda"))
        ca, nt = o.past_key_values, int(o.logits[0, -1].argmax())
    return tok.decode(out)


def parse_answer(text):
    """Pull a normalized number / choice letter / yes-no out of generated text."""
    m = re.findall(r"-?\d+\.?\d*|[A-J]\b|yes|no", text.replace(",", ""), re.I)
    return m[0].lower().rstrip(".") if m else None


def answer(prompt, graft=None, n=12):
    """generate + parse, kept as a convenience wrapper."""
    return parse_answer(generate(prompt, graft=graft, n=n))


if __name__ == "__main__":
    import random, collections
    random.seed(0)
    pool = collections.defaultdict(list)
    for r in torch.load("/home/claudeuser/cot_lab/rollouts_v2.pt", weights_only=False):
        if r["kind"] != "agentic" and r.get("correct") and r.get("complete"):
            pool[r["bench"]].append(r)
    hits = {"graft": 0, "filler": 0, "n": 0}
    for b, rs in sorted(pool.items()):
        r = random.choice(rs)
        g = extract(r["prompt"], r["cot"])
        gt = str(r["gt"]).replace(",", "").lower()
        a_g, a_f = answer(r["prompt"], graft=g), answer(r["prompt"])
        hits["graft"] += a_g == gt; hits["filler"] += a_f == gt; hits["n"] += 1
        print(f"{b:12s} gt={gt:8s} graft={a_g} filler={a_f}")
    print(f"\ngraft {hits['graft']}/{hits['n']}   filler {hits['filler']}/{hits['n']}")
