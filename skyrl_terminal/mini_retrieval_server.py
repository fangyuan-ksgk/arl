"""
Tiny drop-in retrieval server for the SkyRL `search` env — no faiss, no conda, no
GPU. Embeds a small corpus with e5-base-v2 once at startup (CPU) and answers
/retrieve with brute-force cosine top-k. Matches the official server's API exactly:

  POST /retrieve  {"query": str, "topk": int, "return_scores": bool}
   ->  {"result": [ {"document": {"id","contents"}, "score": float}, ... ]}

    /home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/mini_retrieval_server.py \
        --corpus ~/data/searchR1_mini/corpus.jsonl --port 8000
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()
STATE = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(last_hidden, mask):
    last_hidden = last_hidden.masked_fill(~mask[..., None].bool(), 0.0)
    return last_hidden.sum(1) / mask.sum(1)[..., None]


@torch.no_grad()
def embed(texts, is_query):
    pre = "query: " if is_query else "passage: "
    tok, model = STATE["tok"], STATE["model"]
    out = []
    for i in range(0, len(texts), 64):
        batch = [pre + t for t in texts[i : i + 64]]
        enc = tok(batch, max_length=256, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        emb = mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
        out.append(F.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


class QueryRequest(BaseModel):
    query: str
    topk: int | None = None
    return_scores: bool = False


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    k = req.topk or 3
    q = embed([req.query], is_query=True)            # (1, d)
    scores = (STATE["corpus_emb"] @ q[0])            # (N,)
    idx = np.argsort(-scores)[:k]
    docs = [{"document": STATE["corpus"][int(i)], "score": float(scores[int(i)])} for i in idx]
    # The tool treats result as a list of PER-QUERY result-lists (result[i] = docs for
    # query i) and passes each inner list to _passages2string. Wrap accordingly.
    return {"result": [docs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    corpus = [json.loads(l) for l in open(args.corpus)]
    STATE["corpus"] = corpus
    STATE["tok"] = AutoTokenizer.from_pretrained(args.model)
    STATE["model"] = AutoModel.from_pretrained(args.model).eval().to(DEVICE)
    print(f"[mini-retriever] {len(corpus)} passages, e5 on {DEVICE}", flush=True)

    emb_path = args.corpus + ".emb.npy"
    if os.path.exists(emb_path) and np.load(emb_path).shape[0] == len(corpus):
        STATE["corpus_emb"] = np.load(emb_path)
        print(f"[mini-retriever] loaded cached embeddings {STATE['corpus_emb'].shape}", flush=True)
    else:
        print("[mini-retriever] embedding corpus...", flush=True)
        STATE["corpus_emb"] = embed([d["contents"] for d in corpus], is_query=False)
        np.save(emb_path, STATE["corpus_emb"])
        print(f"[mini-retriever] embedded + cached {STATE['corpus_emb'].shape}", flush=True)

    print(f"[mini-retriever] ready: serving on :{args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
