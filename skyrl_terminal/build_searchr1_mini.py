"""
Build a self-contained mini SearchR1 dataset + corpus from HotpotQA.

Why HotpotQA: each question ships its own context paragraphs (gold + distractors),
so a corpus built from those paragraphs is GUARANTEED to contain the answers — the
search agent can actually find them and earn reward, even though the corpus is tiny
(~10k passages vs the 21M-passage full wiki index that won't fit this box).

Outputs to $DATA_DIR (default ~/data/searchR1_mini):
  train.parquet / validation.parquet  — SearchR1 schema (search env)
  corpus.jsonl                         — {"id", "contents": "title\\ntext"} per line

    /home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/build_searchr1_mini.py \
        --n_train 800 --n_val 100
"""
import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

# Exact SearchR1 prompt the `search` env expects (from searchr1_dataset.py)
SYSTEM_CONTENT = "You are a helpful and harmless assistant."
USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and "
    "</information>. You can search as many times as you want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def make_row(question, answer, idx, split):
    return {
        "data_source": "searchR1_hotpotqa",
        "prompt": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": USER_PREFIX + question},
        ],
        "env_class": "search",
        # Search-R1 qa_em reward reads ground_truth["target"] (list of accepted answers)
        "reward_spec": {"method": "rule", "ground_truth": {"target": [answer]}},
        "extra_info": {"index": idx, "split": split, "question": question},
    }


def collect(ds, n, split, corpus, seen):
    """SQuAD example: question, context (paragraph), answers={'text':[...]}, title."""
    rows = []
    # stride-sample across the whole split so contexts span many articles
    stride = max(1, len(ds) // n)
    for i in range(0, len(ds), stride):
        if len(rows) >= n:
            break
        ex = ds[i]
        q = ex["question"]
        answers = ex["answers"]["text"]
        if not answers:
            continue
        a = answers[0]
        key = ex["title"].replace("_", " ") + "\n" + ex["context"].strip()
        if key not in seen:
            seen[key] = len(corpus)
            corpus.append({"id": len(corpus), "contents": key})
        rows.append(make_row(q, a, len(rows), split))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/data/searchR1_mini"))
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_val", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    print("loading SQuAD...", flush=True)
    train = load_dataset("rajpurkar/squad", split="train")
    val = load_dataset("rajpurkar/squad", split="validation")

    corpus, seen = [], {}
    train_rows = collect(train, args.n_train, "train", corpus, seen)
    val_rows = collect(val, args.n_val, "validation", corpus, seen)

    pd.DataFrame(train_rows).to_parquet(os.path.join(args.data_dir, "train.parquet"))
    pd.DataFrame(val_rows).to_parquet(os.path.join(args.data_dir, "validation.parquet"))
    with open(os.path.join(args.data_dir, "corpus.jsonl"), "w") as f:
        for d in corpus:
            f.write(json.dumps(d) + "\n")

    print(f"train={len(train_rows)} val={len(val_rows)} corpus_passages={len(corpus)}", flush=True)
    print("sample answer:", train_rows[0]["reward_spec"]["ground_truth"])
    print("wrote ->", args.data_dir)


if __name__ == "__main__":
    main()
