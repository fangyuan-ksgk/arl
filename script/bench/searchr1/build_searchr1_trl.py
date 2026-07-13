"""Build the SearchR1-mini dataset in a TRL/HF-datasets-friendly format.

Adapted from arl/skyrl_terminal/build_searchr1_mini.py (SQuAD -> SearchR1
schema). SQuAD because every question ships its own context paragraph, so a
corpus built from those paragraphs GUARANTEES the answer is findable by the
search agent even at mini scale.

Outputs to --data_dir (default ~/data/searchr1_trl):
  train.parquet / validation.parquet with columns:
    prompt        — chat messages [{role: system}, {role: user}] with the
                    SearchR1 instructions (TRL conversational prompt column)
    ground_truth  — {"target": [accepted answers]} (qa_em reads ["target"])
    data_source, question — passthrough metadata
  corpus.jsonl    — {"id", "contents": "title\ntext"} per line, for
                    mini_retrieval_server.py

    CUDA_VISIBLE_DEVICES="" python build_searchr1_trl.py --n_train 2000 --n_val 300
"""

import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

# Exact SearchR1 prompt (from Search-R1 / SkyRL searchr1_dataset.py)
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


def make_row(question, answers, split):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": USER_PREFIX + question},
        ],
        # qa_em reads ground_truth["target"] (list of accepted answers)
        "ground_truth": {"target": answers},
        "data_source": "searchR1_squad",
        "question": question,
    }


def collect(ds, n, split, corpus, seen):
    """SQuAD example: question, context (paragraph), answers={'text':[...]}, title."""
    rows = []
    stride = max(1, len(ds) // n)  # stride-sample so contexts span many articles
    for i in range(0, len(ds), stride):
        if len(rows) >= n:
            break
        ex = ds[i]
        answers = list(dict.fromkeys(ex["answers"]["text"]))  # dedupe, keep order
        if not answers:
            continue
        key = ex["title"].replace("_", " ") + "\n" + ex["context"].strip()
        if key not in seen:
            seen[key] = len(corpus)
            corpus.append({"id": len(corpus), "contents": key})
        rows.append(make_row(ex["question"], answers, split))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/data/searchr1_trl"))
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=300)
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
    print("sample ground_truth:", train_rows[0]["ground_truth"])
    print("wrote ->", args.data_dir)


if __name__ == "__main__":
    main()
