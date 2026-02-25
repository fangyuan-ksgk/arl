"""
Analyze successful vs unsuccessful rollouts from a trained GRPO/IBRL checkpoint.

Metrics:
  1. Confidence trace — per-token log P(token_t | prefix) across the generation
  2. MBE trace — representation rank/diversity at each position (patch-based)
  3. P(correct answer | prefix_t) — how early the model "knows" the answer
  4. End-of-trace cosine similarity — final hidden state vs correct-answer token embedding

Usage:
    python scripts/analyze_rollouts.py --checkpoint grpo_gsm8k_output
    python scripts/analyze_rollouts.py --model Qwen/Qwen3-0.6B        # base model
    python scripts/analyze_rollouts.py --checkpoint grpo_gsm8k_output --n_samples 100
"""

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mbe import mbe_reverse_gram


# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------
def extract_answer_from_completion(text: str) -> str:
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def extract_gold_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


# ---------------------------------------------------------------------------
# Forward pass with all outputs
# ---------------------------------------------------------------------------
@torch.no_grad()
def full_forward(model, input_ids):
    """Single forward pass returning logits and all hidden states."""
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    return outputs.logits, outputs.hidden_states


# ---------------------------------------------------------------------------
# Metric 1: Per-token log-prob confidence trace
# ---------------------------------------------------------------------------
def compute_logprob_trace(logits, input_ids, prompt_len):
    """Per-completion-token log P(token_t | prefix)."""
    log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
    token_ids = input_ids[0, 1:]
    token_logprobs = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return token_logprobs[prompt_len - 1:]  # completion only


# ---------------------------------------------------------------------------
# Metric 2: MBE trace — patch-based representation diversity over generation
# ---------------------------------------------------------------------------
def compute_mbe_trace(hidden_states, prompt_len, patch_size=8, layer=-1):
    """
    Compute MBE on sliding patches of hidden states across the completion.
    Returns one MBE value per patch, showing how representation diversity
    evolves during generation.
    """
    h = hidden_states[layer][0, prompt_len:, :]  # (T_comp, D)
    T, D = h.shape
    usable = (T // patch_size) * patch_size
    if usable == 0:
        return torch.tensor([0.0])
    h = h[:usable].reshape(-1, patch_size, D)  # (num_patches, patch_size, D)
    mbe_vals = mbe_reverse_gram(h)  # (num_patches,)
    return mbe_vals


# ---------------------------------------------------------------------------
# Metric 3: P(correct answer | prefix_t) across generation
# ---------------------------------------------------------------------------
def compute_answer_prob_trace(logits, input_ids, prompt_len, gold_token_ids, n_points=10):
    """
    At n_points evenly-spaced positions during generation, compute
    log P(gold_answer_tokens | prefix up to that point).
    Shows when the model "locks in" on the correct answer.
    """
    comp_len = input_ids.shape[1] - prompt_len
    if comp_len <= 0 or len(gold_token_ids) == 0:
        return [float("-inf")] * n_points

    log_probs = F.log_softmax(logits[0].float(), dim=-1)  # (T, V)
    positions = [prompt_len + int(i / (n_points - 1) * (comp_len - 1)) for i in range(n_points)]
    positions = [min(p, logits.shape[1] - 2) for p in positions]

    trace = []
    for pos in positions:
        # log P(gold_tokens | everything up to pos) using teacher-forcing from pos
        total_lp = 0.0
        for offset, tid in enumerate(gold_token_ids):
            idx = pos + offset
            if idx < log_probs.shape[0]:
                total_lp += log_probs[idx, tid].item()
            else:
                total_lp += -20.0  # penalty for running past end
        trace.append(total_lp / len(gold_token_ids))  # normalize by answer length
    return trace


# ---------------------------------------------------------------------------
# Metric 4: End-of-trace hidden state similarity to correct answer embedding
# ---------------------------------------------------------------------------
def compute_end_similarity(model, hidden_states, prompt_len, gold_token_ids, layer=-1):
    """
    Cosine similarity between the final hidden state of the completion
    and the embedding of the correct answer's first token.
    High similarity = model's final representation "points toward" the answer.
    """
    h_end = hidden_states[layer][0, -1, :].float()  # (D,)

    # Get the embedding of the gold answer's first token
    embed_weight = model.get_input_embeddings().weight
    if len(gold_token_ids) == 0:
        return 0.0
    gold_embed = embed_weight[gold_token_ids[0]].float()  # (D,)

    cos_sim = F.cosine_similarity(h_end.unsqueeze(0), gold_embed.unsqueeze(0)).item()
    return cos_sim


# ---------------------------------------------------------------------------
# Binning helper
# ---------------------------------------------------------------------------
def bin_trace(values, n_bins):
    """Average values into n_bins equally-spaced segments."""
    if len(values) == 0:
        return [0.0] * n_bins
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    chunk_size = max(1, len(values) // n_bins)
    bins = []
    for i in range(n_bins):
        start = i * chunk_size
        end = start + chunk_size if i < n_bins - 1 else len(values)
        if start < len(values):
            chunk = values[start:end]
            bins.append(sum(chunk) / len(chunk))
        else:
            bins.append(bins[-1] if bins else 0.0)
    return bins


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def run_rollout(model, tokenizer, question, gold_answer, max_new_tokens=512, n_bins=10):
    """Generate completion and compute all 4 metrics."""
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=0.7, top_p=0.9,
    )
    full_ids = output_ids[0].unsqueeze(0)
    completion_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    comp_len = full_ids.shape[1] - prompt_len

    # Correctness
    predicted = extract_answer_from_completion(completion_text)
    try:
        correct = float(predicted) == float(gold_answer)
    except (ValueError, TypeError):
        correct = False

    # Gold answer token ids (for metrics 3 & 4)
    gold_token_ids = tokenizer.encode(" " + gold_answer, add_special_tokens=False)

    # Single forward pass for all metrics
    logits, hidden_states = full_forward(model, full_ids)

    # Metric 1: confidence trace
    logprob_trace_raw = compute_logprob_trace(logits, full_ids, prompt_len)
    logprob_trace = bin_trace(logprob_trace_raw, n_bins)

    # Metric 2: MBE trace
    mbe_trace_raw = compute_mbe_trace(hidden_states, prompt_len, patch_size=8)
    mbe_trace = bin_trace(mbe_trace_raw, n_bins)

    # Metric 3: P(answer | prefix_t) trace
    answer_prob_trace = compute_answer_prob_trace(logits, full_ids, prompt_len, gold_token_ids, n_bins)

    # Metric 4: end similarity
    end_cos_sim = compute_end_similarity(model, hidden_states, prompt_len, gold_token_ids)

    return {
        "correct": correct,
        "predicted": predicted,
        "gold": gold_answer,
        "completion_len": comp_len,
        "completion_text": completion_text[:200],
        # Metric 1
        "mean_logprob": logprob_trace_raw.mean().item(),
        "logprob_trace": logprob_trace,
        # Metric 2
        "mean_mbe": mbe_trace_raw.mean().item(),
        "mbe_trace": mbe_trace,
        # Metric 3
        "answer_prob_trace": answer_prob_trace,
        # Metric 4
        "end_cos_sim": end_cos_sim,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_trace(label, correct_traces, incorrect_traces, n_bins=10, higher_is_better=True):
    hint = "(higher = better)" if higher_is_better else "(lower = better)"
    print(f"\n--- {label} {hint} ---")
    print(f"{'Position':>10}", end="")
    for i in range(n_bins):
        pct = int((i + 0.5) / n_bins * 100)
        print(f"  {pct:>3}%", end="")
    print()
    print("-" * (10 + n_bins * 6))

    def avg_trace(traces):
        return [sum(t[i] for t in traces) / len(traces) for i in range(n_bins)]

    if correct_traces:
        avg_c = avg_trace(correct_traces)
        print(f"{'Correct':>10}", end="")
        for v in avg_c:
            print(f" {v:>5.2f}", end="")
        print(f"  (n={len(correct_traces)})")

    if incorrect_traces:
        avg_i = avg_trace(incorrect_traces)
        print(f"{'Incorrect':>10}", end="")
        for v in avg_i:
            print(f" {v:>5.2f}", end="")
        print(f"  (n={len(incorrect_traces)})")

    if correct_traces and incorrect_traces:
        avg_c = avg_trace(correct_traces)
        avg_i = avg_trace(incorrect_traces)
        diff = [avg_c[i] - avg_i[i] for i in range(n_bins)]
        print(f"{'Delta':>10}", end="")
        for v in diff:
            print(f" {v:>+5.2f}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze rollout traces (4 metrics)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model_path = args.checkpoint if args.checkpoint else args.model

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main")["test"]
    indices = torch.randperm(len(dataset))[:args.n_samples].tolist()

    results = []
    n_correct = 0
    for idx_i, idx in enumerate(indices):
        example = dataset[idx]
        gold = extract_gold_answer(example["answer"])
        r = run_rollout(model, tokenizer, example["question"], gold, args.max_new_tokens)
        results.append(r)
        n_correct += r["correct"]
        if (idx_i + 1) % 10 == 0:
            print(f"  [{idx_i+1}/{args.n_samples}] acc={n_correct/(idx_i+1):.1%}")

    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]
    n_bins = 10

    # ---- Report ----
    print(f"\n{'=' * 70}")
    print(f"Model: {model_path}")
    print(f"Samples: {len(results)}  |  Correct: {len(correct)}  |  Incorrect: {len(incorrect)}")
    print(f"Accuracy: {len(correct)/len(results):.1%}")
    print(f"{'=' * 70}")

    # Scalar summaries
    print(f"\n--- Scalar metrics ---")
    print(f"{'':>20} {'Correct':>10} {'Incorrect':>10} {'Delta':>10}")
    print("-" * 55)
    for key in ["mean_logprob", "mean_mbe", "end_cos_sim", "completion_len"]:
        vc = sum(r[key] for r in correct) / len(correct) if correct else 0
        vi = sum(r[key] for r in incorrect) / len(incorrect) if incorrect else 0
        print(f"{key:>20} {vc:>10.4f} {vi:>10.4f} {vc - vi:>+10.4f}")

    # Trace comparisons
    print_trace("1. Confidence (log-prob)", [r["logprob_trace"] for r in correct],
                [r["logprob_trace"] for r in incorrect], n_bins, higher_is_better=True)

    print_trace("2. MBE (representation diversity)", [r["mbe_trace"] for r in correct],
                [r["mbe_trace"] for r in incorrect], n_bins, higher_is_better=True)

    print_trace("3. P(correct answer | prefix)", [r["answer_prob_trace"] for r in correct],
                [r["answer_prob_trace"] for r in incorrect], n_bins, higher_is_better=True)

    # Metric 4 summary (scalar, not a trace)
    print(f"\n--- 4. End-of-trace cosine similarity to correct answer embedding ---")
    if correct:
        vals = [r["end_cos_sim"] for r in correct]
        print(f"  Correct:   mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    if incorrect:
        vals = [r["end_cos_sim"] for r in incorrect]
        print(f"  Incorrect: mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

    # Save
    out_path = os.path.join(model_path, "rollout_analysis.json") if os.path.isdir(model_path) else "rollout_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_path,
            "n_samples": len(results),
            "accuracy": len(correct) / len(results),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
