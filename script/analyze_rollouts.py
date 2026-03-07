"""
Analyze successful vs unsuccessful rollouts from a trained GRPO/IBRL checkpoint.

Metrics:
  1. Confidence trace — per-token log P(token_t | prefix) across the generation
  2. MBE trace — representation rank/diversity at each position (patch-based)
  3. P(correct answer | prefix_t) — how early the model "knows" the answer
  4. End-of-trace cosine similarity — final hidden state vs correct-answer token embedding

Usage:
    # Single GPU:
    python script/analyze_rollouts.py --checkpoint grpo_gsm8k_output

    # Multi-GPU (4x faster with 4 GPUs):
    python script/analyze_rollouts.py --checkpoint grpo_gsm8k_output --n_gpus 4

    # Specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python script/analyze_rollouts.py --checkpoint grpo_gsm8k_output --n_gpus 4
"""

import argparse
import json
import os
import re
import sys
import tempfile
import multiprocessing as mp

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
    h = hidden_states[layer][0, prompt_len:, :]  # (T_comp, D)
    T, D = h.shape
    usable = (T // patch_size) * patch_size
    if usable == 0:
        return torch.tensor([0.0])
    h = h[:usable].reshape(-1, patch_size, D)
    mbe_vals = mbe_reverse_gram(h)
    return mbe_vals


# ---------------------------------------------------------------------------
# Metric 3: P(correct answer | prefix_t) across generation
# ---------------------------------------------------------------------------
def compute_answer_prob_trace(logits, input_ids, prompt_len, gold_token_ids, n_points=10):
    comp_len = input_ids.shape[1] - prompt_len
    if comp_len <= 0 or len(gold_token_ids) == 0:
        return [float("-inf")] * n_points

    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    positions = [prompt_len + int(i / (n_points - 1) * (comp_len - 1)) for i in range(n_points)]
    positions = [min(p, logits.shape[1] - 2) for p in positions]

    trace = []
    for pos in positions:
        total_lp = 0.0
        for offset, tid in enumerate(gold_token_ids):
            idx = pos + offset
            if idx < log_probs.shape[0]:
                total_lp += log_probs[idx, tid].item()
            else:
                total_lp += -20.0
        trace.append(total_lp / len(gold_token_ids))
    return trace


# ---------------------------------------------------------------------------
# Metric 4: End-of-trace hidden state similarity to correct answer embedding
# ---------------------------------------------------------------------------
def compute_end_similarity(model, hidden_states, prompt_len, gold_token_ids, layer=-1):
    h_end = hidden_states[layer][0, -1, :].float()
    embed_weight = model.get_input_embeddings().weight
    if len(gold_token_ids) == 0:
        return 0.0
    gold_embed = embed_weight[gold_token_ids[0]].float()
    cos_sim = F.cosine_similarity(h_end.unsqueeze(0), gold_embed.unsqueeze(0)).item()
    return cos_sim


# ---------------------------------------------------------------------------
# Binning helper
# ---------------------------------------------------------------------------
def bin_trace(values, n_bins):
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

    predicted = extract_answer_from_completion(completion_text)
    try:
        correct = float(predicted) == float(gold_answer)
    except (ValueError, TypeError):
        correct = False

    gold_token_ids = tokenizer.encode(" " + gold_answer, add_special_tokens=False)
    logits, hidden_states = full_forward(model, full_ids)

    logprob_trace_raw = compute_logprob_trace(logits, full_ids, prompt_len)
    logprob_trace = bin_trace(logprob_trace_raw, n_bins)
    mbe_trace_raw = compute_mbe_trace(hidden_states, prompt_len, patch_size=8)
    mbe_trace = bin_trace(mbe_trace_raw, n_bins)
    answer_prob_trace = compute_answer_prob_trace(logits, full_ids, prompt_len, gold_token_ids, n_bins)
    end_cos_sim = compute_end_similarity(model, hidden_states, prompt_len, gold_token_ids)

    return {
        "correct": correct,
        "predicted": predicted,
        "gold": gold_answer,
        "completion_len": comp_len,
        "completion_text": completion_text[:200],
        "mean_logprob": logprob_trace_raw.mean().item(),
        "logprob_trace": logprob_trace,
        "mean_mbe": mbe_trace_raw.mean().item(),
        "mbe_trace": mbe_trace,
        "answer_prob_trace": answer_prob_trace,
        "end_cos_sim": end_cos_sim,
    }


# ---------------------------------------------------------------------------
# Worker: runs on a single GPU, processes a shard of indices
# ---------------------------------------------------------------------------
def _load_tokenizer(model_path):
    """Load tokenizer, fixing Qwen3 extra_special_tokens bug if needed."""
    # TRL saves extra_special_tokens as a list, but transformers 5.3 expects a dict.
    # Fix it in-place before loading.
    tc_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        if isinstance(tc.get("extra_special_tokens"), list):
            print(f"  Fixing extra_special_tokens (list->removing) in {tc_path}")
            del tc["extra_special_tokens"]
            with open(tc_path, "w") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
    return AutoTokenizer.from_pretrained(model_path)


def worker_fn(gpu_id, model_path, indices, max_new_tokens, seed, output_file):
    """Load model on gpu_id, process assigned indices, save results to output_file."""
    torch.manual_seed(seed + gpu_id)
    device = f"cuda:{gpu_id}"

    tokenizer = _load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main")["test"]

    results = []
    n_correct = 0
    for i, idx in enumerate(indices):
        example = dataset[idx]
        gold = extract_gold_answer(example["answer"])
        r = run_rollout(model, tokenizer, example["question"], gold, max_new_tokens)
        results.append(r)
        n_correct += r["correct"]
        if (i + 1) % 10 == 0:
            print(f"  [GPU {gpu_id}] [{i+1}/{len(indices)}] acc={n_correct/(i+1):.1%}")

    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"  [GPU {gpu_id}] Done: {len(results)} samples, {n_correct} correct ({n_correct/len(results):.1%})")


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


def print_report(results, model_path):
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]
    n_bins = 10

    print(f"\n{'=' * 70}")
    print(f"Model: {model_path}")
    print(f"Samples: {len(results)}  |  Correct: {len(correct)}  |  Incorrect: {len(incorrect)}")
    if results:
        print(f"Accuracy: {len(correct)/len(results):.1%}")
    else:
        print("Accuracy: N/A (no results)")
        return
    print(f"{'=' * 70}")

    print(f"\n--- Scalar metrics ---")
    print(f"{'':>20} {'Correct':>10} {'Incorrect':>10} {'Delta':>10}")
    print("-" * 55)
    for key in ["mean_logprob", "mean_mbe", "end_cos_sim", "completion_len"]:
        vc = sum(r[key] for r in correct) / len(correct) if correct else 0
        vi = sum(r[key] for r in incorrect) / len(incorrect) if incorrect else 0
        print(f"{key:>20} {vc:>10.4f} {vi:>10.4f} {vc - vi:>+10.4f}")

    print_trace("1. Confidence (log-prob)", [r["logprob_trace"] for r in correct],
                [r["logprob_trace"] for r in incorrect], n_bins, higher_is_better=True)

    print_trace("2. MBE (representation diversity)", [r["mbe_trace"] for r in correct],
                [r["mbe_trace"] for r in incorrect], n_bins, higher_is_better=True)

    print_trace("3. P(correct answer | prefix)", [r["answer_prob_trace"] for r in correct],
                [r["answer_prob_trace"] for r in incorrect], n_bins, higher_is_better=True)

    print(f"\n--- 4. End-of-trace cosine similarity to correct answer embedding ---")
    if correct:
        vals = [r["end_cos_sim"] for r in correct]
        print(f"  Correct:   mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    if incorrect:
        vals = [r["end_cos_sim"] for r in incorrect]
        print(f"  Incorrect: mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")


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
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs to use. Samples are split evenly across GPUs.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model_path = args.checkpoint if args.checkpoint else args.model

    # Generate sample indices
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    indices = torch.randperm(len(dataset))[:args.n_samples].tolist()

    n_gpus = min(args.n_gpus, torch.cuda.device_count(), len(indices))

    if n_gpus <= 1:
        # Single GPU path (original behavior)
        print(f"Loading model: {model_path}")
        tokenizer = _load_tokenizer(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
        model.eval()

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
    else:
        # Multi-GPU path: spawn one process per GPU
        print(f"Distributing {len(indices)} samples across {n_gpus} GPUs")

        # Split indices into shards
        shards = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(indices):
            shards[i % n_gpus].append(idx)

        # Create temp files for each worker's results
        tmp_dir = tempfile.mkdtemp(prefix="rollout_analysis_")
        tmp_files = [os.path.join(tmp_dir, f"gpu_{i}.json") for i in range(n_gpus)]

        # Spawn workers (must use 'spawn' to avoid CUDA fork issues)
        ctx = mp.get_context("spawn")
        processes = []
        for gpu_id in range(n_gpus):
            p = ctx.Process(
                target=worker_fn,
                args=(gpu_id, model_path, shards[gpu_id], args.max_new_tokens, args.seed, tmp_files[gpu_id]),
            )
            p.start()
            processes.append(p)
            print(f"  Started worker on GPU {gpu_id}: {len(shards[gpu_id])} samples")

        # Wait for all workers
        for p in processes:
            p.join()

        # Collect results
        results = []
        for tmp_file in tmp_files:
            if os.path.exists(tmp_file):
                with open(tmp_file) as f:
                    results.extend(json.load(f))
                os.remove(tmp_file)
        os.rmdir(tmp_dir)

        print(f"\nCollected {len(results)} results from {n_gpus} GPUs")

    # Report
    print_report(results, model_path)

    # Save
    out_path = os.path.join(model_path, "rollout_analysis.json") if os.path.isdir(model_path) else "rollout_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_path,
            "n_samples": len(results),
            "accuracy": len([r for r in results if r["correct"]]) / len(results) if results else 0,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
