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

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mbe import mbe_reverse_gram, OnlineMBE


# ---------------------------------------------------------------------------
# Multi-dataset registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "gsm8k": {
        "hf_path": "openai/gsm8k", "hf_name": "main", "split": "test", "category": "math",
    },
    "humaneval": {
        "hf_path": "openai/openai_humaneval", "hf_name": None, "split": "test", "category": "coding",
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc", "hf_name": "ARC-Challenge", "split": "test", "category": "reasoning",
    },
    "mmlu": {
        "hf_path": "cais/mmlu", "hf_name": "all", "split": "test", "category": "knowledge",
    },
}


def _format_question(example, dataset_name):
    """Format a dataset example into a question prompt string."""
    if dataset_name == "gsm8k":
        return example["question"]
    elif dataset_name == "humaneval":
        return f"Complete the following Python function:\n\n{example['prompt']}"
    elif dataset_name == "arc_challenge":
        choices = example["choices"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
        return f"{example['question']}\n\n{opts}\n\nAnswer with the letter only."
    elif dataset_name == "mmlu":
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(example["choices"]))
        return f"{example['question']}\n\n{opts}\n\nAnswer with the letter only."
    return example.get("question", example.get("prompt", ""))


def _extract_gold(example, dataset_name):
    """Extract the gold answer string from a dataset example."""
    if dataset_name == "gsm8k":
        return extract_gold_answer(example["answer"])
    elif dataset_name == "humaneval":
        return example["entry_point"]
    elif dataset_name == "arc_challenge":
        return str(example["answerKey"]).strip()
    elif dataset_name == "mmlu":
        return chr(65 + int(example["answer"]))
    return str(example.get("answer", ""))


def _check_correctness(completion, gold, dataset_name):
    """Dataset-specific correctness check."""
    if dataset_name == "gsm8k":
        predicted = extract_answer_from_completion(completion)
        try:
            return float(predicted) == float(gold)
        except (ValueError, TypeError):
            return False
    elif dataset_name == "humaneval":
        return gold in completion  # heuristic: entry_point function appears in output
    elif dataset_name in ("arc_challenge", "mmlu"):
        matches = re.findall(r'\b([A-D])\b', completion.upper())
        predicted = matches[-1] if matches else ""
        return predicted == gold.upper()
    return False


def load_dataset_examples(dataset_name, n_samples, seed=42):
    """Load n_samples examples from the named dataset.

    Returns list of (question_str, gold_str) tuples.
    """
    cfg = DATASET_REGISTRY[dataset_name]
    if cfg["hf_name"]:
        ds = load_dataset(cfg["hf_path"], cfg["hf_name"])[cfg["split"]]
    else:
        ds = load_dataset(cfg["hf_path"])[cfg["split"]]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False).tolist()
    return [(_format_question(ds[int(i)], dataset_name), _extract_gold(ds[int(i)], dataset_name))
            for i in indices]


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


def compute_per_layer_mbe(hidden_states, prompt_len, patch_size=8):
    per_layer = []
    for layer_idx in range(1, len(hidden_states)):
        mbe_vals = compute_mbe_trace(hidden_states, prompt_len, patch_size=patch_size, layer=layer_idx)
        per_layer.append(mbe_vals.mean().item())
    return per_layer


# ---------------------------------------------------------------------------
# MBE dynamics: per-token cumulative MBE across all layers + growth profile
# (aligned with rollout.ipynb)
# ---------------------------------------------------------------------------
def compute_per_token_mbe(hidden_states, prompt_len):
    """
    Compute cumulative MBE for each token position using OnlineMBE, across all layers.
    For each layer and prefix length t, returns MBE(hidden[prompt_len : prompt_len+t]).

    Returns:
        np.array of shape (n_layers, comp_len)
    """
    n_layers = len(hidden_states) - 1  # skip embedding layer [0]
    seq_len = hidden_states[1].shape[1]
    comp_len = seq_len - prompt_len

    if comp_len < 1:
        return np.zeros((n_layers, 0))

    mbe = np.zeros((n_layers, comp_len))

    for layer_idx in range(n_layers):
        h = hidden_states[layer_idx + 1][0, prompt_len:, :].float()  # (comp_len, D)
        D = h.shape[1]

        tracker = OnlineMBE(D, device=h.device)
        for t in range(comp_len):
            tracker.update(h[t])
            mbe[layer_idx, t] = tracker.mbe().item()

    return mbe


def compute_growth_profile(mbe_matrix):
    """
    For each layer, compute:
      - total_growth: MBE(final) - MBE(initial)
      - half_life: completion % at which MBE reaches 50% of total growth

    Args:
        mbe_matrix: np.array (n_layers, comp_len)
    Returns:
        growth: np.array (n_layers,)
        half_life: np.array (n_layers,)  — in % of completion (0-100)
    """
    n_layers, comp_len = mbe_matrix.shape
    growth = np.zeros(n_layers)
    half_life = np.zeros(n_layers)

    for layer in range(n_layers):
        trace = mbe_matrix[layer]
        if comp_len < 2:
            continue

        total = trace[-1] - trace[0]
        growth[layer] = total

        mid = trace[0] + total * 0.5
        crossed = np.where(trace >= mid)[0]
        if len(crossed) > 0:
            half_life[layer] = crossed[0] / (comp_len - 1) * 100
        else:
            half_life[layer] = 100.0

    return growth, half_life


def interpolate_mbe(mbe_matrix, n_points=50):
    """Interpolate a per-token MBE matrix to a common normalized axis [0, 1]."""
    comp_len = mbe_matrix.shape[1]
    if comp_len < 2:
        return None
    n_layers = mbe_matrix.shape[0]
    norm_pos = np.linspace(0, 1, comp_len)
    common_axis = np.linspace(0, 1, n_points)
    interp = np.zeros((n_layers, n_points))
    for layer in range(n_layers):
        interp[layer] = np.interp(common_axis, norm_pos, mbe_matrix[layer])
    return interp


def compute_mbe_velocity(mbe_matrix, layer_idx=-1):
    """Mean ΔMBE per token for a given layer (last layer by default)."""
    if mbe_matrix.shape[1] < 2:
        return 0.0
    delta = np.diff(mbe_matrix[layer_idx])
    return float(delta.mean())


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
    per_layer_mbe = compute_per_layer_mbe(hidden_states, prompt_len, patch_size=8)
    answer_prob_trace = compute_answer_prob_trace(logits, full_ids, prompt_len, gold_token_ids, n_bins)
    end_cos_sim = compute_end_similarity(model, hidden_states, prompt_len, gold_token_ids)

    # MBE dynamics — cumulative MBE across all layers (aligned with rollout.ipynb)
    mbe_matrix = compute_per_token_mbe(hidden_states, prompt_len)  # (n_layers, comp_len)
    n_layers = mbe_matrix.shape[0]
    growth, half_life = compute_growth_profile(mbe_matrix)

    # Binned trajectory for last layer
    last_layer_traj = mbe_matrix[-1].tolist() if n_layers > 0 else []
    last_layer_traj_binned = bin_trace(last_layer_traj, n_bins)

    # ΔMBE (rate of change) for last layer
    if len(last_layer_traj) >= 2:
        delta_mbe = np.diff(mbe_matrix[-1]).tolist()
    else:
        delta_mbe = [0.0]
    delta_mbe_binned = bin_trace(delta_mbe, n_bins)

    # MBE at checkpoints (last layer)
    mbe_checkpoints = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        t = max(1, int(mbe_matrix.shape[1] * frac)) - 1
        t = min(t, mbe_matrix.shape[1] - 1)
        if mbe_matrix.shape[1] > 0:
            mbe_checkpoints[f"mbe@{int(frac*100)}%"] = float(mbe_matrix[-1, t])

    # Interpolated MBE for cross-rollout averaging
    interp = interpolate_mbe(mbe_matrix, n_points=50)

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
        "per_layer_mbe": per_layer_mbe,
        "answer_prob_trace": answer_prob_trace,
        "end_cos_sim": end_cos_sim,
        # MBE dynamics (aligned with rollout.ipynb)
        "mbe_trajectory": last_layer_traj_binned,
        "delta_mbe": delta_mbe_binned,
        "mbe_checkpoints": mbe_checkpoints,
        "mbe_growth": growth.tolist(),       # per-layer total growth
        "mbe_half_life": half_life.tolist(),  # per-layer half-life (% of completion)
        "mbe_interp": interp.tolist() if interp is not None else None,  # (n_layers, 50)
    }


# ---------------------------------------------------------------------------
# vLLM batch generation + dataset rollout entry point
# ---------------------------------------------------------------------------
def batch_generate_vllm(vllm_engine, tokenizer, questions, max_new_tokens=512):
    """Batch-generate completions with vLLM offline engine.

    Returns list of (completion_text, prompt_text) tuples.
    """
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_new_tokens)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for q in questions
    ]
    outputs = vllm_engine.generate(prompts, sampling_params)
    return [(out.outputs[0].text, prompt) for prompt, out in zip(prompts, outputs)]


def build_result_from_completion(model, tokenizer, question, gold, completion_text, prompt_text,
                                  dataset_name="gsm8k"):
    """Run HF forward pass on a pre-generated completion to get hidden states + MBE."""
    full_text = prompt_text + completion_text
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]

    correct = _check_correctness(completion_text, gold, dataset_name)
    _, hidden_states = full_forward(model, full_ids)
    mbe = compute_per_token_mbe(hidden_states, prompt_len)

    comp_ids = full_ids[0, prompt_len:].tolist()
    tokens = [tokenizer.decode([tid]) for tid in comp_ids]

    return {
        "dataset": dataset_name,
        "question": question[:100],
        "gold": gold,
        "correct": correct,
        "comp_len": full_ids.shape[1] - prompt_len,
        "completion_text": completion_text[:300],
        "tokens": tokens,
        "mbe": mbe,
        "mbe_velocity": compute_mbe_velocity(mbe),
    }


def run_dataset_rollouts(model, tokenizer, dataset_name, n_samples,
                         vllm_engine=None, max_new_tokens=512, seed=42):
    """Run rollouts for a dataset; returns list of result dicts with MBE dynamics.

    Uses vLLM for fast batch generation when vllm_engine is provided,
    then HF forward passes for hidden-state extraction.
    """
    examples = load_dataset_examples(dataset_name, n_samples, seed=seed)
    results = []

    if vllm_engine is not None:
        questions = [q for q, _ in examples]
        vllm_outputs = batch_generate_vllm(vllm_engine, tokenizer, questions, max_new_tokens)
        for (question, gold), (completion_text, prompt_text) in zip(examples, vllm_outputs):
            r = build_result_from_completion(
                model, tokenizer, question, gold, completion_text, prompt_text, dataset_name
            )
            results.append(r)
    else:
        for question, gold in examples:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                )
            completion_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
            r = build_result_from_completion(
                model, tokenizer, question, gold, completion_text, prompt_text, dataset_name
            )
            results.append(r)

    return results


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

    print_trace("2. MBE patch trace", [r["mbe_trace"] for r in correct],
                [r["mbe_trace"] for r in incorrect], n_bins, higher_is_better=True)

    print_trace("2b. Cumulative MBE trajectory (last layer)", [r["mbe_trajectory"] for r in correct],
                [r["mbe_trajectory"] for r in incorrect], n_bins, higher_is_better=True)

    print_trace("2c. ΔMBE rate (last layer)", [r["delta_mbe"] for r in correct],
                [r["delta_mbe"] for r in incorrect], n_bins, higher_is_better=True)

    # MBE checkpoints
    if results and "mbe_checkpoints" in results[0] and results[0]["mbe_checkpoints"]:
        ckpt_keys = sorted(results[0]["mbe_checkpoints"].keys())
        print(f"\n--- 2d. MBE at completion checkpoints (last layer) ---")
        print(f"{'':>12}", end="")
        for k in ckpt_keys:
            print(f" {k:>10}", end="")
        print()
        print("-" * (12 + 11 * len(ckpt_keys)))
        for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
            if not group:
                continue
            print(f"{label:>12}", end="")
            for k in ckpt_keys:
                avg = sum(r["mbe_checkpoints"].get(k, 0) for r in group) / len(group)
                print(f" {avg:>10.4f}", end="")
            print(f"  (n={len(group)})")
        if correct and incorrect:
            print(f"{'Delta':>12}", end="")
            for k in ckpt_keys:
                vc = sum(r["mbe_checkpoints"].get(k, 0) for r in correct) / len(correct)
                vi = sum(r["mbe_checkpoints"].get(k, 0) for r in incorrect) / len(incorrect)
                print(f" {vc - vi:>+10.4f}", end="")
            print()

    # Per-layer MBE growth profile
    if results and "mbe_growth" in results[0] and results[0]["mbe_growth"]:
        n_layers = len(results[0]["mbe_growth"])
        print(f"\n--- 2e. Per-layer MBE growth & half-life ---")
        print(f"{'Layer':>8} {'Growth(C)':>10} {'Growth(I)':>10} {'Delta':>10} {'HL%(C)':>8} {'HL%(I)':>8}")
        print("-" * 62)
        for li in range(n_layers):
            gc = sum(r["mbe_growth"][li] for r in correct) / len(correct) if correct else 0
            gi = sum(r["mbe_growth"][li] for r in incorrect) / len(incorrect) if incorrect else 0
            hc = sum(r["mbe_half_life"][li] for r in correct) / len(correct) if correct else 0
            hi = sum(r["mbe_half_life"][li] for r in incorrect) / len(incorrect) if incorrect else 0
            print(f"{li + 1:>8} {gc:>10.4f} {gi:>10.4f} {gc - gi:>+10.4f} {hc:>8.1f} {hi:>8.1f}")

    print_trace("3. P(correct answer | prefix)", [r["answer_prob_trace"] for r in correct],
                [r["answer_prob_trace"] for r in incorrect], n_bins, higher_is_better=True)

    print(f"\n--- 4. End-of-trace cosine similarity to correct answer embedding ---")
    if correct:
        vals = [r["end_cos_sim"] for r in correct]
        print(f"  Correct:   mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    if incorrect:
        vals = [r["end_cos_sim"] for r in incorrect]
        print(f"  Incorrect: mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

    if results and "per_layer_mbe" in results[0] and results[0]["per_layer_mbe"]:
        n_layers = len(results[0]["per_layer_mbe"])
        labels = torch.tensor([1.0 if r["correct"] else 0.0 for r in results], dtype=torch.float32)
        label_std = labels.std(unbiased=False)
        print(f"\n--- 5. Per-layer MBE vs answer correctness ---")
        print(f"{'Layer':>8} {'Correct':>10} {'Incorrect':>10} {'Delta':>10} {'Corr':>10}")
        print("-" * 54)
        for layer_idx in range(n_layers):
            layer_vals = torch.tensor([r["per_layer_mbe"][layer_idx] for r in results], dtype=torch.float32)
            correct_vals = [r["per_layer_mbe"][layer_idx] for r in correct]
            incorrect_vals = [r["per_layer_mbe"][layer_idx] for r in incorrect]
            vc = sum(correct_vals) / len(correct_vals) if correct_vals else 0.0
            vi = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else 0.0
            val_std = layer_vals.std(unbiased=False)
            if len(results) > 1 and val_std.item() > 0 and label_std.item() > 0:
                corr = ((layer_vals - layer_vals.mean()) * (labels - labels.mean())).mean() / (val_std * label_std)
                corr_value = corr.item()
            else:
                corr_value = 0.0
            print(f"{layer_idx + 1:>8} {vc:>10.4f} {vi:>10.4f} {vc - vi:>+10.4f} {corr_value:>10.4f}")


def summarize_per_layer_mbe(results):
    if not results or "per_layer_mbe" not in results[0] or not results[0]["per_layer_mbe"]:
        return []

    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]
    n_layers = len(results[0]["per_layer_mbe"])
    labels = torch.tensor([1.0 if r["correct"] else 0.0 for r in results], dtype=torch.float32)
    label_std = labels.std(unbiased=False)
    summary = []

    for layer_idx in range(n_layers):
        layer_vals = torch.tensor([r["per_layer_mbe"][layer_idx] for r in results], dtype=torch.float32)
        correct_vals = [r["per_layer_mbe"][layer_idx] for r in correct]
        incorrect_vals = [r["per_layer_mbe"][layer_idx] for r in incorrect]
        vc = sum(correct_vals) / len(correct_vals) if correct_vals else 0.0
        vi = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else 0.0
        val_std = layer_vals.std(unbiased=False)
        if len(results) > 1 and val_std.item() > 0 and label_std.item() > 0:
            corr = ((layer_vals - layer_vals.mean()) * (labels - labels.mean())).mean() / (val_std * label_std)
            corr_value = corr.item()
        else:
            corr_value = 0.0
        summary.append({
            "layer": layer_idx + 1,
            "correct_mean": vc,
            "incorrect_mean": vi,
            "delta": vc - vi,
            "corr_with_correctness": corr_value,
        })

    return summary


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

    # Auto-detect max_new_tokens from training args if not explicitly set
    training_args_path = os.path.join(model_path, "training_args.bin")
    if os.path.exists(training_args_path) and args.max_new_tokens == 512:
        training_args = torch.load(training_args_path, weights_only=False)
        if hasattr(training_args, "max_completion_length"):
            args.max_new_tokens = training_args.max_completion_length
            print(f"Auto-detected max_new_tokens={args.max_new_tokens} from training args")

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

    # Save — strip mbe_interp (large) from per-result to keep file size manageable
    out_path = os.path.join(model_path, "rollout_analysis.json") if os.path.isdir(model_path) else "rollout_analysis.json"
    per_layer_mbe_summary = summarize_per_layer_mbe(results)
    results_compact = []
    for r in results:
        rc = {k: v for k, v in r.items() if k != "mbe_interp"}
        results_compact.append(rc)

    # Aggregate MBE dynamics summary (aligned with rollout.ipynb)
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]
    mbe_dynamics_summary = {}
    for label, group in [("correct", correct), ("incorrect", incorrect), ("all", results)]:
        if not group:
            continue
        n_layers = len(group[0].get("mbe_growth", []))
        entry = {}
        if "mbe_checkpoints" in group[0] and group[0]["mbe_checkpoints"]:
            ckpt_keys = sorted(group[0]["mbe_checkpoints"].keys())
            entry["checkpoints"] = {
                k: sum(r["mbe_checkpoints"].get(k, 0) for r in group) / len(group) for k in ckpt_keys
            }
        if n_layers > 0:
            entry["growth_per_layer"] = [
                sum(r["mbe_growth"][li] for r in group) / len(group) for li in range(n_layers)
            ]
            entry["half_life_per_layer"] = [
                sum(r["mbe_half_life"][li] for r in group) / len(group) for li in range(n_layers)
            ]
        # Averaged interpolated MBE (n_layers, 50) for plotting
        interps = [np.array(r["mbe_interp"]) for r in group if r.get("mbe_interp") is not None]
        if interps:
            entry["avg_interp_mbe"] = np.mean(interps, axis=0).tolist()
        mbe_dynamics_summary[label] = entry

    with open(out_path, "w") as f:
        json.dump({
            "model": model_path,
            "n_samples": len(results),
            "accuracy": len([r for r in results if r["correct"]]) / len(results) if results else 0,
            "per_layer_mbe_summary": per_layer_mbe_summary,
            "mbe_dynamics_summary": mbe_dynamics_summary,
            "results": results_compact,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()