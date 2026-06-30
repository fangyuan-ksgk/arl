"""
Phase-2 MBE computation — reads rollouts.jsonl, computes per-token MBE/CE traces,
writes mbe_dynamics.jsonl. Run after training completes (or while training, on a
separate GPU).

Usage:
    python script/compute_mbe.py \
        --rollouts output/grpo_pcre_correct/rollouts.jsonl \
        --output   output/grpo_pcre_correct/mbe_dynamics.jsonl \
        --model    Qwen/Qwen3-4B \
        --device   cuda:0
"""

import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.mbe_logger import _compute_per_token_mbe, _compute_per_token_ce, _f4
from src.mbe_reward import full_forward


def process_record(model, tokenizer, rec, device):
    prompt_text = rec["prompt"]
    comp_text   = rec["completion"]
    full_text   = prompt_text + comp_text

    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids   = tokenizer(full_text,   return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    comp_len   = full_ids.shape[1] - prompt_len

    if comp_len < 2:
        return None

    with torch.no_grad():
        logits, hidden = full_forward(model, full_ids)

    mbe      = _compute_per_token_mbe(hidden, prompt_len)
    mbe_diff = __import__("numpy").diff(mbe, axis=1)
    ce       = _compute_per_token_ce(logits, full_ids, prompt_len)

    return {
        "step"      : rec["step"],
        "correct"   : rec["correct"],
        "response"  : comp_text,
        "n_tokens"  : int(comp_len),
        "mbe"       : [_f4(mbe[li])      for li in range(mbe.shape[0])],
        "mbe_diff"  : [_f4(mbe_diff[li]) for li in range(mbe_diff.shape[0])],
        "ce"        : _f4(ce),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True, help="Path to rollouts.jsonl from RolloutRecorder")
    parser.add_argument("--output",   required=True, help="Path to write mbe_dynamics.jsonl")
    parser.add_argument("--model",    default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device",   default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model {args.model} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": args.device}
    )
    model.eval()

    records = [json.loads(l) for l in open(args.rollouts)]
    print(f"Processing {len(records)} rollout records ...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    with open(args.output, "w") as f_out:
        for i, rec in enumerate(records):
            try:
                result = process_record(model, tokenizer, rec, args.device)
            except Exception as e:
                print(f"  [SKIP] record {i} (step={rec.get('step')}): {e}")
                continue
            if result is not None:
                f_out.write(json.dumps(result) + "\n")
                written += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(records)} done, {written} written")

    print(f"Done. {written}/{len(records)} records written to {args.output}")


if __name__ == "__main__":
    main()
