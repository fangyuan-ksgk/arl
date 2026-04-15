"""
MBE Dynamics Logger — records full per-token traces during GRPO training.

Integrates as a side-effect inside the reward function (no extra generation):
    logger = MBEDynamicsLogger(tokenizer, log_path="mbe_dynamics.jsonl")
    trainer = GRPOTrainer(model=..., reward_funcs=[correctness_reward, logger.as_reward()], ...)
    logger.set_model(trainer.model)
    trainer.train()

Each JSONL record (one per logged step) contains:
    {
        "step"    : int,
        "rollouts": [
            {
                "correct"  : bool,
                "response" : str,           # raw completion text
                "n_tokens" : int,           # completion length in tokens
                "mbe"      : [[float,...],...],  # (n_layers, n_tokens)  cumulative MBE per prefix
                "mbe_diff" : [[float,...],...],  # (n_layers, n_tokens-1) ΔMBE per token
                "ce"       : [float,...],        # (n_tokens,) cross-entropy per completion token
            },
            ...
        ]
    }

Load for analysis:
    import json, numpy as np
    records = [json.loads(l) for l in open("mbe_dynamics.jsonl")]
    # Per rollout at step 10:
    r = records[10]["rollouts"][0]
    mbe = np.array(r["mbe"])        # (n_layers, n_tokens)
    ce  = np.array(r["ce"])         # (n_tokens,)
"""

import json
import random
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.mbe import OnlineMBE
from src.mbe_reward import full_forward


# ---------------------------------------------------------------------------
# Per-token trace computation
# ---------------------------------------------------------------------------

def _compute_per_token_mbe(hidden_states, prompt_len):
    """Cumulative MBE at each token position for every transformer layer.

    Returns np.ndarray (n_layers, comp_len).
    hidden_states: tuple of len n_layers+1, each tensor (1, seq_len, D).
    """
    n_layers = len(hidden_states) - 1
    comp_len = hidden_states[0].shape[1] - prompt_len
    if comp_len < 2:
        return np.zeros((n_layers, 0))

    mbe = np.zeros((n_layers, comp_len))
    for li in range(n_layers):
        h = hidden_states[li + 1][0, prompt_len:, :].float()  # (comp_len, D)
        tracker = OnlineMBE(h.shape[1], device=h.device)
        for t in range(comp_len):
            tracker.update(h[t])
            mbe[li, t] = tracker.mbe().item()
    return mbe


def _compute_per_token_ce(logits, full_ids, prompt_len):
    """Cross-entropy loss at each completion token position.

    logits  : (1, seq_len, vocab_size)
    full_ids: (1, seq_len)
    Returns np.ndarray (comp_len,).
    """
    comp_len = full_ids.shape[1] - prompt_len
    if comp_len < 1:
        return np.zeros(0)
    # logits[t] predicts token t+1; shift by 1
    pred_logits = logits[0, prompt_len - 1 : prompt_len - 1 + comp_len, :]  # (comp_len, V)
    target_ids  = full_ids[0, prompt_len : prompt_len + comp_len]            # (comp_len,)
    ce = F.cross_entropy(pred_logits, target_ids, reduction="none")          # (comp_len,)
    return ce.cpu().float().numpy()


def _f4(arr):
    """Round array to 4 decimal places for compact JSON storage."""
    return np.round(arr, 4).tolist()


# ---------------------------------------------------------------------------
# MBEDynamicsLogger
# ---------------------------------------------------------------------------

class MBEDynamicsLogger:
    """Logs full per-token MBE, MBE-diff, and CE traces during GRPO training.

    Called as a side-effect inside the reward function — no extra generation.
    Subsamples `sample_k` rollouts per step to bound overhead.

    Args:
        tokenizer  : HF tokenizer
        log_path   : path to output JSONL file
        log_steps  : log every N reward-function calls (1 = every step)
        sample_k   : max rollouts to process per logged step (default: 4)
    """

    def __init__(self, tokenizer, log_path="mbe_dynamics.jsonl", log_steps=1, sample_k=4):
        self.tokenizer = tokenizer
        self.log_path  = log_path
        self.log_steps = log_steps
        self.sample_k  = sample_k
        self.model     = None
        self._call_idx = 0

        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def set_model(self, model):
        self.model = model

    # ------------------------------------------------------------------ #
    # Core: single rollout                                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _analyse_rollout(self, prompt, completion_text, correct, device):
        """Run forward pass and return a per-token trace dict for one rollout."""
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + completion_text

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        full_ids   = self.tokenizer(full_text,   return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]
        comp_len   = full_ids.shape[1] - prompt_len

        if comp_len < 2:
            return None

        logits, hidden = full_forward(self.model, full_ids)

        mbe      = _compute_per_token_mbe(hidden, prompt_len)       # (n_layers, comp_len)
        mbe_diff = np.diff(mbe, axis=1)                              # (n_layers, comp_len-1)
        ce       = _compute_per_token_ce(logits, full_ids, prompt_len)  # (comp_len,)

        return {
            "correct"  : bool(correct),
            "response" : completion_text,
            "n_tokens" : int(comp_len),
            "mbe"      : [_f4(mbe[li])      for li in range(mbe.shape[0])],
            "mbe_diff" : [_f4(mbe_diff[li]) for li in range(mbe_diff.shape[0])],
            "ce"       : _f4(ce),
        }

    # ------------------------------------------------------------------ #
    # Called from reward function                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def log_batch(self, prompts, completions, correct_flags):
        """Compute and log per-token traces for a training batch.

        Args:
            prompts       : list of list-of-dicts (chat messages, prompt only)
            completions   : list of list-of-dicts (chat messages, completion only)
            correct_flags : list of bool — one per completion
        """
        self._call_idx += 1
        if self._call_idx % self.log_steps != 0:
            return
        if self.model is None:
            return

        # Run on CPU to avoid OOM — training model occupies GPU memory
        # during reward computation. Slower but reliable.
        device = torch.device("cpu")

        indices = list(range(len(completions)))
        if len(indices) > self.sample_k:
            indices = random.sample(indices, self.sample_k)

        rollout_records = []
        for i in indices:
            try:
                rec = self._analyse_rollout(
                    prompt          = prompts[i],
                    completion_text = completions[i][0]["content"],
                    correct         = correct_flags[i],
                    device          = device,
                )
            except Exception as e:
                print(f"[MBELogger] _analyse_rollout failed for idx {i}: {e}")
                rec = None
            if rec is not None:
                rollout_records.append(rec)

        record = {
            "step"    : self._call_idx,
            "rollouts": rollout_records,
        }

        # In multi-GPU (accelerate/FSDP) runs, only rank-0 writes to avoid duplicate records
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        n_correct = sum(1 for r in rollout_records if r["correct"])
        print(
            f"[MBEDynamicsLogger] step={self._call_idx}  "
            f"saved {len(rollout_records)} rollouts "
            f"({n_correct} correct, {len(rollout_records)-n_correct} incorrect) "
            f"→ {self.log_path}"
        )

    # ------------------------------------------------------------------ #
    # Reward-function wrapper (logs as side-effect, returns zeros)        #
    # ------------------------------------------------------------------ #

    def as_reward(self, correctness_fn=None):
        """Return a TRL-compatible reward function that logs as a side-effect.

        If correctness_fn is provided it is called to derive correct_flags.
        Always returns 0.0 per completion — no reward signal.
        """
        logger = self

        def _reward(prompts, completions, gold_answer=None, **kwargs):
            if correctness_fn is not None and gold_answer is not None:
                rewards       = correctness_fn(
                    completions=completions, gold_answer=gold_answer,
                    prompts=prompts, **kwargs,
                )
                correct_flags = [r > 0.0 for r in rewards]
            else:
                correct_flags = [False] * len(completions)

            logger.log_batch(prompts, completions, correct_flags)
            return [0.0] * len(completions)

        _reward.__name__ = "mbe_dynamics_logger"
        return _reward
