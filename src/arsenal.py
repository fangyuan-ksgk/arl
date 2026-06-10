import math
from collections import defaultdict, deque
from typing import Dict, Sequence

import numpy as np
import torch

def confident_failure_rare_success(correct, logp_o, D_q=1.0,
                                   clip_logp: float = -1e-3,
                                   pos_scale: float = 1.0,
                                   neg_scale: float = 1.0) -> np.ndarray:
    """Per-rollout shaped reward — penalize confident-failure (accuracy) + encourage rare-success (diversity)
       - to break the dead locking zero advantage group via introducing continuously changing reward
    """
    correct = np.asarray(correct, float)
    lp = np.minimum(np.asarray(logp_o, float), clip_logp)   # strictly < 0
    Dq = np.asarray(D_q, float)
    succ, fail = (-lp), (1.0 / lp)
    succ = succ * Dq * pos_scale                            # C3: difficulty-weighted success
    fail = fail / np.maximum(Dq, 1e-3)                      # penalize EASY failures more
    fail = fail * neg_scale                                 # C2: no difficulty on penalty by default
    return np.where(correct > 0.5, succ, fail)


def virtual_rollout_advantages(rewards, corrects, num_generations,
                               max_reward: float = 1.2,
                               mode: str = "insert_max", eps: float = 1e-4):
    # Per GRPO group: append one no-gradient virtual rollout to the reward vector,
    #   insert_max               -> append max_reward to every group
    #   insert_min               -> append MIN (0.0) to all-correct
    #   insert_max_min           -> append MIN (0.0) to all-correct, max_reward otherwise
    #   insert_max_all_incorrect -> append max_reward only to all-incorrect groups
    #   insert_max_mixed         -> append max_reward only to mixed groups

    rew = rewards.view(-1, num_generations)
    cor = corrects.view(-1, num_generations)
    rows = []
    for r, c in zip(rew, cor):
        all_correct = bool(c.all())
        all_incorrect = not bool(c.any())
        mixed = not all_correct and not all_incorrect

        if mode == "insert_min":
            v, insert = 0.0, all_correct
        elif mode == "insert_max_min":
            v, insert = (0.0 if all_correct else max_reward), True
        elif mode == "insert_max_all_incorrect":
            v, insert = max_reward, all_incorrect
        elif mode == "insert_max_mixed":
            v, insert = max_reward, mixed
        else:  # insert_max
            v, insert = max_reward, True

        if insert:
            aug = torch.cat([r, r.new_tensor([v])])
            adv = ((aug - aug.mean()) / (aug.std() + eps))[:-1]
        else:
            adv = (r - r.mean()) / (r.std() + eps)
        rows.append(adv)
    return torch.stack(rows).reshape(-1)


# Buffer Idea. (Tentative)
# Q1. is this buffer even necessary? I thought we already have a PrefixTrie object tracking the previous rollouts stat for each query? 
class RewardBuffer:
    """Per-prompt running buffer of past rewards -> a stable baseline & std that
    don't collapse when a single on-policy group happens to be all-same.

    mode='window' keeps the last `size` rewards per prompt; mode='ema' keeps an
    exponential moving mean/var. `baseline(key)` returns (mean, std) or None if cold.
    """
    def __init__(self, mode: str = "window", size: int = 256, ema: float = 0.9):
        assert mode in ("window", "ema")
        self.mode, self.size, self.ema = mode, size, ema
        self._win: Dict = defaultdict(lambda: deque(maxlen=size))
        self._m: Dict = {}          # ema mean
        self._v: Dict = {}          # ema var

    def update(self, key, rewards: Sequence[float]) -> None:
        for r in rewards:
            r = float(r)
            if self.mode == "window":
                self._win[key].append(r)
            else:
                m = self._m.get(key, r)
                v = self._v.get(key, 0.0)
                d = r - m
                m = m + (1 - self.ema) * d
                v = self.ema * (v + (1 - self.ema) * d * d)
                self._m[key], self._v[key] = m, v

    def baseline(self, key):
        if self.mode == "window":
            xs = self._win.get(key)
            if not xs:
                return None
            a = np.asarray(xs, float)
            return float(a.mean()), float(a.std())
        if key not in self._m:
            return None
        return self._m[key], math.sqrt(max(self._v[key], 0.0))


# Idea 3. Re-use the prefix trie cached for each query, pick "under-explored" (with low-end quantile childs) and "high potential" (has >0 correct rollouts, not just format rewarded, but correct)
#         "Resample" all-correct rollouts with a prefix conditioned rollouts group (using the prefix sampled from above logic)