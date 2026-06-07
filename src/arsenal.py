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
    # z-score, drop it. insert_max_min appends MIN (0.0) when the group is all-correct.
    rew = rewards.view(-1, num_generations)
    cor = corrects.view(-1, num_generations)
    rows = []
    for r, c in zip(rew, cor):
        v = 0.0 if (mode == "insert_max_min" and bool(c.all())) else max_reward
        aug = torch.cat([r, r.new_tensor([v])])
        rows.append(((aug - aug.mean()) / (aug.std() + eps))[:-1])
    return torch.stack(rows).reshape(-1)

# Idea 3. Re-use the prefix trie cached for each query, pick "under-explored" (with low-end quantile childs) and "high potential" (has >0 correct rollouts, not just format rewarded, but correct)
#         "Resample" all-correct rollouts with a prefix conditioned rollouts group (using the prefix sampled from above logic)