import numpy as np

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