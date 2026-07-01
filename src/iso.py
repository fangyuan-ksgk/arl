import torch

def iso_loss(rep: torch.Tensor) -> torch.Tensor:
    """VICReg-style isotropy on a representation.

    Args:
        rep: (..., D) hidden states (any leading shape; flattened over tokens).
    Returns:
        scalar loss = variance-floor term + decorrelation term.
            variance-floor: hinge pushing every coordinate's std up to >= 1  (recruit dormant dims)
            decorrelation : squared off-diagonal covariance, normalized by D  (spread, don't duplicate)
    Add to the task loss as:  total = ce + lambda_iso * iso_loss(rep)   (lambda_iso ~ 0.10 worked best).
    """
    z = rep.reshape(-1, rep.size(-1))
    z = z - z.mean(0)
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    var = torch.diagonal(cov)
    v_term = torch.relu(1.0 - torch.sqrt(var + 1e-6)).mean()
    c_term = (cov - torch.diag(var)).pow(2).sum() / d
    return v_term + c_term