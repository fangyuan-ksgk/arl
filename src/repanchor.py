import torch
from torch import nn


class RepAnchor:
    """Representation-space anchoring against catastrophic forgetting.

    Snapshots, per ``nn.Linear`` layer, the *input* representation subspace that
    the anchor task actually uses (``M``, an orthonormal basis of the top input
    directions), that subspace's per-direction importance (``Om``), and the
    anchor weights (``Ws = W*``). During later training, ``penalty()`` charges a
    quadratic cost for moving the layer's weights *along the protected input
    directions*, weighted by importance:

        penalty = lam * sum_i  Om_i . ((W_i - W*_i) @ M_i) ** 2

    so the model is free to change how it maps unused directions but is pinned on
    the directions the anchor task relies on.

    Architecture-agnostic: activations feeding each ``nn.Linear`` are captured
    with forward pre-hooks (not a hand-rolled sequential replay), so this works
    on transformers as well as plain MLPs. Intended for full fine-tuning; with
    LoRA the base ``weight`` is frozen so the penalty has no gradient.
    """

    def __init__(self, model, lam=10.0, rank_mult=1.0, load=4):
        self.model = model
        # All nn.Linear leaves, in module-registration order. Order only has to
        # be internally consistent between capture and penalty (it is: same list).
        self.lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.lam, self.rank_mult, self.load = lam, rank_mult, load
        self.M = [None] * len(self.lin)        # protected representation directions [in, r]
        self.Om = [None] * len(self.lin)       # per-direction importance [r]
        self.Ws = [None] * len(self.lin)       # anchor weights W* [out, in]
        self.ranks = [0] * len(self.lin)       # track effective rank (EMA)

    def penalty(self):
        dev = self.lin[0].weight.device
        loss = torch.zeros((), device=dev, dtype=torch.float32)
        for i, l in enumerate(self.lin):
            if self.M[i] is None:
                continue
            D = (l.weight.float() - self.Ws[i]) @ self.M[i]      # [out, r]
            loss = loss + (self.Om[i].unsqueeze(0) * D ** 2).sum()
        return self.lam * loss

    # ------------------------------------------------------------------
    # Activation capture via forward pre-hooks (any architecture).
    # ------------------------------------------------------------------
    def _capture(self, inputs, grad=False):
        """Run one forward of ``self.model(**inputs)`` and grab the input
        activation feeding every tracked ``nn.Linear``.

        Returns (acts, out) where acts[i] is the input tensor of self.lin[i].
        When grad=True, retain_grad is set so acts[i].grad is populated after a
        backward from a scalar built on ``out``.
        """
        acts = [None] * len(self.lin)
        handles = []

        def mk(i):
            def pre_hook(module, args, kwargs):
                a = args[0] if args else kwargs.get("input")
                if grad and torch.is_grad_enabled() and a.requires_grad:
                    a.retain_grad()
                acts[i] = a
                return None
            return pre_hook

        for i, l in enumerate(self.lin):
            handles.append(l.register_forward_pre_hook(mk(i), with_kwargs=True))
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                out = self.model(**inputs)
        finally:
            for h in handles:
                h.remove()
        return acts, out

    @staticmethod
    def _flatten(a, mask):
        """Flatten a [..., D] activation to [N, D] float32, dropping padding when
        the leading shape matches ``mask``."""
        flat = a.reshape(-1, a.size(-1)).float()
        if mask is not None and tuple(a.shape[:-1]) == tuple(mask.shape):
            flat = flat[mask.reshape(-1).bool()]
        return flat

    @staticmethod
    def _eff_rank(S):                              # participation ratio of eigenvalues s^2
        return ((S**2).sum()**2 / (S**4).sum().clamp_min(1e-12)).item()

    @torch.no_grad()
    def _directions_from_cov(self, C):
        """Top input directions from an accumulated covariance C = R^T R ([in, in]).

        Equivalent to the left singular vectors U of R^T (svd(R.t())) but computed
        from the [in, in] Gram matrix, so memory is independent of the number of
        token rows N. Singular values S = sqrt(eigenvalues); the effective-rank
        cutoff is unchanged.
        """
        evals, evecs = torch.linalg.eigh(C)         # ascending eigenvalues, orthonormal evecs
        evals = evals.flip(0).clamp_min(0)          # descending, PSD
        evecs = evecs.flip(1)
        S = evals.sqrt()
        if S.sum() < 1e-12:
            return None, 0
        r = max(1, int(round(self.rank_mult * self._eff_rank(S))))  # effective rank
        r = min(r, S.numel())
        return evecs[:, :r].contiguous(), r

    @torch.no_grad()
    def _truncate_anchor(self, i):                 # cap at ~ load x effective rank, keep most important
        M, budget = self.M[i], int(round(self.load * self.ranks[i]))
        if M is None or self.ranks[i] == 0 or M.shape[1] <= budget: return
        keep = self.Om[i].topk(budget).indices.sort().values
        self.M[i], self.Om[i] = M[:, keep], self.Om[i][keep]

    @staticmethod
    def _as_batches(inputs, chunk):
        """Normalise `inputs` into a re-iterable list of micro-batch dicts.

        A single dict is split into chunks of `chunk` sequences; a list/tuple of
        dicts is used as-is (each element already a micro-batch, e.g. one per
        generation group).
        """
        if isinstance(inputs, dict):
            n = inputs["input_ids"].shape[0]
            return [{k: v[s:s + chunk] for k, v in inputs.items()}
                    for s in range(0, n, chunk)]
        return list(inputs)

    def update_anchor(self, inputs, chunk=8):
        """Snapshot anchor directions & importance from a batch.

        Processed in micro-batches so memory stays bounded even for long
        prompt+completion sequences: instead of holding every layer's activations
        at once, we accumulate each layer's [in, in] covariance for the
        directions, and a running mean of squared projected gradients for the
        importances.

        Args:
            inputs: dict of model kwargs (e.g. {"input_ids", "attention_mask"})
                    already on the model's device, OR a list of such dicts (one
                    per pre-built micro-batch). ``attention_mask`` (if given) is
                    used to drop padding tokens from the activation stats.
            chunk:  micro-batch size when `inputs` is a single dict.
        """
        batches = self._as_batches(inputs, chunk)
        was_training = self.model.training
        self.model.eval()
        try:
            # (1) Protected directions M: accumulate per-layer covariance R^T R.
            C = [None] * len(self.lin)
            for sub in batches:
                m_sub = sub.get("attention_mask")
                acts, _ = self._capture(sub, grad=False)
                for i, a in enumerate(acts):
                    R = self._flatten(a, m_sub)          # [n_b, in] float32
                    Ci = R.t() @ R
                    C[i] = Ci if C[i] is None else C[i] + Ci
            for i in range(len(self.lin)):
                self.M[i], r = self._directions_from_cov(C[i])
                self.ranks[i] = r
            C = None

            # (2) Per-direction importance: running mean of (grad @ M)^2, where the
            # gradient is of the output magnitude (logits^2) w.r.t. each input act.
            sumsq = [None] * len(self.lin)
            cnt = [0] * len(self.lin)
            for sub in batches:
                m_sub = sub.get("attention_mask")
                self.model.zero_grad(set_to_none=True)
                acts, out = self._capture(sub, grad=True)
                logits = out.logits if hasattr(out, "logits") else out[0]
                logits.float().pow(2).sum(-1).sum().backward()
                for i, a in enumerate(acts):
                    if self.M[i] is None or a.grad is None:
                        continue
                    proj = self._flatten(a.grad, m_sub) @ self.M[i]   # [n_b, r]
                    ss = (proj ** 2).sum(0)
                    sumsq[i] = ss if sumsq[i] is None else sumsq[i] + ss
                    cnt[i] += proj.shape[0]
            for i, l in enumerate(self.lin):
                if self.M[i] is None:
                    continue
                if sumsq[i] is None or cnt[i] == 0:
                    self.Om[i] = torch.ones(self.M[i].shape[1], device=self.M[i].device)
                else:
                    self.Om[i] = sumsq[i] / cnt[i]
                self.Ws[i] = l.weight.detach().float().clone()
                self._truncate_anchor(i)            # cap anchors at ~ load x effective rank
            self.model.zero_grad(set_to_none=True)
        finally:
            self.model.train(was_training)