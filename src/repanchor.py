import torch 

class RepAnchor:
    def __init__(self, model, lam=10.0, cam=0.1, load=4):
        self.model = model
        self.lin = [m for m in model if isinstance(m, nn.Linear)]
        self.lam, self.cam, self.load = lam, cam, load
        self.M = [None] * len(self.lin)        # protected representation directions
        self.Om = [None] * len(self.lin)       # per-direction importance
        self.Ws = [None] * len(self.lin)       # anchor weights W*
        self.ranks = [0] * len(self.lin) # track effective rank

    def penalty(self):
        loss = torch.zeros((), device=DEV)
        for i, l in enumerate(self.lin):
            if self.M[i] is None:
                continue
            D = (l.weight - self.Ws[i]) @ self.M[i]
            loss = loss + (self.Om[i].unsqueeze(0) * D ** 2).sum()
        return self.lam * loss

    def _inputs(self, x, grad=False):          # input activation feeding each Linear
        outs = []; h = x.requires_grad_(True) if grad else x
        for m in self.model:
            if isinstance(m, nn.Linear):
                if grad: h.retain_grad()
                outs.append(h); h = m(h)
            else:
                h = m(h)
        return outs, h

    @staticmethod
    def _eff_rank(S):                              # exp(MBE_2) = participation ratio of eigenvalues s^2
        return ((S**2).sum()**2 / (S**4).sum().clamp_min(1e-12)).item()

    @torch.no_grad()
    def _accumulate_anchor(self, M, R):
        if M is not None: R = R - (R@M)@M.t()       # deflate what is already protected
        U,S,_=torch.linalg.svd(R.t(), full_matrices=False)
        if S.sum()<1e-12: return M, 0
        r = max(1,int(round(self.rank_mult*self._eff_rank(S)))) # effective rank to replace the hard threshold
        r = min(r, S.numel())
        return U[:,:r] if M is None else torch.cat([M, U[:,:r]], 1), r

    @torch.no_grad()
    def _truncate_anchor(self, i):                 # cap at ~ load x effective rank, keep most important
        M, budget = self.M[i], int(round(self.load * self.ranks[i]))
        if M is None or self.ranks[i] == 0 or M.shape[1] <= budget: return
        keep = self.Om[i].topk(budget).indices.sort().values
        self.M[i], self.Om[i] = M[:, keep], self.Om[i][keep]

    def update_anchor(self, X):                   # update anchor directions & their importance
        x = X[torch.randperm(len(X))[:512]]
        with torch.no_grad():
            ins, _ = self._inputs(x)
        for i in range(len(self.lin)):
            self.M[i], r = self._accumulate_anchor(self.M[i], ins[i])
            self.ranks[i] = r if self.ranks[i] == 0 else 0.9 * self.ranks[i] + 0.1 * r  # EMA of task ranks
        self.model.zero_grad()
        ins, out = self._inputs(x.clone(), grad=True)
        out.pow(2).sum(1).mean().backward()     # importance from output-magnitude sensitivity
        for i, l in enumerate(self.lin):
            self.Om[i] = ((ins[i].grad @ self.M[i]) ** 2).mean(0)
            self.Ws[i] = l.weight.detach().clone()
            self._truncate_anchor(i)            # cap anchors at ~ load x effective rank