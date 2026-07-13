"""Validate: 'XOR is unlearnable pattern-by-pattern (block-to-convergence) but learnable in batch.'

Two experiments:
  1. XOR with a 2-8-1 tanh MLP. Regimes vary batch composition (block size 1/2/3/4)
     and steps-per-block (1 = fine interleave, 200 = train block to convergence).
  2. 'Yes/No collapse' toy: two near-duplicate inputs x_A=(1,eps), x_B=(1,-eps)
     with opposite labels -- the minimal model of two queries with similar
     representations and conflicting targets.

2 Insights from the experiments: 
(1). The more similar the inputs representations are, the more "churning" occur
(2). Under extreme input similarity, batch together (or interleave) similar data points are KEY to better performance

[Thought]. For LLM training, it's not clear how we define 'conflicting targets', however, we can easily 
#          detect 'similar inputs' by cosine similarity of their representations, I suspect we can simply 
#          batch up queries with similar representations and train them in a batch (or always group them up in their appearance order)

# I learn from ICML that GRPO works just well with SGD (and sometimes even better with SGD than with Adam)
# Under this hypothesis, GRPO issues here can be resolved by 'interleaving' similarly represented queries
# in order to avoid 

Metrics (over last 25% of training):
  cur   = avg fraction of patterns currently correct   (~ per-seed accuracy)
  ever  = avg fraction ever correct so far              (~ union / ever-solved envelope)
  gap   = ever - cur                                    (~ the forgetting/lottery gap)
  flips = correct<->wrong transitions per pattern (whole run)
  solved= all patterns correct at every one of the last 500 steps
"""
import numpy as np

H, LR, TOTAL, SEEDS = 8, 0.5, 10000, 10
BLOCK_LOSS_EXIT = 0.05  # 'convergence' threshold for a block


def init(seed, d_in):
    rng = np.random.default_rng(seed)
    return {'W1': rng.normal(0, 1, (d_in, H)), 'b1': np.zeros(H),
            'W2': rng.normal(0, 1, (H, 1)), 'b2': np.zeros(1)}


def fwd(p, x):
    a1 = np.tanh(x @ p['W1'] + p['b1'])
    z2 = a1 @ p['W2'] + p['b2']
    return a1, 1 / (1 + np.exp(-z2))


def grad_loss(p, X, Y, idx):
    x, y = X[idx], Y[idx]
    a1, yh = fwd(p, x)
    n = len(idx)
    d2 = (yh - y[:, None]) / n
    g = {'W2': a1.T @ d2, 'b2': d2.sum(0)}
    d1 = (d2 @ p['W2'].T) * (1 - a1 ** 2)
    g['W1'] = x.T @ d1
    g['b1'] = d1.sum(0)
    loss = -(y * np.log(yh[:, 0] + 1e-12) + (1 - y) * np.log(1 - yh[:, 0] + 1e-12)).mean()
    return g, loss


def correct(p, X, Y):
    _, yh = fwd(p, X)
    return (yh[:, 0] > 0.5) == (Y > 0.5)


def run(seed, X, Y, blocks, k_steps):
    p = init(seed, X.shape[1])
    hist = np.empty((TOTAL, len(Y)), dtype=bool)
    t, bi = 0, 0
    while t < TOTAL:
        idx = blocks[bi % len(blocks)]
        bi += 1
        for _ in range(k_steps):
            g, loss = grad_loss(p, X, Y, idx)
            for k in p:
                p[k] -= LR * g[k]
            hist[t] = correct(p, X, Y)
            t += 1
            if t >= TOTAL:
                break
            if k_steps > 1 and loss < BLOCK_LOSS_EXIT:
                break
    q0 = int(TOTAL * 0.75)
    ever = np.maximum.accumulate(hist, axis=0)
    return dict(cur=hist[q0:].mean(), ever=ever[q0:].mean(),
                flips=(hist[1:] != hist[:-1]).sum() / len(Y),
                solved=bool(hist[-500:].all()))


def agg(X, Y, blocks, k_steps):
    rs = [run(s, X, Y, blocks, k_steps) for s in range(SEEDS)]
    return {m: np.mean([r[m] for r in rs]) for m in ('cur', 'ever', 'flips')} | \
           {'gap': np.mean([r['ever'] - r['cur'] for r in rs]),
            'solved': sum(r['solved'] for r in rs)}


def report(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'regime':<38}{'cur':>7}{'ever':>7}{'gap':>7}{'flips':>8}{'solved':>9}")
    for name, r in rows:
        print(f"{name:<38}{r['cur']:>7.3f}{r['ever']:>7.3f}{r['gap']:>7.3f}"
              f"{r['flips']:>8.1f}{r['solved']:>6}/{SEEDS}")


# ---------- Experiment 1: XOR ----------
X4 = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y4 = np.array([0., 1., 1., 0.])
SUBSETS = {
    1: [[0], [1], [2], [3]],
    2: [[0, 1], [2, 3], [0, 3], [1, 2], [0, 2], [1, 3]],
    3: [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]],
    4: [[0, 1, 2, 3]],
}
rows = [("full batch (bs4)", agg(X4, Y4, SUBSETS[4], 1)),
        ("round-robin bs1, 1 step/block", agg(X4, Y4, SUBSETS[1], 1)),
        ("bs1, 25 steps/block", agg(X4, Y4, SUBSETS[1], 25)),
        ("bs1, to convergence (<=200)", agg(X4, Y4, SUBSETS[1], 200)),
        ("bs2, to convergence (<=200)", agg(X4, Y4, SUBSETS[2], 200)),
        ("bs3, to convergence (<=200)", agg(X4, Y4, SUBSETS[3], 200))]
report("XOR (2-8-1 tanh MLP)", rows)

# ---------- Experiment 2: Yes/No collapse ----------
print("\n=== Yes/No collapse: x_A=(1,eps) -> 1, x_B=(1,-eps) -> 0 ===")
for eps in (0.5, 0.1, 0.02):
    X2 = np.array([[1., eps], [1., -eps]])
    Y2 = np.array([1., 0.])
    rows = [("  full batch", agg(X2, Y2, [[0, 1]], 1)),
            ("  round-robin bs1, 1 step/block", agg(X2, Y2, [[0], [1]], 1)),
            ("  bs1, 200 steps/block", agg(X2, Y2, [[0], [1]], 200))]
    report(f"eps = {eps}  (input cosine sim = {1 - 2 * eps**2 / (1 + eps**2):.4f})", rows)
