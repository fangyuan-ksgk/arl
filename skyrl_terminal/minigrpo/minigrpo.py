"""
miniGRPO — a single-file, hackable GRPO you can actually read. 🔬

No Ray, no vLLM, no FSDP. Just: sample completions in a sandbox, score them,
turn rewards into advantages, and do one policy-gradient step. The whole point
is that the ADVANTAGE is a single, swappable function — exactly the seam your
`arl/src/tree_trainer.py` overrides in TRL.

Pipeline (one GRPO step):

    prompts ──> sample G completions each ──> score in sandbox ──> rewards
                                                                     │
        ┌──────────────── THE SEAM (edit me) ◀────────────────┐     │
        │  rewards ─> scalar advantage  aᵢ = (rᵢ−mean)/std    │ ◀───┘
        │          ─> per-token advantage  (flat | OPA | …)   │
        └──────────────────────────┬──────────────────────────┘
                                    ▼
              loss = −mean( advantageₜ · logp(tokenₜ) )  over completion tokens
                                    ▼                          (REINFORCE w/ group baseline = GRPO)
                              backward + step

Hack the advantage: write a function `f(rewards, group_ids, token_seqs) -> list[list[float]]`
and register it in ADVANTAGE_MODES. Two are built in: "flat" (vanilla GRPO) and
"opa" (your tree_trainer's Optimistic Prefix Advantage).

Run:
    PY=/home/claudeuser/SkyRL/.venv/bin/python
    $PY minigrpo.py --eval                      # baseline pass@1 on toybox (needs a GPU/model)
    $PY minigrpo.py --train --mode opa --steps 30
    $PY -c "import minigrpo; minigrpo.selftest()"   # no-GPU unit test of the seam
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# in-repo imports (no install needed)
sys.path.insert(0, "/home/claudeuser/SkyRL/skyrl-gym")
sys.path.insert(0, "/home/claudeuser/arl/src")  # tree_trainer.optimistic_prefix_advantages


# ===========================================================================
# 1) THE SEAM — rewards -> scalar advantage -> per-token advantage
#    This is the part you edit. Everything below is plumbing.
# ===========================================================================

def group_zscore(rewards: List[float], group_ids: List[int],
                 norm_by_std: bool = True, eps: float = 1e-6) -> List[float]:
    """Scalar GRPO advantage per rollout: aᵢ = (rᵢ − mean_g) / (std_g + eps).

    Identical to TRL/your tree_trainer's group z-score, just dependency-free.
    `group_ids[i]` says which prompt-group rollout i belongs to (prefix sharing
    and the baseline are per-group).
    """
    by_group: Dict[int, List[float]] = {}
    for r, g in zip(rewards, group_ids):
        by_group.setdefault(g, []).append(r)
    stats = {}
    for g, rs in by_group.items():
        mean = sum(rs) / len(rs)
        std = statistics.pstdev(rs) if len(rs) > 1 else 0.0
        stats[g] = (mean, std)
    out = []
    for r, g in zip(rewards, group_ids):
        mean, std = stats[g]
        out.append((r - mean) / (std + eps) if norm_by_std else (r - mean))
    return out


def adv_flat(rewards, group_ids, token_seqs) -> List[List[float]]:
    """Vanilla GRPO: broadcast each rollout's scalar advantage to all its tokens."""
    scal = group_zscore(rewards, group_ids)
    return [[a] * len(toks) for a, toks in zip(scal, token_seqs)]


def adv_opa(rewards, group_ids, token_seqs, mode: str = "max") -> List[List[float]]:
    """Optimistic Prefix Advantage (your tree_trainer): build a per-group prefix
    trie over completions and credit each token with the best (max) reachable
    continuation's scalar advantage. `mode='min'` = pessimistic backup."""
    from tree_trainer import optimistic_prefix_advantages

    scal = group_zscore(rewards, group_ids)
    # group rollouts by prompt-group (prefix sharing only within a group)
    idx_by_group: Dict[int, List[int]] = {}
    for i, g in enumerate(group_ids):
        idx_by_group.setdefault(g, []).append(i)

    per_token: List[Optional[List[float]]] = [None] * len(token_seqs)
    for g, idxs in idx_by_group.items():
        g_seqs = [token_seqs[i] for i in idxs]
        g_adv = [scal[i] for i in idxs]
        pt = optimistic_prefix_advantages(g_seqs, g_adv, mode=mode)
        for i, vals in zip(idxs, pt):
            per_token[i] = vals
    return per_token  # type: ignore


# Registry: add your own here, e.g. ADVANTAGE_MODES["myidea"] = my_fn
ADVANTAGE_MODES: Dict[str, Callable] = {
    "flat": adv_flat,
    "opa": adv_opa,
    "opa_min": lambda r, g, t: adv_opa(r, g, t, mode="min"),
}


# ===========================================================================
# 2) Sandbox rollout + reward  (toybox env; swap env_class for "terminal")
# ===========================================================================

@dataclass
class Rollout:
    task_id: str
    group_id: int
    prompt_text: str
    completion_text: str
    prompt_ids: List[int] = field(default_factory=list)
    completion_ids: List[int] = field(default_factory=list)
    reward: float = 0.0


def score_completion(env_key: str, completion_text: str, env_class: str = "toybox",
                     env_extra: Optional[Dict] = None, max_turns: int = 1) -> float:
    """Run one completion through the sandbox env and return its scalar reward.

    Uses the registered SkyRL-gym env. `env_key` is the toybox task_id or the
    terminal task_path; `env_extra` is the full extra_info dict for terminal
    (carries task_path/task_name/timeouts). Single-turn: the model's text is one
    action, the env scores it. Reward = self-verifying checks, no reward model.
    """
    import skyrl_gym

    if env_class == "toybox":
        from skyrl_gym.envs.toybox import tasks as toytasks
        toytasks.get_task(env_key)  # validates the id, fails loud on typo
        extras: Dict = {"extra_info": {"task_id": env_key}, "max_turns": max_turns}
    elif env_class == "terminal":
        extras = {"extra_info": env_extra or {"task_path": env_key}, "max_turns": max_turns}
    else:
        raise ValueError(f"unknown env_class: {env_class}")

    env = skyrl_gym.make(env_class, env_config={}, extras=extras)
    try:
        env.init([{"role": "user", "content": ""}])
        out = env.step(completion_text)
        return float(out["reward"])
    finally:
        env.close()


def build_prompts(env_class: str = "toybox", limit: Optional[int] = None):
    """(env_key, chat_messages, env_extra) per task. env_key = toybox task_id or
    terminal task_path; env_extra = full extra_info dict (terminal) or None."""
    if env_class == "toybox":
        sys.path.insert(0, "/home/claudeuser/arl/skyrl_terminal")
        from build_toybox_dataset import SYSTEM_PROMPT
        from skyrl_gym.envs.toybox import tasks as toytasks
        out = [(t["id"], [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": t["prompt"]}], None)
               for t in toytasks.TASKS]
    elif env_class == "terminal":
        import pyarrow.parquet as pq
        rows = pq.read_table("/home/claudeuser/data/terminal_bench/validation.parquet").to_pylist()
        out = [(r["extra_info"]["task_path"], list(r["prompt"]), dict(r["extra_info"])) for r in rows]
    else:
        raise ValueError(f"unknown env_class: {env_class}")
    return out[:limit] if limit else out


def sample_rollouts(model, tok, prompts, n_samples: int, max_new_tokens: int = 512,
                    temperature: float = 1.0, env_class: str = "toybox") -> List[Rollout]:
    """Sample n_samples completions per prompt with HF generate, then score each.

    Returns a flat list of Rollouts; `group_id` indexes the originating prompt
    so the advantage seam can compute per-group baselines.
    """
    import torch

    rollouts: List[Rollout] = []
    model.eval()
    for gid, (env_key, messages, env_extra) in enumerate(prompts):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=False)
        if not torch.is_tensor(enc):       # transformers 5.x may return a BatchEncoding
            enc = enc["input_ids"]
        prompt_ids = enc.to(model.device)
        plen = prompt_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                prompt_ids.repeat(n_samples, 1),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for row in gen:
            comp_ids = row[plen:].tolist()
            # strip trailing pad
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            while comp_ids and comp_ids[-1] == pad:
                comp_ids.pop()
            comp_text = tok.decode(comp_ids, skip_special_tokens=True)
            r = score_completion(env_key, comp_text, env_class=env_class, env_extra=env_extra)
            rollouts.append(Rollout(env_key, gid, "", comp_text,
                                    prompt_ids[0].tolist(), comp_ids, r))
    return rollouts


# ===========================================================================
# 3) GRPO loss + update   (REINFORCE with group baseline == GRPO, no critic)
# ===========================================================================

def grpo_step(model, tok, rollouts: List[Rollout], advantage_mode: str = "flat",
              optimizer=None):
    """One GRPO update over a batch of rollouts. Returns (loss, mean_reward)."""
    import torch
    import torch.nn.functional as F

    rewards = [r.reward for r in rollouts]
    groups = [r.group_id for r in rollouts]
    comp_seqs = [r.completion_ids for r in rollouts]

    # --- THE SEAM: rewards -> per-token advantages ---
    adv_fn = ADVANTAGE_MODES[advantage_mode]
    per_token_adv = adv_fn(rewards, groups, comp_seqs)

    model.train()
    total_loss = 0.0
    n_tok = 0
    optimizer.zero_grad()
    for ro, adv in zip(rollouts, per_token_adv):
        if not ro.completion_ids:
            continue
        ids = torch.tensor([ro.prompt_ids + ro.completion_ids], device=model.device)
        logits = model(ids).logits[0]  # (L, V)
        plen = len(ro.prompt_ids)
        # logp of each completion token under the policy (teacher-forced)
        comp_logits = logits[plen - 1:-1]               # predicts completion tokens
        comp_ids = torch.tensor(ro.completion_ids, device=model.device)
        logp = F.log_softmax(comp_logits, dim=-1).gather(1, comp_ids[:, None]).squeeze(1)
        a = torch.tensor(adv[: logp.shape[0]], device=model.device, dtype=logp.dtype)
        # GRPO objective: maximize advantage-weighted logp  ->  minimize negative
        loss = -(a * logp).sum()
        loss.backward()
        total_loss += loss.item()
        n_tok += logp.shape[0]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss / max(n_tok, 1), sum(rewards) / len(rewards)


# ===========================================================================
# 4) Eval (multi-model sweep) + train loop
# ===========================================================================

def evaluate(model, tok, n_samples: int = 4, env_class: str = "toybox",
             max_new_tokens: int = 512, limit: Optional[int] = None) -> Dict[str, float]:
    """pass@1 (mean reward of the first sample), pass@k (any sample solves), mean reward."""
    prompts = build_prompts(env_class, limit=limit)
    rollouts = sample_rollouts(model, tok, prompts, n_samples, max_new_tokens, env_class=env_class)
    by_group: Dict[int, List[float]] = {}
    for r in rollouts:
        by_group.setdefault(r.group_id, []).append(r.reward)
    pass1 = statistics.mean(rs[0] for rs in by_group.values())
    passk = statistics.mean(1.0 if max(rs) >= 1.0 else 0.0 for rs in by_group.values())
    meanr = statistics.mean(r.reward for r in rollouts)
    return {"pass@1": pass1, f"pass@{n_samples}": passk, "mean_reward": meanr,
            "n_tasks": len(by_group)}


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--env", default="toybox")
    ap.add_argument("--mode", default="flat", choices=list(ADVANTAGE_MODES))
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    if args.eval or not args.train:
        print(f"[eval] {args.model} on {args.env}: ",
              evaluate(model, tok, n_samples=args.n_samples, env_class=args.env,
                       max_new_tokens=args.max_new_tokens))
    if args.train:
        import torch
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        prompts = build_prompts(args.env)
        for step in range(args.steps):
            ros = sample_rollouts(model, tok, prompts, args.n_samples,
                                  args.max_new_tokens, env_class=args.env)
            loss, mr = grpo_step(model, tok, ros, advantage_mode=args.mode, optimizer=opt)
            print(f"step {step:3d} | loss {loss:+.4f} | mean_reward {mr:.3f} | mode {args.mode}",
                  flush=True)


# ===========================================================================
# 5) No-GPU self-test of the seam + sandbox scoring (no model needed)
# ===========================================================================

def selftest():
    # (a) group z-score
    r = [1.0, 0.0, 1.0, 0.0]
    g = [0, 0, 1, 1]
    z = group_zscore(r, g)
    assert z[0] > 0 > z[1] and abs(z[0] + z[1]) < 1e-6, z
    print("✓ group_zscore:", [round(x, 3) for x in z])

    # (b) flat broadcast
    seqs = [[10, 11], [10, 12], [20], [21]]
    flat = adv_flat(r, g, seqs)
    assert flat[0] == [z[0], z[0]] and flat[2] == [z[2]], flat
    print("✓ adv_flat shapes:", [len(x) for x in flat])

    # (c) OPA: shared prefix token [10] should inherit the BEST continuation's adv.
    # group0 seqs share prefix [10]; rollout0 adv>0, rollout1 adv<0 -> token[10] gets max=adv0.
    opa = adv_opa(r, g, seqs, mode="max")
    assert opa[0][0] == max(z[0], z[1]), (opa[0][0], z[0], z[1])
    assert opa[1][0] == max(z[0], z[1]), opa[1][0]  # the other rollout's shared [10] too
    print("✓ adv_opa shared-prefix credit:", [[round(v, 3) for v in s] for s in opa])

    # (d) sandbox scoring: a correct toybox solution must score 1.0, wrong < 1.0
    good = "```python\ndef is_prime(n):\n    return n>1 and all(n%i for i in range(2,int(n**.5)+1))\n```\nTASK_COMPLETE"
    s_good = score_completion("prime_oracle", good)
    s_bad = score_completion("prime_oracle", "no idea")
    assert s_good == 1.0 and s_bad < 1.0, (s_good, s_bad)
    print(f"✓ sandbox scoring: prime_oracle good={s_good} bad={s_bad}")
    print("\nALL SEAM + SANDBOX SELF-TESTS PASSED ✅")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
