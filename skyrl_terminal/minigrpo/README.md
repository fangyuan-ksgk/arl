# miniGRPO — a hackable GRPO you can read in one sitting 🔬

`minigrpo.py` is ~250 lines of plain PyTorch + 🤗 Transformers. No Ray, no vLLM,
no FSDP. It exists so you can **edit the GRPO advantage** — the exact thing your
`arl/src/tree_trainer.py` overrides in TRL — without fighting infrastructure.

## The one idea: the advantage is a swappable function

```
prompts ─> sample G completions (sandbox) ─> rewards
                                                │
   ┌──────── THE SEAM (edit me) ◀───────────────┘
   │  group_zscore:  aᵢ = (rᵢ − mean_g)/(std_g+ε)      # scalar, per rollout
   │  to_per_token:  flat | opa | <your idea>           # scalar -> per token
   └───────────────────────┬───────────────────────────┘
                           ▼
   loss = −mean( advantageₜ · logp(tokenₜ) )  over completion tokens   ==  GRPO
```

Two advantage modes ship in `ADVANTAGE_MODES`:
- **`flat`** — vanilla GRPO: broadcast the scalar z-score to every token.
- **`opa`** — your **Optimistic Prefix Advantage**: imports
  `optimistic_prefix_advantages` straight from `arl/src/tree_trainer.py`, builds
  a per-group prefix trie, credits each token with the best reachable
  continuation. (`opa_min` = pessimistic backup.)

**Add your own** in three lines:
```python
def adv_myidea(rewards, group_ids, token_seqs):
    scal = group_zscore(rewards, group_ids)        # reuse the scalar step
    return [...]                                    # return per-token list[list[float]]
ADVANTAGE_MODES["myidea"] = adv_myidea
# then: python minigrpo.py --train --mode myidea
```

This is the same seam as `tree_trainer.TreeTrainer._compute_loss`, which rewrites
`inputs["advantages"]` from the scalar z-score to the per-token OPA tensor. Here
it's just a free function you can unit-test on toy sequences.

## Run it
```bash
PY=/home/claudeuser/SkyRL/.venv/bin/python
cd /home/claudeuser/arl/skyrl_terminal/minigrpo

$PY minigrpo.py selftest                         # no GPU: checks z-score, flat, OPA, sandbox scoring
$PY minigrpo.py --eval --model Qwen/Qwen2.5-1.5B-Instruct        # baseline pass@1 on toybox
$PY minigrpo.py --train --mode opa --steps 30 --model Qwen/Qwen2.5-0.5B-Instruct
```
`--env toybox` (default) or `--env terminal`. Reward is the env's self-verifying
checks — no reward model.

## Map to your `tree_trainer.py`
| tree_trainer (TRL) | miniGRPO |
|---|---|
| `aᵢ = (rᵢ−mean)/std` (TRL internal) | `group_zscore()` |
| `optimistic_prefix_advantages()` / `PrefixTrie` | imported & used directly by `adv_opa` |
| `TreeTrainer._compute_loss` rewrites `inputs["advantages"]` | `grpo_step()` calls `ADVANTAGE_MODES[mode]` |
| `_tree_token_advantages` (group by prompt, scatter) | `adv_opa` (group by `group_id`, return ragged) |
| `virtual_rollout` / `shaped_reward` hooks | add a mode, or preprocess `rewards` before `group_zscore` |

---

## Scaling up: patching SkyRL's real GRPO (Ray/FSDP/vLLM)

SkyRL already has a **plug-in registry** — you do *not* edit core files.

1. The GRPO advantage lives in
   `SkyRL/skyrl/backends/skyrl_train/utils/ppo_utils.py :: compute_grpo_outcome_advantage`.
   It computes the scalar z-score then **flat-broadcasts**: `scores.unsqueeze(-1) * response_mask`
   — i.e. exactly miniGRPO's `adv_flat`.

2. Register your own without touching SkyRL (pattern in
   `examples/train/algorithms/custom_advantage_estimator/main_custom_adv_est.py`):
   ```python
   from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator

   @register_advantage_estimator("opa")
   def compute_opa_advantage(token_level_rewards, response_mask, index, **kwargs):
       # 1. scalar z-score per group (same as grpo)
       # 2. build prefix trie per group, walk -> per-token advantage tensor
       return advantages, returns       # both (batch, seqlen)
   ```
   then run with `trainer.algorithm.advantage_estimator=opa`.

3. **⚠️ The one wrinkle (important):** the estimator is handed only
   `token_level_rewards`, `response_mask`, and `index` (= prompt-group uids) — **not the
   response token ids**, which OPA needs to build the prefix trie. The token ids do
   exist one level up, in the `TrainingInputBatch` passed to
   `Trainer.compute_advantages_and_returns` (`SkyRL/skyrl/train/trainer.py:960`,
   `data["responses"]`). So to do OPA at scale you must **thread the response token ids
   into the estimator** — either:
   - pass them through `**kwargs` by lightly wrapping `compute_advantages_and_returns`, or
   - subclass the trainer and override `compute_advantages_and_returns` to call your
     trie code directly (closest analogue to `TreeTrainer._compute_loss`).

   miniGRPO sidesteps this because its `Rollout` already carries `completion_ids` — which
   is exactly why it's the right place to prototype the trie logic before scaling.

**Suggested path:** prototype + unit-test your advantage in `minigrpo.py` (fast, no infra),
then port the validated function into a `@register_advantage_estimator` with the token-ids
threaded in. Same math, two backends.
