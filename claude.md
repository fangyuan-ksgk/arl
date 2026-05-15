# Environment Setup

## Verified Working Stack

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.8.0+cu128 | Pre-installed in RunPod CUDA 12.8 base image — **do not upgrade** |
| vllm | 0.11.0 | Compatible with torch 2.8.0; upgrading to 0.17.x breaks cu128 binding |
| trl | 1.1.0 | Requires transformers==4.56.2 |
| transformers | 4.56.2 | Satisfies both trl 1.1.0 and vllm 0.11.0 lower bounds |
| CUDA | 12.8 | — |

## Setup

```bash
pip install flash-attn --no-build-isolation
python -c "import torch; import flash_attn_2_cuda; print('✅ Pytorch and flash-attn are compatible')"

git clone https://github.com/fangyuan-ksgk/arl.git && cd arl
pip install -r requirements.txt
```

## Why This Exact Stack

### vLLM version lock
`vllm==0.11.0` is the highest version that does **not** force a torch upgrade.
`vllm>=0.17.x` pulls `torch==2.10.0+cu126`, replacing the base cu128 build and
breaking any flash_attn binary compiled against cu128.

### transformers version lock
`transformers==4.56.2` is the intersection of:
- `trl==1.1.0` → requires `transformers==4.56.2`
- `vllm==0.11.0` → requires `transformers>=4.55.x`
- vLLM also requires `transformers<5`

### flash_attn binary compatibility
flash_attn ships precompiled wheels targeting specific (torch, CUDA) pairs.
No prebuilt wheel exists for torch 2.8.0+cu128, so it **must** be compiled from source
with `--no-binary flash-attn`. The `--no-build-isolation` flag ensures the build uses
the current environment's torch headers, not a fresh torch pulled into an isolated env.
`--no-deps` prevents pip from re-pulling torch during the flash_attn install.

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `undefined symbol: c10_cuda_check_implementation` | flash_attn wheel compiled for wrong torch/CUDA | Rebuild: `pip install flash-attn --no-build-isolation --no-binary flash-attn --no-deps` |
| torch version changes after vllm install | vllm 0.17.x upgrades torch | Use `vllm==0.11.0` |
| torch loses cu128 after vllm install | vllm pulled cu126 torch wheel | Pin vllm to 0.11.0 (cu128-safe) or reinstall torch with `--index-url .../cu128 --no-deps` |
| `ImportError: flash_attn_2_cuda` after any pip install | pip silently upgraded torch | Rebuild flash_attn from source after settling all other packages |
| vLLM server won't die from `pkill -f vllm`; orphan `VLLM::EngineCore` survives | EngineCore cmdline is bare `python`; comm `VLLM::EngineCore` is 16 chars, exceeding pkill's 15-char comm match | **Golden line:** `ps -ef \| grep 'VLLM::EngineCore' \| grep -v grep \| awk '{print $2}' \| xargs kill -9` |
| vLLM `EngineCore` pegs CPU at 99 %, never allocates GPU | torch.compile / CUDA-graph capture stalled on cold cache | Launch with `--enforce-eager` (skips compile + graph capture; ~15 % decode throughput cost) |
| `trl vllm-serve` hangs at "Waiting for application startup" after repeated kill/restart cycles | Leaked CUDA contexts from orphan EngineCores deadlock vLLM's NCCL init even though `nvidia-smi` shows GPU free | **Restart the container.** `vllm serve` may still work while `trl vllm-serve` doesn't — the trl path stresses NCCL setup more. |

## TRL GRPOTrainer Modes

| Mode | Flag | Use case |
|------|------|----------|
| Colocate | `vllm_mode="colocate"` | Single GPU — vLLM shares VRAM with training model |
| Server | `vllm_mode="server"` | Multi-GPU — vLLM runs on dedicated GPU(s), training on others |

For server mode, set `CUDA_VISIBLE_DEVICES` to control which physical GPU maps to which role
(e.g., `CUDA_VISIBLE_DEVICES=0,1` → vLLM on cuda:0, training on cuda:1).

## MBE Reward

The MBE reward requires a live reference to the training model (for hidden-state forward passes).
Use the deferred binding pattern — **do not** pass the model at construction time:

```python
mbe_reward = MBEReward(tokenizer, scale=40.0, clip=2.0)
trainer = GRPOTrainer(model=model, reward_funcs=[correctness_reward, mbe_reward], ...)
mbe_reward.set_model(trainer.model)   # bind after trainer init
trainer.train()
```

veRL's reward API (`compute_score`) receives only text — it **cannot** support model-dependent
rewards like MBE. Stay on TRL for any reward that needs hidden states.

## vLLM `--max-model-len` vs `max_completion_length` (silent truncation)

vLLM has two independent caps:

- `--max-model-len` (server, hard ceiling on `prompt_len + gen_len`)
- `SamplingParams.max_tokens` (per-request, set by TRL from `max_completion_length`)

Effective generation cap on every request:

```
effective_max_new_tokens = min(max_tokens, max_model_len - prompt_len)
```

If you launch vLLM with `--max-model-len LEN` and the trainer with
`max_completion_length=LEN`, the server silently caps generation at
`LEN - prompt_len`. For Game-of-24 the chat-templated prompt is ~130
tokens, so `LEN=512` actually yields ~382 effective new tokens.

### Why this materially changes GRPO dynamics

The Game-of-24 reward is gated on the `####` answer marker:

- truncated rollout (no `####`)        → 0
- formatted but wrong (`####` present) → 0.2
- formatted and correct                → 1.2

With a tight effective cap, almost every rollout in early training
truncates to 0. Within-group advantages collapse, and the only
non-degenerate signal is "be brief enough to emit `####`". GRPO
optimizes that proxy first, pulling CoT length **down** before
correctness improves. Observable in `eval_df` as falling `pct_truncated`
*and* falling mean `n_cot_tokens` over training.

With the cap loose enough that most rollouts fit, the within-group
reward distribution spans all three levels from step 0. Format and
correctness gradients fire simultaneously, GRPO discovers
"longer CoT → more often correct", and length **grows** — eventually
saturating whatever cap is left.

### Fix

Do **not** pass `--max-model-len` to `trl vllm-serve` unless you
explicitly want it tighter than the model's native context. Let vLLM
fall back to the model's native window and let `max_completion_length`
(per-request) be the only effective cap. Applied in
`script/run_game24_sweep.sh::start_vllm_server` (2026-05-15).

Runs in `logs/game24_sweep1/` were collected under the buggy regime
(`--max-model-len LEN`) and are **not** comparable to clean baselines.
