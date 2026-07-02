# Environment Setup

## Verified Working Stack (2026-07-02)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.11.0 (+cu130 on CUDA-13 drivers, else +cu128) | — |
| vllm | 0.23.0 | 0.24 breaks TRL server-mode weight sync — do not upgrade |
| trl | 1.7.0 | GRPOTrainer with steps_per_generation, sleep mode, profiling hooks |
| transformers | 5.12.1 | — |
| flash-attn | **not installed** | transformers 5 defaults to SDPA; vllm 0.23 ships flashinfer kernels. The old source-compile dance is obsolete. |

`bash env.sh` is idempotent (checks the stack first) and driver-aware:
CUDA-13 drivers (>= 580) get the cu130 PyPI wheels; older lottery hosts get
torch cu128 + vLLM's cu129 wheel (the pairing SkyRL uses).

The pre-2026-07 stack (torch 2.8.0+cu128 / vllm 0.11.0 / trl 1.1.0 /
transformers 4.56.2 + source-built flash-attn) still works but no current
script assumes it; see git history of env.sh for its pin rationale.

## GRPO speed (2026-07-02 profiling, Qwen3-0.6B, 2×H100, 64 rollouts/step)

Full breakdowns in `output/bench_profile/*/profile.json`
(`script/bench_grpo_profile.py`); SkyRL control in
`skyrl_terminal/run_gsm8k_bench.sh`.

| Config | s/step |
|--------|--------|
| server mode + `--enforce-eager` (old default) | 20.9 |
| server mode, CUDA graphs | 10.6 |
| server mode, CUDA graphs + bs16 micro-batches | 9.6 |
| colocate 2-GPU DDP + sleep mode, micro-bs 16 (latent OOM after evals) | 6.3 |
| **colocate, micro-bs 8 (new default)** | **~6.5** |
| SkyRL colocated GRPO, same config (control) | ~7.5 |

### Why each knob works (2026-07-02 profiling)

1. **Never `--enforce-eager` a small model.** The flag disables vLLM's CUDA
   graphs (whole decode step replayed as ONE graph launch) and affects
   GENERATION ONLY — the trainer never runs through vLLM. Small-model decode
   is kernel-LAUNCH-bound (hundreds of tiny kernels/token), so graphs are
   3.2× on generation for 0.6B (15.4s → 4.8s per 64×1024-tok batch) — the
   "~15%" folklore is only true for big models whose kernels dominate launch
   overhead. What enforce-eager buys instead: ~40s faster engine startup, no
   cold-cache compile wedge, a few hundred MB of graph-pool VRAM. Irrelevant
   over a multi-hour run.
2. **Fewer, fatter micro-batches.** Grad accumulation is sequential; each
   micro-pass pays fixed overhead and bs8 GEMMs don't saturate an H100.
   Same 64-seq optimizer batch as 4×bs16 instead of 8×bs8 ≈ −1 s/step.
3. **fp32 logits cap the micro-batch (the vocab is the memory wall, not the
   model).** TRL casts logits to fp32: bs × seq × 151k vocab × 4B ≈ 9.3 GB at
   bs16, and SEVERAL such buffers coexist during loss+backward (fp32 cast,
   log_softmax, entropies, grads) ⇒ bs64 = instant 37 GB OOM; bs16 works in
   server mode but is a LATENT OOM in colocate — fine eval-free, dies on the
   first backward after any eval pass (~60 GB transient + eval residue).
   Colocate default = bs8 (accum 4), costs only ~0.2 s/step vs bs16.
4. **Colocate ≠ faster comms, it's zero idle GPUs.** Server layout: the train
   GPU idles during generation, the vLLM GPU idles during fwd/bwd. Colocate
   runs one engine inside each DDP rank → generation is data-parallel
   (4.3→2.6s) AND training is DDP (3.0→1.7s). Weight sync also drops
   0.8→0.28s (in-process CUDA copy vs NCCL+RPC to another process). The tax
   is ~1s of cross-rank gather/scatter + sleep/wake, far below the ~4s saved.
5. **Sleep mode is a VRAM timeshare, not a speedup.** During the optimizer
   phase the engine offloads weights to CPU and unmaps its KV cache, waking
   before the next generation (~0.1–0.3s/cycle). It exists so training's
   fp32-logits peaks can use the VRAM vLLM would otherwise pin. Colocate
   needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and an
   `empty_cache()` after eval (both wired into the scripts).
6. **Scaling colocate up:** generation side scales (sleep mode; SkyRL trains
   32B colocated). The constraint is the TRAINING side — TRL colocate is DDP,
   so full model + Adam states (~8 B/param) must fit per GPU: full-FT is
   comfortable ≤ ~1.7B, feasible to ~4B w/ grad ckpt, dead ≥ ~8B on 80 GB.
   Bigger: keep colocation but shard the trainer (DeepSpeed ZeRO via
   accelerate) or use LoRA (easy to ~30B).

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `undefined symbol: c10_cuda_check_implementation` | flash_attn wheel compiled for wrong torch/CUDA | Rebuild: `pip install flash-attn --no-build-isolation --no-binary flash-attn --no-deps` |
| torch version changes after vllm install | vllm 0.17.x upgrades torch | Use `vllm==0.11.0` |
| torch loses cu128 after vllm install | vllm pulled cu126 torch wheel | Pin vllm to 0.11.0 (cu128-safe) or reinstall torch with `--index-url .../cu128 --no-deps` |
| `ImportError: flash_attn_2_cuda` after any pip install | pip silently upgraded torch | Rebuild flash_attn from source after settling all other packages |
| vLLM server won't die from `pkill -f vllm`; orphan `VLLM::EngineCore` survives | EngineCore cmdline is bare `python`; comm `VLLM::EngineCore` is 16 chars, exceeding pkill's 15-char comm match | **Golden line:** `ps -ef \| grep 'VLLM::EngineCore' \| grep -v grep \| awk '{print $2}' \| xargs kill -9` |
| vLLM `EngineCore` pegs CPU at 99 %, never allocates GPU | torch.compile / CUDA-graph capture stalled on cold cache | Launch with `--enforce-eager` as a LAST resort — on a 0.6B model it costs 3.2× generation throughput (see GRPO speed table). Try once more with a warm cache first. |
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
