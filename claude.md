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

## MILESTONE (2026-07-03): the lottery gap IS a forgetting gap

**Finding.** On GSM8K/Qwen3-0.6B, the cross-seed *lottery gap* (8 GRPO seeds:
union-of-solved − mean-per-seed = **21.5 pt** on validation) is quantitatively
matched by the within-run *forgetting gap* of a single Dr.GRPO run
(ever-solved − currently-solved by step 300 = **23.9 pt** validation / 22.1 pt
train). Same magnitude, same signature (a run's ever-solved envelope ≈ the
8-seed union ≈ 92%). Interpretation: each seed samples a *churning* solvable
set at its own stopping point; the "lottery" across seeds is the same
per-query forgetting churn seen within one run. ~24% of validation queries are
solved at some checkpoint then wrong at the end; the truly-never-solved set is
only ~7% (the real capacity ceiling). This reframes the whole
routing-fails/soup-wins story: the exploitable diversity is temporal, not
semantic.

**The figure** (`output/forgetting/dr_grpo_dense/forgetting_grouped_g100.{png,gif}`):
two panels (train, validation), queries grouped into class-pure bars of 100 by
their trajectory-vs-step-0 class {KEPT, IMPROVED, REGRESSED, TRANSIENT, NEVER}
(NEVER = never greedy-correct at any checkpoint → ~0). Bar solid = current
running-avg greedy acc (RdYlGn: green 1.0 → red 0.0); grey shade on top =
forgetting gap (per-query running-best − current, avg in group). Validation
panel annotates `forgetting gap X pt / lottery gap Y pt`.

### Reproduce
```bash
# 1. dense Dr.GRPO run (seed 1), checkpoint every 15 steps
#    -> output/dr_grpo_dense/checkpoint-{15,30,...,300}   (output/pipeline_2b*.sh)
# 2. greedy-eval every ckpt over 2000 train + full 1319 val, per-query correctness
CUDA_VISIBLE_DEVICES=0 python script/eval_forgetting.py --run output/dr_grpo_dense --train_subset 2000
#    -> output/forgetting/dr_grpo_dense/step{S}_{train,test}.jsonl
# 3. the 8-seed lottery-gap numbers (union/mean) come from:
#    output/lottery_gap/seed{0..7}_step200_test.jsonl  (script/eval_lottery_gap.py)
# 4. build the diagram + gif (fast static PNG + optional GIF)
python script/forgetting_viz.py --group_size 100                 # PNG only (instant)
python script/forgetting_viz.py --group_size 100 --gif --fps 2 --hold_s 4
```

**`script/forgetting_viz.py` knobs** (built to iterate): `--group_size` (queries
per bar, class-pure/truncated), `--gif`, `--fps` (lower = slower), `--hold_s`
(seconds to hold the end frame). Class definitions live in `classify()`; the
lottery-gap reference is recomputed live by `lottery_gap()` from the seed
jsonls. `step2trainidx.json` (seed-1 training order, validated 300/300 steps vs
logged rollout golds) supports the train-chunk analysis in `make_gif_v3.py`.

---

# Cross-domain Dr.GRPO training — setup experiences, bugs to avoid, speed

Hard-won lessons from standing up GRPO on GSM8K / MATH / code / SearchR1 /
Terminal-Bench. Read this before starting a new domain — most of these cost hours.

## Environment (incl. the CUDA-12.8 fallback)
- **`bash env.sh` is idempotent + driver-aware.** It picks cu130 wheels only if
  the driver is ≥580; otherwise **cu128 torch + vLLM's cu129 wheel** (CUDA
  minor-version compat — the pairing SkyRL uses). Don't hand-install torch.
- **Pin the torch trio with a constraints file** during every `pip install`.
  The single worst env bug we hit: a later package (vllm/deepspeed) silently
  moved torch to a mismatched CUDA build → import-time CUDA errors that look
  like driver problems. `-c constraints.txt` freezing torch/vision/audio kills it.
- **Versions that matter:** vllm **0.23.0** (0.24 breaks TRL weight-sync), trl
  **1.7.0**, transformers **5.12**. The old 2.8/cu128 + vllm 0.11 + trl 1.1
  stack runs but current scripts assume trl-1.7 APIs.
- **flash-attn is NOT needed** — transformers 5.x defaults to SDPA and vllm 0.23
  ships flashinfer kernels. Skipping it removes a fragile, slow-to-build dep.

## Bugs that silently waste GPU (verify, don't assume)
- **vLLM sleep mode is BROKEN with trl 1.7 + vllm 0.23.** `vllm_enable_sleep_mode`
  makes weight-sync a silent no-op → gradients collapse to ~1e-4 in ~10 steps,
  training looks alive but learns nothing. Keep it **OFF everywhere**. Gate new
  drivers with `script/test_vllm_sync.py`. (SkyRL's *own* colocate sleep/wake is
  separate and does work — that's the FSDP backend, not TRL.)
- **Never gate object-sharing on a container's truthiness.**
  `self.buf = kwargs.get("buf") or ReplayBuffer()` swaps in a fresh buffer when
  the passed one is empty (`__len__`==0 → falsy) → producer and consumer hold
  different objects, the term never fires. Use `x if x is not None else ...`.
- **Instrument the term and print a value the first few steps** (e.g. the online
  loss magnitude). Two separate no-ops this project were only caught by a stray
  print. When logging a quantity that *ramps* (KL vs a periodic ref), sample the
  whole window, not one endpoint — post-refresh KL is 0 by construction.
- **`save_only_model` is incompatible with SHARDED_STATE_DICT**; FSDP1 × paged
  8-bit Adam mismatches optimizer-state device. Use FULL_STATE_DICT to checkpoint.
- **LoRA lr ≠ full-FT lr.** 5e-6 (full-FT) silently undertrains LoRA; use ~1e-4.
- **SFT on *gold* terse CoT degrades the model** below base (overwrites native
  reasoning). If you want self-improvement, SFT/distill on the model's OWN
  correct rollouts (best-of-N), not gold.

## Speed (Qwen3-0.6B, 2×H100 — 20.9→6.5 s/step, beats SkyRL's 7.27)
- **Drop `--enforce-eager`** — CUDA graphs are ~3.2× on 0.6B decode. Biggest single win.
- **Colocate 2-GPU DDP** (vllm_mode=colocate) > server mode for small models:
  no cross-process weight-sync latency, both GPUs do policy fwd/bwd.
- **bs 8 × accum 4** sweet spot; raising micro-bs hits the **fp32-logits memory
  wall** (bs × seq × 151k vocab × 4B) — the real cap on long-seq domains, not params.
- `vllm_gpu_memory_utilization` 0.25–0.5 for colocate (leave room for training).
- Eval: greedy over vLLM, one engine rebuild per ckpt is the cost — batch ckpts,
  don't `--enforce-eager` the eval either.
- **Orchestration:** distinct `MASTER_PORT` per GPU (`29510+gpu`), distinct
  output dirs/logs, and reap orphan `VLLM::EngineCore` + `multiprocessing.resource_tracker`
  after any kill (they reparent to init and hold 70GB). Never point a crash-monitor
  at the shared append-only `pipeline_status.txt` — stale `EXIT=1` lines false-fire.

## Per-domain entry points & critical spots
All default to Qwen3-0.6B + colocate vLLM; `--max_steps -1` = one full epoch;
server mode = `trl vllm-serve` on GPU0 + `--vllm_mode server` on GPU1.

- **MATH** (`script/grpo_math.py`, Hendrycks): `--max_completion_length 3072`
  (long proofs), `per_device_bs 2 × accum 4`. Long seq → fp32-logits wall is the
  binding constraint; keep micro-bs tiny. Smoke: `python script/grpo_math.py` (20 steps).
- **Code** (`script/grpo_code.py`): `--dataset mbpp` (374/500, default) or `apps`
  (harder). **Execution-based reward runs untrusted model code** — it uses full
  process-group reaping (`_run` helper); a runaway generation will spawn/hang
  procs without it. `--enable_thinking` defaults **False** for code. Smoke: 20 steps.
- **SearchR1** (`searchr1_trl/run_searchr1_trl.sh`, default Qwen2.5-3B-Instruct):
  multi-turn `rollout_func` — model emits `<search>q</search>` → **mini retrieval
  server** (`skyrl_terminal/mini_retrieval_server.py`, e5 over `corpus.jsonl`,
  cpu or cuda) returns docs as `<information>` injected with **`env_mask=0` so
  retrieved tokens are excluded from the GRPO loss** (critical — don't train on
  the retriever's text). Reward = `qa_em` exact match. The script builds the
  dataset once (`build_searchr1_trl.py`) and health-checks the retriever via a
  `/retrieve` curl before launching. Sleep mode OFF here too.
- **Terminal-Bench** (`skyrl_terminal/`, see `SETUP.md`): **different stack —
  SkyRL FSDP+vLLM backend, not TRL** (`uv sync --extra fsdp`; torch 2.11+cu128,
  vllm 0.20.2, its own `.venv`). Biggest gotcha: **unprivileged container = no
  Docker/bubblewrap**, so the sandbox is **`proot`** (userspace chroot, zero
  privilege) and the verifier runs pytest in a *separate* venv
  (`TBENCH_VERIFIER_PYTHON`). Register the custom `terminal` SkyRL-gym env,
  `curate_tasks.py` (keeps ~32 of 241) → `build_dataset.py` → `run_terminal_grpo.sh`.
  Sandbox timeouts via `TBENCH_EXEC_TIMEOUT`/`TBENCH_VERIFY_TIMEOUT` keep steps short.
