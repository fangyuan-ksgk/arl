# q8_minimal — lottery-gap experiments on the 4 domains (Qwen3-4B) at low VRAM

## The VRAM stack (the actual content of this bundle)

Full-FT GRPO on 4B does not fit consumer/40GB cards naively (Adam fp32 alone
is 32GB). Five levers together get it there; all are wired into
`grpo_domain_lowvram.py`:

| lever | saving | note |
|---|---|---|
| liger kernel | ~20-25GB activations | `apply_liger_kernel_to_qwen3()`; TRL's GRPO loss still materializes logits — the fused-CE path needs TRL's `use_liger_loss`, otherwise savings come from the fused module kernels |
| gradient checkpointing | ~10GB | |
| `optim=paged_adamw_8bit` | 32GB → ~8GB | bitsandbytes |
| bs1 × ga32, completion cap 1024 | activations scale | effective batch 32 kept |
| colocate vLLM `util=0.15` + `enable_thinking=False` | ~12GB + starvation fix | thinking mode exhausts the completion budget → clipped → reward 0 → zero-variance GRPO death; this cost us a week |

**Measured**: ~54GB peak at bs2/ga16 on 80GB. Defaults (bs1/ga32) are sized
for one 40–48GB card. Disk: `save_only_model=True` always (optimizer states
are ~25GB/ckpt at 4B).

## Self-contained domains (train + gap-eval in this folder)

```
python grpo_domain_lowvram.py --domain math --out out_math_4b          # ~1 GPU-day
python eval_gap_domain.py     --domain math --run out_math_4b          # MATH-500
python grpo_domain_lowvram.py --domain mbpp --out out_mbpp_4b --seed 2
python eval_gap_domain.py     --domain mbpp --run out_mbpp_4b
```

Eval is cheap by design: `--vllm_util 0.3 --enforce_eager` runs a 4B gap-eval
in ~14GB. Records keep FULL completions, so the q9_minimal harvest/install
stages apply unchanged if you want to exploit the gap, not just measure it.

Reference numbers (our runs): MATH final 70.0 / union 82.4 (**gap 12.4**,
n=500); MBPP s2 final 46 / union 62 (**gap 16**, n=100).

## SEARCH-R1 and TERMINAL-BENCH (bundled: `searchr1/`, `tbench/`)

Their training loops need external infrastructure, so the RELEVANT FILES are
included here rather than pretending a single file suffices.

**SEARCH-R1** (`searchr1/`): `trl_searchr1.py` (driver, low-VRAM flags),
`searchr1_rollout.py` (multi-turn <search> protocol with loss-masked injected
docs), `retrieval_client.py`, `qa_em.py` (EM reward), `build_searchr1_trl.py`
(dataset parquets), `mini_retrieval_server.py` (the retriever backend),
`eval_forgetting_searchr1.py` (per-ckpt gap eval), `.

```
python searchr1/build_searchr1_trl.py                       # datasets
python searchr1/mini_retrieval_server.py --corpus corpus.jsonl --port 8000 &
python searchr1/trl_searchr1.py --optim paged_adamw_8bit --gradient_checkpointing     --per_device_train_batch_size 1 --gradient_accumulation_steps 32
python searchr1/eval_forgetting_searchr1.py --run <run>     # union/final/gap
```

**TERMINAL-BENCH** (`tbench/`): `trl_tbench.py` (driver, `--liger` wired),
`sandbox.py` (reaped-subprocess terminal sandbox), `tbench_reward.py`,
`build_tbench_trl.py`, `eval_forgetting_tbench.py`, `.

```
python tbench/build_tbench_trl.py
python tbench/trl_tbench.py --liger --optim paged_adamw_8bit --gradient_checkpointing     --per_device_train_batch_size 1 --gradient_accumulation_steps 32
python tbench/eval_forgetting_tbench.py --run <run>
```

## Caveats (earned the hard way)

- Caps are part of the instrument: match `max_completion_length`, eval
  `max_tokens`, and any harvest cap. A 512/1024 mismatch mis-measured
  accuracy by ~23 pts on 0.6B.
- Stay on ONE greedy instrument (vLLM here): batched left-padded HF
  `generate` can read tens of points below vLLM on the same RL-trained
  weights (fragile states).
- n≥2 seeds before believing any per-domain gap number; MBPP cross-seed
  finals ranged 35–46 in our runs.
