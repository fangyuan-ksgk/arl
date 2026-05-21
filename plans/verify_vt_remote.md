# Plan: verify + benchmark `compute_vt_vllm_remote`

**Goal.** Confirm that `compute_vt_vllm_remote` (in `src/velocity.py`) is a
valid drop-in replacement for `compute_vt_batched` along two axes:

1. **Numerical equivalence** — same `R_T` and per-chunk `vt` (within
   floating-point tolerance) as the local HF path on identical inputs.
2. **Throughput** — measurably faster wall-clock on a realistic batch.

Produce a single deliverable: `notebook/verify_vt_remote.ipynb` (or a
self-contained `script/verify_vt_remote.py` with results pickled to
`output/verify_vt_remote/`). The script must be **idempotent** and
**self-checking** (prints PASS/FAIL based on tolerances below).

---

## Preconditions

- TRL >= 1.4 installed (verified: `/get_sequence_logprobs/` endpoint exists).
- A single GPU with >= 4 GB free.
- Model: `Qwen/Qwen3-0.6B` (small, fits twice if needed).
- `transformers`, `torch`, `requests`, `numpy` already in env.

---

## Step 1 — Spin up the vLLM server

In a dedicated tmux/screen session (or as a background process):

```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen3-0.6B \
  --gpu-memory-utilization 0.4 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  > /tmp/vllm_serve.log 2>&1 &
echo $! > /tmp/vllm_serve.pid
```

Wait for readiness (poll until 200):

```bash
for i in $(seq 1 60); do
  curl -fs http://localhost:8000/health/ && break
  sleep 2
done
```

If `/health/` never returns 200 within 120 s, abort and dump
`/tmp/vllm_serve.log`.

**Cleanup at the end of the run** (always — use `trap` or a `finally:`):

```bash
kill -TERM "$(cat /tmp/vllm_serve.pid)" 2>/dev/null
pkill -f "trl vllm-serve" 2>/dev/null
```

GPU memory budget: vLLM holds ~40% (≈ 4 GB on a 10 GB card). The HF model
loaded for the local baseline takes another ~1.5 GB in bf16 — fits.

---

## Step 2 — Build a fixed test batch

Use **real rollouts** from the workspace, not synthetic prompts (the
chat-template + tokenization gotchas only show up on real data).

Source: `output/game24_grpo_baseline/rollouts.jsonl`. Load the first 8
records (or fewer if the file is short). Each record should contain
fields like `prompt` (chat-templated), `completion` (raw CoT), and the
ground-truth answer string used as `reference`. If the field names differ,
inspect one record with `head -1 ... | python -m json.tool` first.

Pin the random subset so the run is reproducible:

```python
N_ROLLOUTS = 8
records = [json.loads(l) for l in open("output/game24_grpo_baseline/rollouts.jsonl")][:N_ROLLOUTS]
prompts     = [r["prompt"]     for r in records]
completions = [r["completion"] for r in records]
references  = [r["reference"]  for r in records]   # adapt field name as needed
```

Print a summary: number of rollouts, min/median/max CoT length in tokens
(via `tok(...).input_ids`).

---

## Step 3 — Numerical-equivalence test

Run both paths on the same `(prompts, completions, references)`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.velocity import compute_vt_batched, compute_vt_vllm_remote

MODEL = "Qwen/Qwen3-0.6B"
tok   = AutoTokenizer.from_pretrained(MODEL)
hf    = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16").cuda().eval()

CHUNK = 4
local  = compute_vt_batched     (prompts, completions, references, hf,  tok, chunk_size=CHUNK)
remote = compute_vt_vllm_remote (prompts, completions, references,      tok, chunk_size=CHUNK,
                                 server_url="http://localhost:8000")
```

**Assertions** (per rollout):

- `len(local) == len(remote) == N_ROLLOUTS`
- Same `len(vt)` and `len(logps)` per rollout — failure means mismatched
  t-grid construction (bug, not numerics).
- `|R_T_local - R_T_remote|` median ≤ **0.05**, max ≤ **0.5**.
- `|vt_local - vt_remote|` per-element max ≤ **0.5** on at least 95% of
  positions; report any rollout that violates.

Tolerances reflect bf16 + flash-attn vs HF eager differences; tighter is
suspiciously good and probably means the test isn't actually hitting both
paths. **Do not lower the tolerances to make the test pass.** If failures
exceed the bounds, dump the offending rollout's full `logps` arrays
side-by-side and inspect — usually the cause is:

1. Chat-template mismatch (the prompt string fed to vLLM was retokenized
   differently than what HF saw). Fix: ensure both paths receive the
   exact same `prompts` strings, already chat-templated upstream.
2. `add_special_tokens` mismatch in tokenization. Both paths in
   `velocity.py` use `add_special_tokens=False` — verify nothing upstream
   re-templates.
3. BOS/EOS handling difference inside vLLM. Try logging
   `len(seq)` from the manifest vs `len(prompt_token_ids)` returned by
   the server (they should be equal).

**Print a clear PASS/FAIL summary** with median and max diffs. Save
`output/verify_vt_remote/equivalence.json` with per-rollout diffs.

---

## Step 4 — Throughput benchmark

Time only the `compute_vt_*` calls (not model loading). Use a fresh batch
each repeat to avoid HF KV-cache reuse from prior calls. Three repeats,
report the **median**.

```python
import time
def time_fn(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts)//2]

t_local  = time_fn(lambda: compute_vt_batched     (prompts, completions, references, hf, tok, chunk_size=CHUNK))
t_remote = time_fn(lambda: compute_vt_vllm_remote (prompts, completions, references,     tok, chunk_size=CHUNK,
                                                   server_url="http://localhost:8000"))
print(f"local  : {t_local :6.2f}s")
print(f"remote : {t_remote:6.2f}s")
print(f"speedup: {t_local / t_remote:.2f}x")
```

Also benchmark **at two scales** to characterize how it scales:

- `N_ROLLOUTS = 8`,  `chunk_size = 4`  (small, default exploratory case)
- `N_ROLLOUTS = 32`, `chunk_size = 4`  (closer to a GRPO step)

Report a 2x2 table: `{rollouts × method}` → `seconds`. Save it as
`output/verify_vt_remote/throughput.json`.

**Expected outcome** (rough — don't fail on this, just flag if violated):

- Remote should be **at least 3× faster** at `N=32` once prefix caching
  kicks in. If remote is slower than local, something is wrong —
  most likely prefix caching is disabled on the server, or
  `max_sequences_per_request` is being hit and serializing the request.
  Inspect with the diagnostics in Step 5.

---

## Step 5 — Diagnostics (run on FAIL)

Only execute if Step 3 fails or Step 4 shows remote slower than local.

1. **Confirm prefix caching is on**: `curl -X POST http://localhost:8000/reset_prefix_cache/`
   should return successfully. The endpoint's existence implies the
   feature is live, but the launch flag may have disabled it. Check
   `/tmp/vllm_serve.log` for "prefix caching" lines.

2. **Sanity-check one sequence end-to-end**: pick one rollout, manually
   construct `q + o[:T] + a` token IDs, POST to `/get_sequence_logprobs/`
   with `top_logprobs=1`, `response_format="json"` (easier to read than
   binary), and print the returned `logprobs` for the last `|a|`
   positions. Sum them and compare against `local[i]['logps'][-1]` (the
   `t = T` endpoint).

3. **Bisect** with `chunk_size = T` (only endpoints, 2 forwards per
   rollout). If equivalence holds at coarse chunking but fails at
   `chunk_size = 1`, the bug is in how intermediate t-points construct
   their prefix.

---

## Deliverables

- `notebook/verify_vt_remote.ipynb` — runnable end-to-end, prints
  PASS/FAIL, includes the throughput table inline.
- `output/verify_vt_remote/equivalence.json` — per-rollout diffs.
- `output/verify_vt_remote/throughput.json` — timing table.
- A 5-line summary at the bottom of the notebook: pass/fail, median
  R_T diff, speedup at N=32. This is what the human will read first.

## Non-goals

- Do **not** modify `src/velocity.py` to make the test pass. If a real
  bug is found, file it as a comment in the notebook and stop.
- Do **not** test against a model other than Qwen3-0.6B in this run.
- Do **not** train anything. This is read-only on the model.
