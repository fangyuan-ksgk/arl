# HANDOFF — SkyRL Terminal-Bench + Vision GRPO (resume after restart)

_Last updated: 2026-06-10 ~02:40 UTC. Single A100-80GB, unprivileged container (NO Docker)._

## TL;DR status

| Goal | Status | Result |
|---|---|---|
| **1. Terminal-Bench GRPO (Qwen2.5-Coder-3B)** | ✅ **achieved (non-zero)** | baseline **0.356** mean test-pass, **6/32 tasks fully solved**, pass@1 0.56. Trained 8 GRPO steps (flat at lr 1e-6 → see insight). |
| **2. Vision GRPO (Geometry-3K)** | ✅ **achieved**; Qwen3.5-4B image-format bug **FIXED + validated** | **Qwen3-VL-4B baseline = 0.4742 pass@1**. Qwen3.5-4B rollout/eval now renders **crash-free** after the `_normalize_mm_content` fix (validated 2026-06-10; pass@1 logging in `/tmp/geo3k_qwen35_validation.log`). |
| **4. ToyBox toy agentic env** | ✅ **built + offline-verified** | Fun 12-puzzle multi-turn python/bash GRPO env (`id="toybox"`); all tasks pass the no-GPU smoke test. See `TOYBOX.md`. GPU training run still TODO. |
| **3. Notebook deliverable** | 🔄 80% built | `arl/SkyRL_Terminal_and_Vision_GRPO.ipynb` + figures; needs final curves + rebuild. |

## The core trick (why this works without Docker)
This container has **no CAP_SYS_ADMIN** and **blocks user namespaces** → Docker / rootless / bubblewrap all fail.
**`proot`** (userspace ptrace chroot, zero privilege) binds a per-rollout temp dir to guest `/app`:
`proot -b $SANDBOX:/app -w /app bash -c "<agent script>"`. The agent's writes and the task's pytest
verifier's absolute `/app/...` checks resolve to the same isolated dir. Tests stay on host (not copied in).

## File map
- **Custom SkyRL-gym env** (Goal 1): `SkyRL/skyrl-gym/skyrl_gym/envs/terminal/{sandbox.py,env.py,__init__.py}`,
  registered as `id="terminal"` in `SkyRL/skyrl-gym/skyrl_gym/envs/__init__.py`.
- **Scripts** (`arl/skyrl_terminal/`): `curate_tasks.py`, `build_dataset.py`, `run_terminal_grpo.sh`,
  `run_geo3k_1gpu.sh`, `parse_results.py`, `make_figures.py`, `build_notebook.py`, `SETUP.md`.
- **Data**: `~/data/terminal_bench/{train,validation}.parquet` (32 tasks);
  `~/data/geometry_3k/{train,test,test_small}.parquet`.
- **Curated task list**: `arl/skyrl_terminal/local_tasks.json` (32 kept of 241).
- **Results so far**: `~/exports/terminal_v2/dumped_evals/` (terminal, steps 0 & 5);
  `~/exports/geo3k_qwen3vl_lora_1gpu/dumped_evals/` (geo baseline 0.4742).
- **Figures**: `arl/skyrl_terminal/figures/` (pipeline_architecture, terminal_curve, terminal_heatmap).
- **Logs**: `/tmp/terminal_grpo.log`, `/tmp/geo3k_grpo.log`.

## Verified stack (in SkyRL/.venv via `uv sync --extra fsdp`)
- torch 2.11.0+cu128, **vLLM 0.20.2**, **transformers 5.8.0** (NOT 4.56 — newer than arl/claude.md says),
  ray, flash-attn. vLLM registers `Qwen3_5ForConditionalGeneration` and `Qwen3VLForConditionalGeneration`.
- Verifier venv: `/home/claudeuser/tbench-venv/bin/python` (pytest, pyyaml). `proot` at `/usr/bin/proot`.
- Always run SkyRL via `uv run --isolated --extra fsdp` (see scripts).

## ⚠️ Disk hazard (FIXED — keep it fixed)
SkyRL force-saves a full FSDP checkpoint **at every epoch end** when `ckpt_interval>0` (ignores the value),
no cleanup → filled 200 GB disk with 38 GB checkpoints (one per 2 steps). **Both run scripts now set
`ckpt_interval=0 hf_save_interval=0 max_ckpts_to_keep=1`.** Do NOT re-enable checkpointing without
`max_ckpts_to_keep`. Watch `df -h /`.

## Learning deliverables + miniGRPO (2026-06-10 night)
- **`minigrpo/minigrpo.py`** — single-file plain-PyTorch GRPO (no Ray/vLLM). The advantage is a swappable
  function (`ADVANTAGE_MODES`): `flat` (vanilla) + `opa` (imports `optimistic_prefix_advantages` from
  `arl/src/tree_trainer.py`). Drives BOTH `toybox` (task_id) and `terminal` (task_path) via HF generate +
  the sandbox. `python minigrpo.py selftest` = no-GPU check of z-score/flat/OPA/sandbox scoring (passes).
- **`minigrpo/README.md`** — maps miniGRPO ↔ tree_trainer, and the SkyRL scale-up path: register
  `@register_advantage_estimator("opa")` (ppo_utils.py; example in
  `examples/train/algorithms/custom_advantage_estimator/`), run `trainer.algorithm.advantage_estimator=opa`.
  ⚠️ wrinkle: estimator gets only (rewards, response_mask, index) — thread response token ids from
  `data["responses"]` in `Trainer.compute_advantages_and_returns` (trainer.py:960) for the OPA trie.
- **`tutorial.html`** — self-contained figures-first tutorial (6 trick PNGs base64-embedded + ToyBox +
  miniGRPO + results). Rebuild: `tbench-venv/bin/python make_tutorial.py`. Figures from
  `make_learning_figures.py` → `figures/learn/`. Companion: `LEARN_THE_TRICKS.md`.
- **`model_sweep.py`** — multi-model pass@1/reward sweep on toybox+terminal via miniGRPO eval (cheap).
  Writes `model_results.json` → re-run `make_tutorial.py` to inject the tables. **BLOCKED on GPU** until the
  user's terminal run concludes (background waiter `b4f38cxyw` fires on conclusion).

## How to resume each goal
```bash
cd /home/claudeuser/arl/skyrl_terminal
# Goal 1 (works; ~150s/step). For the IMPROVEMENT experiment use higher lr + full batch:
NUM_GPUS=1 bash run_terminal_grpo.sh trainer.policy.optimizer_config.lr=5.0e-6 \
    trainer.train_batch_size=32 trainer.epochs=20
# Goal 2 (works on Qwen3-VL-4B):
bash run_geo3k_1gpu.sh   # MODEL_PATH defaults to Qwen/Qwen3.5-4B — see bug below; override to Qwen3-VL-4B to run clean:
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct bash run_geo3k_1gpu.sh \
    data.val_data="['$HOME/data/geometry_3k/test_small.parquet']" trainer.eval_batch_size=96
# Finalize deliverable:
/home/claudeuser/tbench-venv/bin/python make_figures.py
/home/claudeuser/tbench-venv/bin/python build_notebook.py   # -> arl/SkyRL_Terminal_and_Vision_GRPO.ipynb
```

## ✅ FIXED — Qwen3.5-4B vision rollout image-format bug (2026-06-10)
- **Root cause (found):** NOT model-specific code — it's a **parquet round-trip artifact**. The geo3k
  content list mixes an `image_url` part and a `text` part (different dict shapes), so HF
  `datasets.to_parquet` can't build one Arrow struct and **JSON-encodes each content item to a string**.
  Verified directly: `train.parquet` row `prompt[0]['content']` is `[str, str]`, each
  `'{"type":"image_url",...}'`. The local prompt-length filter uses HF `apply_chat_template` (lenient, so
  dataset build/filter passed), but the rollout renders via vLLM's `/v1/chat/completions/render` endpoint
  whose **strict Pydantic validation rejects string content parts** → the crash. Qwen3-VL happened to
  tolerate it on its path; Qwen3.5's processor does not.
- **Fix (applied):** `skyrl/train/generators/skyrl_vlm_generator.py` — added `_normalize_mm_content()` that
  decodes JSON-string content parts back into dicts, called once in `agent_loop` right after `env.init`
  (before any render). Idempotent; leaves plain-string `content` (text turns/observations) untouched.
  Unit-validated against the real parquet + added 2 regression tests in
  `tests/train/generators/test_skyrl_vlm_generator.py` (`test_normalize_mm_content_*`).
- **⏳ Still needs GPU validation:** end-to-end Qwen3.5-4B rollout (run `run_geo3k_1gpu.sh` with default
  `MODEL_PATH=Qwen/Qwen3.5-4B`) once the terminal GRPO run frees the A100. Fallback still stands:
  **Qwen3-VL-4B gives 0.4742** for Goal 2.

## ✅ ROOT-CAUSED + FIXED — the host-RAM OOM was runaway agent bash, not infra (2026-06-11)
Every OOM today (terminal@step30, geo3k@step9, LoRA@step21, the 74GB "runaway" this morning) had ONE
cause: **agent-generated bash scripts that loop/fork/background were never reaped.** The sandbox ran them
via `subprocess.run(timeout=)`, which kills only the direct child on timeout — descendants orphan to init
and accumulate across rollouts. Found **2,965 live bash holding ~50GB**; killing them dropped the cgroup
68→10GB (active_anon 51→2.2GB). NOT FSDP optimizer states / vLLM / page cache (those were red herrings).
**Diagnosis method:** per-process RSS vs step showed all tracked procs FLAT while cgroup climbed → the
growth was in untracked (bash) procs. **Fix:** `terminal/sandbox.py` + `toybox/sandbox.py` now run each
script with `start_new_session=True` and `os.killpg()` the whole group on timeout AND normal exit
(`_run_reaped`/`_run`). **Validated end-to-end:** 0.5B terminal run, bash bounded at peak 4 (was 2965),
cgroup FLAT 36-38GB through step 18 (prior runs died 21-30). Unit reap-tests pass for both envs; smoke
tests still green. Tradeoff: a script that backgrounds a process now waits out its timeout (child holds
the stdout pipe) then gets reaped — correct & safe.
Also added a (now-secondary) driver-side `gc.collect()` per step in `trainer.py`; harmless, keep or drop.

### Terminal-specific note
The lr=5e-6 **full-batch (32 prompts × 8 samples = 256 traj/step)** run reproduced baseline
(eval avg_score 0.3401, pass@1 0.5625) then **crashed at global_step 4 with a Ray host-RAM OOM** — node
mem hit 110.8/116.4 GB (the container's Ray-visible cgroup limit, NOT the 1007 GB host), with
`FSDPPolicyWorkerBase.forward_backward` alone at 58 GB. Full fine-tuning of the 3B under FSDP + a 256-traj
batch overflowed CPU RAM. **Relaunch memory-safe**, e.g. `TRAIN_BS=16 N_SAMPLES=6` (96 traj/step) or
`RAY_memory_usage_threshold=0.97`, and watch `free -g` / ray's node-mem. GPU was never the bottleneck here.

## Insights to share
1. **~~lr=1e-6 too conservative~~ DISPROVEN (2026-06-11 lr sweep).** A full sweep — lr 1e-6 / 5e-6 / 1e-5,
   n_samples 6/12, model 1.5B/3B — left pass@1 **flat at ~0.55 in ALL configs** (5 clean runs, zero OOM).
   Higher lr does NOT lift pass@1. The plateau is **structural**: the 3B coder reliably solves ~6/32 tasks
   single-turn and can't touch the rest; GRPO sharpens reliability but can't manufacture capability, so
   pass@1 saturates at the solvable fraction. **Real levers = more ability to ACT, not more lr**: multi-turn
   ReAct (now testing via `MAX_TURNS` knob + `run_experiments2.sh`), task curriculum, or a bigger model.
   The 5-run sweep also conclusively validated the bash-reaping leak fix (6h, 0 OOM).
2. **8B VLM OOM-kills the colocated vLLM engine-core on one 80GB GPU** at init (silent SIGKILL,
   `Failed core proc(s): {}`). 4B + `max_model_len=16384` + `enforce_eager=true` fits.
3. **Curation filter**: only 32/241 terminal-bench tasks run in the base proot sandbox (no Docker image
   deps); the rest need their Dockerfile's installed packages. Kept set validated by running each task's own
   `solution.sh` → all-pass.

## Session-only loop
A 1h cron (`/loop`, job `cbacffba`) was firing the "keep GPU busy + push goals" prompt; it dies on restart —
recreate with `/loop 1h <prompt>` if wanted.

## Task list (rebuild after restart)
1. ✅ SkyRL env setup. 2. ✅ Goal 1 terminal non-zero. 3. ✅ Goal 2 vision non-zero (Qwen3-VL-4B);
   ⏳ Qwen3.5-4B rollout fix. 4. ⏳ Finalize notebook with both curves.

## ✅ Scaled-down SearchR1 on 1 GPU (2026-06-12)
Full SearchR1 won't fit (64GB e5 flat index > 52GB disk; retriever+train > 116GB cgroup; no conda).
Built a faithful MINI version that runs on 1 A100:
- `arl/skyrl_terminal/build_searchr1_mini.py` — SQuAD → SearchR1 schema (2000 train/300 val), corpus from
  the questions' own context paragraphs (2300 passages → answers guaranteed findable). reward_spec
  ground_truth MUST be `{"target":[answer]}` (qa_em compute_score reads `["target"]`).
- `mini_retrieval_server.py` — faiss-free e5-base-v2 retriever (GPU embed, caches `corpus.jsonl.emb.npy`,
  brute-force cosine). ⚠️ response MUST be `{"result": [[{"document":{...},"score":..}, ...]]}` — a list of
  PER-QUERY lists; the search tool iterates `result` as queries and passes each inner list to
  `_passages2string`. A flat list → "string indices must be integers" error in every <information>.
- `run_search_1gpu.sh` — Qwen2.5-3B + LoRA, 1 engine TP=1, multi-turn (max_turns=4), GRPO, console logger.
- **Result: works end-to-end, 0 OOM, completed 31 steps. Baseline pass@1 ~0.26 → ~0.31 after GRPO**
  (0 retrieval errors, 276/300 got real docs, 94/300 solved). Real learning — unlike the saturated terminal env.
