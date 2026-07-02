#!/bin/bash
# Sweep reward shaping designs for Qwen3 on GSM8K.
#
# Layout (LAYOUT=colocate, default since 2026-07-02):
#   Both GPUs run training (accelerate DDP) AND generation (one vLLM engine
#   per rank, data-parallel, sleep mode). Benchmarked 6.3 s/optimizer-step vs
#   20.9 s for the old split layout — 3.3x. Breakdown & comparison vs skyRL:
#   output/bench_profile/. Same GRPO semantics (8 prompts x 8 gens / step).
#
# Layout (LAYOUT=server, legacy fallback):
#   GPU 0 = vLLM server, GPU 1 = training (--vllm_mode server). ~9.6 s/step
#   after the 2026-07-02 fixes (CUDA graphs on, bs16 micro-batches).
#
# Variants supported by run_experiment's mode switch:
#   "trajectory"  / "rollercoaster"  — MBE velocity (hidden-state geometry).
#                                       raw_v = MBE(q+r) − MBE(q)   [trajectory]
#                                       or sum of positive MBE jumps  [rollercoaster].
#   "invlog"                          — InvLogLength baseline (1/log(T_comp), no model fwd).
#   "entvelo_roll" / "entvelo_traj"   — Rationale-internal entropy velocity.
#                                       Per-token Δ_t = H(o_{t+1}) − H(o_t),
#                                       aggregated rollercoaster (Σ max(0,Δ)) or trajectory
#                                       (= H(o_last) − H(o_first)).
#   "pplxvelo_roll" / "pplxvelo_traj" — Same shape, X = NLL of realised next token.
#   "entdensity"                      — Phase contrast: mean(H over rationale) − mean(H over answer).
#                                       "Reason hard, commit confidently."
#   "predvelo"                        — Predictive velocity: log p(a|q,o) − log p(a|q),
#                                       per-token & length-normalised. Two forward passes.
#
# Weight magnitude w is implemented via --*_scale = ±1/w with clip=1.0,
# so the reward is bounded in [-w, +w]. Negative scale flips sign of the reward.
#
# Eval is on every run (--eval_steps), and the rollout logger writes train/eval
# rollouts to <output_dir>/{rollouts,eval_rollout}.jsonl regardless of which
# reward-shaping variant is active. TRL logs reward fns under their __name__:
#   rewards/mbe_velocity_<mode>/{mean,std}, rewards/entropy_velocity/{...}, etc.
#
# Usage:
#   bash script/sweep_mbe_velocity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
# Colocate mode interleaves vLLM's big allocations with training's fp32-logits
# peaks in one process; expandable segments prevents fragmentation OOMs after
# large eval generation batches (observed 2026-07-02).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BASE_OUTPUT="${BASE_OUTPUT:-${PROJECT_DIR}/output/sweep_mbe_velocity}"
TIMESTAMP=$(date +%Y%m%d_%H%M)

MODEL="Qwen/Qwen3-0.6B"
MODEL_TAG="0.6b"

# Layout: "colocate" (default; both GPUs train+generate, 3.3x faster) or
# "server" (legacy; GPU0 = vLLM server, GPU1 = training).
LAYOUT="${LAYOUT:-colocate}"
N_PROC=2                        # colocate: accelerate DDP world size

# GPU layout & vLLM server (LAYOUT=server only)
VLLM_GPU=0
TRAIN_GPU=1
VLLM_HOST="0.0.0.0"
VLLM_PORT=8950
VLLM_STARTUP_TIMEOUT=300        # seconds; cold start + CUDA-graph capture
VLLM_LOG_DIR="${PROJECT_DIR}/output/sweep_mbe_velocity/vllm_logs"

# Training config
MAX_TOKENS=1024
LR=5e-6
NUM_GEN=8
# Micro-batching (2026-07-02 profile): keep the optimizer batch at 64
# completions (8 unique prompts x NUM_GEN). Micro-bs is bounded by fp32
# logits-scale buffers (bs x ~1.1k tok x 151k vocab x 4B, and TRL/torch keep
# several alive across the loss+backward): bs16 peaks ~60 GB per rank and
# OOMs colocate whenever eval residue or resident KV eats the margin; bs8
# peaks ~30 GB and is safe. Server mode (vLLM on the other GPU) can afford
# bs16. colocate: 8 x 4 accum x 2 ranks = 64; server: 16 x 4 = 64.
if [ "${LAYOUT}" = "colocate" ]; then MICRO_BS=8; GRAD_ACCUM=4; else MICRO_BS=16; GRAD_ACCUM=4; fi
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1319}"  # full GSM8K test set (× NUM_GEN rollouts)
EVAL_EVERY="${EVAL_EVERY:-100}"       # log eval every N steps
# per_device_eval_batch_size. The trainer's eval prediction_step is generation-
# only (no forward pass), so this just controls how many prompts we hand the
# vLLM server per call. Small batches (the old NUM_GEN*16=128 default = 16 unique
# prompts) starve vLLM's continuous batching and make the full-test eval take
# ~30 min. A big batch lets vLLM saturate → full eval in ~5 min. vLLM admission-
# controls the KV cache, so an oversized batch queues rather than OOMs. Must be a
# multiple of NUM_GEN. Here: 512 unique prompts/call → whole test set in ~3 calls.
EVAL_BATCH=$(( NUM_GEN * 512 ))

# RepAnchor anti-forgetting penalty weight (src/repanchor.py). 0 = disabled.
# Override per-run via `LAMBDA_REPANCHOR=10 bash script/run_gsm8k_multiseed_fix.sh`,
# or set per-cell below.
LAMBDA_REPANCHOR="${LAMBDA_REPANCHOR:-0.0}"

# Seeds for the multi-seed run. Each seed reshuffles the TRAINING data ordering
# (passed as --seed); the eval set is sampled sequentially so validation data is
# identical across seeds. SEED is set per-iteration by the driver loop.
SEEDS=(0 1 2 3 4 5 6 7)
SEED=0

# Checkpoint grid: early-training dynamics on a log-ish schedule (1,2,4,8) plus
# mid-training and end. Forced via grpo_gsm8k.py's SaveAtStepsCallback regardless
# of --save_strategy. Saved to <run_dir>/checkpoint-<step>.
SAVE_STEPS_LIST="1,4,${MAX_STEPS}"

# MBE velocity defaults
VELO_STRIDE=8
VELO_LAYERS="-1"
VELO_CLIP=1.0

mkdir -p "${BASE_OUTPUT}" "${VLLM_LOG_DIR}"

# =============================================
# vLLM server lifecycle
#
# Pattern adapted from script/run_game24_sweep.sh:
#   - setsid    : isolates server + EngineCore in a fresh process group so
#                 we can SIGKILL the whole group on shutdown.
#   - NO --enforce-eager (2026-07-02): for a 0.6B model decode is kernel-launch
#                 bound, so CUDA graphs are a 3.2x generation speedup (15.4s ->
#                 4.8s per 64-rollout batch), not the ~15% claude.md estimated
#                 on bigger models. Graph capture adds ~40s to startup, paid
#                 once per run. If a cold-cache host wedges during capture
#                 (EngineCore pegs CPU, never allocates), re-add --enforce-eager.
#   - stop is a three-stage nuke because EngineCore is stubborn:
#       (1) pkill -f vllm                  — catches the launcher
#       (2) ps | grep VLLM::EngineCore     — catches workers (16-char comm,
#                                            exceeds pkill's 15-char limit)
#       (3) nvidia-smi compute-apps        — catches anything still pinning VRAM
# =============================================
VLLM_PID=""

start_vllm_server() {
    local log_file=$1
    stop_vllm_server
    echo ">>> [vllm] starting  model=${MODEL}  gpu=${VLLM_GPU}  port=${VLLM_PORT}"
    CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
        setsid trl vllm-serve \
            --model "${MODEL}" \
            --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
            > "${log_file}" 2>&1 &
    VLLM_PID=$!
    echo ">>> [vllm] pid=${VLLM_PID} (pgid=${VLLM_PID})  log=${log_file}"

    local waited=0
    while (( waited < VLLM_STARTUP_TIMEOUT )); do
        if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo ">>> [vllm] ready after ${waited}s"
            return 0
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo ">>> [vllm] !! server died during startup; see ${log_file}"
            return 1
        fi
        sleep 3
        waited=$(( waited + 3 ))
    done
    echo ">>> [vllm] !! timeout after ${VLLM_STARTUP_TIMEOUT}s; see ${log_file}"
    return 1
}

stop_vllm_server() {
    echo ">>> [vllm] stopping server"
    # Stage 1: SIGKILL the process group (setsid'd at start).
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -9 -- "-${VLLM_PID}" 2>/dev/null || true
    fi
    # Stage 2: pkill anything matching 'vllm' in cmdline.
    pkill -9 -f vllm 2>/dev/null || true
    # Stage 3: EngineCore workers (their cmdline is bare 'python', so
    # match by comm via ps).
    ps -ef | grep 'VLLM::EngineCore' | grep -v grep \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    # Stage 4: anything still holding GPU memory.
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | grep -E '^[0-9]+$' \
            | xargs -r kill -9 2>/dev/null || true
    fi
    # Free the port if anything still bound.
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    fi
    sleep 2
    VLLM_PID=""
}
trap stop_vllm_server EXIT INT TERM

# =============================================
# Run one experiment (GPU 1)
#
# Args:
#   $1  name          — short tag, used for output dir
#   $2  mode          — "trajectory" | "rollercoaster" | "invlog"
#                       "invlog" → InvLogLength baseline (MBE velocity disabled).
#   $3  scale         — float; sign matters. Pass "off" to disable velocity reward
#                       (logs zeros). With clip=1.0, |reward| ≤ 1/|scale| = weight.
#                       Negative scale → reward penalises high MBE velocity / length.
#   $4  clip          — optional; raw-value clip (default $VELO_CLIP). |reward| ≤ clip/|scale|.
#                       Raise to escape saturation (e.g. predvelo railed at ±1.0).
#   $5  norm_mode     — optional; predvelo length denominator: log_total (default)
#                       or cot_len (=> log[p/p]/(l_a·l_o), linear CoT-length pressure).
# =============================================
run_experiment() {
    local name=$1
    local mode=$2
    local scale=$3
    local clip=${4:-$VELO_CLIP}
    local norm_mode=${5:-log_total}
    local run_dir="${BASE_OUTPUT}/${name}"
    local train_log="${run_dir}/train.log"
    mkdir -p "${run_dir}"

    echo ""
    echo ">>> [${name}] mode=${mode}, scale=${scale}, clip=${clip}, norm_mode=${norm_mode}"
    echo ">>>   Output: ${run_dir}"

    # Build reward-shaping args. Regimes (mutually exclusive — exactly one
    # reward shaper is active per cell, others disabled):
    #   mode="invlog"                       → InvLogLength baseline (no fwd).
    #   mode="entvelo_roll"|"entvelo_traj"  → EntropyVelo with that aggregation.
    #   mode="pplxvelo_roll"|"pplxvelo_traj"→ PerplexityVelo with that aggregation.
    #   mode="entdensity"                   → EntropyDensity (phase contrast).
    #   mode="predvelo"                     → PredictiveVelo (two forwards).
    #   scale="off"                         → both off (pure GRPO baseline).
    #   otherwise                           → MBE velocity (mode ∈ traj/roll).
    # All non-MBE shapers reuse --mbe_velocity_stride as the guard threshold.
    local velo_args=""
    if [ "${mode}" = "invlog" ]; then
        velo_args="--no-mbe_velocity_reward \
            --inv_log_length_reward \
            --inv_log_length_scale ${scale} \
            --inv_log_length_clip ${clip} \
            --mbe_velocity_stride ${VELO_STRIDE}"
    elif [ "${mode}" = "entvelo_roll" ] || [ "${mode}" = "entvelo_traj" ]; then
        local agg="rollercoaster"
        [ "${mode}" = "entvelo_traj" ] && agg="trajectory"
        velo_args="--no-mbe_velocity_reward \
            --entropy_velocity_reward \
            --entropy_velocity_scale ${scale} \
            --entropy_velocity_clip ${clip} \
            --entropy_velocity_aggregation ${agg} \
            --mbe_velocity_stride ${VELO_STRIDE}"
    elif [ "${mode}" = "pplxvelo_roll" ] || [ "${mode}" = "pplxvelo_traj" ]; then
        local agg="rollercoaster"
        [ "${mode}" = "pplxvelo_traj" ] && agg="trajectory"
        velo_args="--no-mbe_velocity_reward \
            --perplexity_velocity_reward \
            --perplexity_velocity_scale ${scale} \
            --perplexity_velocity_clip ${clip} \
            --perplexity_velocity_aggregation ${agg} \
            --mbe_velocity_stride ${VELO_STRIDE}"
    elif [ "${mode}" = "entdensity" ]; then
        velo_args="--no-mbe_velocity_reward \
            --entropy_density_reward \
            --entropy_density_scale ${scale} \
            --entropy_density_clip ${clip} \
            --mbe_velocity_stride ${VELO_STRIDE}"
    elif [ "${mode}" = "predvelo" ]; then
        velo_args="--no-mbe_velocity_reward \
            --predictive_velocity_reward \
            --predictive_velocity_scale ${scale} \
            --predictive_velocity_clip ${clip} \
            --predictive_norm_mode ${norm_mode} \
            --mbe_velocity_stride ${VELO_STRIDE}"
    elif [ "${mode}" = "longshort" ]; then
        # Correctness-gated InvLogLength (asymmetric shaping):
        #   scale_correct   = +${scale}   → e.g. 0.1 ⇒ w_correct = +10
        #   scale_incorrect = -${scale}   → e.g. 0.1 ⇒ w_incorrect = -10
        # Reward is positive on correct rollouts (push shorter) and negative
        # on incorrect ones (push longer). Symmetric magnitude.
        local sc_inc=$(python -c "print(-(${scale}))")
        velo_args="--no-mbe_velocity_reward \
            --gated_inv_log_length_reward \
            --gated_inv_log_length_scale_correct ${scale} \
            --gated_inv_log_length_scale_incorrect ${sc_inc} \
            --gated_inv_log_length_clip ${clip} \
            --mbe_velocity_stride ${VELO_STRIDE}"
    elif [ "${scale}" = "off" ]; then
        velo_args="--no-mbe_velocity_reward"
    else
        velo_args="--mbe_velocity_reward \
            --mbe_velocity_mode ${mode} \
            --mbe_velocity_scale ${scale} \
            --mbe_velocity_clip ${clip} \
            --mbe_velocity_stride ${VELO_STRIDE} \
            --mbe_velocity_layers ${VELO_LAYERS}"
    fi

    # Launcher + vLLM wiring per layout. Colocate runs one vLLM engine inside
    # each of the two DDP ranks; server talks to the external `trl vllm-serve`
    # started by run_cell.
    local launch vllm_args
    if [ "${LAYOUT}" = "colocate" ]; then
        launch="accelerate launch --num_processes ${N_PROC} --mixed_precision bf16 ${SCRIPT_DIR}/grpo_gsm8k.py"
        vllm_args="--use_vllm --vllm_mode colocate \
            --vllm_gpu_memory_utilization 0.25"
    else
        launch="env CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python ${SCRIPT_DIR}/grpo_gsm8k.py"
        vllm_args="--use_vllm --vllm_mode server \
            --vllm_server_host ${VLLM_HOST} --vllm_server_port ${VLLM_PORT} \
            --train_device 0"
    fi

    local start_time=$(date +%s)
    ${launch} \
        --model ${MODEL} \
        --output_dir "${run_dir}" \
        --max_steps ${MAX_STEPS} \
        ${vllm_args} \
        --num_generations ${NUM_GEN} \
        --max_completion_length ${MAX_TOKENS} \
        --per_device_train_batch_size ${MICRO_BS} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --logging_steps 10 \
        --save_strategy no \
        --save_steps_list "${SAVE_STEPS_LIST}" \
        --seed ${SEED} \
        --report_to none \
        --eval_steps ${EVAL_EVERY} \
        --eval_samples ${EVAL_SAMPLES} \
        --eval_batch_size ${EVAL_BATCH} \
        --lambda_repanchor ${LAMBDA_REPANCHOR} \
        ${velo_args} \
        2>&1 | tee "${train_log}"
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    # Extract a few headline metrics.
    local final_reward=$(grep "'reward'" "${train_log}" | tail -1 | grep -oP "'reward': [0-9.]+" | grep -oP "[0-9.]+$" || echo "N/A")
    local peak_reward=$(grep "'reward'" "${train_log}" | grep -oP "'reward': [0-9.]+" | grep -oP "[0-9.]+$" | sort -n | tail -1 || echo "N/A")
    local final_correct=$(grep "rewards/correctness_reward/mean" "${train_log}" | tail -1 | grep -oP "[0-9.]+" | tail -1 || echo "N/A")

    {
      echo "${name}: mode=${mode}, scale=${scale}"
      echo "  final_reward=${final_reward}, peak_reward=${peak_reward}, "\
"final_correctness=${final_correct}, time=${elapsed}s"
      echo ""
    } | tee -a "${SUMMARY_FILE}"
}

# =============================================
# Summary header
# =============================================
SUMMARY_FILE="${BASE_OUTPUT}/summary_${TIMESTAMP}.txt"
cat > "${SUMMARY_FILE}" <<EOF
MBE Velocity Reward Sweep — ${MODEL}
Started: $(date)
Config: layout=${LAYOUT}, tok=${MAX_TOKENS}, lr=${LR}, gen=${NUM_GEN}, micro_bs=${MICRO_BS}, grad_accum=${GRAD_ACCUM}, steps=${MAX_STEPS}
        eval_every=${EVAL_EVERY}, eval_samples=${EVAL_SAMPLES}
        velocity: stride=${VELO_STRIDE}, layers=${VELO_LAYERS}, clip=${VELO_CLIP}
Weight w → scale = ±1/w; with clip=1.0, |reward| ≤ w (sign of scale flips reward).
==========================================

EOF

# =============================================
# Driver: stop+start vLLM around every run so a wedged server from a prior
# experiment can't silently corrupt the next one. The cost is ~30-60s of
# vLLM startup per run; over 9 runs that's <10 min, negligible vs MAX_STEPS=200
# training time. Far cheaper than debugging a half-dead server mid-sweep.
# =============================================
FAILED_RUNS=""

run_cell() {
    local name=$1
    local mode=$2
    local scale=$3
    local clip=${4:-$VELO_CLIP}
    local norm_mode=${5:-log_total}
    local vllm_log="${VLLM_LOG_DIR}/${name}.log"

    if [ "${LAYOUT}" = "server" ]; then
        if ! start_vllm_server "${vllm_log}"; then
            echo ">>> [${name}] ✗ vLLM failed to start; skipping"
            FAILED_RUNS="${FAILED_RUNS} ${name}:vllm"
            return
        fi
    else
        # Colocate: engines live inside the training processes; just make sure
        # nothing stale is pinning VRAM from a previous (possibly crashed) run.
        stop_vllm_server
    fi

    if ! run_experiment "${name}" "${mode}" "${scale}" "${clip}" "${norm_mode}"; then
        echo ">>> [${name}] ✗ training failed; continuing sweep"
        FAILED_RUNS="${FAILED_RUNS} ${name}:train"
    fi

    if [ "${LAYOUT}" = "server" ]; then stop_vllm_server; fi
}

# =============================================
# Experiments — multi-seed (2026-06-14)
#
# Two cells only: baseline GRPO and InvLogLength-short (w=10), each across
# SEEDS=(0 1 2 3). The seed is threaded to grpo_gsm8k.py via --seed: it
# reshuffles training data ordering only — the eval set is sampled sequentially
# so validation data is identical across all runs. Intermediate checkpoints are
# saved at SAVE_STEPS_LIST=(1,2,4,8,mid,end) into <run_dir>/checkpoint-<step>.
#
# Output dirs are per-seed: <name>_seed<S> so runs never collide.
# =============================================
# --- Requested layout (2026-07-01) ------------------------------------------
#   * Baseline GRPO         : 1 seed  (BASELINE_SEED)
#   * GRPO + RepAnchor      : 4 seeds (REPANCHOR_SEEDS), lambda_repanchor=10
# Output dirs are per-seed (<name>_seed<S>) so runs never collide.
# ---------------------------------------------------------------------------
BASELINE_SEED=0
REPANCHOR_SEEDS=(0 1 2 3)

# 1) Baseline GRPO (no MBE velocity reward, no RepAnchor) — single seed.
echo ""
echo "############################################################"
echo "# BASELINE GRPO — seed ${BASELINE_SEED}"
echo "############################################################"
SEED=${BASELINE_SEED}
run_cell "baseline_grpo_seed${SEED}"     trajectory     off

# 2) GRPO + RepAnchor ablation, lambda_repanchor=10 (representation-space
#    anti-forgetting penalty; full fine-tuning only). LAMBDA_REPANCHOR is read
#    inside run_experiment; set it just for this cell.
for SEED in "${REPANCHOR_SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# GRPO + REPANCHOR (lambda_repanchor=10) — seed ${SEED}"
    echo "############################################################"
    LAMBDA_REPANCHOR=10 run_cell "repanchor_w10_seed${SEED}"  trajectory  off
done

# MBE velocity run

# =============================================
# Final summary
# =============================================
echo ""
echo "############################################################"
echo "# SWEEP COMPLETE"
echo "############################################################"
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo ">>> Results: ${BASE_OUTPUT}"
echo ">>> Summary: ${SUMMARY_FILE}"
if [ -n "${FAILED_RUNS}" ]; then
    echo ">>> FAILED runs:${FAILED_RUNS}"
fi
