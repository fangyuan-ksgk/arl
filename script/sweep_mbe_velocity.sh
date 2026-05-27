#!/bin/bash
# Sweep MBE *velocity* reward (+ InvLogLength baseline) for Qwen3-1.7B on GSM8K.
# Layout: GPU 0 = vLLM server, GPU 1 = training (--vllm_mode server).
#
# Variants (12 runs total):
#   1. baseline                       — pure GRPO (correctness + format), no MBE velocity
#   2-7. positive MBE velocity        — modes ∈ {trajectory, rollercoaster}
#                                       weights ∈ {10, 100, 10000}
#                                       (raw reward ~0.01-0.03, vs correctness ~1.0
#                                        and format ~0.1, so w<10 is too small to
#                                        meaningfully affect group-relative advantage.)
#   8-9. negative MBE velocity        — modes ∈ {trajectory, rollercoaster}
#                                       weight = 10000 (penalises high MBE velocity)
#   10-12. InvLogLength baseline      — pure 1/log(min(T_comp, D)), MBE velocity disabled.
#                                       Tests whether the diversity numerator in MBE
#                                       velocity adds signal beyond length normalisation;
#                                       same weights {10, 100, 10000} as the positive arm.
#
# Weight magnitude w is implemented via --mbe_velocity_scale = ±1/w with clip=1.0,
# so the reward is bounded in [-w, +w]. Negative scale flips sign of the reward.
#
# Eval is on every run; the MBE velocity reward function is *always* attached
# (even on baseline, just for logging) so its mean/std appears in TRL logs as
#   rewards/mbe_velocity_<mode>/{mean,std}.
# To make it a no-op signal on baseline we set its scale extremely large.
#
# Usage:
#   bash script/sweep_mbe_velocity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
BASE_OUTPUT="${PROJECT_DIR}/output/sweep_mbe_velocity"
TIMESTAMP=$(date +%Y%m%d_%H%M)

MODEL="Qwen/Qwen3-1.7B"
MODEL_TAG="1.7b"

# GPU layout & vLLM server
VLLM_GPU=0
TRAIN_GPU=1
VLLM_HOST="0.0.0.0"
VLLM_PORT=8950
VLLM_STARTUP_TIMEOUT=300        # seconds; cold start of 1.7B + KV alloc
VLLM_LOG_DIR="${PROJECT_DIR}/output/sweep_mbe_velocity/vllm_logs"

# Training config
MAX_TOKENS=1024
LR=5e-6
NUM_GEN=8
GRAD_ACCUM=8
MAX_STEPS=200
EVAL_SAMPLES=200            # subset of test set, keeps eval cheap
EVAL_EVERY=50               # log eval (incl. MBE velocity) every N steps

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
#   - --enforce-eager: skip torch.compile + CUDA-graph capture. Trades ~15%
#                 decode throughput for instant, reliable startup across cold
#                 caches — well worth it for a multi-run sweep.
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
            --enforce-eager \
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
# =============================================
run_experiment() {
    local name=$1
    local mode=$2
    local scale=$3
    local run_dir="${BASE_OUTPUT}/${name}"
    local train_log="${run_dir}/train.log"
    mkdir -p "${run_dir}"

    echo ""
    echo ">>> [${name}] mode=${mode}, scale=${scale}"
    echo ">>>   Output: ${run_dir}"

    # Build reward-shaping args. Three regimes:
    #   mode="invlog"        → InvLogLength on,  MBE velocity off.
    #   scale="off"          → both off (pure GRPO baseline).
    #   otherwise            → MBE velocity on,  InvLogLength off (default sweep).
    local velo_args=""
    if [ "${mode}" = "invlog" ]; then
        velo_args="--no-mbe_velocity_reward \
            --inv_log_length_reward \
            --inv_log_length_scale ${scale} \
            --inv_log_length_clip ${VELO_CLIP} \
            --mbe_velocity_stride ${VELO_STRIDE}"   # reused as the guard threshold
    elif [ "${scale}" = "off" ]; then
        velo_args="--no-mbe_velocity_reward"
    else
        velo_args="--mbe_velocity_reward \
            --mbe_velocity_mode ${mode} \
            --mbe_velocity_scale ${scale} \
            --mbe_velocity_clip ${VELO_CLIP} \
            --mbe_velocity_stride ${VELO_STRIDE} \
            --mbe_velocity_layers ${VELO_LAYERS}"
    fi

    local start_time=$(date +%s)
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python "${SCRIPT_DIR}/grpo_gsm8k.py" \
        --model ${MODEL} \
        --output_dir "${run_dir}" \
        --max_steps ${MAX_STEPS} \
        --use_vllm --vllm_mode server \
        --vllm_server_host "${VLLM_HOST}" --vllm_server_port "${VLLM_PORT}" \
        --train_device 0 \
        --num_generations ${NUM_GEN} \
        --max_completion_length ${MAX_TOKENS} \
        --per_device_train_batch_size ${NUM_GEN} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --logging_steps 10 \
        --save_strategy no \
        --report_to none \
        --eval_steps ${EVAL_EVERY} \
        --eval_samples ${EVAL_SAMPLES} \
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
Config: tok=${MAX_TOKENS}, lr=${LR}, gen=${NUM_GEN}, grad_accum=${GRAD_ACCUM}, steps=${MAX_STEPS}
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
    local vllm_log="${VLLM_LOG_DIR}/${name}.log"

    if ! start_vllm_server "${vllm_log}"; then
        echo ">>> [${name}] ✗ vLLM failed to start; skipping"
        FAILED_RUNS="${FAILED_RUNS} ${name}:vllm"
        return
    fi

    if ! run_experiment "${name}" "${mode}" "${scale}"; then
        echo ">>> [${name}] ✗ training failed; continuing sweep"
        FAILED_RUNS="${FAILED_RUNS} ${name}:train"
    fi

    stop_vllm_server
}

# =============================================
# Experiments
# =============================================
# NOTE: existing MBE velocity cells (baseline / traj_w* / roll_w* / *_neg_*)
# are commented out for this run — only the new InvLogLength baseline ladder
# is active. Re-enable them when running the full 12-cell sweep.

# # 1) Baseline GRPO (no MBE velocity reward).
# run_cell "baseline_grpo"             trajectory     off
#
# # 2-7) Positive MBE velocity — 2 modes × 3 weight magnitudes (10, 100, 10000)
# #      scale = 1/w, so weight 10=0.1, 100=0.01, 10000=0.0001.
# run_cell "traj_w10"                  trajectory     0.1
# run_cell "traj_w100"                 trajectory     0.01
# run_cell "traj_w10000"               trajectory     0.0001
# run_cell "roll_w10"                  rollercoaster  0.1
# run_cell "roll_w100"                 rollercoaster  0.01
# run_cell "roll_w10000"               rollercoaster  0.0001
#
# # 8-9) "Negative" MBE velocity — penalise high MBE velocity, weight = 10000.
# #      scale = -1/w → reward sign flipped.
# run_cell "traj_neg_w10000"           trajectory     -0.0001
# run_cell "roll_neg_w10000"           rollercoaster  -0.0001

# 10-12) InvLogLength baseline — 1/log(min(T_comp, D)) with MBE velocity disabled.
#        Same weight ladder as the positive MBE velocity arm so length-vs-acc curves
#        can be overlaid directly. If these reproduce traj_w*'s length reduction,
#        the diversity numerator is doing no work — see analysis note 2026-05-27.
run_cell "invlog_w10"                invlog         0.1
run_cell "invlog_w100"               invlog         0.01
run_cell "invlog_w10000"             invlog         0.0001

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
