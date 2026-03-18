#!/bin/bash
# Sweep MBE reward scaling for Qwen3-0.6B on GSM8K
# Layout: GPU 0 = vLLM server, GPU 1 = training
# Tests: plain MBE reward (various scale/clip) + correctness-gated MBE reward
#
# Usage:
#   bash script/sweep_mbe_reward.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BASE_PORT=8900
PORT=${BASE_PORT}
EXPERIMENT_INDEX=0
BASE_OUTPUT="${PROJECT_DIR}/output/sweep_mbe"
TIMESTAMP=$(date +%Y%m%d_%H%M)

MODEL="Qwen/Qwen3-0.6B"
MODEL_TAG="0.6b"

# Training config (matches sweep_all tok512_gen8 for 0.6B)
MAX_TOKENS=512
LR=5e-6
NUM_GEN=8
GRAD_ACCUM=8
MAX_STEPS=200

mkdir -p "${BASE_OUTPUT}"

VLLM_PID=""

# =============================================
# Cleanup
# =============================================
cleanup_vllm() {
    echo ">>> Cleaning up vLLM..."
    [ -n "${VLLM_PID}" ] && kill ${VLLM_PID} 2>/dev/null || true
    [ -n "${VLLM_PID}" ] && wait ${VLLM_PID} 2>/dev/null || true
    ps aux | grep -iE "vllm|EngineCore" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 3
}
trap cleanup_vllm EXIT

# =============================================
# Start vLLM (GPU 0)
# =============================================
start_vllm() {
    local log_file=$1
    cleanup_vllm
    PORT=$(( BASE_PORT + EXPERIMENT_INDEX ))
    EXPERIMENT_INDEX=$(( EXPERIMENT_INDEX + 1 ))
    echo ">>> Starting vLLM for ${MODEL} on port ${PORT}, GPU 0..."
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
        --model ${MODEL} \
        --port ${PORT} \
        > "${log_file}" 2>&1 &
    VLLM_PID=$!
    echo ">>> Waiting for vLLM (PID ${VLLM_PID})..."
    until curl -s http://localhost:${PORT}/health > /dev/null 2>&1; do sleep 2; done
    echo ">>> vLLM ready."
}

# =============================================
# Check vLLM alive
# =============================================
ensure_vllm() {
    local log_file=$1
    if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo ">>> vLLM died! Restarting..."
        start_vllm "${log_file}"
    fi
}

# =============================================
# Run one experiment (GPU 1)
# =============================================
run_experiment() {
    local name=$1
    local mbe_flag=$2       # "--mbe_reward" or "--gated_mbe_reward" or ""
    local mbe_scale=$3
    local mbe_clip=$4
    local run_dir="${BASE_OUTPUT}/${name}"
    local train_log="${run_dir}/train.log"
    mkdir -p "${run_dir}"

    echo ""
    echo ">>> [${name}] mbe_flag=${mbe_flag}, scale=${mbe_scale}, clip=${mbe_clip}"
    echo ">>>   Output: ${run_dir}"

    local mbe_args=""
    if [ -n "${mbe_flag}" ]; then
        mbe_args="${mbe_flag} --mbe_scale ${mbe_scale} --mbe_clip ${mbe_clip}"
    fi

    local start_time=$(date +%s)
    CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/grpo_gsm8k.py" \
        --model ${MODEL} \
        --output_dir "${run_dir}" \
        --max_steps ${MAX_STEPS} \
        --use_vllm --vllm_mode server --vllm_server_port ${PORT} \
        --num_generations ${NUM_GEN} \
        --max_completion_length ${MAX_TOKENS} \
        --per_device_train_batch_size ${NUM_GEN} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --logging_steps 10 \
        --save_strategy no \
        --report_to none \
        ${mbe_args} \
        2>&1 | tee "${train_log}"
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    # Extract metrics
    local final_reward=$(grep "'reward'" "${train_log}" | tail -1 | grep -oP "'reward': [0-9.]+" | grep -oP "[0-9.]+$" || echo "N/A")
    local peak_reward=$(grep "'reward'" "${train_log}" | grep -oP "'reward': [0-9.]+" | grep -oP "[0-9.]+$" | sort -n | tail -1 || echo "N/A")

    echo "${name}: final_reward=${final_reward}, peak_reward=${peak_reward}, time=${elapsed}s" | tee -a "${SUMMARY_FILE}"
    echo "  mbe_flag=${mbe_flag}, scale=${mbe_scale}, clip=${mbe_clip}" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
}

# =============================================
# Summary file
# =============================================
SUMMARY_FILE="${BASE_OUTPUT}/summary_${TIMESTAMP}.txt"
cat > "${SUMMARY_FILE}" <<EOF
MBE Reward Sweep — ${MODEL}
Started: $(date)
Config: tok=${MAX_TOKENS}, lr=${LR}, gen=${NUM_GEN}, grad_accum=${GRAD_ACCUM}, steps=${MAX_STEPS}
==========================================

EOF

# =============================================
# Start vLLM
# =============================================
VLLM_LOG="${BASE_OUTPUT}/vllm.log"
start_vllm "${VLLM_LOG}"

# =============================================
# Experiments
# =============================================
# Format: name, mbe_flag, scale, clip
#
# Baseline (no MBE)
run_experiment "baseline"          ""                   0   0

# Plain MBE reward — sweep scale (clip fixed at 2.0)
ensure_vllm "${VLLM_LOG}"
run_experiment "mbe_s20_c2"        "--mbe_reward"       20  2.0

ensure_vllm "${VLLM_LOG}"
run_experiment "mbe_s40_c2"        "--mbe_reward"       40  2.0

ensure_vllm "${VLLM_LOG}"
run_experiment "mbe_s100_c2"       "--mbe_reward"       100 2.0

# Plain MBE reward — sweep clip (scale fixed at 40)
ensure_vllm "${VLLM_LOG}"
run_experiment "mbe_s40_c1"        "--mbe_reward"       40  1.0

ensure_vllm "${VLLM_LOG}"
run_experiment "mbe_s40_c3"        "--mbe_reward"       40  3.0

# Correctness-gated MBE reward — sweep scale
ensure_vllm "${VLLM_LOG}"
run_experiment "gated_s20_c2"      "--gated_mbe_reward" 20  2.0

ensure_vllm "${VLLM_LOG}"
run_experiment "gated_s40_c2"      "--gated_mbe_reward" 40  2.0

ensure_vllm "${VLLM_LOG}"
run_experiment "gated_s100_c2"     "--gated_mbe_reward" 100 2.0

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
