#!/bin/bash
# ============================================================================
# run GRPO on: Qwen3 1.7B | Qwen3 4B | on GSM8K
# 4 GPUs total — vLLM server mode (TP=1 on GPU 0, train on GPU 1),
# run sequentially: 1.7B first, then 4B
# Then run rollout analysis on the resulting checkpoints
#
# Layout per run:
#   GPU 0  → vLLM server (generation)
#   GPU 1  → training (forward/backward/optimizer)
#   After training, GPUs 2,3 → rollout analysis (parallel)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Config ---
MODEL_1="Qwen/Qwen3-1.7B"
MODEL_2="Qwen/Qwen3-4B"
OUTPUT_1="${PROJECT_DIR}/output/grpo_qwen3_1.7b"
OUTPUT_2="${PROJECT_DIR}/output/grpo_qwen3_4b"
MAX_STEPS=200
N_ANALYSIS_SAMPLES=50
VLLM_PORT=8000
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run1_$(date +%Y%m%d_%H%M%S).log"

# Tee all output to log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================"
echo "  GRPO Training + Rollout Analysis Pipeline"
echo "  vLLM server mode (TP=1 generation + training)"
echo "============================================"
echo "Models:  ${MODEL_1}, ${MODEL_2}"
echo "GPUs:    4"
echo "Steps:   ${MAX_STEPS}"
echo ""

# ============================================================================
# Helper: start vLLM server, wait for it, run training, kill server
# ============================================================================
run_grpo() {
    local MODEL=$1
    local OUTPUT_DIR=$2
    local VLLM_GPUS=$3       # e.g. "0,1" for TP=2 or "0" for TP=1
    local TRAIN_GPUS=$4       # e.g. "2,3"
    local TP_SIZE=$5
    local BATCH_SIZE=$6
    local GRAD_ACCUM=$7
    local LR=$8
    local PORT=$9

    local N_TRAIN_GPUS
    N_TRAIN_GPUS=$(echo "${TRAIN_GPUS}" | tr ',' '\n' | wc -l | tr -d ' ')

    echo "  Starting vLLM server for ${MODEL} (GPUs ${VLLM_GPUS}, TP=${TP_SIZE}, port ${PORT}) ..."
    set -m  # enable job control so background job gets its own process group
    CUDA_VISIBLE_DEVICES=${VLLM_GPUS} trl vllm-serve \
        --model "${MODEL}" \
        --tensor-parallel-size ${TP_SIZE} \
        --data-parallel-size 1 \
        --port ${PORT} &
    VLLM_PID=$!
    set +m

    # Wait for vLLM to be ready
    echo "  Waiting for vLLM server (PID ${VLLM_PID}) to be ready ..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "  vLLM server ready after ${i}s"
            break
        fi
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "  ERROR: vLLM server died!"
            return 1
        fi
        sleep 1
    done

    if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  ERROR: vLLM server failed to start within 120s"
        kill ${VLLM_PID} 2>/dev/null || true
        return 1
    fi

    echo "  Starting training on GPUs ${TRAIN_GPUS} (${N_TRAIN_GPUS} processes) ..."
    CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
        --num_processes ${N_TRAIN_GPUS} \
        "${SCRIPT_DIR}/grpo_gsm8k.py" \
        --model "${MODEL}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_steps ${MAX_STEPS} \
        --use_vllm \
        --vllm_mode server \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --logging_steps 10 \
        --save_strategy "no" \
        --report_to "none"
    local TRAIN_EXIT=$?

    echo "  Stopping vLLM server (PID ${VLLM_PID}) and its children ..."
    # Kill the entire process group spawned by vLLM (child workers hold GPU memory)
    kill -- -${VLLM_PID} 2>/dev/null || kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    sleep 3

    # Safety net: kill any orphaned GPU processes (e.g. vLLM EngineCore workers)
    local ORPHANS
    ORPHANS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [[ -n "${ORPHANS}" ]]; then
        echo "  Killing orphaned GPU processes: ${ORPHANS}"
        echo "${ORPHANS}" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi

    echo "  GPU memory released."
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

    return ${TRAIN_EXIT}
}

# ============================================================================
# Phase 1: Train Qwen3-1.7B
#   vLLM: GPU 0 (TP=1) | Training: GPUs 1,2,3
# ============================================================================
echo "[Phase 1/4] Training ${MODEL_1} ..."
echo "  vLLM server: GPU 0 (TP=1)"
echo "  Training:    GPUs 1,2,3"

run_grpo "${MODEL_1}" "${OUTPUT_1}" "0" "1,2,3" 1 2 4 5e-6 ${VLLM_PORT}
echo "[Phase 1/4] Done. Checkpoint: ${OUTPUT_1}"

# ============================================================================
# Phase 2: Train Qwen3-4B
#   vLLM: GPUs 0,1 (TP=2) | Training: GPUs 2,3
# ============================================================================
echo ""
echo "[Phase 2/4] Training ${MODEL_2} ..."
echo "  vLLM server: GPUs 0,1 (TP=2)"
echo "  Training:    GPUs 2,3"

run_grpo "${MODEL_2}" "${OUTPUT_2}" "0,1" "2,3" 2 1 8 3e-6 ${VLLM_PORT}
echo "[Phase 2/4] Done. Checkpoint: ${OUTPUT_2}"

# ============================================================================
# Phase 3: Rollout analysis (parallel on GPUs 2,3)
# ============================================================================
echo ""
echo "[Phase 3/4] Analyzing rollouts in parallel ..."

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/analyze_rollouts.py" \
    --checkpoint "${OUTPUT_1}" \
    --n_samples ${N_ANALYSIS_SAMPLES} &
PID_A1=$!

CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/analyze_rollouts.py" \
    --checkpoint "${OUTPUT_2}" \
    --n_samples ${N_ANALYSIS_SAMPLES} &
PID_A2=$!

echo "  PID ${PID_A1}: analyzing ${MODEL_1} on GPU 0"
echo "  PID ${PID_A2}: analyzing ${MODEL_2} on GPU 1"
wait ${PID_A1} ${PID_A2}
echo "[Phase 3/4] Done."

# ============================================================================
# Phase 4: Summary
# ============================================================================
echo ""
echo "============================================"
echo "  All done!"
echo "  Checkpoints: ${OUTPUT_1}"
echo "               ${OUTPUT_2}"
echo "  Analysis:    ${OUTPUT_1}/rollout_analysis.json"
echo "               ${OUTPUT_2}/rollout_analysis.json"
echo "  Log:         ${LOG_FILE}"
echo "============================================"