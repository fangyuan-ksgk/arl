#!/bin/bash
# Train 8 independent LoRA adaptors on GSM8K — one per seed.
#
# Each seed reshuffles the TRAINING data ordering (--seed); the eval set is
# sampled sequentially so validation data is identical across all 8 runs. The
# only thing that differs between adaptors is the optimisation trajectory
# induced by data order + RNG, giving 8 diverse LoRA experts over the same base.
#
# Layout: GPU 0 = vLLM server, GPU 1 = training (--vllm_mode server).
# Runs are sequential (one adaptor at a time); vLLM is restarted around each so
# a wedged server can't corrupt the next run.
#
# Output: <BASE_OUTPUT>/seed<S>/  — a PEFT adapter dir (adapter_config.json +
#         adapter_model.safetensors) plus rollouts.jsonl / eval_rollout.jsonl.
#
# Usage:
#   bash script/run_gsm8k_lora_8seed.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
BASE_OUTPUT="${PROJECT_DIR}/output/gsm8k_lora_8seed"
TIMESTAMP=$(date +%Y%m%d_%H%M)

MODEL="Qwen/Qwen3-0.6B"

# GPU layout & vLLM server
VLLM_GPU=0
TRAIN_GPU=1
VLLM_HOST="0.0.0.0"
VLLM_PORT=8950
VLLM_STARTUP_TIMEOUT=300
VLLM_LOG_DIR="${BASE_OUTPUT}/vllm_logs"

# Training config
MAX_TOKENS=1024
LR=5e-6
NUM_GEN=8
GRAD_ACCUM=8
MAX_STEPS=300
EVAL_SAMPLES=1319           # full GSM8K test set
EVAL_EVERY=100

# LoRA config
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 8 seeds → 8 adaptors
SEEDS=(0 1 2 3 4 5 6 7)

mkdir -p "${BASE_OUTPUT}" "${VLLM_LOG_DIR}"

# =============================================
# vLLM server lifecycle (adapted from run_gsm8k_multiseed.sh)
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
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -9 -- "-${VLLM_PID}" 2>/dev/null || true
    fi
    pkill -9 -f vllm 2>/dev/null || true
    ps -ef | grep 'VLLM::EngineCore' | grep -v grep \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | grep -E '^[0-9]+$' \
            | xargs -r kill -9 2>/dev/null || true
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    fi
    sleep 2
    VLLM_PID=""
}
trap stop_vllm_server EXIT INT TERM

# =============================================
# Train one LoRA adaptor (GPU 1) for a given seed.
# --save_strategy epoch triggers grpo_gsm8k.py's end-of-train save_model,
# which for a PEFT model writes only the adapter weights to <run_dir>.
# =============================================
run_seed() {
    local seed=$1
    local run_dir="${BASE_OUTPUT}/seed${seed}"
    local train_log="${run_dir}/train.log"
    mkdir -p "${run_dir}"

    echo ""
    echo ">>> [seed${seed}] LoRA r=${LORA_R} alpha=${LORA_ALPHA} → ${run_dir}"

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
        --use_lora \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --save_strategy epoch \
        --seed ${seed} \
        --report_to none \
        --no-mbe_velocity_reward \
        --eval_steps ${EVAL_EVERY} \
        --eval_samples ${EVAL_SAMPLES} \
        2>&1 | tee "${train_log}"
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    local final_correct=$(grep "rewards/correctness_reward/mean" "${train_log}" | tail -1 | grep -oP "[0-9.]+" | tail -1 || echo "N/A")
    {
      echo "seed${seed}: final_correctness=${final_correct}, time=${elapsed}s, adapter=${run_dir}"
    } | tee -a "${SUMMARY_FILE}"
}

# =============================================
# Summary header
# =============================================
SUMMARY_FILE="${BASE_OUTPUT}/summary_${TIMESTAMP}.txt"
cat > "${SUMMARY_FILE}" <<EOF
GSM8K 8-seed LoRA training — ${MODEL}
Started: $(date)
Config: tok=${MAX_TOKENS}, lr=${LR}, gen=${NUM_GEN}, grad_accum=${GRAD_ACCUM}, steps=${MAX_STEPS}
        lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA}, lora_dropout=${LORA_DROPOUT}
        eval_every=${EVAL_EVERY}, eval_samples=${EVAL_SAMPLES}
        seeds=${SEEDS[*]}
==========================================

EOF

# =============================================
# Driver: one adaptor per seed, sequential.
# =============================================
FAILED_RUNS=""

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# SEED ${SEED}  (adaptor $((SEED + 1)) / ${#SEEDS[@]})"
    echo "############################################################"

    if ! start_vllm_server "${VLLM_LOG_DIR}/seed${SEED}.log"; then
        echo ">>> [seed${SEED}] ✗ vLLM failed to start; skipping"
        FAILED_RUNS="${FAILED_RUNS} seed${SEED}:vllm"
        continue
    fi

    if ! run_seed "${SEED}"; then
        echo ">>> [seed${SEED}] ✗ training failed; continuing"
        FAILED_RUNS="${FAILED_RUNS} seed${SEED}:train"
    fi

    stop_vllm_server
done

# =============================================
# Final summary
# =============================================
echo ""
echo "############################################################"
echo "# 8-SEED LoRA TRAINING COMPLETE"
echo "############################################################"
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo ">>> Adaptors: ${BASE_OUTPUT}/seed{0..7}"
echo ">>> Summary:  ${SUMMARY_FILE}"
if [ -n "${FAILED_RUNS}" ]; then
    echo ">>> FAILED runs:${FAILED_RUNS}"
fi
