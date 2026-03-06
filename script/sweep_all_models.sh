#!/bin/bash
# Sweep GRPO across Qwen3 0.6B / 1.5B / 4B
# For each model: restart vLLM server, run all configs, then analyze
# Layout: GPUs 2,3 = vLLM (TP=2), GPUs 0,1 = training (accelerate)

# Issue #1. "port already in use" -- we'd better increment the PORT no. for each experiment, to bypass ths issue of "port already in use"
# Issue #2. Qwen3-1.7B | Qwen3-0.6B shoudl be used, no reason to stay behind the newest ver. of model too far behind

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BASE_PORT=8880
PORT=${BASE_PORT}
EXPERIMENT_INDEX=0
BASE_OUTPUT="${PROJECT_DIR}/output/sweep_all"
TIMESTAMP=$(date +%Y%m%d_%H%M)
ANALYSIS_SAMPLES=100

mkdir -p "${BASE_OUTPUT}"

VLLM_PID=""

# =============================================
# Cleanup function
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
# Helper: start vLLM for a given model
# =============================================
start_vllm() {
    local model_name=$1
    local log_file=$2
    cleanup_vllm
    PORT=$(( BASE_PORT + EXPERIMENT_INDEX ))
    EXPERIMENT_INDEX=$(( EXPERIMENT_INDEX + 1 ))
    echo ">>> Starting vLLM server for ${model_name} on port ${PORT}, GPUs 2,3 (TP=2)..."
    CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
        --model ${model_name} \
        --tensor-parallel-size 2 \
        --port ${PORT} \
        > "${log_file}" 2>&1 &
    VLLM_PID=$!
    echo ">>> Waiting for vLLM server (PID ${VLLM_PID})..."
    until curl -s http://localhost:${PORT}/health > /dev/null 2>&1; do sleep 2; done
    echo ">>> vLLM server ready."
}

# =============================================
# Helper: check vLLM alive, restart if needed
# =============================================
ensure_vllm() {
    local model_name=$1
    local log_file=$2
    if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo ">>> vLLM server died! Restarting..."
        start_vllm "${model_name}" "${log_file}"
    fi
}

# =============================================
# Helper: run one training config
# =============================================
run_config() {
    local model_name=$1
    local name=$2
    local max_tokens=$3
    local lr=$4
    local num_gen=$5
    local grad_accum=$6
    local max_steps=$7
    local run_dir=$8
    local train_log="${run_dir}/train.log"
    mkdir -p "${run_dir}"

    echo ">>>   ${name}: max_tokens=${max_tokens}, lr=${lr}, num_gen=${num_gen}, grad_accum=${grad_accum}, max_steps=${max_steps}"
    echo ">>>   Output: ${run_dir}"

    local start_time=$(date +%s)
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
        "${SCRIPT_DIR}/grpo_gsm8k.py" \
        --model ${model_name} \
        --output_dir "${run_dir}" \
        --max_steps ${max_steps} \
        --use_vllm --vllm_mode server --vllm_server_port ${PORT} \
        --num_generations ${num_gen} \
        --max_completion_length ${max_tokens} \
        --per_device_train_batch_size ${num_gen} \
        --gradient_accumulation_steps ${grad_accum} \
        --learning_rate ${lr} \
        --logging_steps 10 \
        --save_strategy no \
        --report_to none \
        2>&1 | tee "${train_log}"
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    # Extract metrics
    local final_reward=$(grep "'reward'" "${train_log}" | tail -1 | grep -oP "'reward': [0-9.]+" | grep -oP "[0-9.]+$" || echo "N/A")
    local peak_reward=$(grep "'reward'" "${train_log}" | grep -oP "'reward': [0-9.]+" | grep -oP "[0-9.]+$" | sort -n | tail -1 || echo "N/A")
    local clipped=$(grep "'completions/clipped_ratio'" "${train_log}" | tail -1 | grep -oP "'completions/clipped_ratio': [0-9.]+" | grep -oP "[0-9.]+$" || echo "N/A")

    echo "${name}: final_reward=${final_reward}, peak_reward=${peak_reward}, clipped=${clipped}, time=${elapsed}s" | tee -a "${SUMMARY_FILE}"
    echo "  max_tokens=${max_tokens}, lr=${lr}, num_gen=${num_gen}" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
}

# =============================================
# Summary file
# =============================================
SUMMARY_FILE="${BASE_OUTPUT}/summary_${TIMESTAMP}.txt"
echo "Sweep started: $(date)" > "${SUMMARY_FILE}"
echo "==========================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# =============================================================================
#  MODEL 1: Qwen/Qwen3-0.6B
# =============================================================================
# MODEL="Qwen/Qwen3-0.6B"
# MODEL_TAG="0.6b"
# MODEL_DIR="${BASE_OUTPUT}/${MODEL_TAG}"
# VLLM_LOG="${MODEL_DIR}/vllm.log"
# mkdir -p "${MODEL_DIR}"

# echo ""
# echo "############################################################"
# echo "# ${MODEL}"
# echo "############################################################"
# echo "Model: ${MODEL}" >> "${SUMMARY_FILE}"
# echo "-------------------------------------------" >> "${SUMMARY_FILE}"

# start_vllm "${MODEL}" "${VLLM_LOG}"

# # 0.6B: sweep num_generations at tok1024
# # Configs: (name, max_tokens, lr, num_gen, grad_accum, max_steps)
# CONFIGS_06B=(
#     "tok256_gen8,256,5e-6,8,8,200"
#     "tok512_gen8,512,5e-6,8,8,200"
# )

# for cfg in "${CONFIGS_06B[@]}"; do
#     IFS=',' read -r NAME MAX_TOKENS LR NUM_GEN GRAD_ACCUM MAX_STEPS <<< "${cfg}"
#     ensure_vllm "${MODEL}" "${VLLM_LOG}"
#     run_config "${MODEL}" "${MODEL_TAG}/${NAME}" "${MAX_TOKENS}" "${LR}" "${NUM_GEN}" "${GRAD_ACCUM}" "${MAX_STEPS}" "${MODEL_DIR}/${NAME}"
# done

# echo "" >> "${SUMMARY_FILE}"

# # =============================================================================
# #  MODEL 2: Qwen/Qwen3-1.7B
# # =============================================================================
# MODEL="Qwen/Qwen3-1.7B"
# MODEL_TAG="1.7b"
# MODEL_DIR="${BASE_OUTPUT}/${MODEL_TAG}"
# VLLM_LOG="${MODEL_DIR}/vllm.log"
# mkdir -p "${MODEL_DIR}"

# echo ""
# echo "############################################################"
# echo "# ${MODEL}"
# echo "############################################################"
# echo "Model: ${MODEL}" >> "${SUMMARY_FILE}"
# echo "-------------------------------------------" >> "${SUMMARY_FILE}"

# start_vllm "${MODEL}" "${VLLM_LOG}"

# # 1.5B: sweep completion length + num_generations
# CONFIGS_17B=(
#     "tok256_gen8,256,3e-6,8,8,200"
#     "tok512_gen8,512,3e-6,8,8,200"
#     "tok1024_gen4,1024,3e-6,4,16,200"
# )

# for cfg in "${CONFIGS_17B[@]}"; do
#     IFS=',' read -r NAME MAX_TOKENS LR NUM_GEN GRAD_ACCUM MAX_STEPS <<< "${cfg}"
#     ensure_vllm "${MODEL}" "${VLLM_LOG}"
#     run_config "${MODEL}" "${MODEL_TAG}/${NAME}" "${MAX_TOKENS}" "${LR}" "${NUM_GEN}" "${GRAD_ACCUM}" "${MAX_STEPS}" "${MODEL_DIR}/${NAME}"
# done

# echo "" >> "${SUMMARY_FILE}"

# =============================================================================
#  MODEL 3: Qwen/Qwen3-4B
# =============================================================================
MODEL="Qwen/Qwen3-4B"
MODEL_TAG="4b"
MODEL_DIR="${BASE_OUTPUT}/${MODEL_TAG}"
VLLM_LOG="${MODEL_DIR}/vllm.log"
mkdir -p "${MODEL_DIR}"

echo ""
echo "############################################################"
echo "# ${MODEL}"
echo "############################################################"
echo "Model: ${MODEL}" >> "${SUMMARY_FILE}"
echo "-------------------------------------------" >> "${SUMMARY_FILE}"

start_vllm "${MODEL}" "${VLLM_LOG}"

# 4B — tighter on VRAM, sweep completion length
CONFIGS_4B=(
    "tok256_gen4,256,3e-6,4,16,200"
    "tok1024_gen4,1024,3e-6,4,16,200"
    "tok1024_gen8,1024,3e-6,8,8,200"
    "tok2048_gen4,2048,3e-6,4,8,200"
    "tok4096_gen4,4096,3e-6,4,4,200"
)

for cfg in "${CONFIGS_4B[@]}"; do
    IFS=',' read -r NAME MAX_TOKENS LR NUM_GEN GRAD_ACCUM MAX_STEPS <<< "${cfg}"
    ensure_vllm "${MODEL}" "${VLLM_LOG}"
    run_config "${MODEL}" "${MODEL_TAG}/${NAME}" "${MAX_TOKENS}" "${LR}" "${NUM_GEN}" "${GRAD_ACCUM}" "${MAX_STEPS}" "${MODEL_DIR}/${NAME}"
done

echo "" >> "${SUMMARY_FILE}"

# =============================================================================
#  Analysis on all completed runs (4 GPUs)
# =============================================================================
echo ""
echo "############################################################"
echo "# Rollout Analysis (${ANALYSIS_SAMPLES} samples, 4 GPUs)"
echo "############################################################"
cleanup_vllm

for model_tag in 0.6b 1.5b 4b; do
    model_dir="${BASE_OUTPUT}/${model_tag}"
    for run_dir in "${model_dir}"/*/; do
        if [ -f "${run_dir}/config.json" ]; then
            name=$(basename "${run_dir}")
            echo ""
            echo ">>> Analyzing: ${model_tag}/${name}"
            python "${SCRIPT_DIR}/analyze_rollouts.py" \
                --checkpoint "${run_dir}" --n_samples ${ANALYSIS_SAMPLES} --n_gpus 4 \
                2>&1 | tee "${run_dir}/analysis.log"
        fi
    done
done

# =============================================================================
#  Final summary
# =============================================================================
echo ""
echo "############################################################"
echo "# SWEEP COMPLETE"
echo "############################################################"
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo ">>> Full results: ${BASE_OUTPUT}"
echo ">>> Summary: ${SUMMARY_FILE}"
