#!/bin/bash
# GRPO on GSM8K: Qwen3-4B
# vLLM server on GPUs 2,3 (TP=2) — training on GPUs 0,1 (accelerate)
# Logs: output/grpo_qwen3_4b/vllm.log, output/grpo_qwen3_4b/train.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_NAME="Qwen/Qwen3-4B"
PORT=8880
OUTPUT_DIR="${PROJECT_DIR}/output/grpo_qwen3_4b"

# ---------------------------------------------------------------
# MBE dynamics logging (optional)
# Set MBE_LOG=1 to log per-token MBE/CE traces during training.
# Records are written to ${OUTPUT_DIR}/mbe_dynamics.jsonl
# ---------------------------------------------------------------
MBE_LOG=0           # 1 = enable, 0 = disable
MBE_LOG_STEPS=5     # log every N reward-function calls
MBE_LOG_SAMPLE_K=4  # rollouts to analyse per logged step

MBE_FLAGS=""
if [ "${MBE_LOG}" = "1" ]; then
    MBE_FLAGS="--mbe_log --mbe_log_steps ${MBE_LOG_STEPS} --mbe_log_sample_k ${MBE_LOG_SAMPLE_K}"
    echo ">>> MBE dynamics logging ON  (every ${MBE_LOG_STEPS} steps, ${MBE_LOG_SAMPLE_K} samples) → ${OUTPUT_DIR}/mbe_dynamics.jsonl"
fi

mkdir -p "${OUTPUT_DIR}"

# Kill any leftover processes from previous runs
fuser -k ${PORT}/tcp 2>/dev/null || true
ps aux | grep -iE "vllm|EngineCore" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 3
# Verify port is free
if fuser ${PORT}/tcp 2>/dev/null; then
    echo "ERROR: Port ${PORT} still in use. Waiting 10s..."
    sleep 10
    fuser -k ${PORT}/tcp 2>/dev/null || true
fi

# =============================================
# (1) Spin up vLLM server (TP=2 on GPUs 2,3)
# =============================================
VLLM_LOG="${OUTPUT_DIR}/vllm.log"
echo ">>> Starting vLLM server for ${MODEL_NAME} on GPUs 2,3 (TP=2) ..."
echo ">>> vLLM logs: ${VLLM_LOG}"
CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 2 \
    --port ${PORT} \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo ">>> Waiting for vLLM server (PID ${VLLM_PID}) ..."
until curl -s http://localhost:${PORT}/health > /dev/null 2>&1; do sleep 2; done
echo ">>> vLLM server ready."

# =============================================
# (2) Train on GPUs 0,1 via accelerate
# =============================================
TRAIN_LOG="${OUTPUT_DIR}/train.log"
echo ">>> Training ${MODEL_NAME} on GPUs 0,1 ..."
echo ">>> Training logs: ${TRAIN_LOG}"
echo ">>> Follow live: tail -f ${TRAIN_LOG}"
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    "${SCRIPT_DIR}/grpo_gsm8k.py" \
    --model ${MODEL_NAME} \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 200 \
    --use_vllm --vllm_mode server --vllm_server_port ${PORT} \
    --num_generations 4 \
    --max_completion_length 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-6 \
    --logging_steps 10 \
    --save_strategy no \
    --report_to none \
    ${MBE_FLAGS} \
    2>&1 | tee "${TRAIN_LOG}"

# =============================================
# (3) Kill vLLM server
# =============================================
echo ">>> Killing vLLM server ..."
kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true
ps aux | grep -iE "vllm|EngineCore" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 3

# =============================================
# (4) Rollout analysis
# =============================================
ANALYSIS_LOG="${OUTPUT_DIR}/analysis.log"
echo ">>> Rollout analysis ..."
echo ">>> Analysis logs: ${ANALYSIS_LOG}"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/analyze_rollouts.py" \
    --checkpoint "${OUTPUT_DIR}" --n_samples 50 \
    2>&1 | tee "${ANALYSIS_LOG}"

echo ""
echo ">>> All done."
echo ">>> Logs:"
echo ">>>   vLLM:     ${VLLM_LOG}"
echo ">>>   Training: ${TRAIN_LOG}"
echo ">>>   Analysis: ${ANALYSIS_LOG}"
