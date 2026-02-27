#!/bin/bash
# GRPO on GSM8K: Qwen3-4B then Qwen3-4B (4 GPUs, vLLM server mode)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_NAME="Qwen/Qwen3-4B"
PORT=8888

# Kill any leftover processes from previous runs
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

# =============================================
# (1) Spin up vLLM server
# =============================================
echo ">>> Starting vLLM server for ${MODEL_NAME} on GPUs 0,1 (TP=2) ..."
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 2 \
    --port ${PORT} &
VLLM_PID=$!
echo ">>> Waiting for vLLM server (PID ${VLLM_PID}) ..."
until curl -s http://localhost:${PORT}/health > /dev/null 2>&1; do sleep 1; done
echo ">>> vLLM server ready."

# =============================================
# (2) Train model on GPUs 2,3
# =============================================
echo ">>> Training ${MODEL_NAME} on GPUs 2,3 ..."
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 \
    "${SCRIPT_DIR}/grpo_gsm8k.py" \
    --model ${MODEL_NAME} \
    --output_dir "${PROJECT_DIR}/output/grpo_qwen3_4b" \
    --max_steps 200 \
    --use_vllm --vllm_mode server --vllm_server_port ${PORT} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-6 \
    --logging_steps 10 \
    --save_strategy no \
    --report_to none

# =============================================
# (3) Kill vLLM server
# =============================================
echo ">>> Killing vLLM server ..."
kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true
sleep 3


# =============================================
# (4) Rollout analysis
# =============================================
echo ">>> Rollout analysis ..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/analyze_rollouts.py" \
    --checkpoint "${PROJECT_DIR}/output/grpo_qwen3_4b" --n_samples 50 &
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/analyze_rollouts.py" \
    --checkpoint "${PROJECT_DIR}/output/grpo_qwen3_4b" --n_samples 50 &
wait

echo ">>> All done."