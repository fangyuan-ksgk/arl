# GRPO on GSM8K: Qwen3-4B
# vLLM server on GPUs 2,3 (TP=2) — training on GPUs 0,1 (accelerate)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_NAME="Qwen/Qwen3-4B"
PORT=8880

# Kill any leftover processes from previous runs
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

# =============================================
# (1) Spin up vLLM server (TP=2 on GPUs 2,3)
# =============================================
echo ">>> Starting vLLM server for ${MODEL_NAME} on GPUs 2,3 (TP=2) ..."
CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 2 \
    --port ${PORT} &
VLLM_PID=$!
echo ">>> Waiting for vLLM server (PID ${VLLM_PID}) ..."
until curl -s http://localhost:${PORT}/health > /dev/null 2>&1; do sleep 2; done
echo ">>> vLLM server ready."

# =============================================
# (2) Train on GPUs 0,1 via accelerate
# =============================================
echo ">>> Training ${MODEL_NAME} on GPUs 0,1 ..."
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    "${SCRIPT_DIR}/grpo_gsm8k.py" \
    --model ${MODEL_NAME} \
    --output_dir "${PROJECT_DIR}/output/grpo_qwen3_4b" \
    --max_steps 200 \
    --use_vllm --vllm_mode server --vllm_server_port ${PORT} \
    --num_generations 4 \
    --max_completion_length 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-6 \
    --logging_steps 10 \
    --save_strategy no \
    --report_to none

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
echo ">>> Rollout analysis ..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/analyze_rollouts.py" \
    --checkpoint "${PROJECT_DIR}/output/grpo_qwen3_4b" --n_samples 50 &
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/analyze_rollouts.py" \
    --checkpoint "${PROJECT_DIR}/output/grpo_qwen3_4b" --n_samples 50 &
wait

echo ">>> All done."
