#!/bin/bash
# GRPO on GSM8K — 2-GPU layout
#   GPU 1 : vLLM server (TP=1)
#   GPU 0 : training   (accelerate, 1 process)
#
# Override any variable from the command line, e.g.:
#   PCRE=1 PREFIX_FROM=correct MBE_LOG=1 bash script/run_grpo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_NAME="Qwen/Qwen3-1.7B"
PORT=8880

# ---------------------------------------------------------------
# Experiment knobs — all overridable via env vars
# ---------------------------------------------------------------
MBE_LOG=${MBE_LOG:-0}               # 1 = log per-token MBE/CE traces
MBE_LOG_STEPS=${MBE_LOG_STEPS:-5}
MBE_LOG_SAMPLE_K=${MBE_LOG_SAMPLE_K:-4}

PCRE=${PCRE:-0}                          # 1 = prefix-conditioned rollout exploration
PREFIX_AUGMENT_PROB=${PREFIX_AUGMENT_PROB:-0.3}   # fraction of queries to replace with prefixes
PREFIX_BUFFER_SIZE=${PREFIX_BUFFER_SIZE:-500}
PREFIX_MIN_FRAC=${PREFIX_MIN_FRAC:-0.15}
PREFIX_MAX_FRAC=${PREFIX_MAX_FRAC:-0.75}
PREFIX_FROM=${PREFIX_FROM:-all}           # all | correct | incorrect

# Derive a unique output dir from the experiment configuration
EXP_TAG="baseline"
if [ "${PCRE}" = "1" ]; then EXP_TAG="pcre_${PREFIX_FROM}"; fi
if [ "${MBE_LOG}" = "1" ]; then EXP_TAG="${EXP_TAG}_mbe"; fi
OUTPUT_DIR="${PROJECT_DIR}/output/grpo_${EXP_TAG}"

# ---------------------------------------------------------------
MBE_FLAGS=""
if [ "${MBE_LOG}" = "1" ]; then
    MBE_FLAGS="--mbe_log --mbe_log_steps ${MBE_LOG_STEPS} --mbe_log_sample_k ${MBE_LOG_SAMPLE_K}"
    echo ">>> MBE logging ON  (every ${MBE_LOG_STEPS} steps, ${MBE_LOG_SAMPLE_K} samples)"
fi

PCRE_FLAGS=""
if [ "${PCRE}" = "1" ]; then
    PCRE_FLAGS="--prefix_rollout \
        --prefix_augment_prob ${PREFIX_AUGMENT_PROB} \
        --prefix_buffer_size ${PREFIX_BUFFER_SIZE} \
        --prefix_min_frac ${PREFIX_MIN_FRAC} \
        --prefix_max_frac ${PREFIX_MAX_FRAC} \
        --prefix_from_correct ${PREFIX_FROM}"
    echo ">>> PCRE ON  (prob=${PREFIX_AUGMENT_PROB}, buf=${PREFIX_BUFFER_SIZE}, frac=[${PREFIX_MIN_FRAC},${PREFIX_MAX_FRAC}], from=${PREFIX_FROM})"
fi

echo ">>> Output dir: ${OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

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
# (1) Spin up vLLM server (TP=1 on GPU 1)
# =============================================
VLLM_LOG="${OUTPUT_DIR}/vllm.log"
echo ">>> Starting vLLM server for ${MODEL_NAME} on GPU 1 (TP=1) ..."
echo ">>> vLLM logs: ${VLLM_LOG}"
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 1 \
    --port ${PORT} \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo ">>> Waiting for vLLM server (PID ${VLLM_PID}) ..."
until curl -s http://localhost:${PORT}/health > /dev/null 2>&1; do sleep 2; done
echo ">>> vLLM server ready."

# =============================================
# (2) Train on GPU 0 via accelerate
# =============================================
TRAIN_LOG="${OUTPUT_DIR}/train.log"
echo ">>> Training ${MODEL_NAME} on GPU 0 ..."
echo ">>> Training logs: ${TRAIN_LOG}"
echo ">>> Follow live: tail -f ${TRAIN_LOG}"
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    "${SCRIPT_DIR}/grpo_gsm8k.py" \
    --model ${MODEL_NAME} \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 200 \
    --use_vllm --vllm_mode server --vllm_server_port ${PORT} \
    --num_generations 4 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-6 \
    --logging_steps 10 \
    --save_strategy no \
    --report_to none \
    ${MBE_FLAGS} \
    ${PCRE_FLAGS} \
    2>&1 | tee "${TRAIN_LOG}"

# =============================================
# (3) Kill vLLM server
# =============================================
echo ">>> Killing vLLM server ..."
kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true
ps aux | grep -iE "vllm|EngineCore" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 3

# =============================================
# (4) MBE computation (offline, if enabled)
# =============================================
if [ "${MBE_LOG}" = "1" ]; then
    MBE_OUT="${OUTPUT_DIR}/mbe_dynamics.jsonl"
    echo ">>> Computing MBE traces from rollouts (GPU 0) → ${MBE_OUT} ..."
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/compute_mbe.py" \
        --rollouts "${OUTPUT_DIR}/rollouts.jsonl" \
        --output   "${MBE_OUT}" \
        --model    "${MODEL_NAME}" \
        --device   cuda:0 \
        2>&1 | tee "${OUTPUT_DIR}/compute_mbe.log"
fi

# =============================================
# (5) Rollout analysis
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





