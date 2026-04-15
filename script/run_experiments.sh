#!/bin/bash
# Two controlled experiments on GSM8K with Qwen3-1.7B:
#
#   EXP 2 – Baseline GRPO, long run (1500 steps) → watch collapse
#   EXP 3 – GRPO + prefix from incorrect rollouts, 1500 steps → anti-collapse test
#
# GPU layout: GPU_TRAIN runs training, GPU_VLLM runs the vLLM server.
#
# Usage:
#   bash script/run_experiments.sh                   # run both
#   RUN_EXPS=3 bash script/run_experiments.sh        # only exp 3
#   REPORT_TO=wandb bash script/run_experiments.sh
#   GPU_TRAIN=0 GPU_VLLM=1 bash script/run_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ── Overridable globals ──────────────────────────────────────────────────────
GPU_TRAIN=${GPU_TRAIN:-0}
GPU_VLLM=${GPU_VLLM:-1}
PORT=${PORT:-8881}
REPORT_TO=${REPORT_TO:-none}
RUN_EXPS=${RUN_EXPS:-2,3}

MODEL="Qwen/Qwen3-1.7B"
MAX_STEPS=1500
MAX_COMPLETION_LENGTH=1280        # 1024 think + 256 answer

# ── Helpers ──────────────────────────────────────────────────────────────────
should_run() { echo ",${RUN_EXPS}," | grep -q ",${1},"; }

banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    printf  "║  %-52s  ║\n" "$1"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
}

VLLM_PID=""

kill_vllm() {
    echo ">>> Stopping vLLM server ..."
    [ -n "${VLLM_PID}" ] && { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
    fuser -k ${PORT}/tcp 2>/dev/null || true
    ps aux | grep -iE "vllm|EngineCore" | grep -v grep \
           | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    sleep 3
    VLLM_PID=""
}
trap 'kill_vllm' EXIT

start_vllm() {
    local logfile="$1"
    mkdir -p "$(dirname "${logfile}")"
    kill_vllm   # clean slate
    echo ">>> Starting vLLM server for ${MODEL} on GPU ${GPU_VLLM} ..."
    CUDA_VISIBLE_DEVICES=${GPU_VLLM} trl vllm-serve \
        --model "${MODEL}" \
        --tensor-parallel-size 1 \
        --port ${PORT} \
        > "${logfile}" 2>&1 &
    VLLM_PID=$!
    echo ">>> Waiting for vLLM server (PID ${VLLM_PID}) ..."
    until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do sleep 2; done
    echo ">>> vLLM server ready."
}

run_training() {
    local out_dir="$1"; shift
    local extra_flags="$*"
    echo ">>> Training → ${out_dir}  (max ${MAX_STEPS} steps)"
    echo ">>> tail -f ${out_dir}/train.log"

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${GPU_TRAIN} accelerate launch \
        --num_processes 1 --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no \
        "${SCRIPT_DIR}/grpo_gsm8k.py" \
        --model                      "${MODEL}" \
        --output_dir                 "${out_dir}" \
        --max_steps                  ${MAX_STEPS} \
        --max_completion_length      ${MAX_COMPLETION_LENGTH} \
        --use_vllm --vllm_mode server --vllm_server_port ${PORT} \
        --train_device               "${GPU_TRAIN}" \
        --num_generations            4 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --learning_rate              3e-6 \
        --logging_steps              10 \
        --eval_steps                 50 \
        --eval_samples               200 \
        --save_strategy              no \
        --report_to                  "${REPORT_TO}" \
        --mbe_log \
        --mbe_log_steps              5 \
        --mbe_log_sample_k           4 \
        ${extra_flags} \
        2>&1 | tee "${out_dir}/train.log"
}

compute_mbe_offline() {
    local out_dir="$1"
    local rollouts="${out_dir}/rollouts.jsonl"
    [ -f "${rollouts}" ] || return 0
    echo ">>> Computing MBE traces → ${out_dir}/mbe_dynamics.jsonl ..."
    CUDA_VISIBLE_DEVICES=${GPU_TRAIN} python "${SCRIPT_DIR}/compute_mbe.py" \
        --rollouts "${rollouts}" \
        --output   "${out_dir}/mbe_dynamics.jsonl" \
        --model    "${MODEL}" \
        --device   "cuda:${GPU_TRAIN}" \
        2>&1 | tee "${out_dir}/compute_mbe.log"
}

# ════════════════════════════════════════════════════════════════════════════
# EXP 2 — Baseline long run (collapse reference)
# ════════════════════════════════════════════════════════════════════════════
if should_run 2; then
    banner "EXP 2 · Baseline GRPO · Qwen3-1.7B · ${MAX_STEPS} steps"
    OUT2="${PROJECT_DIR}/output/exp2_baseline_collapse"
    mkdir -p "${OUT2}"

    start_vllm "${OUT2}/vllm.log"
    run_training "${OUT2}"     # no extra flags — pure baseline
    kill_vllm

    compute_mbe_offline "${OUT2}"
    echo ">>> EXP 2 done.  →  ${OUT2}"
fi

# ════════════════════════════════════════════════════════════════════════════
# EXP 3 — Prefix from wrong rollouts (anti-collapse test)
# ════════════════════════════════════════════════════════════════════════════
if should_run 3; then
    banner "EXP 3 · Prefix (incorrect rollouts) · Qwen3-1.7B · ${MAX_STEPS} steps"
    OUT3="${PROJECT_DIR}/output/exp3_prefix_incorrect"
    mkdir -p "${OUT3}"

    start_vllm "${OUT3}/vllm.log"
    run_training "${OUT3}" \
        --prefix_rollout \
        --prefix_augment_prob 0.4 \
        --prefix_buffer_size  500 \
        --prefix_min_frac     0.15 \
        --prefix_max_frac     0.75 \
        --prefix_from_correct incorrect
    kill_vllm

    compute_mbe_offline "${OUT3}"
    echo ">>> EXP 3 done.  →  ${OUT3}"
fi

# ════════════════════════════════════════════════════════════════════════════
banner "Done."
should_run 2 && echo "  EXP 2  ${PROJECT_DIR}/output/exp2_baseline_collapse"
should_run 3 && echo "  EXP 3  ${PROJECT_DIR}/output/exp3_prefix_incorrect"
