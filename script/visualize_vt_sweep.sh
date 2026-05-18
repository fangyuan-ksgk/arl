#!/bin/bash
# Mass-produce per-correct-answer vt sidecars + GIFs across the game24 sweep.
#
# Layout (3 models × 2 context-lengths = 6 jobs) distributed over 2 GPUs.
# At most 2 jobs run concurrently; each job is pinned to one GPU via
# CUDA_VISIBLE_DEVICES so it is single-GPU as far as the .py is concerned.
#
# Usage:
#   bash script/visualize_vt_sweep.sh
#   SWEEP_ROOT=output/game24_sweep STEP=1200 MAX_GIFS=16 \
#       bash script/visualize_vt_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

SWEEP_ROOT=${SWEEP_ROOT:-output/game24_sweep}
STEP=${STEP:-1200}
MAX_GIFS=${MAX_GIFS:-16}
MICRO_BATCH=${MICRO_BATCH:-8}
GIF_ROWS=${GIF_ROWS:-correct}
GPUS=(${GPUS:-0 1})  # parallelism = number of GPUs listed

LOG_DIR="${PROJECT_DIR}/logs/visualize_vt_sweep"
mkdir -p "${LOG_DIR}"

# (model_subdir, len_subdir, hf_id) — len_subdir matches the sweep layout.
JOBS=(
    "len512   Qwen__Qwen3-0.6B   Qwen/Qwen3-0.6B"
    "len512   Qwen__Qwen3-1.7B   Qwen/Qwen3-1.7B"
    "len512   Qwen__Qwen3-4B     Qwen/Qwen3-4B"
    "len1024  Qwen__Qwen3-0.6B   Qwen/Qwen3-0.6B"
    "len1024  Qwen__Qwen3-1.7B   Qwen/Qwen3-1.7B"
    "len1024  Qwen__Qwen3-4B     Qwen/Qwen3-4B"
)

run_one() {
    local gpu="$1" len="$2" model_dir="$3" hf_id="$4"
    local run_dir="${SWEEP_ROOT}/${len}/${model_dir}"
    local tag="${len}_${model_dir}"
    local log="${LOG_DIR}/${tag}.log"

    if [[ ! -f "${run_dir}/eval_rollout.jsonl" ]]; then
        echo "[skip] ${run_dir}/eval_rollout.jsonl missing" | tee -a "${log}"
        return 0
    fi

    echo ">>> [GPU ${gpu}] ${tag}  (scorer=${hf_id})  → ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python script/visualize_vt.py \
        --run-dir "${run_dir}" \
        --scorer-model "${hf_id}" \
        --step ${STEP} \
        --micro-batch ${MICRO_BATCH} \
        --max-gifs ${MAX_GIFS} \
        --gif-rows "${GIF_ROWS}" \
        > "${log}" 2>&1
    echo "<<< [GPU ${gpu}] ${tag} done"
}

# ── simple GPU-pool scheduler: keep one job in flight per GPU ──────────────
declare -A GPU_PID  # gpu -> running pid (empty == free)
for g in "${GPUS[@]}"; do GPU_PID[$g]=""; done

wait_free_gpu() {
    while true; do
        for g in "${GPUS[@]}"; do
            local pid="${GPU_PID[$g]}"
            if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
                if [[ -n "${pid}" ]]; then wait "${pid}" || true; fi
                GPU_PID[$g]=""
                echo "$g"
                return
            fi
        done
        sleep 2
    done
}

for spec in "${JOBS[@]}"; do
    read -r len model_dir hf_id <<< "${spec}"
    gpu=$(wait_free_gpu)
    run_one "${gpu}" "${len}" "${model_dir}" "${hf_id}" &
    GPU_PID[$gpu]=$!
done

# drain
for g in "${GPUS[@]}"; do
    pid="${GPU_PID[$g]}"
    [[ -n "${pid}" ]] && wait "${pid}" || true
done

echo ">>> all jobs done. logs in ${LOG_DIR}"
