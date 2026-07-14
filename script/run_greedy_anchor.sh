#!/bin/bash
# Greedy-anchoring Dr.GRPO (Idea 3 — the winning arm, 2026-07-14):
# every generation step also decodes one T=0 rollout per query and splices it
# over the lowest-advantage slot, so the incumbent greedy chain is always in
# the comparison group; advantages recomputed as r - mean_group(r).
# Implementation: src/group_control_grpo.py (group_control="greedy"), wired
# through script/grpo.py --group_control. Protocol matches baseline run2.
# Result on Qwen3-0.6B/GSM8K: final 73.9 / union 91.8 / gap 17.9
#            vs baseline     final 71.3 / union 91.8 / gap 20.5
#
# Usage: bash script/run_greedy_anchor.sh [SEED] [OUT_DIR]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

SEED="${1:-0}"
OUT="${2:-${PROJECT_DIR}/output/gc_greedy_seed${SEED}}"
mkdir -p "${OUT}"

# accum-4 ZeRO-2 config (fp32 master weights — required at lr 5e-6, see claude.md)
CFG="${OUT}/zero2_ga4.yaml"
sed 's/gradient_accumulation_steps: [0-9]*/gradient_accumulation_steps: 4/' \
    "${PROJECT_DIR}/configs/zero2.yaml" > "${CFG}"

accelerate launch --config_file "${CFG}" --main_process_port 29631 \
    script/grpo.py \
    --output_dir "${OUT}" \
    --group_control greedy \
    --no-mbe_velocity_reward \
    --learning_rate 5e-6 --seed "${SEED}" \
    --max_steps 300 --eval_steps 25 \
    --vllm_gpu_memory_utilization 0.25 \
    --eval_batch_size 4096 --eval_greedy_only \
    2>&1 | tee "${OUT}/train.log"

python script/forgetting_gap_gsm8k.py --run "${OUT}" | tee "${OUT}/forgetting_gap.txt"