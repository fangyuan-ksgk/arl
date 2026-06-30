#!/bin/bash
# ============================================================================
# BIG VALIDATION RUN — three claims, one driver (local_sgd_grpo.py).
#
#   1. Lottery Gap Exists:        union_acc > avg_acc          (independent seeds disagree)
#   2. Soup Merge exploits it:    merged_acc > avg_acc         (dense weight-avg harvests the gap)
#   3. Delayed sync ~ one-shot:   merged_acc(P=1) ≈ merged_acc(P=32) ≈ merged_acc(one-shot)
#                                 (limited gains from more frequent synchronization)
#
# Models:   Qwen3-4B, Qwen3-8B  (server mode: 1 GPU serve + 1 GPU train/branch),
#           google/gemma-4-E4B-it (vLLM CANNOT serve gemma4 -> --no_vllm / HF generation).
# Common:   lr=2e-5, K=4 seeds, MATH datasets eval full test; APPS eval = HumanEval+ (proxy).
#           6 GPUs => server mode runs K=4 branches in 2 waves (3 slots); no_vllm runs 4 in 1 wave.
#
# Per (model, dataset) grid (4 runs):
#   r1  plain Dr.GRPO            one-shot, ONE EPOCH  (--total_steps -1, --period 1)   [lottery gap]
#   r2  Dr.GRPO + MBE-velo+vmax  one-shot, ONE EPOCH  (+ --mbe_velocity_reward --virtual_rollout insert_max)
#   r3  plain Dr.GRPO            P=1,  T=128                                            [delayed-sync sweep]
#   r4  plain Dr.GRPO            P=32, T=128
#
# MATH  -> LoRA r512   (4B/8B full-FT is too heavy);  APPS -> FULL fine-tuning.
#
#   bash script/run_big_validation.sh [GPUS] [SWEEP_T]
#   e.g. bash script/run_big_validation.sh 0,1,2,3,4,5 128
# ============================================================================
set -uo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING

GPUS="${1:-0,1,2,3,4,5}"      # 6 GPUs -> 3 server-mode slots (2 GPUs/branch), K=4 in 2 waves
SWEEP_T="${2:-128}"          # bounded step budget for the P=1 / P=32 delayed-sync sweep
SEEDS="0,1,2,3"              # K=4 branches
LR="2e-5"
NUMGEN=4
GA=4
LORA_R=512
OUTROOT="output/bigval"

MODELS=("Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "google/gemma-4-E4B-it")
DATASETS=("math" "apps")

mkdir -p "$OUTROOT"

# run_cell <task> <model> <cond> <extra-flags...>
#   <cond> determines period/budget; lora vs full and no_vllm are derived from task/model.
run_cell () {
    local task="$1"; local model="$2"; local cond="$3"; shift 3
    local extra=("$@")
    local mtag="${model##*/}"
    local tag="${task}_${mtag}_${cond}"
    local out="$OUTROOT/$tag"

    # per-dataset: MATH = LoRA r512, APPS = full fine-tuning
    local lora=()
    if [[ "$task" == "math" ]]; then
        lora=(--use_lora --lora_r "$LORA_R")
    fi
    # gemma4 cannot be vLLM-served -> HF generation (colocate, 1 GPU/branch)
    local vllm=(--vllm_mode server)
    if [[ "$model" == *gemma-4* || "$model" == *gemma4* ]]; then
        vllm=(--no_vllm)
    fi

    echo "############ $tag ############"
    python "$SD/local_sgd_grpo.py" \
        --tag "$tag" --out "$out" --task "$task" --base_model "$model" \
        --gpus "$GPUS" --seeds "$SEEDS" \
        --num_generations "$NUMGEN" --grad_accum "$GA" --learning_rate "$LR" \
        --loss_type dr_grpo --scale_rewards none \
        --final_pass_k 1 --eval_limit 0 --final_eval_limit 0 \
        "${lora[@]}" "${vllm[@]}" "${extra[@]}" \
        || echo "[WARN] $tag returned nonzero"
}

for task in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        # r1: plain Dr.GRPO, one-shot full epoch  (lottery gap + soup)
        run_cell "$task" "$model" "plain_oneshot_epoch" \
            --period 1 --total_steps -1

        # r2: + MBE velocity reward + virtual max rollout, one-shot full epoch
        run_cell "$task" "$model" "mbevelo_vmax_oneshot_epoch" \
            --period 1 --total_steps -1 \
            --mbe_velocity_reward --mbe_velocity_scale 5.0 --mbe_velocity_clip 1.0 \
            --virtual_rollout insert_max --virtual_max_reward 1.2

        # r3: plain Dr.GRPO, P=1 (fully synchronized), bounded T
        run_cell "$task" "$model" "plain_P1_T${SWEEP_T}" \
            --period 1 --total_steps "$SWEEP_T"

        # r4: plain Dr.GRPO, P=32 (delayed sync), bounded T
        run_cell "$task" "$model" "plain_P32_T${SWEEP_T}" \
            --period 32 --total_steps "$SWEEP_T"
    done
done

echo "############ BIG VALIDATION DONE -> $OUTROOT ############"
echo "Per-run summary CSVs: $OUTROOT/*/results.csv  (avg_acc, union_acc, merge_acc, majority_acc, lottery_gap)"
