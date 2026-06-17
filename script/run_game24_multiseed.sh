#!/bin/bash
# run_game24_grpo_ckpt_seeds.sh
# ---------------------------------------------------------------------------
# Game-of-24, **base GRPO** (vanilla trajectory-level advantage), Qwen3-0.6B.
# Runs 8 seeds; each run force-saves 5 checkpoints at global steps
# {1, 4, 8, 12, MAX_STEPS} into <run_dir>/trl/checkpoint-<step>.
#
#   bash script/run_game24_grpo_ckpt_seeds.sh
#   SEEDS="0 1" MAX_STEPS=120 bash script/run_game24_grpo_ckpt_seeds.sh
#   OUT=$HOME/g24_grpo_ckpts CUDA_VISIBLE_DEVICES=1 bash script/run_game24_grpo_ckpt_seeds.sh
#
# Knobs (defaults):
#   OUT=$HOME/game24_grpo_ckpt_seeds   MODEL=Qwen/Qwen3-0.6B
#   SEEDS="0 1 2 3 4 5 6 7"            MAX_STEPS=200
#   MAX_N=13  BETA=0.0  SCALE=group    GPU/CUDA_VISIBLE_DEVICES respected
#
# The seed reshuffles TRAINING data order only; the eval split is built with a
# FIXED rng (0xE7A15 inside tripo_game24.py) so validation data is identical
# across all seeds. Runs are sequential (each fully owns the GPU via colocate
# vLLM). Output dirs are per-seed so they never collide.
# ---------------------------------------------------------------------------
set -uo pipefail

# --- resolve repo + training script robustly (survives arl/ re-syncs) -------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${ARL:-}" ]; then
  if [ -f "$(dirname "$SCRIPT_DIR")/src/game24utils.py" ]; then ARL="$(dirname "$SCRIPT_DIR")"; else ARL="/home/claudeuser/arl"; fi
fi
RUN="${RUN:-$ARL/script/tripo_game24.py}"
export PYTHONPATH="$ARL${PYTHONPATH:+:$PYTHONPATH}" \
       HF_HUB_ENABLE_HF_TRANSFER=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- config -----------------------------------------------------------------
OUT="${OUT:-$HOME/game24_grpo_ckpt_seeds}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
MAX_STEPS="${MAX_STEPS:-120}"
MAX_N="${MAX_N:-13}"
EVAL_MIN="${EVAL_MIN:-200}"
BETA="${BETA:-0.04}"
SCALE="${SCALE:-group}"
MEM="${MEM:-0.55}"
CVD="${CUDA_VISIBLE_DEVICES:-0}"

# 5 checkpoints: early-training grid {1,4,8,12} + end of training.
SAVE_STEPS_LIST="1,4,8,12,${MAX_STEPS}"

echo "RUN=$RUN"
echo "OUT=$OUT  MODEL=$MODEL  MAX_STEPS=$MAX_STEPS  MAX_N=$MAX_N  EVAL_MIN=$EVAL_MIN  BETA=$BETA  SCALE=$SCALE"
echo "SEEDS=[$SEEDS]  CUDA_VISIBLE_DEVICES=$CVD  save@steps={$SAVE_STEPS_LIST}"
mkdir -p "$OUT"

# --- run loop ---------------------------------------------------------------
for S in $SEEDS; do
  D="$OUT/grpo_seed${S}"; mkdir -p "$D"
  echo ">>> [seed $S | base GRPO | CUDA_VISIBLE_DEVICES=$CVD] $(date)"
  CUDA_VISIBLE_DEVICES="$CVD" \
  python "$RUN" \
    --trainer grpo \
    --output-dir "$D" \
    --model "$MODEL" \
    --seed "$S" \
    --save-steps-list "$SAVE_STEPS_LIST" \
    --max-steps "$MAX_STEPS" \
    --num-generations 8 --num-generations-eval 8 \
    --max-completion-length 1024 --no-think \
    --max-n "$MAX_N" --eval-frac 0.40 --eval-min "$EVAL_MIN" --eval-steps 50 \
    --per-device-batch-size 4 --grad-accum 4 \
    --learning-rate 5e-6 --temperature 1.0 \
    --beta "$BETA" --scale-rewards "$SCALE" \
    --vllm-mode colocate --vllm-gpu-memory-utilization "$MEM" \
    > "$D.log" 2>&1 \
    && echo "    DONE   seed $S  ckpts -> $D/trl/checkpoint-{1,4,8,12,$MAX_STEPS}" \
    || echo "    FAILED seed $S  (tail $D.log)"
done

echo
echo "================== all seeds done =================="
echo "checkpoints: $OUT/grpo_seed<S>/trl/checkpoint-<step>"
echo "rollouts:    $OUT/grpo_seed<S>/{rollouts,eval_rollout}.jsonl"
