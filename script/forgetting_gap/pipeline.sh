#!/usr/bin/env bash
# Forgetting-gap verification + OPD, end to end.
#
#   bash script/forgetting_gap/pipeline.sh all          # everything, in order
#   bash script/forgetting_gap/pipeline.sh control      # 1. Dr.GRPO teacher run
#   bash script/forgetting_gap/pipeline.sh annotate     # 2. FULL train split
#   bash script/forgetting_gap/pipeline.sh opd          # 3. OPD student
#   bash script/forgetting_gap/pipeline.sh eval         # 4. one 8192 pass
#   bash script/forgetting_gap/pipeline.sh report       # 5. the table
#
#   GPU=1 SEED=1 PICK=latest bash script/forgetting_gap/pipeline.sh opd
#
# Read script/forgetting_gap/README.md before changing any number below. Every
# constant here was set by a measurement, and several of them were set by a
# measurement that first came out wrong.
set -uo pipefail
STAGE=${1:?stage: all|control|annotate|opd|eval|report}
ROOT=/home/claudeuser/arl
GPU=${GPU:-0}
SEED=${SEED:-0}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
DATASET=${DATASET:-math}
OUT=${OUT:-output/fgap}
TAG=${TAG:-s$SEED}
CTRL=$OUT/drgrpo_$TAG
OPD=$OUT/opd_${PICK:-best}_$TAG
LOGS=$ROOT/logs/fgap
mkdir -p "$LOGS" "$ROOT/$OUT"; cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$((29700+GPU))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false

# ===========================================================================
# TRAINING BUDGET. 3072 new tokens, for BOTH arms, and also the budget the
# annotation pass runs at. Not a free parameter:
#   - at 1536 the score was ~90% a truncation count. Same checkpoints
#     re-scored 1536 -> 3072 moved seed 0 by +10.4 and the others by <1, and
#     the across-seed sd collapsed from 8.1 to 2.1.
#   - the model's natural CoT is ~2500 tokens with a heavy tail; at 3072 about
#     half the training rollouts still hit the cap. Raising it further changes
#     what RL teaches (a tight cap teaches brevity, a loose one teaches
#     capability), so it is a real hyperparameter, not a measurement detail.
TRAIN_TOKENS=${TRAIN_TOKENS:-3072}
VLLM_LEN=$((TRAIN_TOKENS + 384))

# STEPS. 150 for both arms so both see the same number of prompt draws
# (16 prompts/step x 150 = 2400). With the full 7500-prompt split that is
# ~0.36 epochs and NEITHER arm repeats a prompt. An earlier OPD run on a
# 1000-prompt annotation reached epoch 1.42 while its control was at 0.32 --
# that is a data confound, not a method difference. Watch the `epoch` field.
STEPS=${STEPS:-150}

# GEOMETRY, identical for both arms. 2 x 64 = 128 rollouts/step, G=8 -> 16
# prompts/step. lr 1e-5, not 1e-6: at this scale lr was worth 18 points
# (1e-6 -> 48.6, 1e-5 -> 66.6 on an earlier OPD sweep), so a mismatched lr
# swamps every method effect.
COMMON="--model $MODEL --dataset $DATASET \
  --max_completion_length $TRAIN_TOKENS --lr ${LR:-1e-5} \
  --num_generations 8 --per_device_bs 2 --ga 64 --grad_ckpt 1 \
  --vllm_max_len $VLLM_LEN --max_steps $STEPS --save_steps 25"

# ===========================================================================
# EVALUATION BUDGET. Generate ONCE at 8192 so nothing is censored, then derive
# the train-matched 3072 number from that same pass. Never run a second
# generation at cap 3072 -- see README "the capped-eval trap".
EVAL_TOKENS=${EVAL_TOKENS:-8192}
EVAL_N=${EVAL_N:-1000}

log(){ echo "=== $* $(date -u +%H:%M:%S) ===" | tee -a $LOGS/pipeline.log; }
reap(){ for p in $(ps -ef|grep 'VLLM::EngineCore'|grep -v grep|awk '{print $2}'); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=$GPU$" && kill -9 $p; done
  sleep 15; }

# --------------------------------------------------------------- 1. control
do_control(){
  [ -d "$CTRL/checkpoint-$STEPS" ] && { log "control done, skip"; return; }
  log "1. Dr.GRPO teacher run -> $CTRL"
  python script/forgetting_gap/drgrpo_min.py train --out $CTRL --seed $SEED $COMMON \
    --beta 0.01 --gpu_util 0.35 > $LOGS/train_ctrl_$TAG.log 2>&1
  log "control EXIT=$?"; reap
}

# -------------------------------------------------------------- 2. annotate
# --train_subset 0 means the FULL 7500-prompt train split. This matters twice:
#   (a) OPD only trains on prompts some checkpoint solves, so a 1000-prompt
#       annotation leaves it with ~894 prompts = 12% of what the control sees.
#   (b) --max_tokens MUST equal TRAIN_TOKENS. The table records which
#       checkpoint solves which query; answering that under a different
#       truncation regime than the students train under mislabels the teachers.
do_annotate(){
  [ -f "$CTRL/train_forgetting_annotations.jsonl" ] && [ -z "${FORCE:-}" ] && {
    log "annotation exists, skip (FORCE=1 to redo)"; return; }
  log "2. annotate FULL train split @${TRAIN_TOKENS} over every checkpoint"
  python script/forgetting_gap/drgrpo_min.py annotate --out $CTRL --model $MODEL \
    --dataset $DATASET --train_subset 0 --max_tokens $TRAIN_TOKENS \
    --gpu_util 0.55 --overwrite > $LOGS/annotate_$TAG.log 2>&1
  log "annotate EXIT=$?"
  grep -aE "ever-solved" $LOGS/annotate_$TAG.log | tail -1 | tee -a $LOGS/pipeline.log
  python script/forgetting_gap/report.py --audit --run $CTRL | tee -a $LOGS/pipeline.log
  reap
}

# ------------------------------------------------------------------- 3. OPD
# --pick chooses the teacher for a query among the checkpoints that solve it:
#   best     = the globally strongest solver   (default)
#   latest   = the last solver                 ("last solvable teacher")
#   shortest = the solver that used fewest tokens
# ALWAYS read the per-ckpt distribution the run prints. On a 1000-prompt
# annotation, `best` sent 92% of queries to one checkpoint and `latest` 87% --
# i.e. it was single-teacher distillation wearing a multi-teacher costume.
do_opd(){
  [ -d "$OPD/checkpoint-$STEPS" ] && { log "opd done, skip"; return; }
  [ -f "$CTRL/train_forgetting_annotations.jsonl" ] || { log "annotate first"; exit 1; }
  log "3. OPD (pick=${PICK:-best}) -> $OPD"
  python script/forgetting_gap/opd_min.py train --out $OPD --seed $SEED \
    --teacher_run $CTRL $COMMON --pick ${PICK:-best} --upsample 1 \
    --interleave rr --teacher_dtype fp32 --anchor_dtype bf16 --tf32 1 \
    --teacher_cache 1 --model_dtype bfloat16 --gpu_util 0.30 \
    > $LOGS/train_opd_${PICK:-best}_$TAG.log 2>&1
  log "opd EXIT=$?"
  grep -aoE "'epoch': '[0-9.]+'" $LOGS/train_opd_${PICK:-best}_$TAG.log | tail -1 \
    | tee -a $LOGS/pipeline.log
  reap
}

# ------------------------------------------------------------------ 4. eval
# ONE generous pass per checkpoint, all on the same EVAL_N questions, with the
# question set fingerprinted so mismatched comparisons are refused rather than
# silently printed.
do_eval(){
  log "4. evalx: single ${EVAL_TOKENS}-token pass, n=$EVAL_N"
  for R in $CTRL $OPD; do
    [ -d "$R" ] || continue
    python script/forgetting_gap/evalx.py run --cap $EVAL_TOKENS --n $EVAL_N \
      --dataset $DATASET --util 0.55 --run $R >> $LOGS/eval_$TAG.log 2>&1
    log "eval $R EXIT=$?"; reap
  done
  grep -a "^\[evalx\]" $LOGS/eval_$TAG.log | tail -20 | tee -a $LOGS/pipeline.log
}

do_report(){
  python script/forgetting_gap/report.py --run $CTRL --opd $OPD \
    --budgets $TRAIN_TOKENS $EVAL_TOKENS | tee -a $LOGS/pipeline.log
}

case $STAGE in
  control)  do_control;;
  annotate) do_annotate;;
  opd)      do_opd;;
  eval)     do_eval;;
  report)   do_report;;
  all)      do_control; do_annotate; do_opd; do_eval; do_report;;
  *) echo "unknown stage $STAGE"; exit 1;;
esac
log "STAGE $STAGE DONE"
