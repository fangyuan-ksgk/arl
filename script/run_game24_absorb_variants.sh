#!/bin/bash
# run_game24_absorb_variants.sh
# ---------------------------------------------------------------------------
# Game-of-24, TreeTrainer absorb sweep, Qwen3-0.6B. One command per run.
#
# Each run absorbs a PRE-GROWN pooled buffer, then trains the target seed.
# Sweep axes:
#   (a) grow seeds count X  -> pool SIZE        (GROW_COUNTS)
#   (b) grow seed steps  Y  -> OFF-POLICY-ness  (GROW_STEPS)
#   (c) absorb steps     N                       (ABSORB_STEPS)
#   (d) absorb_clip          none|ppo|m2po@0.04|m2po@0.10  (VARIANTS)
# X,Y just select which pool file to load: $POOL_DIR/X<X>_Y<Y>.json
#
# Pools are grown with the SAME training script (run once per pool beforehand;
# the trie accumulates across grow seeds sharing one --tree-persist-path):
#   for GS in 100 101 102 103; do
#     python tripo_game24.py --trainer tree --use-global-tree \
#       --tree-persist-path $POOL_DIR/X4_Y20.json --seed $GS --max-steps 20 ...
#   done
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
OUT="${OUT:-$HOME/game24_absorb_variants}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
POOL_DIR="${POOL_DIR:-$OUT/pools}"           # pools at $POOL_DIR/X<X>_Y<Y>.json

SEEDS="${SEEDS:-0 1 2 3}"                     # target seeds to absorb+train on
GROW_COUNTS="${GROW_COUNTS:-16 32}"          # axis (a): X grow seeds in the pool
GROW_STEPS="${GROW_STEPS:-5 20}"             # axis (b): Y steps per grow seed
ABSORB_STEPS="${ABSORB_STEPS:-1 2}"          # axis (c): absorb gradient steps
# axis (d): absorb_clip variants "<clip> <tau>" (tau blank for none/ppo)
VARIANTS=(
  "none "
  "ppo "
  "m2po 0.04"
  "m2po 0.10"
)

MAX_STEPS="${MAX_STEPS:-200}"                # training length per target seed
MAX_N="${MAX_N:-13}"
EVAL_MIN="${EVAL_MIN:-200}"
BETA="${BETA:-0.04}"
SCALE="${SCALE:-group}"
MEM="${MEM:-0.55}"
CVD="${CUDA_VISIBLE_DEVICES:-0}"

echo "RUN=$RUN  OUT=$OUT  POOL_DIR=$POOL_DIR"
echo "SEEDS=[$SEEDS]  axes: X={$GROW_COUNTS} Y={$GROW_STEPS} N={$ABSORB_STEPS} clip={${VARIANTS[*]}}"
echo "MAX_STEPS=$MAX_STEPS  CUDA_VISIBLE_DEVICES=$CVD"
mkdir -p "$OUT"

# --- sweep loop: seed x X x Y x absorb-step x clip-variant ------------------
for S in $SEEDS; do
  for X in $GROW_COUNTS; do
    for Y in $GROW_STEPS; do
      POOL="$POOL_DIR/X${X}_Y${Y}.json"
      if [ ! -f "$POOL" ]; then
        echo "!!! skip seed $S X=$X Y=$Y: no pool at $POOL"; continue
      fi
      for N in $ABSORB_STEPS; do
        for V in "${VARIANTS[@]}"; do
          read -r CLIP TAU <<< "$V"
          TAG="$CLIP"; [ -n "$TAU" ] && TAG="${CLIP}${TAU}"
          D="$OUT/seed${S}/X${X}_Y${Y}_step${N}_${TAG}"; mkdir -p "$D"
          # private copy so end-of-run trie write-back never touches the pool
          BUF="$D/tries.json"; cp "$POOL" "$BUF"
          TAU_ARG=(); [ -n "$TAU" ] && TAU_ARG=(--m2po-tau "$TAU")
          echo ">>> [seed $S | X=$X Y=$Y | clip=$CLIP${TAU:+ tau=$TAU} | N=$N | GPU=$CVD] $(date)"
          CUDA_VISIBLE_DEVICES="$CVD" \
          python "$RUN" \
            --trainer tree \
            --output-dir "$D" \
            --model "$MODEL" \
            --seed "$S" \
            --tree-persist-path "$BUF" \
            --absorb-steps "$N" \
            --absorb-clip "$CLIP" "${TAU_ARG[@]}" \
            --absorb-n-pos 1 --absorb-n-neg 1 --absorb-groups-per-query 1 \
            --max-steps "$MAX_STEPS" \
            --num-generations 8 --num-generations-eval 8 \
            --max-completion-length 1024 --no-think \
            --max-n "$MAX_N" --eval-frac 0.40 --eval-min "$EVAL_MIN" --eval-steps 50 \
            --per-device-batch-size 4 --grad-accum 4 \
            --learning-rate 5e-6 --temperature 1.0 \
            --beta "$BETA" --scale-rewards "$SCALE" \
            --vllm-mode colocate --vllm-gpu-memory-utilization "$MEM" \
            > "$D.log" 2>&1 \
            && echo "    DONE   $D" \
            || echo "    FAILED $D  (tail $D.log)"
        done
      done
    done
  done
done

echo
echo "================== all absorb variants done =================="
echo "runs:     $OUT/seed<S>/X<X>_Y<Y>_step<N>_<clip[tau]>/"
echo "rollouts: $OUT/seed<S>/X<X>_Y<Y>_step<N>_<clip[tau]>/{rollouts,eval_rollout}.jsonl"
