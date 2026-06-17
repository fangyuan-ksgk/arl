#!/bin/bash
# run_game24_absorb_variants.sh
# ---------------------------------------------------------------------------
# Game-of-24, TreeTrainer absorb sweep, Qwen3-0.6B.
#
# Pools are grown ONCE up front and SHARED across every absorb experiment
# (growing dominates: ~7 min/grow-seed, so per-seed regrow = 7*MAXX min per
# target, far exceeding the ~hour absorb run). Phase 1 grows MAXX seeds per Y
# (accumulating into one trie, snapshotted at each X) into $POOL_DIR; Phase 2
# loops the target seeds and absorbs those shared pools.
#
# Sweep axes:
#   (a) grow seeds count X  -> pool SIZE        (GROW_COUNTS)
#   (b) grow seed steps  Y  -> OFF-POLICY-ness  (GROW_STEPS)
#   (c) absorb steps     N                       (ABSORB_STEPS)
#   (d) absorb_clip          none|ppo|m2po@0.04|m2po@0.10  (VARIANTS)
# Grow cost is paid once (MAXX*|GROW_STEPS| runs total), not per target seed.
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
POOL_DIR="${POOL_DIR:-$OUT/pools}"           # shared pools: $POOL_DIR/X<X>_Y<Y>.json

SEEDS="${SEEDS:-0 1 2 3}"                     # target seeds to absorb+train on
GROW_COUNTS="${GROW_COUNTS:-16 32}"          # axis (a): X grow seeds in the pool (any value)
GROW_STEPS="${GROW_STEPS:-5 20}"             # axis (b): Y steps per grow seed
ABSORB_STEPS="${ABSORB_STEPS:-1 2}"          # axis (c): absorb gradient steps
# axis (d): absorb_clip variants "<clip> <tau>" (tau blank for none/ppo)
VARIANTS=(
  "none "
  "ppo "
  "m2po 0.04"
  "m2po 0.10"
)

MAX_STEPS="${MAX_STEPS:-120}"                # training length per target seed
MAX_N="${MAX_N:-13}"
EVAL_MIN="${EVAL_MIN:-200}"
BETA="${BETA:-0.04}"
SCALE="${SCALE:-group}"
MEM="${MEM:-0.55}"
CVD="${CUDA_VISIBLE_DEVICES:-0}"

# largest requested pool size -> grow seeds per target = 1..MAXX
MAXX="$(echo "$GROW_COUNTS" | tr ' ' '\n' | sort -n | tail -1)"

# shared python args (grow + absorb); eval added per-call below
COMMON=(--model "$MODEL"
        --num-generations 8 --num-generations-eval 8
        --max-completion-length 1024 --no-think
        --max-n "$MAX_N"
        --per-device-batch-size 4 --grad-accum 4
        --learning-rate 5e-6 --temperature 1.0
        --beta "$BETA" --scale-rewards "$SCALE"
        --vllm-mode colocate --vllm-gpu-memory-utilization "$MEM")

echo "RUN=$RUN  OUT=$OUT  POOL_DIR=$POOL_DIR"
echo "SEEDS=[$SEEDS]  axes: X={$GROW_COUNTS} Y={$GROW_STEPS} N={$ABSORB_STEPS} clip={${VARIANTS[*]}}"
echo "MAX_STEPS=$MAX_STEPS  CUDA_VISIBLE_DEVICES=$CVD"
mkdir -p "$OUT" "$POOL_DIR"

# === Phase 1: grow SHARED pools ONCE (one accumulation per Y, snapshot at X) =
# Reused by every target seed below. Skips a pool that already exists, so the
# script is resumable and grow is never repeated.
for Y in $GROW_STEPS; do
  # if every X-snapshot for this Y is already on disk, skip the whole Y
  need=0; for X in $GROW_COUNTS; do [ -f "$POOL_DIR/X${X}_Y${Y}.json" ] || need=1; done
  if [ "$need" -eq 0 ]; then echo "=== pools for Y=$Y already present, skip grow ==="; continue; fi

  ACCUM="$POOL_DIR/_accum_Y${Y}.json"; rm -f "$ACCUM"
  i=0
  while [ "$i" -lt "$MAXX" ]; do
    i=$((i + 1))
    GD="$POOL_DIR/grow/Y${Y}/g${i}"; mkdir -p "$GD"
    echo ">>> [GROW Y=$Y | pool size -> $i (gseed $i) | GPU=$CVD] $(date)"
    CUDA_VISIBLE_DEVICES="$CVD" \
    python "$RUN" \
      --trainer tree --use-global-tree \
      --tree-persist-path "$ACCUM" \
      --output-dir "$GD" --seed "$i" --max-steps "$Y" \
      --eval-steps 0 \
      "${COMMON[@]}" \
      > "$GD.log" 2>&1 \
      && echo "    DONE   grow $i (Y=$Y)" \
      || echo "    FAILED grow $i (Y=$Y)  (tail $GD.log)"
    for X in $GROW_COUNTS; do
      [ "$i" -eq "$X" ] && cp "$ACCUM" "$POOL_DIR/X${X}_Y${Y}.json"
    done
  done
done

# === Phase 2: absorb sweep over target seeds against the SHARED pools ========
for S in $SEEDS; do
  SDIR="$OUT/seed${S}"
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
          D="$SDIR/X${X}_Y${Y}_step${N}_${TAG}"; mkdir -p "$D"
          BUF="$D/tries.json"; cp "$POOL" "$BUF"   # private copy (write-back safe)
          TAU_ARG=(); [ -n "$TAU" ] && TAU_ARG=(--m2po-tau "$TAU")
          echo ">>> [ABSORB seed $S | X=$X Y=$Y | clip=$CLIP${TAU:+ tau=$TAU} | N=$N | GPU=$CVD] $(date)"
          CUDA_VISIBLE_DEVICES="$CVD" \
          python "$RUN" \
            --trainer tree \
            --output-dir "$D" --seed "$S" \
            --tree-persist-path "$BUF" \
            --absorb-steps "$N" --absorb-clip "$CLIP" "${TAU_ARG[@]}" \
            --absorb-n-pos 1 --absorb-n-neg 1 --absorb-groups-per-query 1 \
            --max-steps "$MAX_STEPS" \
            --eval-frac 0.40 --eval-min "$EVAL_MIN" --eval-steps 50 \
            "${COMMON[@]}" \
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
echo "shared pools: $POOL_DIR/X<X>_Y<Y>.json"
echo "runs:         $OUT/seed<S>/X<X>_Y<Y>_step<N>_<clip[tau]>/"
