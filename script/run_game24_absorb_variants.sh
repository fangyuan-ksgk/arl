#!/bin/bash
# run_game24_absorb_variants.sh
# ---------------------------------------------------------------------------
# Game-of-24, TreeTrainer absorb sweep, Qwen3-0.6B.
#
# For EACH target seed we re-spawn a FRESH pool from scratch (no recycling
# across seeds): grow X seeds (Y steps each, accumulating into one trie), then
# absorb that pool and train the target seed. Grow seeds are made per-target
# distinct (1000*S + i) so each seed's pool is an independent draw, not a
# byte-identical re-run.
#
# Sweep axes:
#   (a) grow seeds count X  -> pool SIZE        (GROW_COUNTS)
#   (b) grow seed steps  Y  -> OFF-POLICY-ness  (GROW_STEPS)
#   (c) absorb steps     N                       (ABSORB_STEPS)
#   (d) absorb_clip          none|ppo|m2po@0.04|m2po@0.10  (VARIANTS)
# Within a seed, the grow accumulation is snapshotted at each X (one grow
# sequence covers every X); pools are NEVER shared between seeds.
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

MAX_STEPS="${MAX_STEPS:-200}"                # training length per target seed
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

echo "RUN=$RUN  OUT=$OUT"
echo "SEEDS=[$SEEDS]  axes: X={$GROW_COUNTS} Y={$GROW_STEPS} N={$ABSORB_STEPS} clip={${VARIANTS[*]}}"
echo "MAX_STEPS=$MAX_STEPS  CUDA_VISIBLE_DEVICES=$CVD"
mkdir -p "$OUT"

for S in $SEEDS; do
  SDIR="$OUT/seed${S}"; PDIR="$SDIR/pools"; mkdir -p "$PDIR"

  # === grow FRESH per-seed pools (one accumulation per Y, snapshot at each X) =
  for Y in $GROW_STEPS; do
    ACCUM="$PDIR/_accum_Y${Y}.json"; rm -f "$ACCUM"
    i=0
    while [ "$i" -lt "$MAXX" ]; do
      i=$((i + 1))
      GSEED=$((1000 * S + i))               # per-target distinct -> independent pool
      GD="$SDIR/grow/Y${Y}/g${i}"; mkdir -p "$GD"
      echo ">>> [GROW seed $S | Y=$Y | pool size -> $i (gseed $GSEED) | GPU=$CVD] $(date)"
      CUDA_VISIBLE_DEVICES="$CVD" \
      python "$RUN" \
        --trainer tree --use-global-tree \
        --tree-persist-path "$ACCUM" \
        --output-dir "$GD" --seed "$GSEED" --max-steps "$Y" \
        --eval-steps 0 \
        "${COMMON[@]}" \
        > "$GD.log" 2>&1 \
        && echo "    DONE   grow $i (Y=$Y)" \
        || echo "    FAILED grow $i (Y=$Y)  (tail $GD.log)"
      for X in $GROW_COUNTS; do
        [ "$i" -eq "$X" ] && cp "$ACCUM" "$PDIR/X${X}_Y${Y}.json"
      done
    done
  done

  # === absorb sweep against THIS seed's fresh pools ==========================
  for X in $GROW_COUNTS; do
    for Y in $GROW_STEPS; do
      POOL="$PDIR/X${X}_Y${Y}.json"
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
echo "pools (per seed): $OUT/seed<S>/pools/X<X>_Y<Y>.json"
echo "runs:            $OUT/seed<S>/X<X>_Y<Y>_step<N>_<clip[tau]>/"
