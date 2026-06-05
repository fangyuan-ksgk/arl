#!/bin/bash
# run_tripo_game24.sh — reproduce the GRPO-vs-TriPO Game-of-24 main tables.
# Self-contained: place in arl/script/ and run from anywhere. No helper files needed.
#
#   bash script/run_tripo_game24.sh                    # from arl/: full 2 models x 5 configs x 3 seeds
#   SEEDS="0" bash script/run_tripo_game24.sh          # quick one-seed run
#   bash script/run_tripo_game24.sh --tables-only      # just (re)print tables from $OUT
#
# Runs one lane per model, each on its own GPU, in parallel; prints per-model
# tables (greedy@1, sample pass@8, collapse rate) at the end.
#
# Env knobs (defaults): OUT=$HOME/tripo_game24_results  BETA=0.04  SCALE=group
#   SEEDS="0 1 2"  MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B"
#   CONFIGS="grpo tripo-flat tripo-tree tripo-flat-min tripo-tree-min"   (also: grpo-tree)
set -uo pipefail

# --- resolve paths relative to THIS file so it runs from any cwd ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../arl/script
ARL="${ARL:-$(dirname "$SCRIPT_DIR")}"                       # .../arl
RUN="${RUN:-$SCRIPT_DIR/tripo_game24.py}"
export PYTHONPATH="$ARL${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="${OUT:-$HOME/tripo_game24_results}"
BETA="${BETA:-0.04}"; SCALE="${SCALE:-group}"
SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B}"
CONFIGS="${CONFIGS:-grpo tripo-flat tripo-tree tripo-flat-min tripo-tree-min}"

# ----------------------------------------------------------------------------
print_tables () {   # $1 = results root
python3 - "$1" <<'PY'
import sys, os, glob, json, statistics as st
ROOT = sys.argv[1]
CONFIGS = ["grpo","tripo-flat","tripo-tree","tripo-flat-min","tripo-tree-min","grpo-tree"]
def metrics(vdir):
    p = os.path.join(vdir, "eval_rollout.jsonl")
    if not os.path.exists(p): return None
    recs = [json.loads(l) for l in open(p)]
    recs = [r for r in recs if r.get("eval_dataset") == "eval"]
    if not recs: return None
    last = max(r["global_step"] for r in recs)
    g  = [r for r in recs if r["global_step"]==last and r["decoding"]=="greedy"]
    sa = [r for r in recs if r["global_step"]==last and r["decoding"]=="sample"]
    greedy = st.mean([r["correct"] for r in g]) if g else None
    byp = {}
    for r in sa: byp.setdefault(tuple(r["numbers"]), []).append(r["correct"])
    passk = st.mean([1.0 if any(v) else 0.0 for v in byp.values()]) if byp else None
    return greedy, passk
def ms(xs):
    xs=[x for x in xs if x is not None]
    return (st.mean(xs), st.pstdev(xs) if len(xs)>1 else 0.0) if xs else None
def f(t): return f"{t[0]:.3f}±{t[1]:.3f}" if t else "   -   "
seed_dirs = sorted(glob.glob(f"{ROOT}/s*"))
models = sorted({os.path.basename(p) for sd in seed_dirs for p in glob.glob(f"{sd}/*") if os.path.isdir(p)})
if not models:
    print(f"  (no runs found under {ROOT}/s*/<model>/<config>/)"); sys.exit(0)
for m in models:
    print(f"\n=========== {m}   (root={ROOT}) ===========")
    print(f"{'config':16s} {'greedy@1':>14} {'pass@8':>14} {'collapse':>9}  per-seed pass@8")
    for c in CONFIGS:
        gs=[]; ps=[]
        for sd in seed_dirs:
            r = metrics(f"{sd}/{m}/{c}")
            if r: gs.append(r[0]); ps.append(r[1])
        if not ps: continue
        coll = sum(1 for x in ps if x is not None and x < 0.20)/len(ps)
        per  = [round(x or 0,2) for x in ps]
        print(f"{c:16s} {f(ms(gs)):>14} {f(ms(ps)):>14} {coll*100:>7.0f}%  {per}")
print()
PY
}

cfg_args () { case "$1" in
  grpo)           echo "--trainer grpo" ;;
  grpo-tree)      echo "--trainer grpo --tree-sampling --tree-branch 2 --tree-steps 3" ;;
  tripo-flat)     echo "--trainer tree --credit-mode max" ;;
  tripo-tree)     echo "--trainer tree --credit-mode max --tree-sampling --tree-branch 2 --tree-steps 3" ;;
  tripo-flat-min) echo "--trainer tree --credit-mode min" ;;
  tripo-tree-min) echo "--trainer tree --credit-mode min --tree-sampling --tree-branch 2 --tree-steps 3" ;;
  *) echo "BAD_CONFIG_$1" ;;
esac; }

run_lane () {   # $1=model  $2=gpu
  local M=$1 GPU=$2 SLUG PDBS GA MEM S C D
  SLUG=${M//\//__}
  case "$M" in *1.7B*|*1\.7B*) PDBS=2; GA=8; MEM=0.45 ;; *) PDBS=4; GA=4; MEM=0.55 ;; esac
  for S in $SEEDS; do
    for C in $CONFIGS; do
      D="$OUT/s$S/$SLUG/$C"; mkdir -p "$D"
      echo ">>> [$M | gpu$GPU | seed $S | $C] $(date)"
      CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$((29600 + GPU + 7*S)) \
      python "$RUN" --output-dir "$D" --model "$M" \
        --max-steps 120 --num-generations 8 --num-generations-eval 8 \
        --max-completion-length 1024 --no-think --max-n 6 --eval-frac 0.5 --eval-min 40 \
        --eval-steps 75 --per-device-batch-size "$PDBS" --grad-accum "$GA" \
        --learning-rate 5e-6 --temperature 1.0 --beta "$BETA" --scale-rewards "$SCALE" \
        --vllm-mode colocate --vllm-gpu-memory-utilization "$MEM" --seed "$S" \
        $(cfg_args "$C") > "$D.log" 2>&1 \
        && echo "    DONE  $M $C s$S" || echo "    FAILED $M $C s$S  (tail $D.log)"
    done
  done
}

# --- tables-only mode ---
if [ "${1:-}" = "--tables-only" ]; then
  print_tables "${2:-$OUT}"; exit 0
fi

echo "RUN=$RUN"; echo "OUT=$OUT  BETA=$BETA  SCALE=$SCALE  SEEDS=[$SEEDS]"
gpu=0
for M in $MODELS; do
  run_lane "$M" "$gpu" &
  gpu=$((gpu + 1))
done
wait

echo; echo "==================  MAIN TABLES (beta=$BETA, scale=$SCALE)  =================="
print_tables "$OUT"
echo "results: $OUT/   |  re-print: bash $SCRIPT_DIR/run_tripo_game24.sh --tables-only $OUT"
