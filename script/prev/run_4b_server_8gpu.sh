#!/bin/bash
# ============================================================================
# 4B sweep + thin lottery — SERVER mode, 8 GPUs = 4 parallel (train+serve) pairs.
# Depends ONLY on the arl repo (script/tripo_game24.py); no tripo_run helpers.
#
#   bash run_4b_server_8gpu.sh
#   OUT=/path/to/target  PAIRS="0:1 2:3 4:5 6:7"  bash run_4b_server_8gpu.sh
#
# Each pair = train_gpu:serve_gpu -> its OWN persistent vLLM server (serve_gpu) +
# training (train_gpu) hitting that server. 4 pairs run in parallel.
#
# OUTPUT (written by the trainer into the target folder $OUT):
#   $OUT/sweep_base/<cfg>_4B_s<seed>/Qwen__Qwen3-4B/run/rollouts.jsonl      <- TRAIN
#   $OUT/sweep_base/<cfg>_4B_s<seed>/Qwen__Qwen3-4B/run/eval_rollout.jsonl  <- EVAL
#   $OUT/lottery_4b/<buf>_a1_s<seed>/Qwen__Qwen3-4B/run/{rollouts,eval_rollout}.jsonl
#   + a .../run.log per run, $OUT/progress.log live, per-pair server_g<gpu>.log
# ============================================================================
set -uo pipefail
ARL=/home/claudeuser/arl
RUN=$ARL/script/tripo_game24.py
export PYTHONPATH=$ARL HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ARL"

# ---- config -----------------------------------------------------------------
OUT=${OUT:-/home/claudeuser/tripo_run}            # TARGET folder for results (created if missing)
MODEL=Qwen/Qwen3-4B ; SLUG=Qwen__Qwen3-4B
read -r -a PAIR_ARR <<< "${PAIRS:-0:1 2:3 4:5 6:7}"   # train:serve pairs (4 parallel)
SRV_UTIL=${SRV_UTIL:-0.85} ; SRV_WAIT=${SRV_WAIT:-300}
OUT_SWEEP=$OUT/sweep_base ; OUT_LOT=$OUT/lottery_4b ; WORK=$OUT_LOT/buf_work.json
QUEUE=$OUT/.4b_queue.txt ; LOCK=$OUT/.4b_queue.lock ; PROGRESS=$OUT/progress.log
mkdir -p "$OUT" "$OUT_SWEEP" "$OUT_LOT"           # create target folders FIRST (no pre-existing dir assumed)

COMMON="--no-think --num-generations 8 --num-generations-eval 8 --max-completion-length 1024 \
--max-steps 120 --max-n 6 --eval-frac 0.5 --eval-min 40 --eval-steps 120 --learning-rate 5e-6 \
--temperature 1.0 --beta 0.04 --scale-rewards group"
MEM="--per-device-batch-size 1 --grad-accum 16 --model-dtype bfloat16 --optim paged_adamw_8bit"
GROW="--no-think --trainer tree --credit-mode base --buffered-baseline --num-generations 8 \
--max-completion-length 512 --max-steps 10 --max-n 6 --eval-frac 0.5 --eval-min 40 --eval-steps 0 \
--learning-rate 5e-6 --temperature 1.0 --beta 0.04 --scale-rewards group"

declare -A CFG=(
 [grpo]="--trainer grpo"
 [base]="--trainer tree --credit-mode base"
 [buf_base]="--trainer tree --credit-mode base --buffered-baseline"
 [inj_roll]="--trainer tree --credit-mode base --inject-rollout"
 [inj_inc]="--trainer tree --credit-mode base --inject-rollout --inject-incorrect"
 [resamp]="--trainer tree --credit-mode base --resample-prefix"
 [insert_max]="--trainer tree --credit-mode base --virtual-rollout insert_max"
 [insert_min]="--trainer tree --credit-mode base --virtual-rollout insert_min"
 [insert_max_min]="--trainer tree --credit-mode base --virtual-rollout insert_max_min"
)
CFG_ORDER="grpo base buf_base inj_roll inj_inc resamp insert_max insert_min insert_max_min"
SEEDS="0 1 2 3"
freeport(){ python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'; }

# ---- build the work queue (TAB: TYPE \t outdir \t args \t srcbuf) ------------
: > "$QUEUE"; : > "$PROGRESS"                       # ($OUT/sweep_base/lottery_4b already created above)
printf 'GROW\t-\t-\t-\n' >> "$QUEUE"                                # one serial grow megajob (32 seeds)
for c in $CFG_ORDER; do for s in $SEEDS; do
  printf 'RUN\t%s\t%s --seed %s\t-\n' "$OUT_SWEEP/${c}_4B_s${s}/$SLUG/run" "${CFG[$c]}" "$s" >> "$QUEUE"
done; done
for buf in buf4 buf16 buf32; do
  if [ "$buf" = buf16 ]; then BS="0 1 2 3"; else BS="0 1"; fi          # buf16 -> 4 seeds (table-cell trust)
  for s in $BS; do
    d="$OUT_LOT/${buf}_a1_s${s}/$SLUG/run"
    printf 'ABSORB\t%s\t--tree-persist-path %s --absorb-steps 1 --absorb-groups-per-query 2 --seed %s --max-steps 119 --eval-steps 119\t%s\n' \
      "$d" "$OUT_LOT/trie_${buf}_a1_s${s}.json" "$s" "$OUT_LOT/${buf}.json" >> "$QUEUE"
  done
done
echo "[queue] $(grep -c . "$QUEUE") items: 1 grow + $(echo "$CFG_ORDER"|wc -w)x4 sweep + (buf4:2 buf16:4 buf32:2) absorb"

# ---- start a vLLM server on a serve GPU; echo its port ----------------------
start_server(){ local sg=$1 port log; port=$(freeport); log="$OUT/server_g${sg}.log"
  CUDA_VISIBLE_DEVICES=$sg nohup trl vllm-serve --model "$MODEL" --host 127.0.0.1 --port "$port" \
    --gpu_memory_utilization "$SRV_UTIL" --enforce_eager True > "$log" 2>&1 &
  echo $! > "$OUT/.srv_g${sg}.pid"
  local n=$((SRV_WAIT/5)); for _ in $(seq 1 "$n"); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "$port"; return 0; }
    kill -0 "$(cat "$OUT/.srv_g${sg}.pid")" 2>/dev/null || return 1; sleep 5; done
  return 1; }

# ---- serial grow of the 4B buffer (32 seeds -> snapshot buf4/16/32) ----------
do_grow(){ local tg=$1 srv="$2" i nm gd
  rm -f "$WORK" "$OUT_LOT"/buf4.json "$OUT_LOT"/buf16.json "$OUT_LOT"/buf32.json
  for i in $(seq 0 31); do gd="$OUT_LOT/grow_s${i}/$SLUG/run"; mkdir -p "$gd"
    CUDA_VISIBLE_DEVICES=$tg MASTER_PORT=$(freeport) python "$RUN" --output-dir "$gd" --model "$MODEL" \
      $GROW $MEM $srv --tree-persist-path "$WORK" --seed "$i" > "$gd.log" 2>&1 || echo "  grow s$i FAILED" >> "$PROGRESS"
    case $((i+1)) in 4) nm=buf4;; 16) nm=buf16;; 32) nm=buf32;; *) nm="";; esac
    [ -n "$nm" ] && cp "$WORK" "$OUT_LOT/$nm.json" && echo "[grow] snapshot $nm ($((i+1)) seeds) $(date)" >> "$PROGRESS"
  done; echo "[grow] DONE 32 seeds $(date)" >> "$PROGRESS"; }

# ---- pair worker: own server, then pop+run jobs until queue empty -----------
pair_worker(){ local tg=$1 sg=$2 PORT SRVPID SRV line type outdir args src trie tag
  PORT=$(start_server "$sg") || { echo "[pair $tg:$sg] SERVER FAILED $(date)" >> "$PROGRESS"; return 1; }
  SRVPID=$(cat "$OUT/.srv_g${sg}.pid"); SRV="--vllm-mode server --vllm-server-host 127.0.0.1 --vllm-server-port $PORT"
  echo "[pair $tg:$sg] server ready port $PORT $(date)" >> "$PROGRESS"
  while :; do
    line=$(flock "$LOCK" bash -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'")
    [ -z "$line" ] && break
    IFS=$'\t' read -r type outdir args src <<< "$line"
    if [ "$type" = GROW ]; then echo "[pair $tg:$sg] GROW (32 seeds) $(date)" >> "$PROGRESS"; do_grow "$tg" "$SRV"; continue; fi
    mkdir -p "$outdir"; tag=$(basename "$(dirname "$(dirname "$outdir")")")
    if [ "$type" = ABSORB ]; then trie=$(echo "$args"|sed -n 's/.*--tree-persist-path \([^ ]*\).*/\1/p'); cp "$src" "$trie"; fi
    echo "[pair $tg:$sg] START $tag $(date)" >> "$PROGRESS"
    CUDA_VISIBLE_DEVICES=$tg MASTER_PORT=$(freeport) python "$RUN" --output-dir "$outdir" --model "$MODEL" \
      $COMMON $MEM $SRV $args > "$outdir.log" 2>&1 \
      && echo "[pair $tg:$sg] DONE  $tag $(date)" >> "$PROGRESS" \
      || echo "[pair $tg:$sg] FAILED $tag $(date)" >> "$PROGRESS"
  done
  kill -9 "$SRVPID" 2>/dev/null; echo "[pair $tg:$sg] retired $(date)" >> "$PROGRESS"; }

# ---- launch the 4 pairs in parallel, wait for all --------------------------
for p in $(pgrep -f "vllm-serve"); do kill -9 "$p" 2>/dev/null; done   # clear any stale servers
echo "[start] ${#PAIR_ARR[@]} pairs: ${PAIR_ARR[*]} | OUT=$OUT $(date)"
for p in "${PAIR_ARR[@]}"; do IFS=':' read -r tg sg <<< "$p"; pair_worker "$tg" "$sg" & done
wait
echo "[ALL DONE] $(date)"
echo "TRAIN/EVAL rollouts in:  $OUT_SWEEP/<cfg>_4B_s<seed>/$SLUG/run/{rollouts,eval_rollout}.jsonl"
echo "             lottery:    $OUT_LOT/<buf>_a1_s<seed>/$SLUG/run/{rollouts,eval_rollout}.jsonl"
touch "$OUT/SWEEP_4B_DONE"
