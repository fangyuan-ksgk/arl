#!/usr/bin/env bash
# Wait for the running terminal GRPO job to release the A100, then auto-launch a
# fast Qwen3.5-4B geo3k validation to confirm the image-format fix
# (_normalize_mm_content) lets the multi-turn rollout/eval render without crashing.
#
# The geo3k script runs eval_before_train=true, and eval uses the same
# agent_loop -> /render path that previously crashed, so the first eval already
# proves the fix; we add EPOCHS=1 to also exercise the training rollout.
set -u
TERMINAL_PID="${1:-1330321}"
LOG=/tmp/geo3k_qwen35_validation.log
WATCH_LOG=/tmp/watch_then_validate.log

echo "[$(date -u +%T)] watching terminal pid $TERMINAL_PID ..." > "$WATCH_LOG"
while kill -0 "$TERMINAL_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date -u +%T)] terminal pid gone; waiting for GPU mem to drain ..." >> "$WATCH_LOG"

# Wait until GPU memory is mostly free (< 5 GB) before grabbing it.
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "[$(date -u +%T)] gpu used=${used}MiB" >> "$WATCH_LOG"
  if [ "${used:-99999}" -lt 5000 ]; then break; fi
  sleep 15
done

echo "[$(date -u +%T)] launching geo3k Qwen3.5-4B validation -> $LOG" >> "$WATCH_LOG"
cd /home/claudeuser/arl/skyrl_terminal
# Fast validation: 1 epoch, small eval set. Defaults to MODEL_PATH=Qwen/Qwen3.5-4B.
EPOCHS=1 RUN_NAME=geo3k_qwen35_validate bash run_geo3k_1gpu.sh \
  data.val_data="['$HOME/data/geometry_3k/test_small.parquet']" \
  trainer.eval_batch_size=64 \
  > "$LOG" 2>&1
echo "[$(date -u +%T)] validation run exited rc=$?" >> "$WATCH_LOG"
