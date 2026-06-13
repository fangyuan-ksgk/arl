#!/usr/bin/env bash
# Sequential config sweep — keeps the single A100 busy back-to-back. Each run is
# the (now leak-free) run_terminal_bench.sh with different knobs; we clean up
# between runs and append a pass@1 trajectory to the summary.
#
#   nohup bash run_experiments.sh > /tmp/experiments.log 2>&1 &
#   watch:  tail -f /tmp/experiments_summary.txt
set -uo pipefail
cd /home/claudeuser/arl/skyrl_terminal
export HOME=/home/claudeuser
SUMMARY=/tmp/experiments_summary.txt
: > "$SUMMARY"

cleanup() {
  ps -eo pid,args | grep -E "main_base|ray::|raylet|gcs_server|vllm|EngineCore" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  pkill -9 -f "#!/bin/bash" 2>/dev/null; pkill -9 -f proot 2>/dev/null
  rm -rf /tmp/ray/session_* /tmp/tb_app_* 2>/dev/null
  sleep 8
}

run_one() {            # run_one <run_name> <ENV=var ...>
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M)] START $name  ($*)" | tee -a "$SUMMARY"
  env "$@" RUN_NAME="$name" EPOCHS="${EPOCHS:-12}" bash run_terminal_bench.sh >/dev/null 2>&1
  cleanup
  local L="/tmp/${name}.$(id -un).log"
  local p1; p1=$(grep "eval/all/pass_at_1" "$L" 2>/dev/null | grep -oE "0\.[0-9]+" | tr '\n' ' ')
  local laststep; laststep=$(grep -oE "'trainer/global_step': [0-9]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  local oom="no"; grep -q "OutOfMemoryError" "$L" 2>/dev/null && oom="YES"
  echo "    [$(date -u +%H:%M)] DONE $name  last_step=${laststep:-?}  OOM=${oom}  pass@1: ${p1:-none}" | tee -a "$SUMMARY"
}

echo "### Terminal-Bench config sweep — $(date -u)" | tee -a "$SUMMARY"
# headline: does higher lr convert pass@8 -> pass@1? (baseline lr=1e-6 stayed flat ~0.55)
run_one tb_lr5e6        LR=5.0e-6
run_one tb_lr1e5        LR=1.0e-5
# more GRPO group signal at the better lr
run_one tb_n12_lr5e6    LR=5.0e-6 N_SAMPLES=12
# smaller model, same lr (capacity vs lr)
run_one tb_1p5b_lr5e6   LR=5.0e-6 MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct
echo "### ALL EXPERIMENTS DONE — $(date -u)" | tee -a "$SUMMARY"
