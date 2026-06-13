#!/usr/bin/env bash
# Round 2: multi-turn ReAct. The lr sweep showed pass@1 is capped at ~0.55 in
# single-turn regardless of lr/group/model. Hypothesis: giving the model TURNS to
# run a command, read the output, and fix its mistake should lift that ceiling.
set -uo pipefail
cd /home/claudeuser/arl/skyrl_terminal
export HOME=/home/claudeuser
SUMMARY=/tmp/experiments2_summary.txt
: > "$SUMMARY"

cleanup() {
  ps -eo pid,args | grep -E "main_base|ray::|raylet|gcs_server|vllm|EngineCore" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  pkill -9 -f "#!/bin/bash" 2>/dev/null; pkill -9 -f proot 2>/dev/null
  rm -rf /tmp/ray/session_* /tmp/tb_app_* 2>/dev/null
  sleep 8
}

run_one() {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M)] START $name  ($*)" | tee -a "$SUMMARY"
  env "$@" RUN_NAME="$name" EPOCHS="${EPOCHS:-10}" bash run_terminal_bench.sh >/dev/null 2>&1
  cleanup
  local L="/tmp/${name}.$(id -un).log"
  local p1; p1=$(grep "eval/all/pass_at_1" "$L" 2>/dev/null | grep -oE "0\.[0-9]+" | tr '\n' ' ')
  local laststep; laststep=$(grep -oE "'trainer/global_step': [0-9]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  local oom="no"; grep -q "OutOfMemoryError" "$L" 2>/dev/null && oom="YES"
  echo "    [$(date -u +%H:%M)] DONE $name  last_step=${laststep:-?}  OOM=${oom}  pass@1: ${p1:-none}" | tee -a "$SUMMARY"
}

echo "### Terminal-Bench multi-turn round — $(date -u)" | tee -a "$SUMMARY"
run_one tb_mt4_lr5e6   LR=5.0e-6 MAX_TURNS=4
run_one tb_mt6_lr5e6   LR=5.0e-6 MAX_TURNS=6
echo "### ROUND 2 DONE — $(date -u)" | tee -a "$SUMMARY"
