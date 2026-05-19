#!/usr/bin/env bash
# Smoke test for the TRL + vLLM + DeepSpeed ZeRO-3 stack on this RunPod box.
#
# Goal: verify that the trl vllm-serve <-> GRPOTrainer NCCL handshake works
# end-to-end before kicking off the full sweep. Uses the smallest model in
# the planned sweep, a single CoT length, and just a handful of steps.
#
# The critical bit is exporting the NCCL environment so both
#   * the trl vllm-serve background process, and
#   * the `accelerate launch ... run_game24_deepspeed.py` workers
# see them. This avoids the P2P/IPC hang we hit on RunPod (no --ipc=host).
#
# Usage:
#     bash script/smoke_test_deepspeed.sh
#     MODEL="Qwen/Qwen3-0.6B" LEN=1024 bash script/smoke_test_deepspeed.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- NCCL workarounds for containerised single-host runs -------------------
# These fix the "init succeeds, first collective hangs" symptom caused by
# CUDA-IPC handles not crossing the container's IPC namespace.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"

# --- smoke-test scope ------------------------------------------------------
export MODELS="${MODEL:-Qwen/Qwen3-0.6B}"
export LENGTHS="${LEN:-1024}"
export MAX_STEPS="${MAX_STEPS:-5}"
export EVAL_STEPS="${EVAL_STEPS:-0}"        # skip eval; we only care about handshake + 1 step
export OUTPUT_ROOT="${OUTPUT_ROOT:-output/game24_smoke_ds}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"        # MUST match configs/zero3.yaml
export SCORE_VT="${SCORE_VT:-0}"            # skip Vt scoring on smoke
export KEEP_CKPT="${KEEP_CKPT:-0}"

export VLLM_GPU="${VLLM_GPU:-0}"
export TRAIN_GPUS="${TRAIN_GPUS:-1,2}"

echo "===================== smoke test config ====================="
echo "  model         : ${MODELS}"
echo "  length        : ${LENGTHS}"
echo "  max_steps     : ${MAX_STEPS}"
echo "  eval_steps    : ${EVAL_STEPS}"
echo "  vllm_gpu      : ${VLLM_GPU}"
echo "  train_gpus    : ${TRAIN_GPUS}"
echo "  output_root   : ${OUTPUT_ROOT}"
echo "  NCCL_P2P_DISABLE  = ${NCCL_P2P_DISABLE}"
echo "  NCCL_CUMEM_ENABLE = ${NCCL_CUMEM_ENABLE}"
echo "  NCCL_DEBUG        = ${NCCL_DEBUG}"
echo "============================================================="
echo

# --- pre-flight: GPU + NCCL sanity -----------------------------------------
echo "[smoke] pre-flight checks"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo

python - <<'PY'
import torch
print("[smoke] torch:", torch.__version__, "cuda:", torch.version.cuda)
print("[smoke] nccl:", torch.cuda.nccl.version(), "available:", torch.distributed.is_nccl_available())
print("[smoke] device count:", torch.cuda.device_count())
PY
echo

# --- ensure no leftover vllm processes from prior runs ---------------------
echo "[smoke] cleaning stale vllm processes"
pkill -9 -f "trl vllm-serve" 2>/dev/null || true
pkill -9 -f "vllm_serve_minimal" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 2
echo "[smoke] port 8000 status:"
ss -ltn | grep ':8000 ' || echo "  port free"
echo

# --- delegate to the real sweep script with the smoke overrides ------------
echo "[smoke] handing off to run_game24_sweep_deepspeed.sh"
echo
bash "${SCRIPT_DIR}/run_game24_sweep_deepspeed.sh"
rc=$?

echo
if [[ $rc -eq 0 ]]; then
  echo "===================== smoke test PASSED ====================="
  echo "Sweep should be safe to run with the same NCCL env exported."
else
  echo "===================== smoke test FAILED (rc=${rc}) ============"
  echo "Inspect ${OUTPUT_ROOT}/len${LENGTHS}/logs/  for trainer + vllm logs."
  echo "Common causes:"
  echo "  * NCCL still hung   -> check log for 'via P2P/IPC' (should be 'via SHM')"
  echo "  * vllm timeout      -> increase VLLM_STARTUP_TIMEOUT"
  echo "  * batch shape error -> check num_generations vs effective batch"
fi
exit $rc
