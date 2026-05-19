#!/usr/bin/env bash
# Sweep run_game24_deepspeed.py over (CoT-budget × model) grid using a
# 3-GPU split: vLLM server on one GPU, ZeRO-3 training across the remaining
# two GPUs via `accelerate launch --config_file configs/zero3.yaml`.
#
# Each (length, model) cell runs in its own accelerate launch so the OS
# reclaims all GPU memory between runs.
#
# Per-length results land in:
#     ${OUTPUT_ROOT}/len${LEN}/<model_slug>/...
#     ${OUTPUT_ROOT}/len${LEN}/summary.csv
# and a combined ${OUTPUT_ROOT}/summary_all.csv is written at the end with an
# extra `max_completion_length` column.
#
# Usage:
#   bash script/run_game24_sweep_deepspeed.sh                                # default
#   LENGTHS="1024 2048" bash script/run_game24_sweep_deepspeed.sh
#   MAX_STEPS=30 EVAL_STEPS=30 bash script/run_game24_sweep_deepspeed.sh    # smoke test
#   VLLM_GPU=0 TRAIN_GPUS="1,2" bash script/run_game24_sweep_deepspeed.sh

set -u

# --- locate repo root -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- NCCL workarounds for containerised single-host runs -------------------
# RunPod (and most Docker hosts without --ipc=host) block CUDA-IPC handles
# from crossing the IPC namespace, so NCCL's P2P/IPC transport sets up but
# the first collective hangs forever. Disabling P2P forces NCCL to use SHM
# instead -- still fully NCCL, just a different intra-node transport.
# Override by exporting these before invoking the script.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# --- defaults (override via env) -------------------------------------------
MODELS="${MODELS:-\
Qwen/Qwen3-0.6B \
Qwen/Qwen3-1.7B \
Qwen/Qwen3-4B \
meta-llama/Llama-3.2-3B-Instruct}"

LENGTHS="${LENGTHS:-1024 2048}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/game24_sweep_ds}"
MAX_STEPS="${MAX_STEPS:-1200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SCORE_VT="${SCORE_VT:-1}"
VT_MICRO_BATCH="${VT_MICRO_BATCH:-8}"
RT_PAIR_SEED="${RT_PAIR_SEED:-0}"
SEED="${SEED:-0}"
KEEP_CKPT="${KEEP_CKPT:-0}"  # 1 = retain trained_for_vt checkpoint dir

# --- GPU split + DeepSpeed launcher ----------------------------------------
# vLLM server runs on VLLM_GPU. Training is spread across TRAIN_GPUS (the
# comma-list passed as CUDA_VISIBLE_DEVICES) via accelerate + ZeRO-3.
VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2}"
# Number of training processes — must equal the count of TRAIN_GPUS and the
# `num_processes` entry in the accelerate yaml.
NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F, '{print NF}' <<< "${TRAIN_GPUS}")}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/zero3.yaml}"

# pdbs × grad_accum × num_processes must be a multiple of NUM_GENERATIONS.
EFFECTIVE_BATCH=$((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * NUM_PROCESSES))
if (( EFFECTIVE_BATCH % NUM_GENERATIONS != 0 )); then
  echo "ERROR: effective batch ${EFFECTIVE_BATCH} (pdbs×grad_accum×num_proc) "
  echo "       is not a multiple of NUM_GENERATIONS=${NUM_GENERATIONS}."
  echo "       Adjust PER_DEVICE_BATCH_SIZE, GRAD_ACCUM, NUM_GENERATIONS, or NUM_PROCESSES."
  exit 1
fi

# --- vLLM server settings --------------------------------------------------
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"

DS_SCRIPT="script/run_game24_deepspeed.py"

mkdir -p "${OUTPUT_ROOT}"

echo "Sweep config (DeepSpeed ZeRO-3):"
echo "  output_root  = ${OUTPUT_ROOT}"
echo "  lengths      = ${LENGTHS}"
echo "  max_steps    = ${MAX_STEPS}"
echo "  eval_steps   = ${EVAL_STEPS}"
echo "  num_gen      = ${NUM_GENERATIONS}"
echo "  vllm_gpu     = ${VLLM_GPU}"
echo "  train_gpus   = ${TRAIN_GPUS}  (num_processes=${NUM_PROCESSES})"
echo "  accel_config = ${ACCELERATE_CONFIG}"
echo "  models       ="
for m in ${MODELS}; do echo "    - ${m}"; done
echo

# --- vLLM server lifecycle helpers (same as single-GPU sweep) --------------
VLLM_PID=""
start_vllm_server() {
  local model="$1"; local max_len="$2"; local log_file="$3"
  stop_vllm_server
  echo "  [vllm] starting server  model=${model}  (max-model-len: model default)  gpu=${VLLM_GPU}"
  # NOTE: previously we pinned --max-model-len to the CoT budget (${max_len}).
  # That capped prompt+completion together, which truncated long-CoT runs
  # below --max-completion-length. For the len=2048 sweep we let the server
  # use the model's native context window so completion budget is honoured.
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    setsid trl vllm-serve --model "${model}" \
      --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
      --enforce-eager \
      > "${log_file}" 2>&1 &
  VLLM_PID=$!
  echo "  [vllm] pid=${VLLM_PID} (pgid=${VLLM_PID})  log=${log_file}"
  local waited=0
  while (( waited < VLLM_STARTUP_TIMEOUT )); do
    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
      echo "  [vllm] ready after ${waited}s"
      return 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "  [vllm] !! server process died; see ${log_file}"; return 1
    fi
    sleep 3; waited=$((waited+3))
  done
  echo "  [vllm] !! timeout after ${VLLM_STARTUP_TIMEOUT}s"; return 1
}
stop_vllm_server() {
  echo "  [vllm] stopping server"
  pkill -9 -f vllm 2>/dev/null || true
  ps -ef | grep 'VLLM::EngineCore' | grep -v grep \
    | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Only reap PIDs on the vLLM GPU, so we don't accidentally nuke training
    # processes on TRAIN_GPUS that might overlap with our pkill.
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
      | true  # disabled: too aggressive on shared boxes
  fi
  sleep 2
  VLLM_PID=""
}
trap stop_vllm_server EXIT INT TERM

# --- (length × model) grid -------------------------------------------------
FAILED_CELLS=""

for LEN in ${LENGTHS}; do
  LEN_ROOT="${OUTPUT_ROOT}/len${LEN}"
  LOG_DIR="${LEN_ROOT}/logs"
  mkdir -p "${LOG_DIR}"

  echo "##############################################################"
  echo " CoT budget: ${LEN} tokens"
  echo " out → ${LEN_ROOT}"
  echo "##############################################################"

  for MODEL in ${MODELS}; do
    SLUG="${MODEL//\//__}"
    LOG_FILE="${LOG_DIR}/${SLUG}.log"

    echo "=============================================================="
    echo " len=${LEN}  ${MODEL}"
    echo " log → ${LOG_FILE}"
    echo "=============================================================="

    VLLM_LOG="${LOG_DIR}/vllm_${SLUG}.log"
    if ! start_vllm_server "${MODEL}" "${LEN}" "${VLLM_LOG}"; then
      echo "  ✗ len=${LEN} ${MODEL} vLLM failed to start; skipping"
      FAILED_CELLS="${FAILED_CELLS} len=${LEN}:${MODEL}:vllm"
      continue
    fi

    CMD=( accelerate launch
          --config_file "${ACCELERATE_CONFIG}"
          --num_processes "${NUM_PROCESSES}"
          "${DS_SCRIPT}"
          --model "${MODEL}"
          --output-root "${LEN_ROOT}"
          --max-steps "${MAX_STEPS}"
          --eval-steps "${EVAL_STEPS}"
          --num-generations "${NUM_GENERATIONS}"
          --max-completion-length "${LEN}"
          --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
          --grad-accum "${GRAD_ACCUM}"
          --learning-rate "${LEARNING_RATE}"
          --vllm-mode server
          --vllm-server-host "${VLLM_HOST}"
          --vllm-server-port "${VLLM_PORT}"
          --vt-micro-batch "${VT_MICRO_BATCH}"
          --rt-pair-seed "${RT_PAIR_SEED}"
          --seed "${SEED}" )

    if [[ "${SCORE_VT}" == "0" ]]; then
      CMD+=( --no-score-vt )
    fi
    if [[ "${KEEP_CKPT}" == "1" ]]; then
      CMD+=( --keep-ckpt )
    fi

    printf '  $ CUDA_VISIBLE_DEVICES=%s %s\n' "${TRAIN_GPUS}" "${CMD[*]}"

    if CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"; then
      echo "  ✓ len=${LEN} ${MODEL} done"
    else
      rc=$?
      echo "  ✗ len=${LEN} ${MODEL} failed (rc=${rc}); continuing"
      FAILED_CELLS="${FAILED_CELLS} len=${LEN}:${MODEL}"
    fi

    stop_vllm_server
  done

  # per-length summary
  echo
  echo " Aggregating len=${LEN} → ${LEN_ROOT}/summary.csv"
  python - <<PY
import json, csv
from pathlib import Path

root = Path("${LEN_ROOT}")
rows = []
for mfile in sorted(root.glob("*/metrics.json")):
    rows.append(json.loads(mfile.read_text()))

if not rows:
    print("No metrics.json files found for len=${LEN}.")
    raise SystemExit(0)

keys = sorted({k for r in rows for k in r.keys()})
out = root / "summary.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"  wrote {out} ({len(rows)} models)")
PY
done

# --- combined summary across all lengths -----------------------------------
echo
echo "=============================================================="
echo " Combined summary → ${OUTPUT_ROOT}/summary_all.csv"
echo "=============================================================="

python - <<PY
import json, csv, re
from pathlib import Path

root = Path("${OUTPUT_ROOT}")
rows = []
for mfile in sorted(root.glob("len*/*/metrics.json")):
    m = re.search(r"len(\d+)", str(mfile))
    if not m:
        continue
    row = json.loads(mfile.read_text())
    row["max_completion_length"] = int(m.group(1))
    rows.append(row)

if not rows:
    print("No metrics.json files found.")
    raise SystemExit(0)

keys = ["max_completion_length"] + sorted({k for r in rows for k in r.keys()} - {"max_completion_length"})
out = root / "summary_all.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Wrote {out} ({len(rows)} rows)")
PY

if [[ -n "${FAILED_CELLS}" ]]; then
  echo
  echo "FAILED:${FAILED_CELLS}"
  exit 1
fi
