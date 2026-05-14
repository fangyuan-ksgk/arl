#!/usr/bin/env bash
# Sweep run_game24_one.py over (CoT-budget × model) grid.
#
# Each (length, model) cell runs in its own python process so the OS reclaims
# all GPU memory between runs (vLLM colocate doesn't release nicely otherwise).
#
# Per-length results land in:
#     ${OUTPUT_ROOT}/len${LEN}/<model_slug>/...
#     ${OUTPUT_ROOT}/len${LEN}/summary.csv
# and a combined ${OUTPUT_ROOT}/summary_all.csv is written at the end with an
# extra `max_completion_length` column.
#
# Usage:
#   bash script/run_game24_sweep.sh                                # default
#   LENGTHS="1024 2048" bash script/run_game24_sweep.sh
#   LENGTHS=2048 MODELS="Qwen/Qwen3-4B" bash script/run_game24_sweep.sh
#   MAX_STEPS=50 bash script/run_game24_sweep.sh                   # smoke test
#
# All env-var overrides are passed through to run_game24_one.py.

set -u  # unset vars are an error; keep going on a single-cell failure

# --- locate repo root -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- defaults (override via env) -------------------------------------------
# Default model list: Qwen3 0.6B/1.7B/4B + Llama-3.2-3B-Instruct as a
# non-thinking baseline. The longer CoT budgets (≥2048) make Llama informative
# again — it was dropped earlier only because 512 tokens gave too few correct
# rollouts.
MODELS="${MODELS:-\
Qwen/Qwen3-0.6B \
Qwen/Qwen3-1.7B \
Qwen/Qwen3-4B \
meta-llama/Llama-3.2-3B-Instruct}"

# CoT-budget sweep. Each length runs the full model list end-to-end.
LENGTHS="${LENGTHS:-512 1024 2048 3096}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/game24_sweep}"
MAX_STEPS="${MAX_STEPS:-1200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"   # pdbs * grad_accum must be a multiple of NUM_GENERATIONS
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
# R_T scoring + figure generation. SCORE_VT=0 skips the post-training scoring
# pass entirely (no R_T fields added, no rt_*.png written). RT_PAIR_SEED
# controls which (correct, incorrect) pair appears in each rt_step{N}.png.
SCORE_VT="${SCORE_VT:-1}"
VT_MICRO_BATCH="${VT_MICRO_BATCH:-64}"   # forward batch for v_t scoring; raise on bigger GPUs
RT_PAIR_SEED="${RT_PAIR_SEED:-0}"
SEED="${SEED:-0}"

# --- vLLM mode --------------------------------------------------------------
# VLLM_MODE=server splits GPUs: VLLM_GPU runs the vLLM server, TRAIN_GPU runs
# training. We (re)start the server with --max-model-len matching the current
# CoT budget so its KV cache is sized correctly.
VLLM_MODE="${VLLM_MODE:-server}"            # 'server' | 'colocate'
VLLM_GPU="${VLLM_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"  # seconds

PY="${PYTHON:-python}"
ONE_SCRIPT="script/run_game24_one.py"

mkdir -p "${OUTPUT_ROOT}"

echo "Sweep config:"
echo "  output_root = ${OUTPUT_ROOT}"
echo "  lengths     = ${LENGTHS}"
echo "  max_steps   = ${MAX_STEPS}"
echo "  eval_steps  = ${EVAL_STEPS}"
echo "  num_gen     = ${NUM_GENERATIONS}"
echo "  vllm_mode   = ${VLLM_MODE}  (vllm_gpu=${VLLM_GPU}, train_gpu=${TRAIN_GPU})"
echo "  models      ="
for m in ${MODELS}; do echo "    - ${m}"; done
echo

# --- vLLM server lifecycle helpers ------------------------------------------
VLLM_PID=""
start_vllm_server() {
  local model="$1"; local max_len="$2"; local log_file="$3"
  stop_vllm_server
  echo "  [vllm] starting server  model=${model}  max_model_len=${max_len}  gpu=${VLLM_GPU}"
  # setsid puts the server + all its EngineCore subprocesses in a fresh
  # process group so we can SIGKILL the whole group on shutdown.
  # --enforce-eager: skip torch.compile + CUDA graph capture. Trades ~15%
  # decode throughput for instant startup and bulletproof reliability across
  # cold caches / image rebuilds. Worth it for a multi-cell sweep.
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    setsid trl vllm-serve --model "${model}" --max-model-len "${max_len}" \
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
  # Single-tenant container: nuke every vllm process by command-line match.
  # Catches the wrapper, EngineCore subprocess, and any worker. Idempotent.
  echo "  [vllm] pkill -9 -f vllm"
  pkill -9 -f vllm 2>/dev/null || true
  sleep 2
  VLLM_PID=""
}
trap stop_vllm_server EXIT INT TERM

# --- (length × model) grid --------------------------------------------------
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

    # In server mode, (re)start vLLM with this length's max-model-len.
    if [[ "${VLLM_MODE}" == "server" ]]; then
      VLLM_LOG="${LOG_DIR}/vllm_${SLUG}.log"
      if ! start_vllm_server "${MODEL}" "${LEN}" "${VLLM_LOG}"; then
        echo "  ✗ len=${LEN} ${MODEL} vLLM failed to start; skipping"
        FAILED_CELLS="${FAILED_CELLS} len=${LEN}:${MODEL}:vllm"
        continue
      fi
    fi

    CMD=( "${PY}" "${ONE_SCRIPT}"
          --model "${MODEL}"
          --output-root "${LEN_ROOT}"
          --max-steps "${MAX_STEPS}"
          --eval-steps "${EVAL_STEPS}"
          --num-generations "${NUM_GENERATIONS}"
          --max-completion-length "${LEN}"
          --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
          --grad-accum "${GRAD_ACCUM}"
          --learning-rate "${LEARNING_RATE}"
          --vllm-mode "${VLLM_MODE}"
          --vt-micro-batch "${VT_MICRO_BATCH}"
          --rt-pair-seed "${RT_PAIR_SEED}"
          --seed "${SEED}" )

    if [[ "${SCORE_VT}" == "0" ]]; then
      CMD+=( --no-score-vt )
    fi

    if [[ "${VLLM_MODE}" == "server" ]]; then
      CMD+=( --vllm-server-host "${VLLM_HOST}"
             --vllm-server-port "${VLLM_PORT}"
             --train-device "${TRAIN_GPU}" )
    fi

    printf '  $ %s\n' "${CMD[*]}"

    if CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"; then
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
  "${PY}" - <<PY
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

# --- combined summary across all lengths ------------------------------------
echo
echo "=============================================================="
echo " Combined summary → ${OUTPUT_ROOT}/summary_all.csv"
echo "=============================================================="

"${PY}" - <<PY
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
