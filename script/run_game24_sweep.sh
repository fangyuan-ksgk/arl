#!/usr/bin/env bash
# Sweep run_game24_one.py over multiple base models.
#
# Each model runs in its own python process so the OS reclaims all GPU
# memory between runs (vLLM colocate doesn't release nicely otherwise).
#
# Usage:
#   bash script/run_game24_sweep.sh                  # default model list
#   MAX_STEPS=50 bash script/run_game24_sweep.sh     # smoke test
#   MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B" bash script/run_game24_sweep.sh
#
# All env-var overrides are passed through to run_game24_one.py.

set -u  # unset vars are an error; keep going on a single-model failure

# --- locate repo root -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- defaults (override via env) -------------------------------------------
MODELS="${MODELS:-\
Qwen/Qwen3-0.6B \
Qwen/Qwen3-1.7B \
Qwen/Qwen3-4B \
meta-llama/Llama-3.2-1B-Instruct \
meta-llama/Llama-3.2-3B-Instruct}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/game24_sweep}"
MAX_STEPS="${MAX_STEPS:-200}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
N_PER_CLASS="${N_PER_CLASS:-50}"
N_PAIRS="${N_PAIRS:-50}"
D4_SAMPLE="${D4_SAMPLE:-16}"
SKIP_VT="${SKIP_VT:-0}"   # set to 1 to skip v_t probes
SEED="${SEED:-0}"

PY="${PYTHON:-python}"
ONE_SCRIPT="script/run_game24_one.py"

mkdir -p "${OUTPUT_ROOT}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "Sweep config:"
echo "  output_root = ${OUTPUT_ROOT}"
echo "  max_steps   = ${MAX_STEPS}"
echo "  num_gen     = ${NUM_GENERATIONS}"
echo "  skip_vt     = ${SKIP_VT}"
echo "  models      ="
for m in ${MODELS}; do echo "    - ${m}"; done
echo

# --- per-model loop ---------------------------------------------------------
FAILED_MODELS=""

for MODEL in ${MODELS}; do
  SLUG="${MODEL//\//__}"
  LOG_FILE="${LOG_DIR}/${SLUG}.log"

  echo "=============================================================="
  echo " ${MODEL}"
  echo " log → ${LOG_FILE}"
  echo "=============================================================="

  CMD=( "${PY}" "${ONE_SCRIPT}"
        --model "${MODEL}"
        --output-root "${OUTPUT_ROOT}"
        --max-steps "${MAX_STEPS}"
        --num-generations "${NUM_GENERATIONS}"
        --max-completion-length "${MAX_COMPLETION_LENGTH}"
        --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
        --grad-accum "${GRAD_ACCUM}"
        --learning-rate "${LEARNING_RATE}"
        --n-per-class "${N_PER_CLASS}"
        --n-pairs "${N_PAIRS}"
        --d4-sample "${D4_SAMPLE}"
        --seed "${SEED}" )

  if [[ "${SKIP_VT}" == "1" ]]; then
    CMD+=( --skip-vt )
  fi

  printf '  $ %s\n' "${CMD[*]}"

  # Stream stdout/stderr to both the terminal and the log file.
  if "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"; then
    echo "  ✓ ${MODEL} done"
  else
    rc=$?
    echo "  ✗ ${MODEL} failed (rc=${rc}); continuing"
    FAILED_MODELS="${FAILED_MODELS} ${MODEL}"
  fi
done

# --- aggregate metrics.json files into summary.csv -------------------------
echo
echo "=============================================================="
echo " Aggregating metrics → ${OUTPUT_ROOT}/summary.csv"
echo "=============================================================="

"${PY}" - <<PY
import json, csv
from pathlib import Path

root = Path("${OUTPUT_ROOT}")
rows = []
for mfile in sorted(root.glob("*/metrics.json")):
    rows.append(json.loads(mfile.read_text()))

if not rows:
    print("No metrics.json files found.")
    raise SystemExit(0)

keys = sorted({k for r in rows for k in r.keys()})
out = root / "summary.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Wrote {out} ({len(rows)} models)")
PY

if [[ -n "${FAILED_MODELS}" ]]; then
  echo
  echo "FAILED:${FAILED_MODELS}"
  exit 1
fi
