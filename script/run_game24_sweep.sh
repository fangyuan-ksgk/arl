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
# Default to Qwen-only: the size-vs-passrate inversion under a fixed CoT
# budget is the active hypothesis. Llama-3.2-Instruct produced too few correct
# rollouts at 512 to be informative; pass MODELS=... to re-include them.
MODELS="${MODELS:-\
Qwen/Qwen3-0.6B \
Qwen/Qwen3-1.7B \
Qwen/Qwen3-4B}"

# CoT-budget sweep. Each length runs the full model list end-to-end.
LENGTHS="${LENGTHS:-1024 2048}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/game24_sweep}"
MAX_STEPS="${MAX_STEPS:-200}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
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

echo "Sweep config:"
echo "  output_root = ${OUTPUT_ROOT}"
echo "  lengths     = ${LENGTHS}"
echo "  max_steps   = ${MAX_STEPS}"
echo "  num_gen     = ${NUM_GENERATIONS}"
echo "  skip_vt     = ${SKIP_VT}"
echo "  models      ="
for m in ${MODELS}; do echo "    - ${m}"; done
echo

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

    CMD=( "${PY}" "${ONE_SCRIPT}"
          --model "${MODEL}"
          --output-root "${LEN_ROOT}"
          --max-steps "${MAX_STEPS}"
          --num-generations "${NUM_GENERATIONS}"
          --max-completion-length "${LEN}"
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

    if "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"; then
      echo "  ✓ len=${LEN} ${MODEL} done"
    else
      rc=$?
      echo "  ✗ len=${LEN} ${MODEL} failed (rc=${rc}); continuing"
      FAILED_CELLS="${FAILED_CELLS} len=${LEN}:${MODEL}"
    fi
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
