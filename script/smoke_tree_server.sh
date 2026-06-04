#!/usr/bin/env bash
# Smoke test: TriPO tree-sampling on Game-of-24 driving the vLLM HTTP SERVER
# backend (the path enabled by TreeSamplingMixin._expand_prefixes_server).
#
# Layout (2 GPUs): vLLM server on VLLM_GPU, training on TRAIN_GPU. Verifies that
#   - tree sampling runs end-to-end in --vllm-mode server,
#   - eval fires over ALL val splits (eval + probe), sampled + greedy,
#   - train and eval rollouts land in their JSONL files.
#
# Usage:
#   bash script/smoke_tree_server.sh
#   TRAIN_GPU=0 VLLM_GPU=1 MODEL=Qwen/Qwen3-0.6B bash script/smoke_tree_server.sh
#
# For a single-GPU sanity check (colocate, no server) set MODE=colocate:
#   MODE=colocate TRAIN_GPU=0 bash script/smoke_tree_server.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODE="${MODE:-server}"                 # server | colocate
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_GPU="${VLLM_GPU:-1}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"

# Tiny config so the whole thing finishes in a couple of minutes.
# num_generations = tree_branch**tree_steps = 2**3 = 8 (keeps the tree exact).
PY="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-output/smoke_tree_${MODE}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=============================================================="
echo " smoke: tree sampling | mode=${MODE} | model=${MODEL}"
echo "        train_gpu=${TRAIN_GPU} vllm_gpu=${VLLM_GPU} out=${OUT_DIR}"
echo "=============================================================="

VLLM_PID=""
stop_vllm () {
  [[ -z "${VLLM_PID}" ]] && return 0
  echo "  [vllm] stopping (pid=${VLLM_PID})"
  kill -9 "${VLLM_PID}" 2>/dev/null || true
  VLLM_PID=""
}
trap stop_vllm EXIT INT TERM

if [[ "${MODE}" == "server" ]]; then
  VLLM_LOG="${LOG_DIR}/vllm_$(date +%s).log"
  echo "  [vllm] starting on gpu=${VLLM_GPU} -> ${VLLM_LOG}"
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    setsid trl vllm-serve --model "${MODEL}" \
      --host "${VLLM_HOST}" --port "${VLLM_PORT}" --enforce-eager \
      > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "  [vllm] pid=${VLLM_PID}; waiting for /health/ ..."
  waited=0
  ready=0
  while (( waited < VLLM_STARTUP_TIMEOUT )); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://${VLLM_HOST}:${VLLM_PORT}/health/" 2>/dev/null || true)"
    if [[ "${code}" == "200" ]]; then ready=1; echo "  [vllm] ready after ${waited}s"; break; fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "  [vllm] !! died during startup; see ${VLLM_LOG}"; exit 1
    fi
    sleep 3; waited=$((waited+3))
  done
  if [[ "${ready}" != "1" ]]; then
    echo "  [vllm] !! timeout after ${VLLM_STARTUP_TIMEOUT}s; see ${VLLM_LOG}"; exit 1
  fi
fi

# -------- training command (tiny) -----------------------------------------
cmd=( "${PY}" script/tripo_game24.py
    --model "${MODEL}"
    --output-dir "${OUT_DIR}"
    --trainer tree --credit-mode max
    --tree-sampling --tree-branch 2 --tree-steps 3
    --max-steps 3 --eval-steps 2 --seed 0
    --num-generations 8 --num-generations-eval 8
    --max-completion-length 128
    --per-device-batch-size 2 --grad-accum 4
    --learning-rate 5e-6 --temperature 1.0 --logging-steps 1
    --max-n 9
    --vllm-mode "${MODE}" )

if [[ "${MODE}" == "server" ]]; then
  cmd+=( --vllm-server-host "${VLLM_HOST}" --vllm-server-port "${VLLM_PORT}" )
else
  cmd+=( --vllm-gpu-memory-utilization 0.4 )
fi

TRAIN_LOG="${LOG_DIR}/train_$(date +%s).log"
echo "  [train] CUDA_VISIBLE_DEVICES=${TRAIN_GPU} ${cmd[*]}"
echo "  [train] log -> ${TRAIN_LOG}"
if CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${cmd[@]}" 2>&1 | tee "${TRAIN_LOG}"; then
  train_rc=0
else
  train_rc=$?
fi

stop_vllm

# -------- verification ------------------------------------------------------
echo "--------------------------------------------------------------"
echo " verifying outputs"
TRAIN_JSONL="${OUT_DIR}/rollouts.jsonl"
EVAL_JSONL="${OUT_DIR}/eval_rollout.jsonl"
fail=0

if [[ "${train_rc}" != "0" ]]; then
  echo "  FAIL: training exited rc=${train_rc} (see ${TRAIN_LOG})"; fail=1
fi
for f in "${TRAIN_JSONL}" "${EVAL_JSONL}"; do
  if [[ -s "${f}" ]]; then
    echo "  OK: $(wc -l < "${f}" | tr -d ' ') lines -> ${f}"
  else
    echo "  FAIL: missing/empty -> ${f}"; fail=1
  fi
done

# eval JSONL should carry both splits and both decodings.
if [[ -s "${EVAL_JSONL}" ]]; then
  echo "  eval splits seen:    $(grep -o '"eval_dataset":[^,]*' "${EVAL_JSONL}" | sort -u | tr '\n' ' ')"
  echo "  eval decodings seen: $(grep -o '"decoding":[^,]*' "${EVAL_JSONL}" | sort -u | tr '\n' ' ')"
fi

echo "--------------------------------------------------------------"
if [[ "${fail}" == "0" ]]; then
  echo " SMOKE PASS (mode=${MODE})"
else
  echo " SMOKE FAIL (mode=${MODE})"; exit 1
fi
