#!/usr/bin/env bash
# Head-to-head ablation: GRPO vs PerTokenAdvantageTrainer on Game-of-24.
#
# Two-GPU layout (default; override with env):
#   GPU 1 → vLLM server  (VLLM_GPU=1)
#   GPU 0 → training     (TRAIN_GPU=0)
#
# vLLM server lifecycle is inherited from script/run_game24_sweep.sh: each
# model gets its own server invocation, and we KILL it (process-group + EngineCore
# sweep) before moving to the next model.
#
# Variant ladder (per model):
#   server-mode variants (use GPU 0 for training, GPU 1 for vLLM):
#     grpo                      vanilla baseline
#     pt-placeholder            PT subclass with ramp reward
#     pt-velocity               + velocity per-token reward
#     pt-velocity (position)    + advantage shaping ablation
#     pt-velocity (progress)    + advantage shaping ablation
#   colocate-mode variants (vLLM server stopped; both on TRAIN_GPU):
#     pt-velocity-prefix p=0.5  full system (PrefixInjector needs colocate)
#     [if EXTENDED=1] knob sweep on p_inject / share_within_group / max_layer
#
# Models default to Qwen3-0.6B / 1.7B / 4B. Each cell runs in its own python
# process so the OS reclaims GPU memory between cells.
#
# Output:
#   ${OUTPUT_ROOT}/<model_slug>/<run_name>/{config.json, rollouts.jsonl, buffers.json}
#
# Usage:
#   bash script/ablate_game24.sh                                   # default
#   MODELS="Qwen/Qwen3-0.6B"          bash script/ablate_game24.sh  # one model
#   STEPS=200                         bash script/ablate_game24.sh  # short
#   EXTENDED=1                        bash script/ablate_game24.sh  # full knob sweep

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- defaults (override via env) -------------------------------------------
MODELS="${MODELS:-\
Qwen/Qwen3-0.6B \
Qwen/Qwen3-1.7B \
Qwen/Qwen3-4B}"

# Inherited from run_game24_sweep.sh (longer-CoT regime).
STEPS="${STEPS:-1200}"
SEED="${SEED:-0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-200}"
VEL_CHUNK_SIZE="${VEL_CHUNK_SIZE:-64}"

# 2-GPU layout: vLLM on GPU 1, training on GPU 0.
VLLM_GPU="${VLLM_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"

# colocate-only knobs (used when vLLM server is down).
VLLM_MEM_COLOCATE="${VLLM_MEM_COLOCATE:-0.4}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/ablate_game24}"
EXTENDED="${EXTENDED:-0}"
PY="${PYTHON:-python}"
ONE_SCRIPT="script/ablate_game24.py"

mkdir -p "${OUTPUT_ROOT}"

echo "Ablation sweep config:"
echo "  output_root  = ${OUTPUT_ROOT}"
echo "  steps        = ${STEPS}"
echo "  max_comp_len = ${MAX_COMPLETION_LENGTH}"
echo "  num_gen      = ${NUM_GENERATIONS}"
echo "  vllm_gpu     = ${VLLM_GPU}    train_gpu = ${TRAIN_GPU}"
echo "  extended     = ${EXTENDED}"
echo "  models       ="
for m in ${MODELS}; do echo "    - ${m}"; done
echo

# --- vLLM server lifecycle (copied from run_game24_sweep.sh) ---------------
VLLM_PID=""
start_vllm_server() {
  local model="$1"; local log_file="$2"
  stop_vllm_server
  echo "  [vllm] starting  model=${model}  gpu=${VLLM_GPU}"
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    setsid trl vllm-serve --model "${model}" \
      --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
      --enforce-eager \
      > "${log_file}" 2>&1 &
  VLLM_PID=$!
  echo "  [vllm] pid=${VLLM_PID}  log=${log_file}"
  local waited=0
  while (( waited < VLLM_STARTUP_TIMEOUT )); do
    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
      echo "  [vllm] ready after ${waited}s"; return 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "  [vllm] !! died early; see ${log_file}"; return 1
    fi
    sleep 3; waited=$((waited+3))
  done
  echo "  [vllm] !! timeout after ${VLLM_STARTUP_TIMEOUT}s"; return 1
}
stop_vllm_server() {
  echo "  [vllm] stopping"
  pkill -9 -f vllm 2>/dev/null || true
  ps -ef | grep 'VLLM::EngineCore' | grep -v grep \
    | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | tr -d ' ' | grep -E '^[0-9]+$' \
      | xargs -r kill -9 2>/dev/null || true
  fi
  sleep 2
  VLLM_PID=""
}
trap stop_vllm_server EXIT INT TERM

# --- per-cell runner -------------------------------------------------------
FAILED=""

run_cell () {
  # $1 = model, $2 = vllm_mode (server|colocate), rest = variant args
  local model="$1"; shift
  local mode="$1"; shift
  local slug="${model//\//__}"
  local model_root="${OUTPUT_ROOT}/${slug}"
  local log_dir="${model_root}/logs"
  mkdir -p "${log_dir}"

  echo "  ----------------------------------------------------------------"
  echo "    model=${model}  mode=${mode}  args: $*"
  echo "  ----------------------------------------------------------------"

  local cmd=( "${PY}" "${ONE_SCRIPT}"
      --model "${model}"
      --output_root "${model_root}"
      --steps "${STEPS}" --seed "${SEED}"
      --num_generations "${NUM_GENERATIONS}"
      --max_completion_length "${MAX_COMPLETION_LENGTH}"
      --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
      --gradient_accumulation_steps "${GRAD_ACCUM}"
      --learning_rate "${LEARNING_RATE}"
      --logging_steps "${LOGGING_STEPS}"
      --eval_steps "${EVAL_STEPS}"
      --vel_chunk_size "${VEL_CHUNK_SIZE}"
      --vllm_mode "${mode}"
      "$@" )

  local cuda_dev log_file
  if [[ "${mode}" == "server" ]]; then
    cmd+=( --vllm_server_host "${VLLM_HOST}"
           --vllm_server_port "${VLLM_PORT}"
           --train_device "${TRAIN_GPU}" )
    cuda_dev="${TRAIN_GPU}"
  else
    cmd+=( --vllm_mem "${VLLM_MEM_COLOCATE}" )
    cuda_dev="${TRAIN_GPU}"   # colocate: training + vLLM both on TRAIN_GPU
  fi

  # Derive run-name suffix for the log file (mirror ablate_game24.py).
  log_file="${log_dir}/$(date +%s)_${RANDOM}.log"
  printf '    $ CUDA_VISIBLE_DEVICES=%s %s\n' "${cuda_dev}" "${cmd[*]}"

  if CUDA_VISIBLE_DEVICES="${cuda_dev}" "${cmd[@]}" 2>&1 | tee "${log_file}"; then
    echo "    ✓ done"
  else
    rc=$?
    echo "    ✗ failed (rc=${rc}); continuing"
    FAILED="${FAILED} ${model}:[$*]"
  fi
}

# --- model × variant grid --------------------------------------------------
for MODEL in ${MODELS}; do
  SLUG="${MODEL//\//__}"
  MODEL_ROOT="${OUTPUT_ROOT}/${SLUG}"
  LOG_DIR="${MODEL_ROOT}/logs"
  mkdir -p "${LOG_DIR}"

  echo "##############################################################"
  echo " MODEL: ${MODEL}"
  echo " out  → ${MODEL_ROOT}"
  echo "##############################################################"

  # ---- server-mode variants (vLLM on GPU 1, training on GPU 0) ----
  VLLM_LOG="${LOG_DIR}/vllm_$(date +%s).log"
  if start_vllm_server "${MODEL}" "${VLLM_LOG}"; then
    run_cell "${MODEL}" server  --variant grpo
    run_cell "${MODEL}" server  --variant pt-placeholder --adv_mode token
    run_cell "${MODEL}" server  --variant pt-velocity    --adv_mode token

    if [[ "${EXTENDED}" == "1" ]]; then
      run_cell "${MODEL}" server  --variant pt-velocity --adv_mode position
      run_cell "${MODEL}" server  --variant pt-velocity --adv_mode progress
    fi
  else
    echo "  !! vLLM server failed to start for ${MODEL}; skipping server variants."
    FAILED="${FAILED} ${MODEL}:vllm-start"
  fi
  stop_vllm_server

  # ---- colocate-mode variants (PrefixInjector requires it) ----
  run_cell "${MODEL}" colocate  --variant pt-velocity-prefix --adv_mode token --p_inject 0.5

  if [[ "${EXTENDED}" == "1" ]]; then
    run_cell "${MODEL}" colocate  --variant pt-velocity-prefix --adv_mode token --p_inject 0.25
    run_cell "${MODEL}" colocate  --variant pt-velocity-prefix --adv_mode token --p_inject 0.75
    run_cell "${MODEL}" colocate  --variant pt-velocity-prefix --adv_mode token --p_inject 0.5 --share_within_group 0
    run_cell "${MODEL}" colocate  --variant pt-velocity-prefix --adv_mode token --p_inject 0.5 --prefix_max_layer 2
    run_cell "${MODEL}" colocate  --variant pt-velocity-prefix --adv_mode token --p_inject 0.5 --prefix_max_layer 4
  fi
done

echo
echo "=============================================================="
echo " All cells finished. Results under ${OUTPUT_ROOT}/"
echo "=============================================================="
if [[ -n "${FAILED}" ]]; then
  echo "FAILED:${FAILED}"
  exit 1
fi
