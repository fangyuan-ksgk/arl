#!/usr/bin/env bash
# Sweep V0–V10 per-token reward variants on Game-of-24.
#
# Each cell runs `pt-placeholder` with a different `--reward vX`, keeping
# every other knob fixed so reward shaping is the only varying axis.
#
# Layout (matches script/ablate_game24.sh):
#   GPU 1 → vLLM server  (VLLM_GPU=1) — shared across all V variants per model
#   GPU 0 → training     (TRAIN_GPU=0)
#
# The vLLM server is started ONCE per model and reused for every V variant,
# saving ~30s × |REWARDS| of startup time per model.
#
# Output:
#   ${OUTPUT_ROOT}/<model_slug>/pt-placeholder_<vX>_adv-<mode>_seed-<seed>/
#       config.json
#       rollouts.jsonl
#       eval_rollouts.jsonl
#       logs/<ts>_<rand>.log
#
# Usage:
#   # full sweep, default model
#   bash script/sweep_reward_variants.sh
#
#   # subset of variants
#   REWARDS="v0 v8 v10"           bash script/sweep_reward_variants.sh
#
#   # different model / shorter run
#   MODELS="Qwen/Qwen3-1.7B" STEPS=400 \
#     bash script/sweep_reward_variants.sh
#
#   # multiple seeds for variance estimate
#   SEEDS="0 1 2" REWARDS="v0 v8 v10" \
#     bash script/sweep_reward_variants.sh
#
#   # single-GPU box (colocate)
#   VLLM_MODE=colocate \
#     bash script/sweep_reward_variants.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- defaults (override via env) -------------------------------------------
MODELS="${MODELS:-Qwen/Qwen3-0.6B}"

# V1 is an alias of V0 (the trainer always passes traj_reward), so skip by default.
REWARDS="${REWARDS:-v0 v3 v4 v5 v6 v7 v8 v9 v10}"
SEEDS="${SEEDS:-0}"

STEPS="${STEPS:-400}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-100}"
ADV_MODE="${ADV_MODE:-token}"

# vLLM mode. "server" keeps training and inference on separate GPUs (recommended
# when you have ≥2 GPUs). "colocate" puts both on TRAIN_GPU.
VLLM_MODE="${VLLM_MODE:-server}"
VLLM_GPU="${VLLM_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"
VLLM_MEM_COLOCATE="${VLLM_MEM_COLOCATE:-0.4}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/sweep_reward_variants}"
PY="${PYTHON:-python}"
ONE_SCRIPT="script/ablate_game24.py"

mkdir -p "${OUTPUT_ROOT}"

echo "Reward-variant sweep config:"
echo "  output_root  = ${OUTPUT_ROOT}"
echo "  steps        = ${STEPS}"
echo "  max_comp_len = ${MAX_COMPLETION_LENGTH}"
echo "  num_gen      = ${NUM_GENERATIONS}"
echo "  adv_mode     = ${ADV_MODE}"
echo "  vllm_mode    = ${VLLM_MODE}"
[[ "${VLLM_MODE}" == "server" ]] && echo "  vllm_gpu     = ${VLLM_GPU}    train_gpu = ${TRAIN_GPU}"
echo "  models       ="
for m in ${MODELS}; do echo "    - ${m}"; done
echo "  rewards      = ${REWARDS}"
echo "  seeds        = ${SEEDS}"
echo

# --- vLLM server lifecycle (server mode only) ------------------------------
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
  [[ "${VLLM_MODE}" != "server" ]] && return 0
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
  # $1 = model, $2 = reward variant, $3 = seed
  local model="$1"; local reward="$2"; local seed="$3"
  local slug="${model//\//__}"
  local model_root="${OUTPUT_ROOT}/${slug}"
  local log_dir="${model_root}/logs"
  mkdir -p "${log_dir}"

  echo "  ----------------------------------------------------------------"
  echo "    model=${model}  reward=${reward}  seed=${seed}"
  echo "  ----------------------------------------------------------------"

  local cmd=( "${PY}" "${ONE_SCRIPT}"
      --variant pt-placeholder
      --reward "${reward}"
      --adv_mode "${ADV_MODE}"
      --model "${model}"
      --output_root "${model_root}"
      --steps "${STEPS}" --seed "${seed}"
      --num_generations "${NUM_GENERATIONS}"
      --max_completion_length "${MAX_COMPLETION_LENGTH}"
      --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
      --gradient_accumulation_steps "${GRAD_ACCUM}"
      --learning_rate "${LEARNING_RATE}"
      --logging_steps "${LOGGING_STEPS}"
      --eval_steps "${EVAL_STEPS}"
      --vllm_mode "${VLLM_MODE}" )

  local cuda_dev
  if [[ "${VLLM_MODE}" == "server" ]]; then
    cmd+=( --vllm_server_host "${VLLM_HOST}"
           --vllm_server_port "${VLLM_PORT}"
           --train_device "${TRAIN_GPU}" )
    cuda_dev="${TRAIN_GPU}"
  else
    cmd+=( --vllm_mem "${VLLM_MEM_COLOCATE}" )
    cuda_dev="${TRAIN_GPU}"
  fi

  local log_file="${log_dir}/$(date +%s)_${reward}_seed${seed}_${RANDOM}.log"
  printf '    $ CUDA_VISIBLE_DEVICES=%s %s\n' "${cuda_dev}" "${cmd[*]}"

  if CUDA_VISIBLE_DEVICES="${cuda_dev}" "${cmd[@]}" 2>&1 | tee "${log_file}"; then
    echo "    ✓ done"
  else
    rc=$?
    echo "    ✗ failed (rc=${rc}); continuing"
    FAILED="${FAILED} ${model}/${reward}/seed${seed}"
  fi
}

# --- model × reward × seed grid --------------------------------------------
for MODEL in ${MODELS}; do
  SLUG="${MODEL//\//__}"
  MODEL_ROOT="${OUTPUT_ROOT}/${SLUG}"
  LOG_DIR="${MODEL_ROOT}/logs"
  mkdir -p "${LOG_DIR}"

  echo "##############################################################"
  echo " MODEL: ${MODEL}"
  echo " out  → ${MODEL_ROOT}"
  echo "##############################################################"

  # Start vLLM once per model (server mode only).
  if [[ "${VLLM_MODE}" == "server" ]]; then
    VLLM_LOG="${LOG_DIR}/vllm_$(date +%s).log"
    if ! start_vllm_server "${MODEL}" "${VLLM_LOG}"; then
      echo "  !! vLLM server failed for ${MODEL}; skipping all variants for this model."
      FAILED="${FAILED} ${MODEL}:vllm-start"
      continue
    fi
  fi

  for REWARD in ${REWARDS}; do
    for SEED in ${SEEDS}; do
      run_cell "${MODEL}" "${REWARD}" "${SEED}"
    done
  done

  stop_vllm_server
done

echo
echo "=============================================================="
echo " Sweep finished. Results under ${OUTPUT_ROOT}/"
echo "=============================================================="
if [[ -n "${FAILED}" ]]; then
  echo "FAILED:${FAILED}"
  exit 1
fi
