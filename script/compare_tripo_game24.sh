#!/usr/bin/env bash
# Head-to-head comparison of GRPO vs TriPO variants on Game-of-24.
#
# Five configurations (each in its own python process so the OS reclaims GPU
# memory between runs).
#
# GPU layout (2-GPU, the impatient default):
#   - Non-tree variants run in vLLM **server** mode: generation on VLLM_GPU
#     (default 1), training on TRAIN_GPU (default 0) — fully parallel, fast.
#   - Tree variants (tripo-tree*) REQUIRE the **colocate** engine (the
#     breadth-first sampler drives `self.vllm_generation.llm` directly), so the
#     server is stopped and they run colocate on TRAIN_GPU.
#   Set VLLM_GPU == TRAIN_GPU (or VLLM_GPU="") to force single-GPU colocate for
#   every variant.
#
#   1. grpo                  vanilla GRPO baseline                (--trainer grpo)
#   2. tripo-flat            TriPO, flat rollouts, OPA (max)      (--trainer tree)
#   3. tripo-tree            TriPO, tree sampling, OPA (max)      (+ --tree-sampling)
#   4. tripo-flat-min        TriPO, flat rollouts, min advantage  (+ --credit-mode min)
#   5. tripo-tree-min        TriPO, tree sampling, min advantage  (+ both)
#
# OPA (max)      = each prefix inherits its BEST reachable continuation's advantage.
# min advantage  = pessimistic backup: each prefix inherits its WORST reachable one.
#
# Evaluation (every --eval-steps, plus a baseline at step 0) for EVERY variant:
#   - runs over ALL validation datasets (in-distribution `eval` + hard `probe`)
#   - logs BOTH regimes to <variant>/eval_rollout.jsonl, tagged by
#     {eval_dataset, decoding, temperature}:
#       * sampled  t=1, n=${NUM_GENERATIONS_EVAL} rollouts/prompt -> t=1 pass@8
#       * greedy   t=0, 1 rollout/prompt                          -> greedy pass@1
#   - train rollouts -> <variant>/rollouts.jsonl
#   Compute pass@k offline from eval_rollout.jsonl (src/velo_viz.py:plot_pass_at_k).
#
# NOTE: Qwen3 has no 0.7B checkpoint; the nearest is Qwen3-0.6B (the default).
#       Override with e.g.  MODEL="Qwen/Qwen3-1.7B"  bash script/compare_tripo_game24.sh
#
# Usage:
#   bash script/compare_tripo_game24.sh                         # all 5, defaults
#   STEPS=100 bash script/compare_tripo_game24.sh               # short smoke run
#   VARIANTS="grpo tripo-tree" bash script/compare_tripo_game24.sh   # a subset
#   MODEL="Qwen/Qwen3-1.7B" OUTPUT_ROOT=output/cmp17 bash script/compare_tripo_game24.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- defaults (override via env) -------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
STEPS="${STEPS:-200}"
SEED="${SEED:-0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
# Sampled-eval group size (pass@K). Default = NUM_GENERATIONS -> t=1 pass@8.
NUM_GENERATIONS_EVAL="${NUM_GENERATIONS_EVAL:-${NUM_GENERATIONS}}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-50}"
MAX_N="${MAX_N:-9}"

# Tree-sampling knobs (shared by tree-sampling variants).
TREE_BRANCH="${TREE_BRANCH:-2}"
TREE_STEPS="${TREE_STEPS:-3}"

# GPU layout. Non-tree variants: vLLM server on VLLM_GPU, training on TRAIN_GPU.
# Tree variants always colocate on TRAIN_GPU. Force single-GPU colocate-for-all
# by setting VLLM_GPU to the same value as TRAIN_GPU (or VLLM_GPU="").
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_GPU="${VLLM_GPU:-1}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"
VLLM_MEM="${VLLM_MEM:-0.4}"   # colocate-only KV-cache fraction (tree variants)

# Two-GPU server mode is used only when VLLM_GPU is set AND differs from TRAIN_GPU.
USE_SERVER=1
if [[ -z "${VLLM_GPU}" || "${VLLM_GPU}" == "${TRAIN_GPU}" ]]; then
  USE_SERVER=0
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-output/compare_tripo_game24}"
PY="${PYTHON:-python}"
ONE_SCRIPT="script/tripo_game24.py"

# Which configurations to run (space-separated). Default = all five.
VARIANTS="${VARIANTS:-grpo tripo-flat tripo-tree tripo-flat-min tripo-tree-min}"

SLUG="${MODEL//\//__}"
RUN_ROOT="${OUTPUT_ROOT}/${SLUG}"
mkdir -p "${RUN_ROOT}"

echo "TriPO comparison config:"
echo "  model        = ${MODEL}"
echo "  output_root  = ${RUN_ROOT}"
echo "  steps        = ${STEPS}    seed = ${SEED}"
echo "  num_gen      = ${NUM_GENERATIONS}    num_gen_eval = ${NUM_GENERATIONS_EVAL}"
echo "  max_comp_len = ${MAX_COMPLETION_LENGTH}    eval_steps = ${EVAL_STEPS}"
echo "  eval         = all val datasets (eval+probe) | sample t=1 pass@${NUM_GENERATIONS_EVAL} + greedy pass@1"
echo "  tree         = branch=${TREE_BRANCH} steps=${TREE_STEPS}"
if [[ "${USE_SERVER}" == "1" ]]; then
  echo "  gpu layout   = vLLM server on GPU ${VLLM_GPU}, training on GPU ${TRAIN_GPU} (non-tree)"
  echo "                 tree variants -> colocate on GPU ${TRAIN_GPU} (vllm_mem=${VLLM_MEM})"
else
  echo "  gpu layout   = single-GPU colocate on GPU ${TRAIN_GPU} (vllm_mem=${VLLM_MEM})"
fi
echo "  variants     = ${VARIANTS}"
echo

# True iff $1 appears in the space-separated VARIANTS list.
want_variant () {
  local v
  for v in ${VARIANTS}; do
    [[ "${v}" == "$1" ]] && return 0
  done
  return 1
}

FAILED=""

# run_variant <name> <extra tripo_game24.py args...>
run_variant () {
  local name="$1"; shift
  local out_dir="${RUN_ROOT}/${name}"
  local log_dir="${out_dir}/logs"
  mkdir -p "${log_dir}"

  echo "  ----------------------------------------------------------------"
  echo "    variant=${name}  args: $*"
  echo "  ----------------------------------------------------------------"

  local cmd=( "${PY}" "${ONE_SCRIPT}"
      --model "${MODEL}"
      --output-dir "${out_dir}"
      --max-steps "${STEPS}" --seed "${SEED}"
      --num-generations "${NUM_GENERATIONS}"
      --num-generations-eval "${NUM_GENERATIONS_EVAL}"
      --max-completion-length "${MAX_COMPLETION_LENGTH}"
      --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
      --grad-accum "${GRAD_ACCUM}"
      --learning-rate "${LEARNING_RATE}"
      --temperature "${TEMPERATURE}"
      --logging-steps "${LOGGING_STEPS}"
      --eval-steps "${EVAL_STEPS}"
      --max-n "${MAX_N}"
      --vllm-mode colocate
      --vllm-gpu-memory-utilization "${VLLM_MEM}"
      "$@" )

  local log_file="${log_dir}/$(date +%s)_${RANDOM}.log"
  printf '    $ CUDA_VISIBLE_DEVICES=%s %s\n' "${TRAIN_GPU}" "${cmd[*]}"

  if CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${cmd[@]}" 2>&1 | tee "${log_file}"; then
    echo "    done -> ${out_dir}"
  else
    rc=$?
    echo "    failed (rc=${rc}); continuing"
    FAILED="${FAILED} ${name}"
  fi
  echo
}

# --- the five configurations ------------------------------------------------
want_variant grpo          && run_variant grpo          --trainer grpo
want_variant tripo-flat    && run_variant tripo-flat    --trainer tree --credit-mode max
want_variant tripo-tree    && run_variant tripo-tree    --trainer tree --credit-mode max \
                                  --tree-sampling --tree-branch "${TREE_BRANCH}" --tree-steps "${TREE_STEPS}"
want_variant tripo-flat-min && run_variant tripo-flat-min --trainer tree --credit-mode min
want_variant tripo-tree-min && run_variant tripo-tree-min --trainer tree --credit-mode min \
                                  --tree-sampling --tree-branch "${TREE_BRANCH}" --tree-steps "${TREE_STEPS}"

echo "=============================================================="
echo " All variants finished. Results under ${RUN_ROOT}/"
echo "   per-variant rollouts -> ${RUN_ROOT}/<variant>/rollouts.jsonl"
echo "   per-variant eval     -> ${RUN_ROOT}/<variant>/eval_rollout.jsonl"
echo "=============================================================="
if [[ -n "${FAILED}" ]]; then
  echo "FAILED:${FAILED}"
  exit 1
fi
