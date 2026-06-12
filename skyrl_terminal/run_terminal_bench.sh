#!/usr/bin/env bash
# Self-contained, memory-safe Terminal-Bench GRPO launcher.
# Works no matter who runs it (root or claudeuser): all paths are absolute and
# HOME is pinned to the project user so the uv/HF/model caches are reused.
#
#   bash run_terminal_bench.sh                       # baseline run
#   LR=5.0e-6 bash run_terminal_bench.sh             # the lr improvement experiment
#   bash run_terminal_bench.sh trainer.epochs=5      # extra hydra overrides pass through
set -uo pipefail

# --- run as claudeuser, never root -----------------------------------------
# This container's root is capability-limited: it can't hardlink/overwrite files
# it doesn't own, and its uv cache lands on /workspace (a quota-limited network FS
# that wedges the build with "Quota exceeded (os error 122)"). claudeuser uses the
# roomy 200 GB overlay cache that's already built, so re-exec there if root.
if [ "$(id -u)" = "0" ]; then
  echo ">> launched as root — re-executing as claudeuser (avoids the /workspace uv-cache quota)."
  exec su claudeuser -c "bash $(printf '%q' "$0") $(printf '%q ' "$@")"
fi

# --- pin the environment to the project user -------------------------------
export HOME="${HOME:-/home/claudeuser}"
PROJECT="${PROJECT:-$HOME}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.cache/uv}"   # never the quota-limited /workspace
# SKYRL_DIR = the (forked) SkyRL repo to train against — pass it to use any checkout.
SKYRL_DIR="${SKYRL_DIR:-$PROJECT/SkyRL}"
ARL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's own dir
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export TBENCH_VERIFIER_PYTHON="${TBENCH_VERIFIER_PYTHON:-$PROJECT/tbench-venv/bin/python}"
# Raise Ray's host-RAM kill line (container cgroup is only ~116 GB).
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.97}"

# --- knobs (override via env) ----------------------------------------------
DATA_DIR="${DATA_DIR:-$PROJECT/data/terminal_bench}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"
RUN_NAME="${RUN_NAME:-terminal_grpo_coder3b}"
EPOCHS="${EPOCHS:-20}"
TRAIN_BS="${TRAIN_BS:-16}"      # memory-safe (full 32 OOM'd the cgroup)
N_SAMPLES="${N_SAMPLES:-6}"     # 16x6 = 96 traj/step
LR="${LR:-1.0e-6}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_TURNS="${MAX_TURNS:-1}"   # >1 = multi-turn ReAct (run cmd, see output, fix, ...)
# LoRA on by default: full fine-tuning a 3B model keeps ~58 GB of AdamW optimizer
# states in host RAM and OOMs this 116 GB cgroup during the backward. LoRA trains
# tiny adapters (<1 GB optimizer state) so training actually survives. LORA_RANK=0
# to force full fine-tuning (will OOM on 3B here).
LORA_RANK="${LORA_RANK:-32}"
LORA_ARGS=""
if [ "${LORA_RANK}" -gt 0 ] 2>/dev/null; then
  LORA_ARGS="trainer.policy.model.lora.rank=${LORA_RANK} trainer.policy.model.lora.alpha=${LORA_RANK}"
fi
# Per-user log path so a file left by another user can never block you (sticky /tmp +
# this container's capability-limited root can't overwrite someone else's file).
LOG="${LOG:-/tmp/${RUN_NAME}.$(id -un).log}"

# --- fail fast with a clear message if inputs are missing ------------------
for split in train validation; do
  if [[ ! -f "$DATA_DIR/$split.parquet" ]]; then
    echo "ERROR: missing $DATA_DIR/$split.parquet"
    echo "       (build it with: $PROJECT/SkyRL/.venv/bin/python -m ... or check DATA_DIR)"
    exit 1
  fi
done
if [[ ! -x "$TBENCH_VERIFIER_PYTHON" ]]; then
  echo "ERROR: verifier python not found/executable at $TBENCH_VERIFIER_PYTHON"; exit 1
fi

echo ">> Terminal-Bench GRPO | model=$MODEL_PATH | bs=$TRAIN_BS x n=$N_SAMPLES | lr=$LR | epochs=$EPOCHS | lora_rank=$LORA_RANK"
echo ">> data=$DATA_DIR | log=$LOG | RAY_memory_usage_threshold=$RAY_memory_usage_threshold"

cd "$SKYRL_DIR"

# Logging must never break the run: only tee if the log is writable, else stdout-only.
if : >> "$LOG" 2>/dev/null; then
  TEE=(tee -a "$LOG")
else
  echo "WARN: cannot write $LOG (owned by another user?) — logging to stdout only."
  TEE=(cat)
fi

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  ${LORA_ARGS} \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.65 \
  generator.batched=false \
  generator.max_turns=$MAX_TURNS \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  trainer.epochs=$EPOCHS \
  trainer.train_batch_size=$TRAIN_BS \
  trainer.policy_mini_batch_size=$TRAIN_BS \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.algorithm.use_kl_loss=false \
  environment.env_class=terminal \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.eval_batch_size=32 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_path="$PROJECT/ckpts/$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.logger=console \
  trainer.project_name="terminal_bench" \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$PROJECT/exports/$RUN_NAME" \
  trainer.dump_eval_results=true \
  "$@" 2>&1 | "${TEE[@]}"

# auto-plot results into clean figures + CSV (no more log-soup grepping)
PLOT_PY="$PROJECT/tbench-venv/bin/python"; [ -x "$PLOT_PY" ] || PLOT_PY="$SKYRL_DIR/.venv/bin/python"
"$PLOT_PY" "$ARL/plot_run.py" --run "$RUN_NAME" 2>/dev/null \
  && echo ">> figures: $PROJECT/exports/$RUN_NAME/${RUN_NAME}_curves.png" || true
