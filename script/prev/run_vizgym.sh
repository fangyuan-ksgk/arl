#!/usr/bin/env bash
# Multi-turn VLM RL on VisGym maze_2d/easy with Qwen3.5-4B on 1 GPU.
#
# Produces:
#   - Training log:       /tmp/skyrl-logs/...
#   - Per-episode GIFs:   /tmp/skyrl-logs/gifs/visgym_maze/ep_NNNNNN_*.gif
#                          (every 16th episode by default; tunable via the
#                           `sample_rate` arg in env_instruct.py)
#
# GPU plan (Qwen3.5-4B, 8.7 GB weights):
#   - FSDP trainer    ~30 GB
#   - Colocated vLLM @ gpu_memory_utilization=0.5  ~40 GB
#   - Total           ~70 GB  → fits 80GB A100 with ~10 GB headroom
#
# max_generate_length is bumped to 1024 because Qwen3.5-4B emits a
# <think>...</think> reasoning prelude that eats budget before the
# <action>...</action> tag.

set -euo pipefail
set -x

SKYRL_DIR="${SKYRL_DIR:-/home/claudeuser/SkyRL}"
cd "$SKYRL_DIR"

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
source .venv/bin/activate
ray start --head 2>/dev/null || true

: "${MODEL_PATH:=Qwen/Qwen2.5-VL-3B-Instruct}"
# Tried on this hardware (1× A100-80GB):
#   - Qwen/Qwen3.5-4B          ❌  FSDP forward: image-token count mismatch
#                                  (tokens: 384, features: 512). Vision
#                                  processor disagrees with SkyRL's collator.
#   - Qwen/Qwen3-VL-4B-Instruct ❌ OOM at FSDP optim_step (76.5 GB used / 80 GB)
#                                  even with gpu_memory_utilization=0.35.
#   - Qwen/Qwen2.5-VL-3B-Instruct ✅ Works. ~60s/step, full epoch completes.
# To try 4B again, also drop train_batch_size to 4 and n_samples to 2.
: "${EPOCHS:=1}"
: "${NUM_GPUS:=1}"
: "${NUM_DATASET_ROWS:=64}"
: "${MAX_TURNS:=8}"
: "${TRAIN_BATCH:=8}"
: "${N_SAMPLES:=4}"
: "${DATA_DIR:=$HOME/data/visgym_maze}"
: "${EVAL_DIR:=$HOME/data/visgym_maze_eval}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  uv run --isolated examples/train/visgym/dataset.py \
    --env_id maze_2d/easy --num_rows "$NUM_DATASET_ROWS" --output_dir "$DATA_DIR"
fi

if [ ! -f "$EVAL_DIR/train.parquet" ]; then
  uv run --isolated examples/train/visgym/dataset.py \
    --env_id maze_2d/easy --num_rows 16 --seed --output_dir "$EVAL_DIR"
fi

_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra fsdp \
  python examples/train/visgym/entrypoint.py \
  --env_variant instruct \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$EVAL_DIR/train.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.ref.fsdp_config.cpu_offload=false \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=16000 \
  trainer.epochs=$EPOCHS \
  trainer.train_batch_size=$TRAIN_BATCH \
  trainer.policy_mini_batch_size=$TRAIN_BATCH \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1 \
  generator.max_turns=$MAX_TURNS \
  generator.max_input_length=12000 \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.vision_language_generator=true \
  generator.batched=false \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.005 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class=visgym \
  trainer.logger=console \
  trainer.project_name=vlm_maze_2d_easy \
  trainer.run_name=maze_2d_easy_qwen35 \
  trainer.resume_mode=null \
  trainer.log_path=/tmp/skyrl-logs \
  trainer.eval_before_train=false \
  trainer.dump_eval_results=false \
  trainer.ckpt_interval=-1 \
  trainer.eval_interval=5 \
  trainer.use_sample_packing=false \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  "$@"