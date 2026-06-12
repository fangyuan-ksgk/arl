#!/usr/bin/env bash
set -x
# Single-GPU colocated GRPO on real Terminal-Bench tasks, scored by each task's
# own pytest verifier inside a container-free proot /app sandbox.
#
#   bash run_terminal_grpo.sh
#
# Override knobs:  NUM_GPUS=1 MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct ...

: "${DATA_DIR:="$HOME/data/terminal_bench"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}"
: "${MODEL_PATH:=Qwen/Qwen2.5-Coder-3B-Instruct}"
: "${RUN_NAME:=terminal_grpo_coder3b}"
: "${EPOCHS:=40}"
: "${TRAIN_BS:=16}"
: "${N_SAMPLES:=8}"
: "${CKPT_DIR:="$HOME/ckpts/${RUN_NAME}"}"

export TBENCH_VERIFIER_PYTHON=/home/claudeuser/tbench-venv/bin/python
export TOKENIZERS_PARALLELISM=false
# proot sandboxes are CPU-cheap but numerous; keep them snappy.
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /home/claudeuser/SkyRL

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
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
  generator.max_turns=1 \
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
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  environment.env_class=terminal \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.eval_batch_size=32 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name="terminal_bench" \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.dump_eval_results=true \
  "$@"
