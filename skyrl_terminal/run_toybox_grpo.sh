#!/usr/bin/env bash
set -x
# Single-GPU colocated multi-turn GRPO on the ToyBox agentic puzzle pack. 🎮
# A small model acts as a coding agent: runs python/bash snippets in a throwaway
# sandbox, iterates on the output, and finishes with <answer> or TASK_COMPLETE.
# Reward = mean of each task's self-verifying checks (partial credit).
#
#   bash run_toybox_grpo.sh
#
# Override knobs:  MODEL_PATH=Qwen/Qwen2.5-3B-Instruct EPOCHS=60 TRAIN_BS=24 ...

: "${DATA_DIR:="$HOME/data/toybox"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}"
# A small, snappy instruct model is plenty for these toy puzzles.
: "${MODEL_PATH:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${RUN_NAME:=toybox_grpo_qwen15b}"
: "${EPOCHS:=60}"
: "${TRAIN_BS:=12}"
: "${N_SAMPLES:=8}"
: "${MAX_TURNS:=4}"
: "${LR:=1.0e-6}"
: "${CKPT_DIR:="$HOME/ckpts/${RUN_NAME}"}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
# Keep each agent snippet quick to execute inside the rollout.
export TOYBOX_EXEC_TIMEOUT=10

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
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
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
  environment.env_class=toybox \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.eval_batch_size=12 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name="toybox" \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.dump_eval_results=true \
  "$@"
