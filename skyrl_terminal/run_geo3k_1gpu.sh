#!/usr/bin/env bash
set -x
# Goal 2: Single-GPU multi-turn GRPO (LoRA) on Geometry-3K visual-math with the
# vision-capable Qwen3-VL model. Reward is binary (correct \boxed{} answer).
# Adapted from examples/train/geometry3k/run_geometry3k_lora.sh for 1x A100-80GB.
#
#   bash run_geo3k_1gpu.sh

: "${DATA_DIR:="$HOME/data/geometry_3k"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}"
# Qwen3.5-4B: natively vision-multimodal (Qwen3_5ForConditionalGeneration, vision_config),
# supported by transformers 5.8.0 + vLLM 0.20.2 in this env. 4B fits colocated on one 80GB GPU
# (the 8B-class model OOM-killed the colocated vLLM engine core at init).
: "${MODEL_PATH:=Qwen/Qwen3.5-4B}"
: "${RUN_NAME:=geo3k_qwen3vl_lora_1gpu}"
: "${EPOCHS:=3}"
: "${EXPORT_PATH:="$HOME/exports/${RUN_NAME}"}"

export TOKENIZERS_PARALLELISM=false
cd /home/claudeuser/SkyRL

_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra fsdp --with pylatexenc \
  python examples/train/geometry3k/geometry3k_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
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
  generator.inference_engine.engine_init_kwargs.max_model_len=16384 \
  generator.inference_engine.engine_init_kwargs.enforce_eager=true \
  generator.batched=false \
  generator.vision_language_generator=true \
  generator.max_turns=3 \
  generator.n_samples_per_prompt=8 \
  generator.sampling_params.max_generate_length=1536 \
  trainer.epochs=$EPOCHS \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=1024 \
  trainer.remove_microbatch_padding=false \
  trainer.policy.optimizer_config.lr=3.0e-5 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  environment.env_class=geometry3k \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.eval_batch_size=128 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry3k" \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  "$@"
