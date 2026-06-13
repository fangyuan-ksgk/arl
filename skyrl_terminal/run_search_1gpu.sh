#!/usr/bin/env bash
# Scaled-down SearchR1 on ONE A100. Multi-turn search-agent GRPO: the model emits
# <search>q</search>, the mini retrieval server (mini_retrieval_server.py on :8000)
# returns <information>, and it answers in <answer>. Reward = exact-match vs gold.
#
# Prereq: the mini retriever must be running on :8000 (over ~/data/searchR1_mini).
#   bash run_search_1gpu.sh
set -uo pipefail

# run as claudeuser, never root (uv-cache quota + perms on this box)
if [ "$(id -u)" = "0" ]; then
  exec su claudeuser -c "bash $(printf '%q' "$0") $(printf '%q ' "$@")"
fi
export HOME="${HOME:-/home/claudeuser}"
PROJECT="${PROJECT:-$HOME}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.cache/uv}"
export TOKENIZERS_PARALLELISM=false
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.97}"

# SKYRL_DIR = the (forked) SkyRL repo to train against — pass it to use any checkout.
SKYRL_DIR="${SKYRL_DIR:-$PROJECT/SkyRL}"
ARL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's own dir
DATA_DIR="${DATA_DIR:-$PROJECT/data/searchR1_mini}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
RUN_NAME="${RUN_NAME:-searchr1_mini_1gpu}"
# Per-run scratch dir: isolates THIS run's Ray session + sandbox temp dirs (Ray +
# tempfile both honor TMPDIR), so concurrent runs / stray cleanups can't collide.
export TMPDIR="/tmp/skyrl-$RUN_NAME"
mkdir -p "$TMPDIR"
EPOCHS="${EPOCHS:-1}"
TRAIN_BS="${TRAIN_BS:-64}"      # 2000 rows / 64 ≈ 31 steps in 1 epoch
N_SAMPLES="${N_SAMPLES:-5}"
LORA_RANK="${LORA_RANK:-32}"
LR="${LR:-1.0e-6}"
LOG="${LOG:-/tmp/${RUN_NAME}.$(id -un).log}"

for s in train validation; do
  [[ -f "$DATA_DIR/$s.parquet" ]] || { echo "ERROR: missing $DATA_DIR/$s.parquet (run build_searchr1_mini.py)"; exit 1; }
done
curl -s -m 5 -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"query":"test","topk":1,"return_scores":true}' >/dev/null 2>&1 \
  || { echo "ERROR: retrieval server not responding on :8000 — start mini_retrieval_server.py first"; exit 1; }

echo ">> SearchR1-mini | model=$MODEL_PATH | bs=$TRAIN_BS x n=$N_SAMPLES | turns=4 | lora=$LORA_RANK | skyrl=$SKYRL_DIR | log=$LOG"
cd "$SKYRL_DIR"
if : >> "$LOG" 2>/dev/null; then TEE=(tee -a "$LOG"); else TEE=(cat); fi

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_RANK \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.55 \
  trainer.epochs=$EPOCHS \
  trainer.train_batch_size=$TRAIN_BS \
  trainer.policy_mini_batch_size=$TRAIN_BS \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=500 \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.max_turns=4 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  environment.env_class="search" \
  environment.skyrl_gym.max_env_workers=8 \
  environment.skyrl_gym.search.log_requests=false \
  environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
  environment.skyrl_gym.search.topk=3 \
  trainer.logger="console" \
  trainer.project_name="searchr1_mini" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_ckpts_to_keep=1 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$PROJECT/ckpts/$RUN_NAME" \
  trainer.eval_before_train=true \
  trainer.eval_batch_size=64 \
  trainer.eval_interval=10 \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=500 \
  trainer.dump_eval_results=true \
  trainer.export_path="$PROJECT/exports/$RUN_NAME" \
  trainer.log_path="/tmp/skyrl-logs" \
  "$@" 2>&1 | "${TEE[@]}"

# auto-plot results into clean figures + CSV
"$SKYRL_DIR/.venv/bin/python" "$ARL/plot_run.py" --run "$RUN_NAME" 2>/dev/null \
  && echo ">> figures: $PROJECT/exports/$RUN_NAME/${RUN_NAME}_curves.png" || true
