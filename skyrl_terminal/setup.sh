#!/usr/bin/env bash
# From-scratch setup to reproduce every run here (terminal-bench, SearchR1, toybox,
# geometry3k) on 1 A100, unprivileged container (no Docker). Idempotent — re-runnable;
# each step is skipped if already done.
#
#   bash setup.sh
#
# Overridable: SKYRL_REPO (default = the fork with the fixes), SKYRL_DIR.
set -uo pipefail

PROJECT=/home/claudeuser
SKYRL_REPO="${SKYRL_REPO:-https://github.com/fangyuan-ksgk/SkyRL.git}"   # fork WITH the env/training fixes
SKYRL_DIR="${SKYRL_DIR:-$PROJECT/SkyRL}"                                  # run scripts expect this path
ARL="$PROJECT/arl/skyrl_terminal"
PY="$SKYRL_DIR/.venv/bin/python"

step(){ echo; echo "==== $* ===="; }

step "1. SkyRL fork (Terminal/ToyBox envs + leak/vision/gc fixes baked in)"
[ -d "$SKYRL_DIR/.git" ] || git clone "$SKYRL_REPO" "$SKYRL_DIR"

step "2. SkyRL python env (FSDP + vLLM backend)"
( cd "$SKYRL_DIR" && uv sync --extra fsdp )
"$PY" -c "import torch,vllm; print('torch',torch.__version__,'vllm',vllm.__version__,'cuda',torch.cuda.is_available())"

step "3. container-free sandbox toolchain (the Docker replacement for terminal-bench)"
if ! command -v proot >/dev/null 2>&1; then sudo apt-get update -y && sudo apt-get install -y proot; fi
if [ ! -x "$PROJECT/tbench-venv/bin/python" ]; then
  python3 -m venv "$PROJECT/tbench-venv"
  "$PROJECT/tbench-venv/bin/pip" -q install pytest pyyaml
fi

step "4. Terminal-Bench task source (241 tasks; we curate 32 that run without Docker)"
[ -d "$PROJECT/terminal-bench" ] || git clone https://github.com/laude-institute/terminal-bench.git "$PROJECT/terminal-bench"

step "5. build datasets (skipped if present)"
# -- terminal-bench: curate locally-runnable tasks, then build parquet
if [ ! -f "$PROJECT/data/terminal_bench/train.parquet" ]; then
  "$PROJECT/tbench-venv/bin/python" "$ARL/curate_tasks.py" --workers 32
  "$PY" "$ARL/build_dataset.py"
fi
# -- SearchR1 mini: SQuAD -> SearchR1 schema + self-covering corpus
[ -f "$PROJECT/data/searchR1_mini/train.parquet" ] || "$PY" "$ARL/build_searchr1_mini.py" --n_train 2000 --n_val 300
# -- ToyBox: the 12-puzzle pack -> parquet
[ -f "$PROJECT/data/toybox/train.parquet" ] || "$PY" "$ARL/build_toybox_dataset.py" --output_dir "$PROJECT/data/toybox" --repeat 4
# -- Geometry3K (vision): builder ships in SkyRL examples
[ -f "$PROJECT/data/geometry_3k/train.parquet" ] || "$PY" "$SKYRL_DIR/examples/train/geometry3k/geometry_3k_dataset.py" --output_dir "$PROJECT/data/geometry_3k"

step "DONE — reproduce a run:"
cat <<EOF
  bash $ARL/run_terminal_bench.sh                                   # Terminal-Bench GRPO (Qwen2.5-Coder-3B + LoRA)
  bash $ARL/reproduce_searchr1.sh                                   # SearchR1 (builds retriever + trains)
  bash $ARL/run_toybox_grpo.sh                                      # ToyBox agentic GRPO
  MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct bash $ARL/run_geo3k_1gpu.sh  # Vision GRPO (Geometry-3K)
Each run auto-plots curves to ~/exports/<run>/<run>_curves.png at the end.
EOF
