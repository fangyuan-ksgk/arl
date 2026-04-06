#!/bin/bash
set -e

# ============================================================
# veRL Full Environment Setup Script
# Prerequisites: Python >= 3.10, CUDA >= 12.8, NVIDIA GPU
# ============================================================

echo "=== Step 1: Clone veRL repository ==="
git clone https://github.com/volcengine/verl.git
cd verl

echo "=== Step 2: Run install script (vLLM + SGLang + Megatron + FlashAttention) ==="
# Set USE_MEGATRON=0 to skip Megatron/TransformerEngine (faster install)
# Set USE_SGLANG=0 to skip SGLang
bash scripts/install_vllm_sglang_mcore.sh

echo "=== Step 3: Fix NumPy version (opencv may upgrade to 2.4+ which breaks numba/vLLM) ==="
pip install "numpy>=1.26,<2.3"

echo "=== Step 4: Remove deprecated pynvml (nvidia-ml-py is the replacement) ==="
pip uninstall pynvml -y 2>/dev/null || true

echo "=== Step 5: Install veRL in editable mode ==="
pip install --no-deps -e .

echo "=== Step 6: Verify installation ==="
python3 -c "import verl; print('veRL version:', verl.__version__)"
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import megatron; print('Megatron-LM: OK')"
python3 -c "import transformer_engine; print('TransformerEngine: OK')"

echo "=== Setup complete ==="
