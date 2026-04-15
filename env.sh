pip install --upgrade torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade
apt-get update && apt-get install -y ninja-build
pip install ninja
pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu128
# -> then 
pip install datasets seaborn matplotlib hf_transfer 
pip install "trl[vllm]"