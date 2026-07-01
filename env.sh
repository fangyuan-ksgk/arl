# pip install --upgrade torch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade
# apt-get update && apt-get install -y ninja-build
# pip install ninja
# pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu128
# # -> then 
# pip install datasets seaborn matplotlib hf_transfer 
# pip install "trl[vllm]"
# pip3 install deepspeed


#!/usr/bin/env bash
set -euo pipefail

export PIP_ROOT_USER_ACTION=ignore
export PIP_BREAK_SYSTEM_PACKAGES=1     # RunPod base images are often PEP-668 "externally managed"

CU=cu128
TORCH_INDEX="https://download.pytorch.org/whl/${CU}"

# --- Pinned, mutually-compatible versions (all cu128-safe) ------------------
TORCH_VER=2.8.0
VISION_VER=0.23.0
AUDIO_VER=2.8.0
TRANSFORMERS_VER=4.56.2
VLLM_VER=0.11.0
TRL_VER=1.1.0

# --- 1. Torch stack: cu128 ONLY. --force-reinstall guarantees a lottery
#        cu126/cu130 build of the same version is actually replaced.
pip install --force-reinstall --index-url "${TORCH_INDEX}" \
    "torch==${TORCH_VER}" "torchvision==${VISION_VER}" "torchaudio==${AUDIO_VER}"

# --- 2. Build tooling for flash-attn.
apt-get update && apt-get install -y ninja-build
pip install ninja

# --- 3. Everything else, under a constraints file that pins the torch trio so
#        no downstream dependency can move it off cu128. The cu128 index is kept
#        as a fallback so any legitimate same-version reinstall stays on cu128.
CONSTRAINTS="$(mktemp)"
cat > "${CONSTRAINTS}" <<EOF
torch==${TORCH_VER}
torchvision==${VISION_VER}
torchaudio==${AUDIO_VER}
EOF

pip install -c "${CONSTRAINTS}" --extra-index-url "${TORCH_INDEX}" \
    "transformers==${TRANSFORMERS_VER}" \
    "vllm==${VLLM_VER}" \
    "trl==${TRL_VER}" \
    datasets seaborn matplotlib hf_transfer deepspeed

# --- 4. flash-attn compiled against the installed cu128 torch.
#        --no-build-isolation -> build with the env's torch headers (no fresh
#                                torch pulled into an isolated build env)
#        --no-deps            -> never let flash-attn drag torch off cu128
#        (claude.md: no prebuilt wheel exists for torch 2.8.0+cu128, so it must
#         compile from source; the golden invocation is below.)
pip install flash-attn --no-build-isolation --no-deps

# --- 5. Fail loudly unless the whole thing landed on cu128.
python - <<'PY'
import torch, torchvision, torchaudio
cuda = torch.version.cuda or ""
assert cuda.startswith("12.8"), f"torch is on CUDA {cuda!r}, expected 12.8 (cu128)"
print(f"OK  torch {torch.__version__} | vision {torchvision.__version__} | "
      f"audio {torchaudio.__version__} | CUDA {cuda}")
try:
    import flash_attn_2_cuda  # noqa: F401
    print("OK  flash-attn importable")
except Exception as e:
    print(f"WARN flash-attn import failed: {e}")
PY