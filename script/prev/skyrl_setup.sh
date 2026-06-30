#!/usr/bin/env bash
# Reproducible from-scratch setup for SkyRL + VisGym + Pokemon Crystal example.
#
# What this script does:
#   1. apt: build-essential + libnuma-dev (build deps for some wheels)
#   2. git clone novasky-ai/SkyRL    -> /home/claudeuser/SkyRL
#   3. git clone anyscale/VisGym     -> /home/claudeuser/SkyRL/VisGym
#   4. Patch SkyRL pyproject.toml to point gymnasium at the local VisGym clone
#      and add pygame + pyboy to skyrl-train deps.
#   5. uv venv + uv sync --extra fsdp
#   6. Drop the Pokemon Crystal example files into examples/train/pokemon_crystal
#      (assumes you have them in $PWD/skyrl_extras/pokemon_crystal — if not, the
#      script skips this step quietly).
#   7. Print next steps.
#
# Tested on Ubuntu 24.04, Python 3.12.3, 1× A100-80GB, driver 550.x (CUDA 12.4).
#
# Usage:   bash /workspace/skyrl_setup.sh
#
# IMPORTANT: this script intentionally does NOT install pyboy via pip directly
# anymore — pyboy is in the patched pyproject so `uv sync` handles it.

set -euo pipefail

SKYRL_DIR="${SKYRL_DIR:-/home/claudeuser/SkyRL}"
VISGYM_DIR="${VISGYM_DIR:-$SKYRL_DIR/VisGym}"

log() { echo "[setup] $*"; }

# --- 1. apt prereqs ---------------------------------------------------------
log "Installing build-essential + libnuma-dev (sudo)..."
sudo apt-get update -qq
sudo apt-get install -y build-essential libnuma-dev

# --- 2. uv -----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# --- 3. SkyRL clone --------------------------------------------------------
if [ ! -d "$SKYRL_DIR" ]; then
  log "Cloning SkyRL into $SKYRL_DIR..."
  git clone https://github.com/novasky-ai/SkyRL.git "$SKYRL_DIR"
else
  log "SkyRL already at $SKYRL_DIR — skipping clone."
fi

# --- 4. VisGym clone -------------------------------------------------------
if [ ! -d "$VISGYM_DIR" ]; then
  log "Cloning VisGym into $VISGYM_DIR..."
  git clone https://github.com/anyscale/VisGym.git "$VISGYM_DIR"
else
  log "VisGym already at $VISGYM_DIR — skipping clone."
fi

# --- 5. Patch pyproject.toml ----------------------------------------------
# Use idempotent sed inserts; bail if the markers we look for aren't present.
PY="$SKYRL_DIR/pyproject.toml"

if ! grep -q '"./VisGym"' "$PY"; then
  log "Patching $PY: add gymnasium source override -> ./VisGym"
  # Insert the uv-source line right after the skyrl-gym source line.
  sed -i '/^skyrl-gym = { path = ".\/skyrl-gym"/a gymnasium = { path = "./VisGym", editable = true }' "$PY"
fi

if ! grep -q '"pygame"' "$PY"; then
  log "Patching $PY: add pygame + pyboy + gymnasium to skyrl-train extra"
  # Insert before the closing ] of the skyrl-train block.
  python3 - "$PY" <<'PYEOF'
import sys, re
path = sys.argv[1]
text = open(path).read()
m = re.search(r'(skyrl-train = \[)(.*?)(\n\])', text, re.S)
if not m:
    raise SystemExit("skyrl-train block not found")
block = m.group(2)
adds = []
for dep in ('gymnasium', 'pygame', 'pyboy'):
    if f'"{dep}"' not in block:
        adds.append(f'    "{dep}",\n')
new = m.group(1) + block.rstrip("\n") + "\n" + "".join(adds) + m.group(3)
open(path, "w").write(text[:m.start()] + new + text[m.end():])
print("[setup] patched skyrl-train deps:", ", ".join(d.strip().rstrip(",").strip('"') for d in adds) or "(no change)")
PYEOF
fi

# --- 6. Create venv + sync ------------------------------------------------
cd "$SKYRL_DIR"
if [ ! -d ".venv" ]; then
  log "Creating uv venv (Python 3.12, seeded with pip)..."
  uv venv --python 3.12 .venv --seed
fi

log "Syncing extras: fsdp ..."
uv sync --extra fsdp

# --- 7. Pokemon Crystal example (optional drop-in) ------------------------
EXTRAS_DIR="${EXTRAS_DIR:-$(dirname "$0")/skyrl_extras}"
TARGET_DIR="$SKYRL_DIR/examples/train/pokemon_crystal"
if [ -d "$EXTRAS_DIR/pokemon_crystal" ] && [ ! -d "$TARGET_DIR" ]; then
  log "Copying Pokemon Crystal example into $TARGET_DIR"
  mkdir -p "$TARGET_DIR"
  cp -a "$EXTRAS_DIR/pokemon_crystal/." "$TARGET_DIR/"
fi

# --- 8. Smoke check --------------------------------------------------------
log "Smoke-checking imports ..."
source .venv/bin/activate
python -c "
import torch, gymnasium, pygame
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('gymnasium:', gymnasium.__file__)
import vllm, transformers
print('vllm:', vllm.__version__, 'transformers:', transformers.__version__)
"

log "DONE. Next steps:"
log "  bash /workspace/run_gsm8k_grpo.sh    # text-only GRPO on GSM8K (Qwen2.5-1.5B)"
log "  bash /workspace/run_visgym.sh        # multi-turn VLM on VisGym maze_2d/easy"
log "  bash /workspace/run_pokemon.sh       # multi-turn VLM on Pokemon Crystal"
