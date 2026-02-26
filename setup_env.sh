#!/usr/bin/env bash
# setup_env.sh — One-shot environment setup for mcdo-vlm-uncertainty
#
# Detects platform (macOS/Apple Silicon vs Linux/CUDA), creates a conda env,
# installs all dependencies, caches model weights, and validates everything.
#
# Usage:
#   bash setup_env.sh            # full setup
#   bash setup_env.sh --skip-models   # skip model weight downloads
#   bash setup_env.sh --env-name foo  # custom conda env name (default: mcdo)
#
set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
ENV_NAME="mcdo"
SKIP_MODELS=false
PYTHON_VERSION="3.11"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)   ENV_NAME="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=true; shift ;;
    -h|--help)
      echo "Usage: bash setup_env.sh [--env-name NAME] [--skip-models]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── platform detection ──────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
  if [[ "$ARCH" == "arm64" ]]; then
    PLATFORM="macos-arm64"
    DEVICE="mps"
    echo "==> Detected macOS Apple Silicon (arm64) — will use MPS backend"
  else
    PLATFORM="macos-x86"
    DEVICE="cpu"
    echo "==> Detected macOS Intel — will use CPU backend (no MPS)"
  fi
elif [[ "$OS" == "Linux" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    PLATFORM="linux-cuda"
    DEVICE="cuda"
    echo "==> Detected Linux with NVIDIA GPU — will use CUDA backend"
  else
    PLATFORM="linux-cpu"
    DEVICE="cpu"
    echo "==> Detected Linux (no GPU) — will use CPU backend"
  fi
else
  echo "==> Unknown OS ($OS) — defaulting to CPU"
  PLATFORM="unknown"
  DEVICE="cpu"
fi

# ── conda check ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install miniforge/miniconda first."
  echo "  macOS: brew install miniforge"
  echo "  Linux: https://github.com/conda-forge/miniforge"
  exit 1
fi

# Source conda so we can activate envs in this script
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── create or reuse conda env ──────────────────────────────────────────────
if conda env list | grep -qw "$ENV_NAME"; then
  echo "==> Conda env '$ENV_NAME' already exists — reusing it"
else
  echo "==> Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION"
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
echo "==> Activated env: $ENV_NAME (Python $(python --version 2>&1))"

# ── install PyTorch ─────────────────────────────────────────────────────────
echo ""
echo "==> Installing PyTorch..."
case "$PLATFORM" in
  macos-arm64|macos-x86)
    # PyTorch nightly/stable ships with MPS support on macOS arm64 by default
    pip install --upgrade torch torchvision
    ;;
  linux-cuda)
    # Use the CUDA 12.1 index for modern GPUs
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ;;
  *)
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ;;
esac

# ── install project deps ───────────────────────────────────────────────────
echo ""
echo "==> Installing project dependencies..."
pip install open_clip_torch Pillow numpy tqdm scipy networkx transformers protobuf sentencepiece

# Install the project itself in editable mode (picks up src/mcdo_clip)
pip install -e "$REPO_DIR"

# ── validate core imports ───────────────────────────────────────────────────
echo ""
echo "==> Validating environment..."
python -c "
import sys, platform
import torch
import torchvision
import open_clip
import transformers
import scipy
import numpy as np

print(f'  Python:       {sys.version.split()[0]}')
print(f'  Platform:     {platform.system()} {platform.machine()}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  torchvision:  {torchvision.__version__}')
print(f'  open_clip:    {open_clip.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  numpy:        {np.__version__}')
print(f'  scipy:        {scipy.__version__}')
print()
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  MPS available:  {torch.backends.mps.is_available()}')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = 'mps'
    # Quick MPS smoke test
    x = torch.randn(2, 2, device='mps')
    y = x @ x.T
    print(f'  MPS smoke:    OK (matmul on device worked)')

print(f'  Best device:  {device}')
"

# ── cache model weights ────────────────────────────────────────────────────
if [[ "$SKIP_MODELS" == false ]]; then
  echo ""
  echo "==> Pre-caching model weights (this may take a few minutes)..."

  # CLIP models (open_clip downloads on first use, but let's warm the cache)
  python -c "
import open_clip
print('  Caching CLIP ViT-B-32...')
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
print('  Caching CLIP ViT-L-14...')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
print('  CLIP models cached.')
"

  # SigLIP2 models — cache model weights, image processor, and tokenizer
  python -c "
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
models = [
    'google/siglip2-base-patch16-224',
    'google/siglip2-so400m-patch14-384',
]
for m in models:
    print(f'  Caching {m}...')
    try:
        AutoImageProcessor.from_pretrained(m)
        AutoTokenizer.from_pretrained(m, use_fast=False)
        AutoModel.from_pretrained(m)
        print(f'    OK: {m}')
    except Exception as e:
        print(f'    WARN: {m} — {e}')
        print(f'    (model may still work if weights were partially cached)')
print('  SigLIP2 cache step done.')
"
fi

# ── check data readiness ───────────────────────────────────────────────────
echo ""
echo "==> Checking data..."
DATA_DIR="$REPO_DIR/data/raw"

check_data() {
  local name="$1" path="$2"
  if [[ -d "$path" ]]; then
    local count
    count="$(find "$path" -maxdepth 1 -mindepth 1 | wc -l | tr -d ' ')"
    echo "  [OK] $name ($count entries)"
    return 0
  else
    echo "  [--] $name (not found at $path)"
    return 1
  fi
}

check_data "ImageNet val"    "$DATA_DIR/imagenet_val"
check_data "ImageNet meta"   "$DATA_DIR/imagenet_meta"
check_data "COCO val 2017"   "$DATA_DIR/coco_val2017" || true

# ── write a convenience run script ─────────────────────────────────────────
RUN_SCRIPT="$REPO_DIR/run.sh"
cat > "$RUN_SCRIPT" << RUNEOF
#!/usr/bin/env bash
# run.sh — Convenience wrapper that activates the conda env and runs experiments.
# Auto-generated by setup_env.sh
#
# Usage:
#   bash run.sh phase1              # run Phase 1 only
#   bash run.sh phase2              # run Phase 2 only
#   bash run.sh all                 # run all phases (1,2,3)
#   bash run.sh all --phases 1,2    # custom phase selection
#   bash run.sh <any extra args>    # passed through to run_all_phases.py
#
set -euo pipefail

REPO_DIR="$REPO_DIR"
CONDA_BASE="$CONDA_BASE"
ENV_NAME="$ENV_NAME"
DEVICE="$DEVICE"
DATA_DIR="$REPO_DIR/data/raw/imagenet_val"
OUT_DIR="$REPO_DIR/outputs/run_\$(date +%Y%m%d_%H%M%S)"

source "\$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "\$ENV_NAME"

cd "\$REPO_DIR"

MODE="\${1:-all}"
shift 2>/dev/null || true

case "\$MODE" in
  phase1|p1)
    echo "==> Running Phase 1 -> \$OUT_DIR"
    python -m phase_one.run_phase1 "\$DATA_DIR" "\$OUT_DIR/phase_one" \\
      --device "\$DEVICE" "\$@"
    ;;
  phase2|p2)
    echo "==> Running Phase 2 -> \$OUT_DIR"
    python -m phase_two.run_phase2 "\$DATA_DIR" "\$OUT_DIR/phase_two" \\
      --device "\$DEVICE" "\$@"
    ;;
  phase3|p3)
    echo "==> Running Phase 3 -> \$OUT_DIR"
    python -m phase_three.run_phase3 "\$DATA_DIR" "\$OUT_DIR/phase_three" \\
      --device "\$DEVICE" "\$@"
    ;;
  phase4|p4)
    echo "==> Running Phase 4 -> \$OUT_DIR"
    python -m phase_four.run_phase4 "\$DATA_DIR" "\$OUT_DIR/phase_four" \\
      --device "\$DEVICE" "\$@"
    ;;
  all)
    echo "==> Running all phases -> \$OUT_DIR"
    python run_all_phases.py "\$DATA_DIR" "\$OUT_DIR" \\
      --device "\$DEVICE" "\$@"
    ;;
  *)
    echo "Usage: bash run.sh {phase1|phase2|phase3|phase4|all} [extra args...]"
    exit 1
    ;;
esac

echo ""
echo "==> Done. Outputs at: \$OUT_DIR"
RUNEOF
chmod +x "$RUN_SCRIPT"

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "  Conda env:  $ENV_NAME"
echo "  Device:     $DEVICE"
echo "  Data dir:   $DATA_DIR/imagenet_val"
echo ""
echo "  To run experiments:"
echo "    bash run.sh phase1          # Phase 1 go/no-go gate"
echo "    bash run.sh all             # all phases"
echo "    bash run.sh phase2 --dropout 0.05   # with extra args"
echo ""
echo "  Or manually:"
echo "    conda activate $ENV_NAME"
echo "    python run_all_phases.py data/raw/imagenet_val outputs/my_run --device $DEVICE"
echo ""
