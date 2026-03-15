#!/bin/bash
set -euo pipefail

# Bootstrap + smoke test for a fresh Runpod instance.
# Minimal, cost-conscious checks to verify environment and InfoRidge CSV output.
#
# Usage:
#   ./bootstrap-smoke.sh [repo_url] [branch] [workdir]
# Defaults:
#   repo_url=https://github.com/smallGreenTree/HRM.git
#   branch=main
#   workdir=/workspace/HRM

REPO_URL="${1:-https://github.com/smallGreenTree/HRM.git}"
BRANCH="${2:-main}"
WORKDIR="${3:-/workspace/HRM}"

echo "=============================================="
echo "Bootstrap + Smoke Test"
echo "=============================================="
echo "Repo:    $REPO_URL"
echo "Branch:  $BRANCH"
echo "Workdir: $WORKDIR"
echo "=============================================="

if ! command -v git >/dev/null 2>&1; then
  echo "Installing git..."
  apt-get update -y
  apt-get install -y git
fi

if [ -d "$WORKDIR/.git" ]; then
  echo "Repo exists, updating..."
  cd "$WORKDIR"
  git fetch --all --prune
else
  echo "Cloning repo..."
  git clone "$REPO_URL" "$WORKDIR"
  cd "$WORKDIR"
fi

git checkout "$BRANCH"
git pull --ff-only || true

echo "Installing build deps..."
apt-get update -y
apt-get install -y build-essential python3-dev ninja-build

echo "Installing Python deps..."
pip install -U pip
pip install packaging wheel setuptools setuptools-scm
pip install -r requirements.txt


echo "Checking CUDA + PyTorch..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo "Skipping FlashAttention install (using PyTorch SDPA fallback)."

echo "Checking core imports..."
python - <<'PY'
import torch
import einops
import pydantic
import omegaconf
import hydra
import wandb
import coolname
import huggingface_hub
print("core imports: OK")
PY

echo "✅ Smoke test complete (no training)."
