#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

apt-get update
apt-get install -y build-essential python3-dev ninja-build git tmux

python -m pip install -U pip packaging wheel setuptools setuptools-scm
python -m pip install -r "${ROOT_DIR}/requirements-runpod.txt"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda devices:", torch.cuda.device_count())
PY
