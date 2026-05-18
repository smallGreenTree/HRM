#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${REPO_DIR:-${ROOT_DIR}}"
REPO_URL="${REPO_URL:-https://github.com/smallGreenTree/HRM.git}"
BRANCH="${BRANCH:-gated-residual}"
DATA_PATH="${DATA_PATH:-data/maze-30x30-hard-1k}"
BUILD_DATASET="${BUILD_DATASET:-1}"
SKIP_PIP="${SKIP_PIP:-0}"
CHECKPOINT_ARTIFACT="${CHECKPOINT_ARTIFACT:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/restored-new-step24304-exact0736}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-step_24304.pt}"

echo "== HRM RunPod setup =="
echo "repo: ${REPO_DIR}"
echo "branch: ${BRANCH}"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  ninja-build \
  python3-dev \
  rsync \
  tmux

if [ ! -d "${REPO_DIR}/.git" ]; then
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch origin --prune
git checkout "${BRANCH}"
git pull --ff-only

if [ "${SKIP_PIP}" != "1" ]; then
  python -m pip install -U pip packaging wheel setuptools setuptools-scm
  python -m pip install -r "${REPO_DIR}/requirements-runpod.txt"
fi

if [ "${BUILD_DATASET}" = "1" ]; then
  if [ ! -f "${REPO_DIR}/${DATA_PATH}/train/dataset.json" ] || [ ! -f "${REPO_DIR}/${DATA_PATH}/test/dataset.json" ]; then
    echo "Building maze dataset at ${DATA_PATH}"
    python dataset/build_maze_dataset.py
  else
    echo "Dataset already exists at ${DATA_PATH}"
  fi
fi

if [ -n "${CHECKPOINT_ARTIFACT}" ]; then
  mkdir -p "${CHECKPOINT_DIR}"
  python - <<PY
from pathlib import Path
import wandb

artifact_ref = "${CHECKPOINT_ARTIFACT}"
target = Path("${CHECKPOINT_DIR}")
print(f"Downloading W&B artifact {artifact_ref} -> {target}")
api = wandb.Api()
artifact = api.artifact(artifact_ref, type="model")
artifact.download(root=str(target))
PY
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda devices:", torch.cuda.device_count())
PY

python - <<'PY'
try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except Exception as exc:
    print("matplotlib unavailable:", exc)
try:
    import wandb
    print("wandb:", wandb.__version__)
except Exception as exc:
    print("wandb unavailable:", exc)
PY

if command -v wandb >/dev/null 2>&1; then
  wandb status || true
fi

checkpoint_path="${CHECKPOINT_DIR}/${CHECKPOINT_FILE}"
echo ""
echo "== Setup complete =="
echo "repo: ${REPO_DIR}"
echo "branch: $(git branch --show-current)"
echo "dataset: ${REPO_DIR}/${DATA_PATH}"
if [ -f "${checkpoint_path}" ]; then
  echo "checkpoint: ${checkpoint_path}"
else
  echo "checkpoint missing: ${checkpoint_path}"
  echo "If needed, rerun with CHECKPOINT_ARTIFACT='entity/project/artifact:version' CHECKPOINT_DIR='${CHECKPOINT_DIR}' bash scripts/setup_runpod.sh"
fi
echo ""
echo "Suggested tmux launch:"
echo "tmux new -s layer-full-step24304"
