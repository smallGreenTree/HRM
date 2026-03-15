#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_PATH="${DATA_PATH:-data/maze-30x30-hard-1k}"
PROJECT_NAME="${PROJECT_NAME:-maze-hrm}"
RUN_NAME="${RUN_NAME:-maze-baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/HRM/checkpoints/maze-baseline}"
EPOCHS="${EPOCHS:-20000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
NPROC="${NPROC:-8}"

cd "${ROOT_DIR}"

if [ ! -d "${DATA_PATH}/train" ] || [ ! -d "${DATA_PATH}/test" ]; then
  python dataset/build_maze_dataset.py
fi

DISABLE_COMPILE=1 OMP_NUM_THREADS=8 torchrun --nproc-per-node "${NPROC}" pretrain.py \
  data_path="${DATA_PATH}" \
  project_name="${PROJECT_NAME}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT_PATH}" \
  checkpoint_every_eval=true \
  epochs="${EPOCHS}" \
  eval_interval="${EVAL_INTERVAL}"
