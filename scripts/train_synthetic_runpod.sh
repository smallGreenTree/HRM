#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_PATH="${DATA_PATH:-data/synthetic-arithmetic-k5}"
PROJECT_NAME="${PROJECT_NAME:-synthetic-arithmetic-hrm}"
RUN_NAME="${RUN_NAME:-synthetic-arithmetic-baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/HRM/checkpoints/synthetic-arithmetic-baseline}"
EPOCHS="${EPOCHS:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NPROC="${NPROC:-8}"
OMP_THREADS="${OMP_THREADS:-2}"

cd "${ROOT_DIR}"

if [ ! -d "${DATA_PATH}/train" ] || [ ! -d "${DATA_PATH}/test" ]; then
  echo "Synthetic dataset not found at ${DATA_PATH}. Build it first with scripts/build_synthetic_arithmetic.sh." >&2
  exit 1
fi

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
DISABLE_COMPILE=1 \
OMP_NUM_THREADS="${OMP_THREADS}" \
torchrun --nproc-per-node "${NPROC}" pretrain.py \
  data_path="${DATA_PATH}" \
  project_name="${PROJECT_NAME}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT_PATH}" \
  checkpoint_every_eval=true \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  eval_interval="${EVAL_INTERVAL}"
