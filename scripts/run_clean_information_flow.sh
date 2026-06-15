#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CHECKPOINT_ARTIFACT="${CHECKPOINT_ARTIFACT:-sotantep-personal-project/maze-hrm/model-maze-rebuild-a100x2-step24304-exact0736:v0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/restored-new-step24304-exact0736}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-step_24304.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/${CHECKPOINT_FILE}}"
DATA_PATH="${DATA_PATH:-data/maze-30x30-hard-1k}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/reports/clean-information-flow-step24304}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
MAX_EXAMPLES="${MAX_EXAMPLES:-128}"
STATE_DTYPE="${STATE_DTYPE:-float16}"
KERNEL_SIGMA="${KERNEL_SIGMA:-1.0}"
RUN_NAME="${RUN_NAME:-clean-information-flow-step24304}"
UPLOAD_ARTIFACT="${UPLOAD_ARTIFACT:-0}"
UPLOAD_SUMMARIES="${UPLOAD_SUMMARIES:-1}"
PROJECT_NAME="${PROJECT_NAME:-maze-hrm-information-flow}"
ARTIFACT_NAME="${ARTIFACT_NAME:-clean-information-flow-step24304}"

cd "${ROOT_DIR}"
mkdir -p "${CHECKPOINT_DIR}" "${OUTPUT_DIR}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Downloading checkpoint artifact ${CHECKPOINT_ARTIFACT}"
  CHECKPOINT_ARTIFACT="${CHECKPOINT_ARTIFACT}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" python - <<'PY'
import os
import wandb

artifact = wandb.Api().artifact(os.environ["CHECKPOINT_ARTIFACT"], type="model")
artifact.download(root=os.environ["CHECKPOINT_DIR"])
PY
fi

if [ ! -f "${DATA_PATH}/train/dataset.json" ] || [ ! -f "${DATA_PATH}/test/dataset.json" ]; then
  echo "Building maze dataset at ${DATA_PATH}"
  python dataset/build_maze_dataset.py
fi

available_kb="$(df -Pk "${OUTPUT_DIR}" | awk 'NR==2 {print $4}')"
dtype_factor=1
if [ "${STATE_DTYPE}" = "float32" ]; then
  dtype_factor=2
fi
required_kb=$(((70 * 1024 * 1024 * MAX_EXAMPLES * dtype_factor + 127) / 128))
if [ "${available_kb}" -lt "${required_kb}" ]; then
  echo "Experiment 1 needs about $((required_kb / 1024 / 1024)) GB free; available: $((available_kb / 1024 / 1024)) GB." >&2
  exit 1
fi

DISABLE_COMPILE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" python scripts/collect_clean_information_flow.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --max-examples "${MAX_EXAMPLES}" \
  --state-dtype "${STATE_DTYPE}" \
  --kernel-sigma "${KERNEL_SIGMA}" \
  --run-name "${RUN_NAME}"

if [ "${UPLOAD_ARTIFACT}" = "1" ]; then
  echo "Uploading the complete result, including activation shards, to W&B."
  WANDB_MODE="${WANDB_MODE:-online}" python scripts/upload_wandb_files.py \
    --project "${PROJECT_NAME}" \
    --artifact-name "${ARTIFACT_NAME}" \
    --artifact-type dataset \
    --path "${OUTPUT_DIR}" \
    --recursive \
    --run-name "artifact-${ARTIFACT_NAME}"
else
  echo "W&B upload skipped. Set UPLOAD_ARTIFACT=1 to upload the roughly 57 GB activation dataset."
fi

if [ "${UPLOAD_SUMMARIES}" = "1" ]; then
  echo "Uploading CSVs, plots, and the manifest to W&B without activation shards."
  WANDB_MODE="${WANDB_MODE:-online}" python scripts/upload_wandb_files.py \
    --project "${PROJECT_NAME}" \
    --artifact-name "${ARTIFACT_NAME}-summaries" \
    --artifact-type analysis \
    --path "${OUTPUT_DIR}" \
    --recursive \
    --exclude-glob "activations/*" \
    --run-name "artifact-${ARTIFACT_NAME}-summaries"
fi

echo "Experiment 1 complete: ${OUTPUT_DIR}"
