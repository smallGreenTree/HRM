#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CHECKPOINT_ARTIFACT="${CHECKPOINT_ARTIFACT:-sotantep-personal-project/maze-hrm/model-maze-rebuild-a100x2-step24304-exact0736:v0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/restored-new-step24304-exact0736}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-step_24304.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/${CHECKPOINT_FILE}}"
DATA_PATH="${DATA_PATH:-data/maze-30x30-hard-1k}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/reports/merge-sensitivity-step24304}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-128}"
SOURCES="${SOURCES:-L:own L:H L:input H:own H:L}"
SCALES="${SCALES:-0 0.5 0.75 1.25 1.5}"
ACT_STEPS="${ACT_STEPS:-}"
PROJECT_NAME="${PROJECT_NAME:-maze-hrm-merge-sensitivity}"
ARTIFACT_NAME="${ARTIFACT_NAME:-merge-sensitivity-step24304}"
UPLOAD_SUMMARIES="${UPLOAD_SUMMARIES:-1}"

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

read -r -a source_args <<< "${SOURCES}"
read -r -a scale_args <<< "${SCALES}"
command=(
  python scripts/run_merge_sensitivity.py
  --checkpoint "${CHECKPOINT_PATH}"
  --data-path "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --max-examples "${MAX_EXAMPLES}"
  --sources "${source_args[@]}"
  --scales "${scale_args[@]}"
)

if [ -n "${ACT_STEPS}" ]; then
  read -r -a act_step_args <<< "${ACT_STEPS}"
  command+=(--act-steps "${act_step_args[@]}")
fi

DISABLE_COMPILE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" "${command[@]}"

if [ "${UPLOAD_SUMMARIES}" = "1" ]; then
  WANDB_MODE="${WANDB_MODE:-online}" python scripts/upload_wandb_files.py \
    --project "${PROJECT_NAME}" \
    --artifact-name "${ARTIFACT_NAME}" \
    --artifact-type analysis \
    --path "${OUTPUT_DIR}" \
    --recursive \
    --run-name "artifact-${ARTIFACT_NAME}"
fi

echo "Merge-sensitivity experiment complete: ${OUTPUT_DIR}"
