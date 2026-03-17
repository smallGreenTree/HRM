#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_PATH="${DATA_PATH:-data/maze-30x30-hard-1k}"
PROJECT_NAME="${PROJECT_NAME:-maze-hrm}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/maze-baseline-b128-rerun}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/checkpoints/inforidge-sweep}"
RUN_PREFIX="${RUN_PREFIX:-inforidge}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MAX_BATCHES="${MAX_BATCHES:-8}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
SAMPLE_MODE="${SAMPLE_MODE:-last_token}"
TARGET_EMBEDDING_MODE="${TARGET_EMBEDDING_MODE:-second_pass}"
WANDB_MODE="${WANDB_MODE:-online}"
STEPS="${STEPS:-}"

cd "${ROOT_DIR}"

mkdir -p "${OUTPUT_ROOT}"

resolve_checkpoints() {
  if [ -n "${STEPS}" ]; then
    for step in ${STEPS}; do
      printf '%s\n' "${CHECKPOINT_DIR}/step_${step}"
    done
    return
  fi

  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'step_*' | sort -V
}

mapfile -t CHECKPOINTS < <(resolve_checkpoints)

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  echo "No checkpoints found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

for checkpoint_path in "${CHECKPOINTS[@]}"; do
  checkpoint_name="$(basename "${checkpoint_path}")"
  if [ ! -f "${checkpoint_path}" ]; then
    echo "Skipping missing checkpoint ${checkpoint_path}" >&2
    continue
  fi

  output_dir="${OUTPUT_ROOT}/${checkpoint_name}"
  run_name="${RUN_PREFIX}-${checkpoint_name}"

  mkdir -p "${output_dir}"
  echo "Running layerwise InfoRidge for ${checkpoint_name}"

  WANDB_MODE="${WANDB_MODE}" \
  DISABLE_COMPILE=1 \
  OMP_NUM_THREADS=8 \
  python pretrain.py \
    data_path="${DATA_PATH}" \
    project_name="${PROJECT_NAME}" \
    run_name="${run_name}" \
    checkpoint_path="${output_dir}" \
    load_checkpoint_path="${checkpoint_path}" \
    analysis_only=true \
    global_batch_size="${GLOBAL_BATCH_SIZE}" \
    inforidge_config.enabled=true \
    inforidge_config.max_batches="${MAX_BATCHES}" \
    inforidge_config.max_tokens="${MAX_TOKENS}" \
    inforidge_config.sample_mode="${SAMPLE_MODE}" \
    inforidge_config.target_embedding_mode="${TARGET_EMBEDDING_MODE}" \
    inforidge_act_mi.enabled=false

  echo "Saved ${checkpoint_name} results in ${output_dir}"
done
