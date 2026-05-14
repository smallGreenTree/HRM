#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the trained checkpoint file, e.g. /workspace/HRM/checkpoints/maze-corrected-rerun/step_12000.pt}"
DATA_PATH="${DATA_PATH:-data/maze-30x30-hard-1k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/checkpoints/layer-intervention-sweep}"
PROJECT_NAME="${PROJECT_NAME:-maze-hrm-layer-interventions}"
RUN_PREFIX="${RUN_PREFIX:-layer-play}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
WANDB_MODE="${WANDB_MODE:-offline}"
INTERVENTION_MODES="${INTERVENTION_MODES:-bypass zero shuffle_batch mean noise}"
LEVELS="${LEVELS:-H L}"
LAYERS="${LAYERS:-0 1 2 3}"
MAX_BATCHES="${MAX_BATCHES:-8}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

checkpoint_name="$(basename "${CHECKPOINT_PATH}")"

run_case() {
  local level="$1"
  local layer="$2"
  local mode="$3"
  local run_name="${RUN_PREFIX}-${checkpoint_name}-${level}${layer}-${mode}"
  local output_dir="${OUTPUT_ROOT}/${run_name}"

  mkdir -p "${output_dir}"
  echo "Running ${run_name}"

  WANDB_MODE="${WANDB_MODE}" \
  DISABLE_COMPILE=1 \
  OMP_NUM_THREADS=8 \
  python pretrain.py \
    data_path="${DATA_PATH}" \
    project_name="${PROJECT_NAME}" \
    run_name="${run_name}" \
    checkpoint_path="${output_dir}" \
    load_checkpoint_path="${CHECKPOINT_PATH}" \
    analysis_only=true \
    global_batch_size="${GLOBAL_BATCH_SIZE}" \
    +arch.layer_intervention_level="${level}" \
    +arch.layer_intervention_layers="[${layer}]" \
    +arch.layer_intervention_mode="${mode}" \
    inforidge_config.enabled=true \
    inforidge_config.max_batches="${MAX_BATCHES}" \
    inforidge_config.max_tokens="${MAX_TOKENS}" \
    inforidge_config.target_embedding_mode=second_pass \
    inforidge_act_mi.enabled=true
}

baseline_name="${RUN_PREFIX}-${checkpoint_name}-baseline"
baseline_output="${OUTPUT_ROOT}/${baseline_name}"
mkdir -p "${baseline_output}"
echo "Running ${baseline_name}"

WANDB_MODE="${WANDB_MODE}" \
DISABLE_COMPILE=1 \
OMP_NUM_THREADS=8 \
python pretrain.py \
  data_path="${DATA_PATH}" \
  project_name="${PROJECT_NAME}" \
  run_name="${baseline_name}" \
  checkpoint_path="${baseline_output}" \
  load_checkpoint_path="${CHECKPOINT_PATH}" \
  analysis_only=true \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  +arch.layer_intervention_mode=none \
  inforidge_config.enabled=true \
  inforidge_config.max_batches="${MAX_BATCHES}" \
  inforidge_config.max_tokens="${MAX_TOKENS}" \
  inforidge_config.target_embedding_mode=second_pass \
  inforidge_act_mi.enabled=true

for level in ${LEVELS}; do
  for layer in ${LAYERS}; do
    for mode in ${INTERVENTION_MODES}; do
      run_case "${level}" "${layer}" "${mode}"
    done
  done
done

echo "Layer intervention sweep complete: ${OUTPUT_ROOT}"
