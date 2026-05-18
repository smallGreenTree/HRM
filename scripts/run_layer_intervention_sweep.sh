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
INTERVENTION_MODES="${INTERVENTION_MODES:-bypass shuffle_batch noise}"
LEVELS="${LEVELS:-H L}"
LAYERS="${LAYERS:-0 1 2 3}"
MAX_BATCHES="${MAX_BATCHES:-8}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
NOISE_STD="${NOISE_STD:-0.5}"
UPLOAD_ARTIFACT="${UPLOAD_ARTIFACT:-auto}"

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
    +arch.layer_intervention_noise_std="${NOISE_STD}" \
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
  +arch.layer_intervention_noise_std="${NOISE_STD}" \
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

summary_csv="${OUTPUT_ROOT}/summary.csv"
echo "Writing combined summary: ${summary_csv}"
python scripts/summarize_layer_interventions.py "${OUTPUT_ROOT}" --output-csv "${summary_csv}"

csv_dir="${OUTPUT_ROOT}/csv"
mkdir -p "${csv_dir}"
for case_dir in "${OUTPUT_ROOT}"/*; do
  [[ -d "${case_dir}" ]] || continue
  case_name="$(basename "${case_dir}")"
  short_case="${case_name#${RUN_PREFIX}-${checkpoint_name}-}"
  layerwise_csv="$(find "${case_dir}" -maxdepth 1 -name 'inforidge_step_*.csv' | sort | tail -n 1)"
  act_csv="$(find "${case_dir}" -maxdepth 1 -name 'inforidge_act_mi_step_*.csv' | sort | tail -n 1)"
  if [[ -n "${layerwise_csv}" ]]; then
    cp "${layerwise_csv}" "${csv_dir}/${short_case}__inforidge_layerwise.csv"
  fi
  if [[ -n "${act_csv}" ]]; then
    cp "${act_csv}" "${csv_dir}/${short_case}__inforidge_act_mi.csv"
  fi
done

plot_dir="${OUTPUT_ROOT}/plots"
mkdir -p "${plot_dir}"
echo "Writing result plots: ${plot_dir}"
python scripts/plot_layer_intervention_results.py "${summary_csv}" --output-dir "${plot_dir}"
python scripts/plot_layer_intervention_inforidge.py "${csv_dir}" --output-dir "${plot_dir}"

if [[ "${UPLOAD_ARTIFACT}" == "1" || ( "${UPLOAD_ARTIFACT}" == "auto" && "${WANDB_MODE}" == "online" ) ]]; then
  artifact_name="${ARTIFACT_NAME:-${RUN_PREFIX}-${checkpoint_name}-full-results}"
  upload_run_name="${UPLOAD_RUN_NAME:-artifact-${artifact_name}}"
  echo "Uploading W&B artifact: ${artifact_name}"
  WANDB_MODE="${WANDB_MODE}" python scripts/upload_wandb_files.py \
    --project "${PROJECT_NAME}" \
    --artifact-name "${artifact_name}" \
    --artifact-type analysis \
    --path "${OUTPUT_ROOT}" \
    --recursive \
    --run-name "${upload_run_name}"
fi

echo "Layer intervention sweep complete: ${OUTPUT_ROOT}"
