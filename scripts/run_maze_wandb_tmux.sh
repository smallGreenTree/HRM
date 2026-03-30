#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SESSION_NAME="${SESSION_NAME:-maze-corrected}"
DATA_PATH="${DATA_PATH:-data/maze-30x30-hard-1k}"
PROJECT_NAME="${PROJECT_NAME:-maze-hrm}"
RUN_NAME="${RUN_NAME:-maze-corrected-rerun}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/checkpoints/${RUN_NAME}}"

EPOCHS="${EPOCHS:-20000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
NPROC="${NPROC:-8}"
OMP_THREADS="${OMP_THREADS:-8}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-96}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NPROC * LOCAL_BATCH_SIZE))}"

LR="${LR:-1e-4}"
PUZZLE_EMB_LR="${PUZZLE_EMB_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0}"
PUZZLE_EMB_WEIGHT_DECAY="${PUZZLE_EMB_WEIGHT_DECAY:-1.0}"

WANDB_MODE="${WANDB_MODE:-online}"
LOG_PATH="${LOG_PATH:-${CHECKPOINT_PATH}/train.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Run scripts/setup_runpod.sh first." >&2
  exit 1
fi

if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb is not installed. Run scripts/setup_runpod.sh first." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_PATH}"

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "Logging in to Weights & Biases with WANDB_API_KEY"
  wandb login --relogin "${WANDB_API_KEY}"
else
  echo "WANDB_API_KEY is not set."
  echo "Assuming this pod is already logged in to Weights & Biases."
  echo "If not, run: wandb login"
fi

LAUNCH_SCRIPT="${CHECKPOINT_PATH}/launch_${SESSION_NAME}.sh"
cat > "${LAUNCH_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail

cd "${ROOT_DIR}"

if [ ! -d "${DATA_PATH}/train" ] || [ ! -d "${DATA_PATH}/test" ]; then
  python dataset/build_maze_dataset.py
fi

mkdir -p "${CHECKPOINT_PATH}"

WANDB_MODE="${WANDB_MODE}" \
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
  eval_interval="${EVAL_INTERVAL}" \
  lr="${LR}" \
  puzzle_emb_lr="${PUZZLE_EMB_LR}" \
  weight_decay="${WEIGHT_DECAY}" \
  puzzle_emb_weight_decay="${PUZZLE_EMB_WEIGHT_DECAY}" \
  2>&1 | tee "${LOG_PATH}"
EOF

chmod +x "${LAUNCH_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" "${LAUNCH_SCRIPT}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Tail logs with: tail -f ${LOG_PATH}"
echo "Launcher script: ${LAUNCH_SCRIPT}"
