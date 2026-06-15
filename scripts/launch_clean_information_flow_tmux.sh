#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-clean-info-flow}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/reports/clean-information-flow-step24304}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/run.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Run scripts/setup_runpod.sh first." >&2
  exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export OUTPUT_DIR
tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${ROOT_DIR}' && bash scripts/run_clean_information_flow.sh 2>&1 | tee '${LOG_PATH}'"

echo "Started Experiment 1 in tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Logs: tail -f ${LOG_PATH}"
