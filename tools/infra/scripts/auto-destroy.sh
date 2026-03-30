#!/bin/bash
set -euo pipefail

# Auto-destroy RunPod after X hours.
# Usage: ./auto-destroy.sh <env_dir> <hours>
# Example: ./auto-destroy.sh sandbox 2

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <env_dir> <hours>"
    echo "Example: $0 sandbox 2  (auto-destroy after 2 hours)"
    exit 1
fi

ENV_DIR="$1"
HOURS="$2"
SECONDS=$((HOURS * 3600))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "⏰ Will auto-destroy RunPod in $HOURS hour(s)"
echo "⏰ Started at: $(date)"
echo "⏰ Will destroy at: $(date -v+${HOURS}H)"
echo ""
echo "Press Ctrl+C to cancel"
echo ""

sleep $SECONDS

echo ""
echo "🔥 Time's up! Destroying RunPod..."
(cd "${INFRA_DIR}/${ENV_DIR}" && terraform destroy -auto-approve)

echo ""
echo "✅ Pod destroyed at: $(date)"
