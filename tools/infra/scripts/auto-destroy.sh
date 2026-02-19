#!/bin/bash
set -e

# Auto-destroy RunPod after X hours
# Usage: ./auto-destroy.sh 2  (destroys after 2 hours)

if [ $# -eq 0 ]; then
    echo "Usage: $0 <hours>"
    echo "Example: $0 2  (auto-destroy after 2 hours)"
    exit 1
fi

HOURS=$1
SECONDS=$((HOURS * 3600))

echo "⏰ Will auto-destroy RunPod in $HOURS hour(s)"
echo "⏰ Started at: $(date)"
echo "⏰ Will destroy at: $(date -v+${HOURS}H)"
echo ""
echo "Press Ctrl+C to cancel"
echo ""

sleep $SECONDS

echo ""
echo "🔥 Time's up! Destroying RunPod..."
cd /Users/antonis-antono/dev/experiment-llm/tools/infra/sandbox
terraform destroy -auto-approve

echo ""
echo "✅ Pod destroyed at: $(date)"
