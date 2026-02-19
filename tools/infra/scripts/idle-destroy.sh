#!/bin/bash
set -euo pipefail

# Idle watchdog: destroys the pod when GPU utilization stays below a threshold
# Usage: ./idle-destroy.sh <env_dir> <idle_minutes> <util_threshold> <poll_seconds>
# Example: ./idle-destroy.sh sandbox 20 5 120

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <env_dir> <idle_minutes> <util_threshold> <poll_seconds>"
    exit 1
fi

ENV_DIR="$1"
IDLE_MINUTES="$2"
UTIL_THRESHOLD="$3"
POLL_SECONDS="$4"

if ! [[ "$IDLE_MINUTES" =~ ^[0-9]+$ ]] || ! [[ "$UTIL_THRESHOLD" =~ ^[0-9]+$ ]] || ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: idle_minutes, util_threshold, poll_seconds must be integers."
    exit 1
fi

POD_ID=$(cd "$ENV_DIR" && terraform output -raw pod_id 2>/dev/null || true)
if [ -z "$POD_ID" ]; then
    echo "ERROR: Pod ID not found. Has the infrastructure been applied?"
    exit 1
fi

IDLE_TARGET_SECONDS=$((IDLE_MINUTES * 60))
IDLE_SECONDS=0
FAILS=0
MAX_FAILS=5

echo "=============================================="
echo "Idle Watchdog"
echo "=============================================="
echo "Pod ID: $POD_ID"
echo "Idle threshold: ${UTIL_THRESHOLD}% for ${IDLE_MINUTES} min"
echo "Poll interval: ${POLL_SECONDS} sec"
echo "=============================================="

while true; do
    set +e
    SSH_KEY="${SSH_KEY:-}"
    if [ -z "$SSH_KEY" ]; then
        if [ -f "$HOME/.ssh/id_ed25519" ]; then
            SSH_KEY="$HOME/.ssh/id_ed25519"
        elif [ -f "$HOME/.ssh/id_rsa" ]; then
            SSH_KEY="$HOME/.ssh/id_rsa"
        fi
    fi

    SSH_TARGET="${POD_ID}@ssh.runpod.io"
    if [ -n "$SSH_KEY" ]; then
        UTIL_RAW=$(ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" \
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)
    else
        UTIL_RAW=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" \
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)
    fi
    STATUS=$?
    set -e

    if [ "$STATUS" -ne 0 ] || [ -z "$UTIL_RAW" ]; then
        FAILS=$((FAILS + 1))
        echo "⚠️  Could not query GPU utilization (fail ${FAILS}/${MAX_FAILS}). Retrying..."
        if [ "$FAILS" -ge "$MAX_FAILS" ]; then
            echo "❌ Too many failures. Not destroying automatically."
            exit 1
        fi
        sleep "$POLL_SECONDS"
        continue
    fi

    FAILS=0
    # Compute max utilization across GPUs
    MAX_UTIL=0
    while IFS= read -r line; do
        util=$(echo "$line" | tr -d '[:space:]')
        if [[ "$util" =~ ^[0-9]+$ ]] && [ "$util" -gt "$MAX_UTIL" ]; then
            MAX_UTIL="$util"
        fi
    done <<< "$UTIL_RAW"

    if [ "$MAX_UTIL" -le "$UTIL_THRESHOLD" ]; then
        IDLE_SECONDS=$((IDLE_SECONDS + POLL_SECONDS))
        echo "Idle: ${MAX_UTIL}% (idle ${IDLE_SECONDS}/${IDLE_TARGET_SECONDS} sec)"
    else
        if [ "$IDLE_SECONDS" -gt 0 ]; then
            echo "Active: ${MAX_UTIL}% (reset idle counter)"
        else
            echo "Active: ${MAX_UTIL}%"
        fi
        IDLE_SECONDS=0
    fi

    if [ "$IDLE_SECONDS" -ge "$IDLE_TARGET_SECONDS" ]; then
        echo "🔥 Idle threshold reached. Destroying pod..."
        (cd "$ENV_DIR" && terraform destroy -auto-approve)
        echo "✅ Pod destroyed due to idle timeout."
        exit 0
    fi

    sleep "$POLL_SECONDS"
done
