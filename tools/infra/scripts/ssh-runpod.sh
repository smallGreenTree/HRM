#!/bin/bash
set -euo pipefail

# SSH helper for RunPod via the proxy endpoint.
# Uses pod_id from Terraform outputs and connects to ssh.runpod.io.
# Usage: ./ssh-runpod.sh [env_dir]

ENV_DIR="${1:-sandbox}"

POD_ID=$(cd "$ENV_DIR" && terraform output -raw pod_id 2>/dev/null || true)
if [ -z "$POD_ID" ]; then
  echo "ERROR: Pod ID not found. Has the infrastructure been applied?"
  exit 1
fi

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
  echo "Connecting via RunPod proxy: ssh $SSH_TARGET -i $SSH_KEY"
  ssh -i "$SSH_KEY" "$SSH_TARGET"
else
  echo "Connecting via RunPod proxy: ssh $SSH_TARGET"
  echo "NOTE: No SSH key found. If auth fails, set SSH_KEY=/path/to/key"
  ssh "$SSH_TARGET"
fi
