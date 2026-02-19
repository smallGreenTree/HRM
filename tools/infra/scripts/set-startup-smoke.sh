#!/bin/bash
set -euo pipefail

# Configure startup_command in sandbox/terraform.tfvars to auto-clone and run bootstrap.
# Usage: ./set-startup-smoke.sh [repo_url] [branch] [workdir]

REPO_URL="${1:-https://github.com/smallGreenTree/HRM.git}"
BRANCH="${2:-exp}"
WORKDIR="${3:-/workspace/HRM}"

ENV_DIR="$(cd "$(dirname "$0")/../sandbox" && pwd)"
TFVARS="${ENV_DIR}/terraform.tfvars"
EXAMPLE="${ENV_DIR}/terraform.tfvars.example"

if [ ! -f "$TFVARS" ]; then
  cp "$EXAMPLE" "$TFVARS"
fi

STARTUP_CMD="bash -lc 'apt-get update && apt-get install -y git && \
git clone ${REPO_URL} ${WORKDIR} && \
cd ${WORKDIR} && git checkout ${BRANCH} && \
bash tools/infra/scripts/bootstrap-smoke.sh'"

python3 - <<PY
import io
from pathlib import Path

tfvars = Path("${TFVARS}")
lines = tfvars.read_text().splitlines()
out = []
replaced = False
for line in lines:
    if line.strip().startswith("startup_command"):
        out.append(f'startup_command = "{STARTUP_CMD}"')
        replaced = True
    else:
        out.append(line)
if not replaced:
    out.append("")
    out.append(f'startup_command = "{STARTUP_CMD}"')
tfvars.write_text("\n".join(out) + "\n")
print(f"Updated startup_command in {tfvars}")
PY
