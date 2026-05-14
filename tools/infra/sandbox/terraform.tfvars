# Example Terraform Configuration
#
# SECURITY: DO NOT PUT YOUR API KEY IN THIS FILE!
# API key is automatically read from environment variable: TF_VAR_runpod_api_key
#
# Usage:
#   1. Copy this file: cp terraform.tfvars.example terraform.tfvars
#   2. Edit the values below (but NOT the API key)
#   3. Run terraform with: export TF_VAR_runpod_api_key="your-key" && terraform apply

# Pod Configuration
pod_name     = "my-runpod"
docker_image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# GPU Configuration
# Get available types with: make test-connection
gpu_type_id = "NVIDIA GeForce RTX 4090"
gpu_count   = 1
cloud_type  = "COMMUNITY"  # "COMMUNITY" (cheaper) or "SECURE" (reliable)

# Storage Configuration
container_disk_gb = 20

# Network Configuration
public_ip = true

# Startup Command (optional)
startup_command = [
  "/bin/bash",
  "-lc",
  "apt-get update && apt-get install -y git && if [ -d /workspace/HRM/.git ]; then cd /workspace/HRM && git fetch --all --prune; else git clone https://github.com/smallGreenTree/HRM.git /workspace/HRM && cd /workspace/HRM; fi && git checkout -B inforidge origin/inforidge && git pull --ff-only || true && sleep infinity"
]

# Environment Variables (optional)
# environment_vars = {
#   MY_VAR = "value"
# }
