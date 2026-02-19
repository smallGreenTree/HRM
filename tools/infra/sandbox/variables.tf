variable "runpod_api_key" {
  description = "RunPod API key (get from https://www.runpod.io/console/user/settings)"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.runpod_api_key) > 0
    error_message = "RunPod API key is required. Set via TF_VAR_runpod_api_key or terraform.tfvars"
  }
}

variable "pod_name" {
  description = "Name for the pod"
  type        = string
  default     = "runpod-instance"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.pod_name))
    error_message = "Pod name must contain only lowercase letters, numbers, and hyphens."
  }
}

# Infrastructure configuration
variable "docker_image" {
  description = "Docker image to use"
  type        = string
  default     = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
}

variable "gpu_type_id" {
  description = "GPU type (e.g., 'NVIDIA RTX A4000', 'NVIDIA A40', 'NVIDIA RTX 4090', 'NVIDIA A100-PCIE-40GB')"
  type        = string
  default     = "NVIDIA RTX A4000"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "cloud_type" {
  description = "COMMUNITY (cheaper) or SECURE (more reliable)"
  type        = string
  default     = "COMMUNITY"
}

variable "container_disk_gb" {
  description = "Container disk size in GB"
  type        = number
  default     = 20
}

variable "public_ip" {
  description = "Enable public IP"
  type        = bool
  default     = true
}

variable "startup_command" {
  description = "Docker startup command (list of args)"
  type        = list(string)
  default     = []
}

variable "environment_vars" {
  description = "Environment variables to set in the pod"
  type        = map(string)
  default     = {}
  sensitive   = true
}
