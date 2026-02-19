variable "pod_name" {
  description = "Name of the RunPod instance"
  type        = string
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.pod_name))
    error_message = "Pod name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "docker_image" {
  description = "Docker image to use for the pod"
  type        = string
  default     = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
}

variable "gpu_type_id" {
  description = "GPU type ID (e.g., 'NVIDIA RTX A4000', 'NVIDIA A40', 'NVIDIA RTX 4090')"
  type        = string
  default     = "NVIDIA RTX A4000"
}

variable "gpu_count" {
  description = "Number of GPUs to allocate"
  type        = number
  default     = 1
  validation {
    condition     = var.gpu_count > 0 && var.gpu_count <= 8
    error_message = "GPU count must be between 1 and 8."
  }
}

variable "cloud_type" {
  description = "Cloud type: 'COMMUNITY' (cheaper, less reliable) or 'SECURE' (more expensive, more reliable)"
  type        = string
  default     = "COMMUNITY"
  validation {
    condition     = contains(["COMMUNITY", "SECURE"], var.cloud_type)
    error_message = "Cloud type must be either 'COMMUNITY' or 'SECURE'."
  }
}

variable "container_disk_gb" {
  description = "Size of container disk in GB"
  type        = number
  default     = 20
  validation {
    condition     = var.container_disk_gb >= 10 && var.container_disk_gb <= 1000
    error_message = "Container disk size must be between 10 and 1000 GB."
  }
}

variable "environment_vars" {
  description = "Environment variables to set in the pod"
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "volume_id" {
  description = "ID of persistent volume to mount (optional)"
  type        = string
  default     = ""
}

variable "volume_mount_path" {
  description = "Path to mount persistent volume"
  type        = string
  default     = "/workspace"
}

variable "startup_command" {
  description = "Docker startup command (list of args)"
  type        = list(string)
  default     = []
}

variable "public_ip" {
  description = "Enable public IP access"
  type        = bool
  default     = true
}
