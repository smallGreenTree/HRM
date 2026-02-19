terraform {
  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = ">= 1.0.0"
    }
  }
}

# RunPod GPU Pod Instance
resource "runpod_pod" "gpu_instance" {
  name       = var.pod_name
  image_name = var.docker_image

  # Compute type
  compute_type = "GPU"
  cloud_type   = var.cloud_type

  # GPU configuration
  gpu_count    = var.gpu_count
  gpu_type_ids = [var.gpu_type_id]  # Provider expects a list

  # Storage
  container_disk_in_gb = var.container_disk_gb

  # Network volume (optional)
  network_volume_id = var.volume_id != "" ? var.volume_id : null
  volume_mount_path = var.volume_mount_path != "" ? var.volume_mount_path : null

  # Environment variables
  env = var.environment_vars

  # Container startup (expects a list)
  docker_start_cmd = length(var.startup_command) > 0 ? var.startup_command : null

  # Networking
  support_public_ip = var.public_ip
}
