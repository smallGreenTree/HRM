terraform {
  required_version = ">= 1.5"

  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = ">= 1.0.0"
    }
  }
}

# Configure RunPod provider
provider "runpod" {
  api_key = var.runpod_api_key
}

# Use the runpod-instance module
module "runpod_pod" {
  source = "../modules/runpod-instance"

  # Basic configuration
  pod_name        = var.pod_name
  docker_image    = var.docker_image
  gpu_type_id     = var.gpu_type_id
  gpu_count       = var.gpu_count
  cloud_type      = var.cloud_type

  # Storage configuration
  container_disk_gb = var.container_disk_gb

  # Network configuration
  public_ip = var.public_ip

  # Environment variables
  environment_vars = var.environment_vars

  # Startup command
  startup_command = var.startup_command
}
