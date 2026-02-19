# Common configuration shared across all environments
# This file is loaded by Terragrunt configurations

locals {
  # Project metadata
  project_name = "llm-benchmarking"
  owner        = "experiment-team"

  # Common tags to apply to all resources
  common_tags = {
    Project   = "LLM-Benchmarking"
    ManagedBy = "Terragrunt"
  }

  # Default experiment settings (can be overridden per environment)
  default_experiment = {
    batch_size    = 1
    n_trials      = 3
    prompt_source = "random"
    framework     = "vllm"
  }

  # Default infrastructure settings
  default_infra = {
    cloud_type         = "COMMUNITY"
    volume_size_gb     = 50
    container_disk_gb  = 20
    keep_alive         = true
  }
}

# Inputs to merge with environment-specific configurations
inputs = {
  # Default values that can be overridden
  batch_size    = local.default_experiment.batch_size
  n_trials      = local.default_experiment.n_trials
  prompt_source = local.default_experiment.prompt_source
  framework     = local.default_experiment.framework
  cloud_type    = local.default_infra.cloud_type
  volume_size_gb = local.default_infra.volume_size_gb
  container_disk_gb = local.default_infra.container_disk_gb
  keep_alive    = local.default_infra.keep_alive
}
