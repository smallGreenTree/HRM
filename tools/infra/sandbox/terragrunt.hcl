# Terragrunt configuration for sandbox environment
# This provides additional automation and DRY principles on top of Terraform

# Automatically load variables from parent directory
locals {
  # Parse environment name from path
  environment = basename(get_terragrunt_dir())

  # Load common variables
  common_vars = read_terragrunt_config(find_in_parent_folders("common.hcl", "${get_terragrunt_dir()}/common.hcl"), {})

  # Default tags
  default_tags = {
    Environment = local.environment
    ManagedBy   = "Terragrunt"
    Project     = "RunPod"
  }
}

# Generate provider configuration
generate "provider" {
  path      = "provider_override.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "runpod" {
  api_key = var.runpod_api_key
}
EOF
}

# Remote state configuration (optional - uncomment if using S3/GCS backend)
# remote_state {
#   backend = "s3"
#   config = {
#     bucket         = "my-terraform-state-bucket"
#     key            = "runpod/${local.environment}/terraform.tfstate"
#     region         = "us-east-1"
#     encrypt        = true
#     dynamodb_table = "terraform-locks"
#   }
#   generate = {
#     path      = "backend.tf"
#     if_exists = "overwrite_terragrunt"
#   }
# }

# Terraform configuration
terraform {
  source = "."

  # Extra arguments to pass to all Terraform commands
  extra_arguments "common_vars" {
    commands = get_terraform_commands_that_need_vars()

    optional_var_files = [
      "${get_terragrunt_dir()}/terraform.tfvars",
      "${get_parent_terragrunt_dir()}/common.tfvars",
    ]
  }

  # Before and after hooks for better DX
  before_hook "validate_api_key" {
    commands = ["apply", "plan"]
    execute  = ["bash", "-c", "test -n \"$TF_VAR_runpod_api_key\" || (echo 'ERROR: TF_VAR_runpod_api_key not set' && exit 1)"]
  }

  after_hook "show_outputs" {
    commands     = ["apply"]
    execute      = ["terraform", "output", "-json"]
    run_on_error = false
  }
}

# Input values to pass to Terraform
inputs = merge(
  local.common_vars.inputs,
  {
    # Environment-specific overrides can go here
  }
)
