terraform {
  required_version = ">= 1.5"

  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = ">= 1.0.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}
