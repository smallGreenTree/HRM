# Makefile for Terraform/Terragrunt operations
# Provides convenient commands for common operations

.PHONY: help init plan apply destroy clean validate fmt check-api-key gpu-types apply-with-timer apply-with-idle-kill startup-smoke

# Default target
.DEFAULT_GOAL := help

# Environment (default: sandbox)
ENV ?= sandbox

# Check if API key is set
check-api-key:
	@if [ -z "$$TF_VAR_runpod_api_key" ]; then \
		echo "ERROR: TF_VAR_runpod_api_key environment variable is not set"; \
		echo "Set it with: export TF_VAR_runpod_api_key='your-api-key'"; \
		exit 1; \
	fi

test-connection: check-api-key ## Test RunPod API connection and show available GPUs
	@bash scripts/test-runpod-connection.sh

help: ## Show this help message
	@echo "RunPod Infrastructure - Makefile Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: ## Initialize Terraform
	cd $(ENV) && terraform init

plan: check-api-key ## Plan infrastructure changes
	cd $(ENV) && terraform plan

apply: check-api-key ## Apply infrastructure changes
	@echo "⚠️  WARNING: Pod will keep running until you destroy it!"
	@echo "⚠️  Cost: ~\$$1.29/hour for A100 80GB"
	@echo "⚠️  Set an alarm or use: make apply-with-timer HOURS=2"
	@echo ""
	cd $(ENV) && terraform apply

apply-auto: check-api-key ## Apply without confirmation
	cd $(ENV) && terraform apply -auto-approve

destroy: check-api-key ## Destroy infrastructure
	cd $(ENV) && terraform destroy

destroy-auto: check-api-key ## Destroy without confirmation
	cd $(ENV) && terraform destroy -auto-approve

validate: ## Validate Terraform configuration
	cd $(ENV) && terraform validate

fmt: ## Format Terraform files
	terraform fmt -recursive .

check: validate fmt ## Run validation and formatting

output: ## Show outputs
	cd $(ENV) && terraform output

output-json: ## Show outputs in JSON format
	cd $(ENV) && terraform output -json

ssh: ## SSH into the pod (RunPod proxy)
	@bash scripts/ssh-runpod.sh $(ENV)

gpu-types: check-api-key ## List available GPU types
	cd $(ENV) && terraform apply -auto-approve && terraform output available_gpu_types

clean: ## Clean Terraform cache
	find . -type d -name ".terraform" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.tfplan" -delete 2>/dev/null || true

# Terragrunt commands
tg-init: ## Initialize with Terragrunt
	cd $(ENV) && terragrunt init

tg-plan: check-api-key ## Plan with Terragrunt
	cd $(ENV) && terragrunt plan

tg-apply: check-api-key ## Apply with Terragrunt
	cd $(ENV) && terragrunt apply

tg-destroy: check-api-key ## Destroy with Terragrunt
	cd $(ENV) && terragrunt destroy

tg-output: ## Show outputs with Terragrunt
	cd $(ENV) && terragrunt output

# Quick commands
quick-start: init apply-auto output ## Quick start: init, apply, and show outputs
	@echo ""
	@echo "Pod is ready! Use 'make ssh' to connect"

quick-stop: destroy-auto ## Quick stop: destroy infrastructure

apply-with-timer: check-api-key ## Apply and auto-destroy after N hours (make apply-with-timer HOURS=2)
	@if [ -z "$(HOURS)" ]; then \
		echo "ERROR: Specify hours with: make apply-with-timer HOURS=2"; \
		exit 1; \
	fi
	@echo "🚀 Starting pod with $(HOURS) hour auto-destroy timer..."
	cd $(ENV) && terraform apply -auto-approve
	@echo ""
	@echo "⏰ Auto-destroy timer set for $(HOURS) hour(s)"
	@echo "⏰ Started at: $$(date)"
	@bash ../scripts/auto-destroy.sh $(HOURS)

apply-with-idle-kill: check-api-key ## Apply and auto-destroy when GPU is idle (make apply-with-idle-kill IDLE_MIN=20 UTIL=5 POLL=120)
	@if [ -z "$(IDLE_MIN)" ]; then \
		echo "ERROR: Specify IDLE_MIN with: make apply-with-idle-kill IDLE_MIN=20 UTIL=5 POLL=120"; \
		exit 1; \
	fi
	@if [ -z "$(UTIL)" ]; then \
		echo "ERROR: Specify UTIL with: make apply-with-idle-kill IDLE_MIN=20 UTIL=5 POLL=120"; \
		exit 1; \
	fi
	@if [ -z "$(POLL)" ]; then \
		echo "ERROR: Specify POLL with: make apply-with-idle-kill IDLE_MIN=20 UTIL=5 POLL=120"; \
		exit 1; \
	fi
	@echo "🚀 Starting pod with idle watchdog..."
	cd $(ENV) && terraform apply -auto-approve
	@echo ""
	@echo "⏰ Idle watchdog armed: IDLE_MIN=$(IDLE_MIN) UTIL=$(UTIL)% POLL=$(POLL)s"
	@bash scripts/idle-destroy.sh $(ENV) $(IDLE_MIN) $(UTIL) $(POLL)

startup-smoke: ## Configure startup_command to clone HRM + run bootstrap smoke test
	@bash scripts/set-startup-smoke.sh
