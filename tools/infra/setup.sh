#!/bin/bash
set -euo pipefail

# Setup script for RunPod Infrastructure
# This script helps you get started quickly

echo "============================================="
echo "RunPod Infrastructure Setup"
echo "============================================="
echo ""

# Check prerequisites
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ $1 not found"
        return 1
    else
        echo "✅ $1 found: $($1 --version 2>&1 | head -n1)"
        return 0
    fi
}

echo "Checking prerequisites..."
check_command terraform || {
    echo "Please install Terraform: https://www.terraform.io/downloads"
    exit 1
}

echo ""
echo "Optional tools:"
check_command terragrunt || echo "  ℹ️  Install Terragrunt for advanced features: https://terragrunt.gruntwork.io/docs/getting-started/install/"
check_command make || echo "  ℹ️  Install make for convenient commands"

echo ""
echo "============================================="

# Check if API key is set
if [ -z "${TF_VAR_runpod_api_key:-}" ]; then
    echo "⚠️  RunPod API key not set"
    echo ""
    echo "Please set your RunPod API key:"
    echo "  export TF_VAR_runpod_api_key='your-api-key-here'"
    echo ""
    echo "Get your API key from: https://www.runpod.io/console/user/settings"
    echo ""
    read -p "Do you want to set it now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your RunPod API key: " api_key
        export TF_VAR_runpod_api_key="$api_key"
        echo ""
        echo "✅ API key set for this session"
        echo ""
        echo "To make it permanent, add to your shell profile:"
        echo "  echo 'export TF_VAR_runpod_api_key=\"$api_key\"' >> ~/.bashrc"
    fi
else
    echo "✅ RunPod API key is set"
fi

echo ""
echo "============================================="

# Create terraform.tfvars if it doesn't exist
if [ ! -f "sandbox/terraform.tfvars" ]; then
    echo "Creating terraform.tfvars from example..."
    cp sandbox/terraform.tfvars sandbox/terraform.tfvars
    echo "✅ Created sandbox/terraform.tfvars"
    echo ""
    echo "⚠️  Please edit sandbox/terraform.tfvars with your configuration"
    echo ""
    read -p "Do you want to edit it now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} sandbox/terraform.tfvars
    fi
else
    echo "✅ terraform.tfvars already exists"
fi

echo ""
echo "============================================="
echo "Setup complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit configuration (if not done already):"
echo "   nano sandbox/terraform.tfvars"
echo ""
echo "2. Initialize Terraform:"
echo "   cd sandbox && terraform init"
echo "   # Or with make: make init"
echo ""
echo "3. Apply infrastructure:"
echo "   terraform apply"
echo "   # Or with make: make apply"
echo ""
echo "4. Connect to pod:"
echo "   make ssh"
echo ""
echo "5. Clean up:"
echo "   make destroy"
echo ""
