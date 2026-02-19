
#!/bin/bash
set -euo pipefail

# Test RunPod API connection and list available resources
# This script verifies your API key works and shows what's available

echo "=============================================="
echo "RunPod Connection Test"
echo "=============================================="
echo ""

# Check if API key is set
if [ -z "${TF_VAR_runpod_api_key:-}" ]; then
    echo "❌ ERROR: RunPod API key not set"
    echo ""
    echo "Please set your API key:"
    echo "  export TF_VAR_runpod_api_key='your-api-key-here'"
    echo ""
    echo "Get your API key from:"
    echo "  https://www.runpod.io/console/user/settings"
    echo ""
    exit 1
fi

API_KEY="$TF_VAR_runpod_api_key"
echo "✅ API key found (${#API_KEY} characters)"
echo ""

# Test API connection by getting account info
echo "Testing API connection..."
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    "https://api.runpod.io/graphql" \
    -d '{"query": "query { myself { id email } }"}')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n1)

if [ "$HTTP_CODE" != "200" ]; then
    echo "❌ API connection failed (HTTP $HTTP_CODE)"
    echo ""
    echo "Response: $BODY"
    echo ""
    echo "Please check:"
    echo "  1. Your API key is correct"
    echo "  2. You have an active RunPod account"
    echo "  3. You have credits in your account"
    echo ""
    exit 1
fi

echo "✅ API connection successful!"
echo ""

# Extract email if present
EMAIL=$(echo "$BODY" | grep -o '"email":"[^"]*"' | cut -d'"' -f4 || echo "N/A")
echo "Connected as: $EMAIL"
echo ""

# Get available GPU types
echo "=============================================="
echo "Fetching available GPU types..."
echo "=============================================="
echo ""

GPU_RESPONSE=$(curl -s \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    "https://api.runpod.io/graphql" \
    -d '{"query": "query GpuTypes { gpuTypes { id displayName memoryInGb secureCloud communityCloud lowestPrice { minimumBidPrice uninterruptablePrice } } }"}')

# Parse and display GPU types
echo "$GPU_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    gpu_types = data.get('data', {}).get('gpuTypes', [])

    if not gpu_types:
        print('⚠️  No GPU types found or API response error')
        sys.exit(1)

    print(f'Found {len(gpu_types)} GPU types:\n')
    print(f'{'GPU Type':<30}  {'Min Price/hr':<15}')
    print('-' * 85)
    for gpu in gpu_types:
        name = gpu.get('id', )
        print(name)
    print('\n✅ = Available, ❌ = Not available')
    print()
except Exception as e:
    print(f'Error parsing response: {e}')
    print('Raw response:', file=sys.stderr)
    print(sys.stdin.read(), file=sys.stderr)
    sys.exit(1)
" || {
    echo "⚠️  Could not parse GPU types. Raw response:"
    echo "$GPU_RESPONSE"
}

echo ""
echo "=============================================="
echo "Checking account balance..."
echo "=============================================="
echo ""

BALANCE_RESPONSE=$(curl -s \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    "https://api.runpod.io/graphql" \
    -d '{"query": "query { myself { id clientBalance } }"}')

echo "$BALANCE_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    balance = data.get('data', {}).get('myself', {}).get('clientBalance')
    if balance is not None:
        print(f'💰 Account balance: \${balance:.2f}')
        if balance < 1:
            print('⚠️  Warning: Low balance. Add credits at https://www.runpod.io/console/user/billing')
        else:
            print('✅ Sufficient balance to run experiments')
    else:
        print('⚠️  Could not retrieve balance')
except:
    print('⚠️  Could not parse balance information')
"

echo ""
echo "=============================================="
echo "Connection Test Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Choose a GPU type from the list above"
echo "  2. Configure your experiment in sandbox/terraform.tfvars"
echo "  3. Run: make init && make apply"
echo ""
