#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-data/synthetic-arithmetic-k5}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
TEST_SIZE="${TEST_SIZE:-5000}"
NUM_TERMS="${NUM_TERMS:-10}"
TRAIN_MODULUS="${TRAIN_MODULUS:-5}"
TEST_MODULUS="${TEST_MODULUS:-}"
NOISE_RANGE="${NOISE_RANGE:-100}"
SEED="${SEED:-0}"

cd "${ROOT_DIR}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --train-size "${TRAIN_SIZE}"
  --test-size "${TEST_SIZE}"
  --num-terms "${NUM_TERMS}"
  --train-modulus "${TRAIN_MODULUS}"
  --noise-range "${NOISE_RANGE}"
  --seed "${SEED}"
)

if [ -n "${TEST_MODULUS}" ]; then
  ARGS+=(--test-modulus "${TEST_MODULUS}")
fi

python dataset/build_synthetic_arithmetic_dataset.py "${ARGS[@]}"
