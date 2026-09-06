#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output/gpu_once}"
ROUNDS="${ROUNDS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
MODELS="${MODELS:-classical_cnn,fedqcnn}"
DATASETS="${DATASETS:-pathmnist,octmnist,pneumoniamnist,retinamnist,breastmnist}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

args=(
  -m fedqtrust run-gpu-once
  --download-data
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
  --rounds "$ROUNDS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --models "$MODELS"
  --datasets "$DATASETS"
)

if [ "$REQUIRE_CUDA" = "1" ]; then
  args+=(--require-cuda --amp)
fi

if [ "${AMP:-}" = "1" ] && [ "$REQUIRE_CUDA" != "1" ]; then
  args+=(--amp)
fi

python "${args[@]}"
