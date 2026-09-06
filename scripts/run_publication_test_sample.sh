#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output/publication_test_sample}"
DEVICE="${DEVICE:-cpu}"
ROUNDS="${ROUNDS:-8}"
SEEDS="${SEEDS:-42}"

echo "[FedQTrust] Fast publication-suite test sample"
echo "[FedQTrust] OUTPUT_DIR=$OUTPUT_DIR DEVICE=$DEVICE ROUNDS=$ROUNDS SEEDS=$SEEDS"

python -m fedqtrust publication-suite \
  --mode test \
  --device "$DEVICE" \
  --rounds "$ROUNDS" \
  --seeds "$SEEDS" \
  --output-dir "$OUTPUT_DIR"

python -m fedqtrust audit-publication --output-dir "$OUTPUT_DIR"

echo "[DONE] Test-sample publication artifacts saved under $OUTPUT_DIR"
