#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-paper}"
ROUNDS="${ROUNDS:-}"
SEEDS="${SEEDS:-}"

echo "[FedQTrust] Publication artifact suite"
echo "[FedQTrust] OUTPUT_DIR=$OUTPUT_DIR DEVICE=$DEVICE MODE=$MODE"

echo "[1/4] Unit tests"
python -m pytest -q

echo "[2/4] Smoke test"
python -m fedqtrust smoke-test --download-data --device "$DEVICE" --output-dir "$OUTPUT_DIR/smoke_test"

echo "[3/4] Generate publication figures, tables, statistics, and experiment artifacts"
args=(
  -m fedqtrust publication-suite
  --mode "$MODE"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
  --download-data
)

if [ -n "$ROUNDS" ]; then
  args+=(--rounds "$ROUNDS")
fi

if [ -n "$SEEDS" ]; then
  args+=(--seeds "$SEEDS")
fi

python "${args[@]}"

echo "[4/4] Audit publication artifact completeness"
python -m fedqtrust audit-publication --output-dir "$OUTPUT_DIR"

echo "[DONE] Saved publication artifact bundle under $OUTPUT_DIR"
