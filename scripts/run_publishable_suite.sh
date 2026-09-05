#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output}"
DEVICE="${DEVICE:-cuda}"

echo "[1/7] Paper preflight"
python -m fedqtrust preflight --profile paper --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "[2/7] Unit tests"
python -m pytest -q

echo "[3/7] Smoke test"
python -m fedqtrust smoke-test --download-data --device "$DEVICE" --output-dir "$OUTPUT_DIR/smoke_test"

echo "[4/7] Download data"
python -m fedqtrust download-data --profile paper --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "[5/7] Prepare partitions"
python -m fedqtrust prepare-data --profile paper --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "[6/7] Run E1-E9"
python -m fedqtrust run-all --profile paper --device "$DEVICE" --resume --output-dir "$OUTPUT_DIR"

echo "[7/7] Generate report and audit"
python -m fedqtrust generate-report --profile paper --device "$DEVICE" --output-dir "$OUTPUT_DIR"
python -m fedqtrust audit-publication --strict-infra --output-dir "$OUTPUT_DIR"
