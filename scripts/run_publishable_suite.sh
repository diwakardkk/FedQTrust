#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output}"
DEVICE="${DEVICE:-cuda}"

echo "[1/7] Environment preflight"
python -m fedqtrust preflight --profile phase2 --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "[2/7] Unit tests"
python -m pytest -q

echo "[3/7] Smoke test"
python -m fedqtrust smoke-test --download-data --device "$DEVICE" --output-dir "$OUTPUT_DIR/smoke_test"

echo "[4/7] Download data"
python -m fedqtrust download-data --profile paper --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "[5/7] Prepare partitions"
python -m fedqtrust prepare-data --profile paper --device "$DEVICE" --output-dir "$OUTPUT_DIR"

echo "[6/7] Run implemented GPU result bundle"
OUTPUT_DIR="$OUTPUT_DIR/gpu_once" DEVICE="$DEVICE" bash scripts/run_gpu_once.sh

echo "[7/7] Generate report and non-blocking publication audit"
python -m fedqtrust generate-report --profile paper --device "$DEVICE" --output-dir "$OUTPUT_DIR"
python -m fedqtrust run-all --profile paper --device "$DEVICE" --resume --output-dir "$OUTPUT_DIR"
python -m fedqtrust audit-publication --strict-infra --output-dir "$OUTPUT_DIR" || true

echo "[DONE] Saved GPU result bundle under $OUTPUT_DIR/gpu_once/"
echo "[DONE] Publication audit, if not passing, is saved as a readiness report and does not block result collection."
