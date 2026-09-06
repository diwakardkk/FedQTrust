#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output/smoke_test_cpu}"

echo "[FedQTrust] CPU smoke test"
echo "[FedQTrust] OUTPUT_DIR=$OUTPUT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m fedqtrust smoke-test --download-data --device cpu --output-dir "$OUTPUT_DIR"

echo "[DONE] CPU smoke test saved under $OUTPUT_DIR"
