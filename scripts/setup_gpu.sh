#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -r requirements-gpu.txt
python3 -m pip install -e .
python3 -m fedqtrust preflight --profile smoke --device auto --output-dir output
