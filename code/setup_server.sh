#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# The server mixes Volta V100 and Ampere GPUs. CUDA 13 wheels omit Volta.
if [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]; then
  .venv/bin/python -m pip install "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" \
    --index-url https://download.pytorch.org/whl/cu126
fi
.venv/bin/python -m pip install -r requirements-manuscript.txt
.venv/bin/python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())'
printf '%s\n' 'Installed Python dependencies. Activate .venv, configure settings.json, and run preflight.'
