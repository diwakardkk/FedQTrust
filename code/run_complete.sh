#!/usr/bin/env bash
# Start the complete configured suite in a detached supervisor.
set -euo pipefail
cd "$(dirname "$0")"
GPUS="${1:-2,4,6}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv. Run this inside the existing FedQTrust_GPU_server folder.' >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" .venv/bin/python run.py preflight --config settings.json
.venv/bin/python server.py start --gpus "$GPUS"
printf '%s\n' 'Detached supervisor started. It verifies data, runs the suite, and exports measured results.'
printf '%s\n' 'Monitor: .venv/bin/python server.py status'
