#!/usr/bin/env bash
set -euo pipefail

PYTORCH_CUDA="${PYTORCH_CUDA:-cu124}"
OUTPUT_DIR="${OUTPUT_DIR:-output/gpu_once}"
DEVICE="${DEVICE:-cuda}"

echo "[FedQTrust] GPU one-command results runner"
echo "[FedQTrust] PYTORCH_CUDA=$PYTORCH_CUDA OUTPUT_DIR=$OUTPUT_DIR DEVICE=$DEVICE"
echo "[FedQTrust] This runs the implemented GPU result bundle, not the strict E1-E9 publication audit."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. This GPU results runner requires an NVIDIA GPU server." >&2
  exit 1
fi

nvidia-smi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

case "$PYTORCH_CUDA" in
  cu118|cu121|cu124|cu126)
    python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
    ;;
  cpu)
    echo "[ERROR] cpu selected, but this one-command collaborator run requires CUDA." >&2
    exit 1
    ;;
  *)
    echo "[ERROR] Unsupported PYTORCH_CUDA value: $PYTORCH_CUDA" >&2
    echo "Allowed examples: cu118, cu121, cu124, cu126" >&2
    exit 1
    ;;
esac

python -m pip install -r requirements-dev.txt
python -m pip install -e .

OUTPUT_DIR="$OUTPUT_DIR" DEVICE="$DEVICE" bash scripts/run_gpu_once.sh
