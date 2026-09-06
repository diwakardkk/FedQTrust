#!/usr/bin/env bash
set -euo pipefail

PYTORCH_CUDA="${PYTORCH_CUDA:-cu124}"
OUTPUT_DIR="${OUTPUT_DIR:-output/publication_suite}"
MODE="${MODE:-paper}"
ROUNDS="${ROUNDS:-}"
SEEDS="${SEEDS:-}"
if [ "$PYTORCH_CUDA" = "cpu" ] && [ -z "${DEVICE:-}" ]; then
  DEVICE="cpu"
else
  DEVICE="${DEVICE:-cuda}"
fi

echo "[FedQTrust] Publication one-command runner"
echo "[FedQTrust] PYTORCH_CUDA=$PYTORCH_CUDA OUTPUT_DIR=$OUTPUT_DIR DEVICE=$DEVICE MODE=$MODE"

if [ "$PYTORCH_CUDA" != "cpu" ] && [ "$DEVICE" != "cpu" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. Set PYTORCH_CUDA=cpu DEVICE=cpu for a CPU smoke/result run." >&2
    exit 1
  fi
  nvidia-smi
else
  echo "[FedQTrust] CPU mode selected; skipping nvidia-smi."
fi

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
    python -m pip install torch torchvision
    ;;
  *)
    echo "[ERROR] Unsupported PYTORCH_CUDA value: $PYTORCH_CUDA" >&2
    echo "Allowed examples: cu118, cu121, cu124, cu126" >&2
    exit 1
    ;;
esac

python -m pip install -r requirements-dev.txt
python -m pip install -e .

OUTPUT_DIR="$OUTPUT_DIR" DEVICE="$DEVICE" MODE="$MODE" ROUNDS="$ROUNDS" SEEDS="$SEEDS" bash scripts/run_publishable_suite.sh
