#!/usr/bin/env bash
set -euo pipefail
python3 -m fedqtrust run --experiment E1 --profile phase1 --resume
python3 -m fedqtrust run --experiment E5 --profile phase1 --resume
python3 -m fedqtrust run --experiment E7 --profile phase1 --resume

