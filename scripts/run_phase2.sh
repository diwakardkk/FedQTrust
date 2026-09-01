#!/usr/bin/env bash
set -euo pipefail
for exp in E2 E3 E4 E6 E8 E9; do
  python3 -m fedqtrust run --experiment "$exp" --profile phase2 --resume
done

