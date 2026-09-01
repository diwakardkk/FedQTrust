#!/bin/bash
#SBATCH --job-name=fedqtrust-exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

python3 -m fedqtrust run --experiment "${EXPERIMENT:-E1}" --profile paper --device cuda --resume --output-dir output

