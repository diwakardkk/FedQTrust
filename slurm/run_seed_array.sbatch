#!/bin/bash
#SBATCH --job-name=fedqtrust-seeds
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

SEEDS=(42 123 456 789 999)
python3 -m fedqtrust run --experiment "${EXPERIMENT:-E1}" --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" --profile paper --device cuda --resume --output-dir output

