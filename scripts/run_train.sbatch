#!/bin/bash
#SBATCH --job-name e245751
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 05:00:00

date
uv run python scripts/exp_run.py
date