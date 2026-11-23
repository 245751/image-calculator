#!/bin/bash
#SBATCH --job-name visualize_predictions
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 05:00:00

set -euo pipefail

# プロジェクトルートから投入する。ログ用ディレクトリは投入前に作成する。
# mkdir -p outputs/slurm
# sbatch scripts/run_visualize_predictions.sbatch --characters 1 7 --iou-threshold 0.85
cd "${SLURM_SUBMIT_DIR:-$PWD}"

date
uv run python -u scripts/visualize_predictions.py --device cuda "$@"
date
