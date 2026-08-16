#!/bin/bash
#SBATCH --job-name e245751
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 05:00:00

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch scripts/run_train.sbatch <output-directory-name>"
    exit 1
fi

RESULT_DIR_NAME="$1"
RESULT_ROOT="outputs/logs/${RESULT_DIR_NAME}"

date
uv run python scripts/exp_run.py --output-dir-name "$RESULT_DIR_NAME"

for SEED in 42 43 44; do
    uv run python scripts/evaluate_by_class_and_font.py \
        --model "${RESULT_ROOT}/model/ssd_calculator_seed${SEED}_state_dict.pth" \
        --output-dir "${RESULT_ROOT}/csv" \
        --output-suffix "seed${SEED}"
done

date
