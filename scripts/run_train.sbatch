#!/bin/bash
#SBATCH --job-name e245751
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --array=0-2%2
#SBATCH --time 05:00:00

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch scripts/run_train.sbatch <output-directory-name>"
    exit 1
fi

RESULT_DIR_NAME="$1"
RESULT_ROOT="outputs/logs/${RESULT_DIR_NAME}"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

date
echo "array task: ${SLURM_ARRAY_TASK_ID} / seed: ${SEED}"
uv run python scripts/exp_run.py \
    --output-dir-name "$RESULT_DIR_NAME" \
    --seed "$SEED"

uv run python scripts/evaluate_by_class_and_font.py \
    --model "${RESULT_ROOT}/model/ssd_calculator_seed${SEED}_state_dict.pth" \
    --output-dir "${RESULT_ROOT}/csv" \
    --output-suffix "seed${SEED}"

date
