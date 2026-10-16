#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --job-name=juggle_arm_ampc
#SBATCH --output=hpclogs/%x_%A_%a.log
set -euo pipefail
source "${ROOT:?Set ROOT}/examples/juggling_ampc/claix_env.sh"
export WARP_CACHE_PATH="$ROOT/.warp_cache"
DURATIONS=(5 10 20)
SECONDS_SIM="${DURATIONS[$SLURM_ARRAY_TASK_ID]}"
RESULT="$ROOT/results/arm15_${SECONDS_SIM}s"
"$PYTHON" -u examples/juggling_ampc/run_gpu_pipeline.py \
 --factors "$REFERENCE/gpu_factors" --source "$FRANKA_SOURCE" --calibration "$CALIBRATION" \
 --eigen "$EIGEN_DIR" \
 --collection-multiplier 1 --collection-arm-multiplier 1.5 --validation-multiplier 1 --validation-arm-multiplier 1 \
 --batch 60000 --seconds "$SECONDS_SIM" --eval-batch 1024 --dataset-gib "${DATASET_GIB:-3}" \
 --epochs "${TRAIN_EPOCHS:-100}" --train-batch 8192 --output "$RESULT"
