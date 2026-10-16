#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=juggle_fast
set -euo pipefail
# ROOT is a private source copy for this one job, never a shared mutable checkout.
source "${ROOT:?Set the private job ROOT}/examples/juggling_ampc/claix_env.sh"
export WARP_CACHE_PATH="$ROOT/.warp_cache"
RESULT="$ROOT/artifacts"
"$PYTHON" -u examples/juggling_ampc/run_gpu_pipeline.py \
 --factors "$REFERENCE/gpu_factors" --source "$FRANKA_SOURCE" --calibration "$CALIBRATION" \
 --eigen "$EIGEN_DIR" \
 --collection-multiplier 1 --collection-arm-multiplier 1.5 \
 --validation-multiplier 1 --validation-arm-multiplier 1 \
 --batch 60000 --seconds "${SECONDS_SIM:?Set 5, 10, or 20}" --eval-batch 1024 \
 --dataset-gib "${DATASET_GIB:-3}" --epochs "${TRAIN_EPOCHS:-100}" --train-batch 8192 \
 --output "$RESULT"
