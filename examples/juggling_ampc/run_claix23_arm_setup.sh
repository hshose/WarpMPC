#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --job-name=juggle_arm_qual
#SBATCH --output=hpclogs/%x_%A_%a.log
set -euo pipefail
source "${ROOT:?Set ROOT}/examples/juggling_ampc/claix_env.sh"
export WARP_CACHE_PATH="$ROOT/.warp_cache"
MODES=(arm15 calibrated)
ARMS=(1.5 1.0)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
"$PYTHON" -u examples/juggling_ampc/check_native_compiled.py "$REFERENCE"
if [[ "$MODE" == arm15 ]]; then
 "$PYTHON" -u examples/juggling_ampc/check_full_batch_memory.py --root "$ROOT" --multiplier 1 --arm-multiplier 1.5
fi
COMMON=(--factors "$REFERENCE/gpu_factors" --observer "$OBSERVER" --source "$FRANKA_SOURCE" --calibration "$CALIBRATION"
 --seconds 30 --multiplier 1 --arm-multiplier "$ARM" --batch 128 --seed 1002026)
for EXPERT in native warp; do
 EXTRA=(--diagnostic)
 if [[ "$EXPERT" == native ]]; then EXTRA=(--native-expert --native-compiled --native-workers 8); fi
 JUGGLING_EVAL_THREADS=1 "$PYTHON" -u examples/juggling_ampc/rollout.py "${COMMON[@]}" "${EXTRA[@]}" \
  --output "$ROOT/results/qualification_${MODE}_${EXPERT}"
done
"$PYTHON" -u examples/juggling_ampc/check_rollout_parity.py "$REFERENCE" "$ROOT/results/qualification_${MODE}_warp" \
 --cases 16 --solve-batch 60000 --output "$ROOT/results/parity_${MODE}_60000.json"
"$PYTHON" -u examples/juggling_ampc/qualify_expert.py \
 --native "$ROOT/results/qualification_${MODE}_native/rollout.json" --warp "$ROOT/results/qualification_${MODE}_warp/rollout.json" \
 --inflight-parity "$ROOT/results/parity_${MODE}_60000.json" --parity "$REFERENCE/gpu_mpc_parity.json" \
 --metadata "$REFERENCE/gpu_factors/metadata.json" --output "$ROOT/results/qualification_${MODE}.json"
