#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=juggle_qualify
#SBATCH --output=hpclogs/%x_%j.log
set -euo pipefail
source "${ROOT:?Set ROOT}/examples/juggling_ampc/claix_env.sh"
cmake -S examples/juggling_ampc/reference/optimization/native -B "$REFERENCE/controller_claix" \
 -DFACTOR_DIR="$REFERENCE/factors" -DEIGEN_DIR="$EIGEN_DIR" \
 -Dcasadi_DIR="$($PYTHON -c 'import casadi,pathlib; print(pathlib.Path(casadi.__file__).parent/"cmake")')" -Dosqp_DIR="$FRANKA_SOURCE/build/osqp_install/lib/cmake/osqp"
cmake --build "$REFERENCE/controller_claix" -j 12
cp "$REFERENCE/controller_claix/libjuggling_controller.so" "$REFERENCE/controller/libjuggling_controller.so"
cmake -S examples/juggling_ampc/observer -B "$(dirname "$OBSERVER")" -DEIGEN_DIR="$EIGEN_DIR"
cmake --build "$(dirname "$OBSERVER")" -j 8
"$PYTHON" examples/juggling_ampc/build_native_factors.py "$REFERENCE/factors" --jobs 12
"$PYTHON" examples/juggling_ampc/check_native_compiled.py "$REFERENCE"
"$PYTHON" examples/juggling_ampc/check_gpu_fixture.py "$REFERENCE"
"$PYTHON" examples/juggling_ampc/check_full_batch_memory.py --root "$ROOT"
"$PYTHON" examples/juggling_ampc/rollout.py --factors "$REFERENCE/gpu_factors" --observer "$OBSERVER" \
 --source "$FRANKA_SOURCE" --calibration "$CALIBRATION" --seconds 30 --multiplier 1.5 --batch 128 --seed 1002026 --diagnostic \
 --output "$ROOT/results/claix_qualification_warp"
"$PYTHON" examples/juggling_ampc/check_rollout_parity.py "$REFERENCE" "$ROOT/results/claix_qualification_warp" \
 --cases 16 --solve-batch 60000 --output "$ROOT/results/claix_full_batch_parity.json"
"$PYTHON" examples/juggling_ampc/qualify_expert.py \
 --native "$ROOT/results/native_qualification128_v2/rollout.json" --warp "$ROOT/results/claix_qualification_warp/rollout.json" \
 --inflight-parity "$ROOT/results/claix_full_batch_parity.json" --parity "$REFERENCE/gpu_mpc_parity.json" \
 --metadata "$REFERENCE/gpu_factors/metadata.json" --output "$ROOT/results/claix_expert_qualification.json"
