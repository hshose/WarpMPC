#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --job-name=juggling_ampc
#SBATCH --output=hpclogs/%A.log
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the juggling experiment directory}"
FRANKA_SOURCE="${FRANKA_SOURCE:?Set FRANKA_SOURCE to the live reference checkout}"
CALIBRATION="${CALIBRATION:?Set CALIBRATION to the fitted Newton calibration artifact}"
DATASET="${DATASET:-$RESULTS_DIR/dataset}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT/.warp_cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT/.jax_cache}"
mkdir -p "$RESULTS_DIR/logs"
cd "$ROOT"
REFERENCE="${REFERENCE:-$RESULTS_DIR/reference}"
OBSERVER="${OBSERVER:-$RESULTS_DIR/observer/libjuggling_batch_observer.so}"
EIGEN_DIR="${EIGEN_DIR:?Set EIGEN_DIR to Eigen headers}"
CMAKE="${CMAKE:-cmake}"
if [[ ! -f "$REFERENCE/native_fixtures.npz" ]]; then
  CMAKE="$CMAKE" EIGEN_DIR="$EIGEN_DIR" FRANKA_PYTHON="$PYTHON" bash examples/juggling_ampc/build_reference.sh "$REFERENCE"
  "$PYTHON" examples/juggling_ampc/check_mpc.py "$REFERENCE" --fixtures-only
  "$PYTHON" examples/juggling_ampc/build_gpu_factors.py \
    examples/juggling_ampc/reference/config/cpp_ball_spec.json "$REFERENCE/gpu_factors"
fi
"$PYTHON" examples/juggling_ampc/check_gpu_fixture.py "$REFERENCE" 2>&1 | tee "$RESULTS_DIR/logs/parity.log"
"$CMAKE" -S examples/juggling_ampc/observer -B "$(dirname "$OBSERVER")" -DEIGEN_DIR="$EIGEN_DIR"
"$CMAKE" --build "$(dirname "$OBSERVER")" -j "${BUILD_JOBS:-8}"
COMMON=(--factors "$REFERENCE/gpu_factors" --observer "$OBSERVER" --source "$FRANKA_SOURCE" --calibration "$CALIBRATION" --seconds 30 --multiplier 1.5)
QUALIFICATION="$RESULTS_DIR/expert_qualification.json"
if [[ ! -f "$QUALIFICATION" ]]; then
  "$PYTHON" examples/juggling_ampc/build_native_factors.py "$REFERENCE/factors" --jobs "${BUILD_JOBS:-8}"
  "$PYTHON" examples/juggling_ampc/check_native_compiled.py "$REFERENCE"
  for EXPERT in native warp; do
    EXTRA=(--diagnostic)
    if [[ "$EXPERT" == native ]]; then EXTRA=(--native-expert --native-compiled --native-workers "${NATIVE_WORKERS:-8}"); fi
    if [[ ! -f "$RESULTS_DIR/qualification_$EXPERT/rollout.json" ]]; then
      JUGGLING_EVAL_THREADS=1 "$PYTHON" examples/juggling_ampc/rollout.py "${COMMON[@]}" "${EXTRA[@]}" \
        --batch "${QUALIFICATION_BATCH_SIZE:-128}" --seed 1002026 --output "$RESULTS_DIR/qualification_$EXPERT" \
        2>&1 | tee "$RESULTS_DIR/logs/qualify_$EXPERT.log"
    fi
  done
fi
"$PYTHON" examples/juggling_ampc/check_rollout_parity.py "$REFERENCE" "$RESULTS_DIR/qualification_warp" \
  --cases 16 --output "$RESULTS_DIR/inflight_parity.json"
"$PYTHON" examples/juggling_ampc/qualify_expert.py \
  --native "$RESULTS_DIR/qualification_native/rollout.json" --warp "$RESULTS_DIR/qualification_warp/rollout.json" \
  --inflight-parity "$RESULTS_DIR/inflight_parity.json" --parity "$REFERENCE/gpu_mpc_parity.json" --metadata "$REFERENCE/gpu_factors/metadata.json" --output "$QUALIFICATION"
"$PYTHON" examples/juggling_ampc/check_rollout_parity.py "$REFERENCE" "$RESULTS_DIR/qualification_warp" \
  --cases 16 --solve-batch "${COLLECT_MICROBATCH_SIZE:-60000}" --output "$RESULTS_DIR/collection_batch_inflight_parity.json"
# A partial final batch has a different compiled solve shape; audit it too.
COLLECTION_TAIL=$(( ${COLLECT_BATCH_SIZE:-60000} % ${COLLECT_MICROBATCH_SIZE:-60000} ))
if (( COLLECTION_TAIL > 0 )); then
  "$PYTHON" examples/juggling_ampc/check_rollout_parity.py "$REFERENCE" "$RESULTS_DIR/qualification_warp" \
    --cases 16 --solve-batch "$COLLECTION_TAIL" --output "$RESULTS_DIR/collection_tail_inflight_parity.json"
fi
COMMON+=(--expert-qualification "$QUALIFICATION")
if [[ ! -f "$DATASET/manifest.json" ]]; then
  "$PYTHON" examples/juggling_ampc/collect.py "${COMMON[@]}" \
    --batch "${COLLECT_BATCH_SIZE:-60000}" --microbatch "${COLLECT_MICROBATCH_SIZE:-60000}" --output "$DATASET" \
    2>&1 | tee "$RESULTS_DIR/logs/collect.log"
fi
# Recheck the live empirical fit immediately before the first optimizer update.
# The trainer rejects a changed source/fit or an expert that failed stabilization.
TRAIN_EXTRA=()
if [[ -n "${TRAIN_AUDIT_DIR:-}" ]]; then TRAIN_EXTRA=(--authoritative-audit "$TRAIN_AUDIT_DIR"); fi
"$PYTHON" examples/juggling_ampc/train_policy.py "${TRAIN_EXTRA[@]}" \
 --source "$FRANKA_SOURCE" --calibration "$CALIBRATION" \
 --dataset "$DATASET" --output "$RESULTS_DIR/policy" \
 --epochs "${TRAIN_EPOCHS:-100}" --batch "${TRAIN_BATCH_SIZE:-8192}" \
 --hidden 256,256,256 2>&1 | tee "$RESULTS_DIR/logs/train.log"
"$PYTHON" examples/juggling_ampc/export_policy.py "$RESULTS_DIR/policy/policy.npz" "$RESULTS_DIR/policy/eigen"
"$PYTHON" examples/juggling_ampc/check_policy_export.py "$RESULTS_DIR/policy/policy.npz" "$RESULTS_DIR/policy/eigen" --eigen "$EIGEN_DIR"
for POLICY in expert student; do
  EXTRA=()
  if [[ "$POLICY" == student ]]; then EXTRA=(--policy "$RESULTS_DIR/policy/policy.npz"); fi
  "$PYTHON" examples/juggling_ampc/rollout.py "${COMMON[@]}" "${EXTRA[@]}" \
    --batch "${EVALUATION_BATCH_SIZE:-1024}" --seed 1002026 --output "$RESULTS_DIR/evaluation_$POLICY" \
    2>&1 | tee "$RESULTS_DIR/logs/evaluate_$POLICY.log"
done
"$PYTHON" examples/juggling_ampc/compare_rollouts.py \
  "$RESULTS_DIR/evaluation_expert/rollout.json" "$RESULTS_DIR/evaluation_student/rollout.json" \
  "$RESULTS_DIR/policy/policy.npz" "$RESULTS_DIR/policy/eigen"
