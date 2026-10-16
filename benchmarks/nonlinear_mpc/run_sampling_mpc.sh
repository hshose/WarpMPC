#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=sampling-mpc
#SBATCH --output=hpclogs/%A.log
#SBATCH --error=hpclogs/%A.log

set -euo pipefail
if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks/nonlinear_mpc" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks/nonlinear_mpc"
export ROOT
SAMPLING_SYSTEM="${SAMPLING_SYSTEM:-cartpole}"
SAMPLING_METHOD="${SAMPLING_METHOD:-mppi}"
case "${SAMPLING_SYSTEM}" in
  cartpole) PREFIX=cartpole_quadratic_mpc ;;
  crazyflie) PREFIX=crazyflie_sqp_rollout ;;
  *) echo "No paper sampling baseline for ${SAMPLING_SYSTEM}" >&2; exit 2 ;;
esac
case "${SAMPLING_METHOD}" in
  mppi|cem) ;;
  *) echo "Unknown sampling method: ${SAMPLING_METHOD}" >&2; exit 2 ;;
esac
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${PREFIX}_${SAMPLING_METHOD}"

# These adapters use the publication prediction grid and control space.
if [[ "${SAMPLING_SYSTEM}" == cartpole ]]; then
  BATCH_SIZES="${CARTPOLE_BATCH_SIZES}"
  SIM_STEPS=20
  CHUNK_SIZE="${SAMPLING_CARTPOLE_CHUNK_SIZE:-1024}"
  REFERENCE_SQP_ITERATIONS="${CARTPOLE_SQP_ITERATIONS:-5}"
else
  if [[ "${CRAZYFLIE_HORIZON_STEPS}" != 40 || "${CRAZYFLIE_CONTROL_DT}" != 0.01 ]]; then
    echo "Sampling paper defaults require the original Quadrotor horizon=40 and control_dt=0.01" >&2
    exit 2
  fi
  BATCH_SIZES="${CRAZYFLIE_BATCH_SIZES}"
  SIM_STEPS="${CRAZYFLIE_SIM_STEPS:-}"
  if [[ -z "${SIM_STEPS}" ]]; then
    SIM_STEPS="$("${PYTHON}" -c 'import math, sys; print(math.ceil(float(sys.argv[1]) / .01))' "${CRAZYFLIE_SIM_TIME}")"
  fi
  CHUNK_SIZE="${SAMPLING_CRAZYFLIE_CHUNK_SIZE:-128}"
  REFERENCE_SQP_ITERATIONS=1
fi
EXTRA_ARGS=()
[[ "${SAMPLING_RESUME:-0}" == 1 ]] && EXTRA_ARGS+=(--resume)
[[ "${SAMPLING_REQUIRE_H100:-0}" == 1 ]] && EXTRA_ARGS+=(--require-h100)
[[ "${SAMPLING_DRY_RUN:-0}" == 1 ]] && EXTRA_ARGS+=(--dry-run)
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run_benchmark "${PREFIX}_${SAMPLING_METHOD}" \
  "${PYTHON}" "${SCRIPT_DIR}/benchmark_sampling_mpc.py" \
  --system "${SAMPLING_SYSTEM}" --method "${SAMPLING_METHOD}" \
  --output-dir "${RESULTS_ROOT}/${PREFIX}_${SAMPLING_METHOD}" \
  --csv-path "${RESULTS_ROOT}/${PREFIX}_${SAMPLING_METHOD}.csv" \
  --batch-sizes "${BATCH_SIZES}" \
  --sim-steps "${SIM_STEPS}" --chunk-size "${CHUNK_SIZE}" \
  --reference-sqp-iterations "${REFERENCE_SQP_ITERATIONS}" \
  --seed "${SAMPLING_SEED:-101}" --repeats "${SAMPLING_REPEATS:-3}" \
  "${EXTRA_ARGS[@]}"
