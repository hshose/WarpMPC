#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=crazy
#SBATCH --output=hpclogs/%A.log
#SBATCH --error=hpclogs/%A.log

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
CRAZYFLIE_OUTPUT_NAME="${CRAZYFLIE_OUTPUT_NAME:-crazyflie_sqp_rollout}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${CRAZYFLIE_OUTPUT_NAME}"

SIM_STEP_ARGS=()
if [[ -n "${CRAZYFLIE_SIM_STEPS}" ]]; then
  SIM_STEP_ARGS+=(--sim-steps "${CRAZYFLIE_SIM_STEPS}")
fi

PLOT_ARGS=()
if [[ "${CRAZYFLIE_SKIP_PLOTS}" == "1" ]]; then
  PLOT_ARGS+=(--skip-plots)
fi

BACKEND_ARGS=()
if [[ -n "${NONLINEAR_QDLDL_VARIANTS:-}" ]]; then
  BACKEND_ARGS+=(--qdldl-variants "${NONLINEAR_QDLDL_VARIANTS}")
elif [[ -n "${QDLDL_BACKEND_PAIRS}" ]]; then
  BACKEND_ARGS+=(--qdldl-backend-pairs "${QDLDL_BACKEND_PAIRS}")
else
  BACKEND_ARGS+=(--qdldl-backends "${QDLDL_BACKENDS}")
fi

run_benchmark "${CRAZYFLIE_OUTPUT_NAME}" \
  "${PYTHON}" benchmarks/nonlinear_mpc/benchmark_crazyflie_sqp_rollout.py \
    --batch-sizes "${CRAZYFLIE_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --qp-solvers "${QP_SOLVERS}" \
    "${BACKEND_ARGS[@]}" \
    --group-modes "grouped" \
    --levelsolve-modes "${LEVELSOLVE_MODES}" \
    --level-scheduled-solve-threshold "${LEVEL_SCHEDULED_SOLVE_THRESHOLD}" \
    --repeat "${REPEAT}" \
    --horizon-steps "${CRAZYFLIE_HORIZON_STEPS}" \
    --sim-time "${CRAZYFLIE_SIM_TIME}" \
    --control-dt "${CRAZYFLIE_CONTROL_DT}" \
    "${SIM_STEP_ARGS[@]}" \
    --max-iter "${MAX_ITER}" \
    "${MPAX_ARGS[@]}" \
    --segment-budget "${CRAZYFLIE_SEGMENT_BUDGET}" \
    --segment-strategy "${SEGMENT_STRATEGY}" \
    --skip-output-npz \
    "${PLOT_ARGS[@]}" \
    --output-dir "${RESULTS_ROOT}/${CRAZYFLIE_OUTPUT_NAME}" \
    --csv-path "${RESULTS_ROOT}/${CRAZYFLIE_OUTPUT_NAME}.csv" \
    --plot-path "${RESULTS_ROOT}/${CRAZYFLIE_OUTPUT_NAME}_summary.png"
