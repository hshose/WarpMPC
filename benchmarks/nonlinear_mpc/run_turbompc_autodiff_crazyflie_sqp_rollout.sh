#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=turboadcrazy
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
CRAZYFLIE_OUTPUT_NAME="${CRAZYFLIE_OUTPUT_NAME:-crazyflie_sqp_rollout_turbompc_autodiff}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${CRAZYFLIE_OUTPUT_NAME}"

TURBOMPC_FORWARD_BACKEND="${TURBOMPC_FORWARD_BACKEND:-admm_fused_cudss}"
TURBOMPC_BACKWARD_BACKEND="${TURBOMPC_BACKWARD_BACKEND:-direct_cudss_ffi}"
TURBOMPC_EPS_ABS="${TURBOMPC_EPS_ABS:-${EPS_ABS}}"
TURBOMPC_EPS_REL="${TURBOMPC_EPS_REL:-${EPS_REL}}"
TURBOMPC_AUTODIFF_CRAZYFLIE_BATCH_SIZES="${TURBOMPC_AUTODIFF_CRAZYFLIE_BATCH_SIZES:-${TURBOMPC_CRAZYFLIE_BATCH_SIZES:-${CRAZYFLIE_BATCH_SIZES}}}"

SIM_STEP_ARGS=()
if [[ -n "${CRAZYFLIE_SIM_STEPS}" ]]; then
  SIM_STEP_ARGS+=(--sim-steps "${CRAZYFLIE_SIM_STEPS}")
fi

PLOT_ARGS=()
if [[ "${CRAZYFLIE_SKIP_PLOTS}" == "1" ]]; then
  PLOT_ARGS+=(--skip-plots)
fi

run_benchmark "${CRAZYFLIE_OUTPUT_NAME}" \
  "${PYTHON}" benchmarks/nonlinear_mpc/benchmark_turbompc_autodiff_crazyflie_sqp_rollout.py \
    --batch-sizes "${TURBOMPC_AUTODIFF_CRAZYFLIE_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --repeat "${REPEAT}" \
    --horizon-steps "${CRAZYFLIE_HORIZON_STEPS}" \
    --sim-time "${CRAZYFLIE_SIM_TIME}" \
    --control-dt "${CRAZYFLIE_CONTROL_DT}" \
    "${SIM_STEP_ARGS[@]}" \
    --max-iter "${MAX_ITER}" \
    --turbompc-eps-abs "${TURBOMPC_EPS_ABS}" \
    --turbompc-eps-rel "${TURBOMPC_EPS_REL}" \
    --rho 0.1 \
    --sigma 1e-6 \
    --alpha 1.6 \
    --line-search-step-min 0.1 \
    --turbompc-forward-backend "${TURBOMPC_FORWARD_BACKEND}" \
    --turbompc-backward-backend "${TURBOMPC_BACKWARD_BACKEND}" \
    --skip-output-npz \
    "${PLOT_ARGS[@]}" \
    --output-dir "${RESULTS_ROOT}/${CRAZYFLIE_OUTPUT_NAME}" \
    --csv-path "${RESULTS_ROOT}/${CRAZYFLIE_OUTPUT_NAME}.csv" \
    --plot-path "${RESULTS_ROOT}/${CRAZYFLIE_OUTPUT_NAME}_summary.png"
