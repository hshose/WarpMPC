#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=turbocart
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
CARTPOLE_OUTPUT_NAME="${CARTPOLE_OUTPUT_NAME:-cartpole_quadratic_mpc_turbompc}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${CARTPOLE_OUTPUT_NAME}"

TURBOMPC_FORWARD_BACKEND="${TURBOMPC_FORWARD_BACKEND:-admm_fused_cudss}"
TURBOMPC_BACKWARD_BACKEND="${TURBOMPC_BACKWARD_BACKEND:-direct_cudss_ffi}"
TURBOMPC_EPS_ABS="${TURBOMPC_EPS_ABS:-${EPS_ABS}}"
TURBOMPC_EPS_REL="${TURBOMPC_EPS_REL:-${EPS_REL}}"
TURBOMPC_CARTPOLE_BATCH_SIZES="${TURBOMPC_CARTPOLE_BATCH_SIZES:-${CARTPOLE_BATCH_SIZES}}"
CARTPOLE_SQP_ITERATIONS="${CARTPOLE_SQP_ITERATIONS:-5}"
CARTPOLE_OSQP_MAX_ITER="${CARTPOLE_OSQP_MAX_ITER:-25}"

run_benchmark "${CARTPOLE_OUTPUT_NAME}" \
  "${PYTHON}" benchmarks/nonlinear_mpc/benchmark_turbompc_cartpole_quadratic_mpc.py \
    --batch-sizes "${TURBOMPC_CARTPOLE_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --horizon-steps 100 \
    --dt-start 0.1 \
    --dt-growth 1.0 \
    --sim-time 2.0 \
    --control-dt 0.1 \
    --integrator-substeps 1 \
    --sqp-iterations "${CARTPOLE_SQP_ITERATIONS}" \
    --osqp-max-iter "${CARTPOLE_OSQP_MAX_ITER}" \
    --turbompc-eps-abs "${TURBOMPC_EPS_ABS}" \
    --turbompc-eps-rel "${TURBOMPC_EPS_REL}" \
    --rho 0.1 \
    --sigma 1e-6 \
    --alpha 1.6 \
    --line-search-step-min 0.1 \
    --turbompc-forward-backend "${TURBOMPC_FORWARD_BACKEND}" \
    --turbompc-backward-backend "${TURBOMPC_BACKWARD_BACKEND}" \
    --enable-rail-constraint \
    --noise-scale 0 \
    --process-noise-scale "0,0,0,0" \
    --input-noise-scale "0" \
    --plot-samples 2048 \
    --output-dir "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}" \
    --csv-path "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}.csv" \
    --plot-path "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}_summary.png"
