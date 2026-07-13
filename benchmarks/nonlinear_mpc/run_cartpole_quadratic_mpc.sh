#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=cartquad
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
CARTPOLE_OUTPUT_NAME="${CARTPOLE_OUTPUT_NAME:-cartpole_quadratic_mpc}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${CARTPOLE_OUTPUT_NAME}"
CARTPOLE_SQP_ITERATIONS="${CARTPOLE_SQP_ITERATIONS:-5}"
CARTPOLE_OSQP_MAX_ITER="${CARTPOLE_OSQP_MAX_ITER:-25}"

BACKEND_ARGS=()
if [[ -n "${NONLINEAR_QDLDL_VARIANTS:-}" ]]; then
  BACKEND_ARGS+=(--qdldl-variants "${NONLINEAR_QDLDL_VARIANTS}")
elif [[ -n "${QDLDL_BACKEND_PAIRS}" ]]; then
  BACKEND_ARGS+=(--qdldl-backend-pairs "${QDLDL_BACKEND_PAIRS}")
fi

run_benchmark "${CARTPOLE_OUTPUT_NAME}" \
  "${PYTHON}" benchmarks/nonlinear_mpc/benchmark_cartpole_quadratic_mpc.py \
    --batch-sizes "${CARTPOLE_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --qp-solvers "${QP_SOLVERS}" \
    "${BACKEND_ARGS[@]}" \
    --group-modes "grouped" \
    --levelsolve-modes "${LEVELSOLVE_MODES}" \
    --level-scheduled-solve-threshold "${LEVEL_SCHEDULED_SOLVE_THRESHOLD}" \
    --horizon-steps 100 \
    --dt-start 0.1 \
    --dt-growth 1.0 \
    --sim-time 2.0 \
    --control-dt 0.1 \
    --integrator-substeps 1 \
    --sqp-iterations "${CARTPOLE_SQP_ITERATIONS}" \
    --osqp-max-iter "${CARTPOLE_OSQP_MAX_ITER}" \
    "${MPAX_ARGS[@]}" \
    --rho 0.1 \
    --sigma 1e-6 \
    --alpha 1.6 \
    --segment-budget "${CARTPOLE_SEGMENT_BUDGET}" \
    --segment-strategy optimal \
    --line-search-step-min 0.1 \
    --enable-rail-constraint \
    --noise-scale 0 \
    --process-noise-scale "0,0,0,0" \
    --input-noise-scale "0" \
    --plot-samples 2048 \
    --output-dir "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}" \
    --csv-path "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}.csv" \
    --plot-path "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}_summary.png"
