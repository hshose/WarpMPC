#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=human
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
HUMANOID_OUTPUT_NAME="${HUMANOID_OUTPUT_NAME:-humanoid_mpc}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${HUMANOID_OUTPUT_NAME}"

BACKEND_ARGS=()
if [[ -n "${NONLINEAR_QDLDL_VARIANTS:-}" ]]; then
  BACKEND_ARGS+=(--qdldl-variants "${NONLINEAR_QDLDL_VARIANTS}")
elif [[ -n "${QDLDL_BACKEND_PAIRS}" ]]; then
  BACKEND_ARGS+=(--qdldl-backend-pairs "${QDLDL_BACKEND_PAIRS}")
fi

run_benchmark "${HUMANOID_OUTPUT_NAME}" \
  "${PYTHON}" benchmarks/nonlinear_mpc/benchmark_humanoid_mpc.py \
    --batch-sizes "${HUMANOID_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --qp-solvers "${QP_SOLVERS}" \
    "${BACKEND_ARGS[@]}" \
    --group-modes "grouped" \
    --levelsolve-modes "${LEVELSOLVE_MODES}" \
    --level-scheduled-solve-threshold "${LEVEL_SCHEDULED_SOLVE_THRESHOLD}" \
    --sim-steps 20 \
    --max-iter "${MAX_ITER}" \
    "${MPAX_ARGS[@]}" \
    --segment-budget "${HUMANOID_SEGMENT_BUDGET}" \
    --segment-strategy "${SEGMENT_STRATEGY}" \
    --output-dir "${RESULTS_ROOT}/${HUMANOID_OUTPUT_NAME}" \
    --csv-path "${RESULTS_ROOT}/${HUMANOID_OUTPUT_NAME}.csv" \
    --plot-path "${RESULTS_ROOT}/${HUMANOID_OUTPUT_NAME}_summary.png"
