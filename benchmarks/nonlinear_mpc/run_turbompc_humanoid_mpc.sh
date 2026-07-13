#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=turbohuman
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
HUMANOID_OUTPUT_NAME="${HUMANOID_OUTPUT_NAME:-humanoid_mpc_turbompc}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${HUMANOID_OUTPUT_NAME}"

TURBOMPC_FORWARD_BACKEND="${TURBOMPC_FORWARD_BACKEND:-admm_fused_cudss}"
TURBOMPC_BACKWARD_BACKEND="${TURBOMPC_BACKWARD_BACKEND:-direct_cudss_ffi}"
TURBOMPC_EPS_ABS="${TURBOMPC_EPS_ABS:-${EPS_ABS}}"
TURBOMPC_EPS_REL="${TURBOMPC_EPS_REL:-${EPS_REL}}"
TURBOMPC_HUMANOID_BATCH_SIZES="${TURBOMPC_HUMANOID_BATCH_SIZES:-${HUMANOID_BATCH_SIZES}}"

run_benchmark "${HUMANOID_OUTPUT_NAME}" \
  "${PYTHON}" benchmarks/nonlinear_mpc/benchmark_turbompc_humanoid_mpc.py \
    --batch-sizes "${TURBOMPC_HUMANOID_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --sim-steps 20 \
    --max-iter "${MAX_ITER}" \
    --turbompc-eps-abs "${TURBOMPC_EPS_ABS}" \
    --turbompc-eps-rel "${TURBOMPC_EPS_REL}" \
    --sigma 1e-6 \
    --turbompc-forward-backend "${TURBOMPC_FORWARD_BACKEND}" \
    --turbompc-backward-backend "${TURBOMPC_BACKWARD_BACKEND}" \
    --output-dir "${RESULTS_ROOT}/${HUMANOID_OUTPUT_NAME}" \
    --csv-path "${RESULTS_ROOT}/${HUMANOID_OUTPUT_NAME}.csv" \
    --plot-path "${RESULTS_ROOT}/${HUMANOID_OUTPUT_NAME}_summary.png"
