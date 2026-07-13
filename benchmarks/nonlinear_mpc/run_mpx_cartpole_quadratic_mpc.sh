#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=cartmpx
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
export CARTPOLE_BATCH_SIZES="${CARTPOLE_BATCH_SIZES:-${MPX_CARTPOLE_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}}"
CARTPOLE_OUTPUT_NAME="${CARTPOLE_OUTPUT_NAME:-cartpole_quadratic_mpc_mpx}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "${CARTPOLE_OUTPUT_NAME}"
ensure_mpx_available
CARTPOLE_SQP_ITERATIONS="${CARTPOLE_SQP_ITERATIONS:-5}"

MPX_MEMORY_ARGS=()
if [[ "${MPX_LIMITED_MEMORY:-1}" != "1" ]]; then
  MPX_MEMORY_ARGS+=(--no-mpx-limited-memory)
fi

PLOT_ARGS=()
if [[ "${CARTPOLE_SKIP_PLOTS:-0}" == "1" ]]; then
  PLOT_ARGS+=(--skip-plots)
fi

run_benchmark "${CARTPOLE_OUTPUT_NAME}" \
  "${PYTHON}" benchmarks/nonlinear_mpc/benchmark_mpx_nonlinear_mpc.py \
    --system cartpole \
    --batch-sizes "${CARTPOLE_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --repeat "${REPEAT}" \
    --mpx-solver-mode primal_dual \
    --mpx-equality-weight "${MPX_EQUALITY_WEIGHT:-1e4}" \
    --mpx-barrier-alpha "${MPX_BARRIER_ALPHA:-0.1}" \
    --mpx-barrier-sigma "${MPX_BARRIER_SIGMA:-1.0}" \
    --mpx-num-alpha "${MPX_NUM_ALPHA:-11}" \
    "${MPX_MEMORY_ARGS[@]}" \
    --horizon-steps 100 \
    --dt-start 0.1 \
    --dt-growth 1.0 \
    --sim-time 2.0 \
    --control-dt 0.1 \
    --integrator-substeps 1 \
    --sqp-iterations "${CARTPOLE_SQP_ITERATIONS}" \
    --enable-rail-constraint \
    --noise-scale 0 \
    --process-noise-scale "0,0,0,0" \
    --input-noise-scale "0" \
    --plot-samples 2048 \
    "${PLOT_ARGS[@]}" \
    --output-dir "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}" \
    --csv-path "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}.csv" \
    --plot-path "${RESULTS_ROOT}/${CARTPOLE_OUTPUT_NAME}_summary.png"
