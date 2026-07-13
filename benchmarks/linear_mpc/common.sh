#!/bin/bash

set -euo pipefail

STUDY_NAME="linear_mpc"
if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks/linear_mpc" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks/linear_mpc"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV="${VENV:-}"
if [[ -n "${VENV}" ]]; then
  PYTHON="${PYTHON:-${VENV}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi
RUN_ROOT="${1:-${ROOT}/results/h100_paper_benchmarks_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_NAME="${2:-${STUDY_NAME}}"
RESULTS_ROOT="${RUN_ROOT}/${STUDY_NAME}"

HORIZON="${HORIZON:-40}"
BATCH_SIZES="${BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}"
DTYPES="${DTYPES:-float32,float64}"
REPEAT="${REPEAT:-3}"
MAX_ITER="${MAX_ITER:-25}"
EPS_ABS="${EPS_ABS:-1e-5}"
EPS_REL="${EPS_REL:-1e-5}"
SEGMENT_BUDGET="${SEGMENT_BUDGET:-16}"
SEGMENT_STRATEGY="${SEGMENT_STRATEGY:-optimal}"
MAX_DEVICE_GB="${MAX_DEVICE_GB:-88.0}"
VARIANTS="${VARIANTS:-baseline,transpose+segmented,transpose+segmented+levelsolve,factor-warp+solve-jax:transpose+segmented,factor-warp+solve-jax:transpose+segmented+levelsolve,factor-warp+solve-warp:transpose+segmented,factor-warp+solve-warp:transpose+segmented+levelsolve}"

command mkdir -p "${RESULTS_ROOT}" "${ROOT}/hpclogs"

BASHRC="${BASHRC:-${HOME}/.bashrc}"
if [[ -f "${BASHRC}" ]]; then
  source "${BASHRC}"
fi
if [[ -n "${VENV}" && -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
fi
cd "${ROOT}" || exit 1

export PYTHONUNBUFFERED=1
export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=multi_output_fusion}"

echo "=== ${BENCHMARK_NAME} ==="
echo "Job ID:  ${SLURM_JOB_ID:-manual}"
echo "Host:    $(uname -n)"
echo "Results: ${RESULTS_ROOT}"
echo "Python:  ${PYTHON}"
echo "Horizon: ${HORIZON}"
echo "Tol:     eps_abs=${EPS_ABS} eps_rel=${EPS_REL}"

run_benchmark() {
  local name="$1"
  shift

  echo
  echo "Command: $*"
  "$@"
}
