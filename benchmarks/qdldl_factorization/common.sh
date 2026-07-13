#!/bin/bash

set -euo pipefail

STUDY_NAME="qdldl_factorization"
if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks/qdldl_factorization" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks/qdldl_factorization"
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

QDLDL_BATCH_SIZES="${QDLDL_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}"
CUDSS_BATCH_SIZES="${CUDSS_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000}"
DENSE_BATCH_SIZES="${DENSE_BATCH_SIZES:-512,2048}"
DTYPES="${DTYPES:-float32,float64}"
REPEAT="${REPEAT:-3}"
MAX_ITER="${MAX_ITER:-25}"
SEGMENT_BUDGET="${SEGMENT_BUDGET:-16}"
SEGMENT_STRATEGY="${SEGMENT_STRATEGY:-optimal}"
MAX_DEVICE_GB="${MAX_DEVICE_GB:-88.0}"
QDLDL_VARIANTS="${QDLDL_VARIANTS:-baseline,transpose,segmented,transpose+segmented,transpose+segmented+levelsolve,factor-warp+solve-jax:transpose+segmented,factor-warp+solve-jax:transpose+segmented+levelsolve,factor-warp+solve-warp:transpose+segmented,factor-warp+solve-warp:transpose+segmented+levelsolve}"

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

run_benchmark() {
  local name="$1"
  shift

  echo
  echo "Command: $*"
  "$@"
}
