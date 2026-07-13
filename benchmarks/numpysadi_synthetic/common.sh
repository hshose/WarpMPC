#!/bin/bash

set -euo pipefail

STUDY_NAME="numpysadi_synthetic"
if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks/numpysadi_synthetic" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks/numpysadi_synthetic"
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

DTYPES="${DTYPES:-float32,float64}"

command mkdir -p "${RESULTS_ROOT}" "${ROOT}/hpclogs"

BASHRC="${BASHRC:-${HOME}/.bashrc}"
if [ -f "${BASHRC}" ]; then
  source "${BASHRC}"
elif [ -f "${HOME}/.bashrc" ]; then
  source "${HOME}/.bashrc"
fi
if [[ -n "${VENV}" && -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
fi
cd "${ROOT}" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.88}"
export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=multi_output_fusion}"
export MPLCONFIGDIR="${RESULTS_ROOT}/matplotlib"

command mkdir -p "${MPLCONFIGDIR}"

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
