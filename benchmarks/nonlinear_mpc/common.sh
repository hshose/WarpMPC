#!/bin/bash

set -euo pipefail

STUDY_NAME="nonlinear_mpc"
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

CARTPOLE_BATCH_SIZES="${CARTPOLE_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}"
CRAZYFLIE_BATCH_SIZES="${CRAZYFLIE_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}"
HUMANOID_BATCH_SIZES="${HUMANOID_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}"
DTYPES="${DTYPES:-float32}"
QP_SOLVERS="${QP_SOLVERS:-jax_osqp}"
QDLDL_BACKENDS="${QDLDL_BACKENDS:-jax}"
QDLDL_BACKEND_PAIRS="${QDLDL_BACKEND_PAIRS:-}"
LEVELSOLVE_MODES="${LEVELSOLVE_MODES:-regular,levelsolve}"
LEVEL_SCHEDULED_SOLVE_THRESHOLD="${LEVEL_SCHEDULED_SOLVE_THRESHOLD:-2}"
REPEAT="${REPEAT:-1}"
MAX_ITER="${MAX_ITER:-25}"
EPS_ABS="${EPS_ABS:-1e-5}"
EPS_REL="${EPS_REL:-1e-5}"
MPAX_ITERATION_LIMIT="${MPAX_ITERATION_LIMIT:-1000}"
MPAX_TERMINATION_EVALUATION_FREQUENCY="${MPAX_TERMINATION_EVALUATION_FREQUENCY:-100}"
MPAX_L_INF_RUIZ_ITERATIONS="${MPAX_L_INF_RUIZ_ITERATIONS:-10}"
MPAX_POCK_CHAMBOLLE_ALPHA="${MPAX_POCK_CHAMBOLLE_ALPHA:-1.0}"
MPAX_REGULARIZATION="${MPAX_REGULARIZATION:-0.0}"
MPAX_UNROLL="${MPAX_UNROLL:-0}"
USER_SEGMENT_BUDGET="${SEGMENT_BUDGET:-}"
SEGMENT_BUDGET="${USER_SEGMENT_BUDGET:-16}"
CARTPOLE_SEGMENT_BUDGET="${CARTPOLE_SEGMENT_BUDGET:-${USER_SEGMENT_BUDGET:-384}}"
CRAZYFLIE_SEGMENT_BUDGET="${CRAZYFLIE_SEGMENT_BUDGET:-${USER_SEGMENT_BUDGET:-256}}"
HUMANOID_SEGMENT_BUDGET="${HUMANOID_SEGMENT_BUDGET:-${USER_SEGMENT_BUDGET:-96}}"
SEGMENT_STRATEGY="${SEGMENT_STRATEGY:-optimal}"
CRAZYFLIE_HORIZON_STEPS="${CRAZYFLIE_HORIZON_STEPS:-40}"
CRAZYFLIE_SIM_TIME="${CRAZYFLIE_SIM_TIME:-1.0}"
CRAZYFLIE_CONTROL_DT="${CRAZYFLIE_CONTROL_DT:-0.01}"
CRAZYFLIE_SIM_STEPS="${CRAZYFLIE_SIM_STEPS:-}"
CRAZYFLIE_SKIP_PLOTS="${CRAZYFLIE_SKIP_PLOTS:-0}"

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
echo "Dtypes:  ${DTYPES}"
echo "Solvers: ${QP_SOLVERS}"
echo "Levelsolve threshold: ${LEVEL_SCHEDULED_SOLVE_THRESHOLD}"

MPAX_ARGS=(
  --mpax-iteration-limit "${MPAX_ITERATION_LIMIT}"
  --mpax-eps-abs "${EPS_ABS}"
  --mpax-eps-rel "${EPS_REL}"
  --mpax-termination-evaluation-frequency "${MPAX_TERMINATION_EVALUATION_FREQUENCY}"
  --mpax-l-inf-ruiz-iterations "${MPAX_L_INF_RUIZ_ITERATIONS}"
  --mpax-pock-chambolle-alpha "${MPAX_POCK_CHAMBOLLE_ALPHA}"
  --mpax-regularization "${MPAX_REGULARIZATION}"
)
if [[ "${MPAX_UNROLL}" == "1" ]]; then
  MPAX_ARGS+=(--mpax-unroll)
fi

run_benchmark() {
  local name="$1"
  shift

  echo
  echo "Command: $*"
  "$@"
}

ensure_mpx_available() {
  if ! "${PYTHON}" -c "import importlib.metadata as metadata; metadata.version('mpx')" >/dev/null 2>&1; then
    echo "MPX is required for this benchmark. Install it in the benchmark environment." >&2
    return 1
  fi
  "${PYTHON}" -c "import mpx, trajax"
}
