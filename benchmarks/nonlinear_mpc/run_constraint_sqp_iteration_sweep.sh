#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=constraintsqp
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
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV="${VENV:-}"
if [[ -n "${VENV}" ]]; then
  PYTHON="${PYTHON:-${VENV}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi
RUN_ROOT="${1:-${ROOT}/results/constraint_sqp_iteration_sweep_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_NAME="${CONSTRAINT_SWEEP_NAME:-constraint_sqp_iteration_sweep}"
RESULTS_DIR="${RUN_ROOT}/nonlinear_mpc/${BENCHMARK_NAME}"

SYSTEMS="${CONSTRAINT_SWEEP_SYSTEMS:-cartpole,crazyflie_obstacle}"
SOLVERS="${CONSTRAINT_SWEEP_SOLVERS:-jax_osqp,mpx}"
SQP_ITERATIONS="${CONSTRAINT_SWEEP_SQP_ITERATIONS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
JAX_OSQP_SQP_ITERATIONS="${CONSTRAINT_SWEEP_JAX_OSQP_SQP_ITERATIONS:-5}"
MPX_SQP_ITERATIONS="${CONSTRAINT_SWEEP_MPX_SQP_ITERATIONS:-5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
CARTPOLE_JAX_OSQP_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CARTPOLE_JAX_OSQP_SQP_ITERATIONS:-5}"
CARTPOLE_MPX_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CARTPOLE_MPX_SQP_ITERATIONS:-5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
CRAZYFLIE_OBSTACLE_JAX_OSQP_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_JAX_OSQP_SQP_ITERATIONS:-1}"
CRAZYFLIE_OBSTACLE_MPX_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_MPX_SQP_ITERATIONS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
BATCH_SIZE="${CONSTRAINT_SWEEP_BATCH_SIZE:-200000}"
DTYPE="${DTYPE:-float32}"
SEED="${SEED:-0}"
SKIP_PLOTS="${CONSTRAINT_SWEEP_SKIP_PLOTS:-0}"
WRITE_NPZ="${CONSTRAINT_SWEEP_WRITE_NPZ:-0}"
CRAZYFLIE_HORIZON_STEPS="${CONSTRAINT_SWEEP_CRAZYFLIE_HORIZON_STEPS:-40}"
CRAZYFLIE_SIM_TIME="${CONSTRAINT_SWEEP_CRAZYFLIE_SIM_TIME:-2.0}"
CRAZYFLIE_CONTROL_DT="${CONSTRAINT_SWEEP_CRAZYFLIE_CONTROL_DT:-0.01}"
CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER:-100}"
CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE:-20.0}"
CRAZYFLIE_OBSTACLE_OSQP_SCALING="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_SCALING:-10}"
CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN:-0.01}"
CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION:-initial_state}"
CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC:-1}"

mkdir -p "${RESULTS_DIR}" "${ROOT}/hpclogs"

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
if [[ ",${SOLVERS}," == *",mpx,"* ]]; then
  "${PYTHON}" -c "import mpx, trajax" >/dev/null 2>&1 || {
    echo "MPX is required for SOLVERS=${SOLVERS}; install it in the benchmark environment." >&2
    exit 1
  }
fi

ARGS=(
  --systems "${SYSTEMS}"
  --solvers "${SOLVERS}"
  --sqp-iterations "${SQP_ITERATIONS}"
  --jax-osqp-sqp-iterations "${JAX_OSQP_SQP_ITERATIONS}"
  --mpx-sqp-iterations "${MPX_SQP_ITERATIONS}"
  --cartpole-jax-osqp-sqp-iterations "${CARTPOLE_JAX_OSQP_SQP_ITERATIONS}"
  --cartpole-mpx-sqp-iterations "${CARTPOLE_MPX_SQP_ITERATIONS}"
  --crazyflie-obstacle-jax-osqp-sqp-iterations "${CRAZYFLIE_OBSTACLE_JAX_OSQP_SQP_ITERATIONS}"
  --crazyflie-obstacle-mpx-sqp-iterations "${CRAZYFLIE_OBSTACLE_MPX_SQP_ITERATIONS}"
  --batch-size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --output-dir "${RESULTS_DIR}"
  --csv-path "${RESULTS_DIR}.csv"
  --cartpole-dt-start 0.1
  --cartpole-control-dt 0.1
  --cartpole-sim-time 10.0
  --cartpole-mpx-barrier-alpha 0.1
  --cartpole-mpx-barrier-sigma 1.0
  --crazyflie-horizon-steps "${CRAZYFLIE_HORIZON_STEPS}"
  --crazyflie-sim-time "${CRAZYFLIE_SIM_TIME}"
  --crazyflie-control-dt "${CRAZYFLIE_CONTROL_DT}"
  --crazyflie-obstacle-osqp-max-iter "${CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER}"
  --crazyflie-obstacle-constraint-scale "${CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE}"
  --crazyflie-obstacle-osqp-scaling "${CRAZYFLIE_OBSTACLE_OSQP_SCALING}"
  --crazyflie-obstacle-line-search-step-min "${CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN}"
  --crazyflie-obstacle-trajectory-initialization "${CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION}"
  --crazyflie-obstacle-mpx-barrier-alpha 0.003
  --crazyflie-obstacle-mpx-barrier-sigma 0.25
  --qdldl-backend warp
  --qdldl-factor-backend warp
  --qdldl-solve-backend warp
  --transpose-work
  --segmented
  --segment-strategy optimal
  --no-level-scheduled-solve
  --group-repeated-stages
)

if [[ "${SKIP_PLOTS}" == "1" ]]; then
  ARGS+=(--skip-plots)
fi
if [[ "${WRITE_NPZ}" == "1" ]]; then
  ARGS+=(--write-npz)
else
  ARGS+=(--no-write-npz)
fi
if [[ "${CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC}" == "1" ]]; then
  ARGS+=(--crazyflie-obstacle-osqp-rho-is-vec)
else
  ARGS+=(--no-crazyflie-obstacle-osqp-rho-is-vec)
fi

echo "Constraint SQP iteration sweep"
echo "Job ID:     ${SLURM_JOB_ID:-manual}"
echo "Host:       $(uname -n)"
echo "Results:    ${RESULTS_DIR}"
echo "CSV:        ${RESULTS_DIR}.csv"
echo "Python:     ${PYTHON}"
echo "Systems:    ${SYSTEMS}"
echo "Solvers:    ${SOLVERS}"
echo "SQP iters:  ${SQP_ITERATIONS}"
echo "JAX SQP:    ${JAX_OSQP_SQP_ITERATIONS}"
echo "MPX SQP:    ${MPX_SQP_ITERATIONS}"
echo "Cart JAX:   ${CARTPOLE_JAX_OSQP_SQP_ITERATIONS}"
echo "Cart MPX:   ${CARTPOLE_MPX_SQP_ITERATIONS}"
echo "Crazy JAX:  ${CRAZYFLIE_OBSTACLE_JAX_OSQP_SQP_ITERATIONS}"
echo "Crazy MPX:  ${CRAZYFLIE_OBSTACLE_MPX_SQP_ITERATIONS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Crazy cfg:  horizon=${CRAZYFLIE_HORIZON_STEPS} sim=${CRAZYFLIE_SIM_TIME} scale=${CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE} osqp_iter=${CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER} osqp_scaling=${CRAZYFLIE_OBSTACLE_OSQP_SCALING} rho_vec=${CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC}"
echo
echo "Command: ${PYTHON} benchmarks/nonlinear_mpc/benchmark_constraint_sqp_iteration_sweep.py ${ARGS[*]}"

"${PYTHON}" benchmarks/nonlinear_mpc/benchmark_constraint_sqp_iteration_sweep.py "${ARGS[@]}"
