#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=crazyrollouts
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
RESULTS_DIR="${RUN_ROOT}/nonlinear_mpc"

BATCH_SIZE="${CONSTRAINT_SWEEP_ROLLOUT_PATH_BATCH_SIZE:-20}"
DTYPE="${DTYPE:-float32}"
SEED="${SEED:-0}"
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
"${PYTHON}" -c "import mpx, trajax" >/dev/null 2>&1 || {
  echo "MPX is required for rollout path comparison; install it in the benchmark environment." >&2
  exit 1
}

COMMON_ARGS=(
  --batch-size "${BATCH_SIZE}"
  --horizon-steps "${CRAZYFLIE_HORIZON_STEPS}"
  --sim-time "${CRAZYFLIE_SIM_TIME}"
  --control-dt "${CRAZYFLIE_CONTROL_DT}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --trajectory-initialization "${CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION}"
  --initial-state-sampling border
  --skip-plots
)

JAX_ARGS=(
  --solver jax_osqp
  --sqp-iterations 1
  --max-iter "${CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER}"
  --osqp-scaling "${CRAZYFLIE_OBSTACLE_OSQP_SCALING}"
  --line-search-step-min "${CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN}"
  --obstacle-constraint-scale "${CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE}"
  --qdldl-backend warp
  --qdldl-factor-backend warp
  --qdldl-solve-backend warp
  --transpose-work
  --segmented
  --segment-strategy optimal
  --output-npz "${RESULTS_DIR}/crazyflie_obstacle_rollout_paths_jax_osqp.npz"
  --summary-json "${RESULTS_DIR}/crazyflie_obstacle_rollout_paths_jax_osqp.json"
)

if [[ "${CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC}" == "1" ]]; then
  JAX_ARGS+=(--osqp-rho-is-vec)
else
  JAX_ARGS+=(--no-osqp-rho-is-vec)
fi

MPX_ARGS=(
  --solver mpx
  --sqp-iterations 20
  --mpx-solver-mode primal_dual
  --mpx-equality-weight 1.0e4
  --mpx-barrier-alpha 0.003
  --mpx-barrier-sigma 0.25
  --mpx-num-alpha 11
  --mpx-limited-memory
  --output-npz "${RESULTS_DIR}/crazyflie_obstacle_rollout_paths_mpx.npz"
  --summary-json "${RESULTS_DIR}/crazyflie_obstacle_rollout_paths_mpx.json"
)

echo "Crazyflie obstacle rollout path generation"
echo "Job ID:     ${SLURM_JOB_ID:-manual}"
echo "Host:       $(uname -n)"
echo "Results:    ${RESULTS_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Python:     ${PYTHON}"
echo

"${PYTHON}" benchmarks/nonlinear_mpc/evaluate_crazyflie_obstacle_mpc.py "${COMMON_ARGS[@]}" "${JAX_ARGS[@]}"
"${PYTHON}" benchmarks/nonlinear_mpc/evaluate_crazyflie_obstacle_mpc.py "${COMMON_ARGS[@]}" "${MPX_ARGS[@]}"
