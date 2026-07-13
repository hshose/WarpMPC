#!/bin/bash
set -euo pipefail

if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_all_paper_benchmarks_h100.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${1:-${ROOT}/results/h100_paper_benchmarks_sequential_${STAMP}}"
LOG_DIR="${RESULTS_ROOT}/logs"
STATUS_CSV="${RESULTS_ROOT}/sequential_status.csv"

PAPER_QDLDL_VARIANTS="${PAPER_QDLDL_VARIANTS:-baseline,transpose,segmented,transpose+segmented,transpose+segmented+levelsolve,factor-warp+solve-jax:transpose+segmented,factor-warp+solve-jax:transpose+segmented+levelsolve,factor-warp+solve-warp:transpose,factor-warp+solve-warp:transpose+segmented,factor-warp+solve-warp:transpose+segmented+levelsolve}"
PAPER_OSQP_VARIANTS="${PAPER_OSQP_VARIANTS:-baseline,transpose+segmented,transpose+segmented+levelsolve,factor-warp+solve-jax:transpose+segmented,factor-warp+solve-jax:transpose+segmented+levelsolve,factor-warp+solve-warp:transpose,factor-warp+solve-warp:transpose+segmented,factor-warp+solve-warp:transpose+segmented+levelsolve}"
PAPER_NONLINEAR_QDLDL_VARIANTS="${PAPER_NONLINEAR_QDLDL_VARIANTS:-baseline,factor-warp+solve-warp:transpose,factor-warp+solve-warp:transpose+segmented,factor-warp+solve-warp:transpose+segmented+levelsolve}"
PAPER_QDLDL_BACKEND_PAIRS="${PAPER_QDLDL_BACKEND_PAIRS:-jax:jax,warp:jax,warp:warp}"
PAPER_LEVELSOLVE_MODES="${PAPER_LEVELSOLVE_MODES:-regular,levelsolve}"
PAPER_NONLINEAR_BATCH_SIZES="${PAPER_NONLINEAR_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}"
SWEEP_BATCH_SIZE="${CONSTRAINT_SWEEP_BATCH_SIZE:-200000}"
SWEEP_CARTPOLE_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CARTPOLE_SQP_ITERATIONS:-5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
SWEEP_CARTPOLE_JAX_OSQP_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CARTPOLE_JAX_OSQP_SQP_ITERATIONS:-5}"
SWEEP_CARTPOLE_MPX_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CARTPOLE_MPX_SQP_ITERATIONS:-5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
SWEEP_CRAZYFLIE_OBSTACLE_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_SQP_ITERATIONS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
SWEEP_CRAZYFLIE_OBSTACLE_JAX_OSQP_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_JAX_OSQP_SQP_ITERATIONS:-1}"
SWEEP_CRAZYFLIE_OBSTACLE_MPX_SQP_ITERATIONS="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_MPX_SQP_ITERATIONS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
SWEEP_CRAZYFLIE_HORIZON_STEPS="${CONSTRAINT_SWEEP_CRAZYFLIE_HORIZON_STEPS:-40}"
SWEEP_CRAZYFLIE_SIM_TIME="${CONSTRAINT_SWEEP_CRAZYFLIE_SIM_TIME:-2.0}"
SWEEP_CRAZYFLIE_CONTROL_DT="${CONSTRAINT_SWEEP_CRAZYFLIE_CONTROL_DT:-0.01}"
SWEEP_CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER:-100}"
SWEEP_CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE:-20.0}"
SWEEP_CRAZYFLIE_OBSTACLE_OSQP_SCALING="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_SCALING:-10}"
SWEEP_CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN:-0.01}"
SWEEP_CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION:-initial_state}"
SWEEP_CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC="${CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC:-1}"
SWEEP_SOLVERS="${CONSTRAINT_SWEEP_SOLVERS:-jax_osqp,mpx}"

command mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}" "${ROOT}/hpclogs"
cd "${ROOT}" || exit 1

printf 'study,name,status,elapsed_s,log,script\n' > "${STATUS_CSV}"
failures=0

run_job() {
  local study="$1"
  local name="$2"
  local script="$3"
  shift 3
  local env_args=("$@")
  local log_path="${LOG_DIR}/${study}_${name}.log"
  local start_s end_s elapsed_s status

  echo "${study}/${name}: running"
  start_s="$(date +%s)"
  set +e
  env "ROOT=${ROOT}" "${env_args[@]}" bash "${script}" "${RESULTS_ROOT}" > "${log_path}" 2>&1
  status="$?"
  set -e
  end_s="$(date +%s)"
  elapsed_s="$((end_s - start_s))"

  printf '%s,%s,%s,%s,%s,%s\n' \
    "${study}" "${name}" "${status}" "${elapsed_s}" "${log_path}" "${script}" \
    >> "${STATUS_CSV}"

  if [[ "${status}" -eq 0 ]]; then
    echo "${study}/${name}: ok (${elapsed_s}s)"
  else
    echo "${study}/${name}: failed with exit ${status} (${elapsed_s}s); log: ${log_path}"
    failures="$((failures + 1))"
  fi
}

echo "H100 paper benchmark sequential runner"
echo "Results: ${RESULTS_ROOT}"
echo "Logs:    ${LOG_DIR}"
echo "Status:  ${STATUS_CSV}"
echo

run_job "numpysadi_synthetic" "numpysadi_synthetic" "${ROOT}/benchmarks/numpysadi_synthetic/run_numpysadi_synthetic.sh"

run_job "qdldl_factorization" "qdldl_factorization" "${ROOT}/benchmarks/qdldl_factorization/run_qdldl_factorization.sh" \
  "QDLDL_VARIANTS=${PAPER_QDLDL_VARIANTS}"
run_job "qdldl_factorization" "cudss_factorization" "${ROOT}/benchmarks/qdldl_factorization/run_cudss_factorization.sh"
run_job "qdldl_factorization" "dense_lu_factorization" "${ROOT}/benchmarks/qdldl_factorization/run_dense_lu_factorization.sh"
run_job "qdldl_factorization" "jaxopt_cg_factorization" "${ROOT}/benchmarks/qdldl_factorization/run_jaxopt_cg_factorization.sh"

run_job "linear_mpc" "jax_osqp_linear_mpc" "${ROOT}/benchmarks/linear_mpc/run_jax_osqp_linear_mpc.sh" \
  "VARIANTS=${PAPER_OSQP_VARIANTS}"
run_job "linear_mpc" "boxosqp_linear_mpc" "${ROOT}/benchmarks/linear_mpc/run_boxosqp_linear_mpc.sh"
run_job "linear_mpc" "qpax_linear_mpc" "${ROOT}/benchmarks/linear_mpc/run_qpax_linear_mpc.sh"
run_job "linear_mpc" "mpax_linear_mpc" "${ROOT}/benchmarks/linear_mpc/run_mpax_linear_mpc.sh"

run_job "linear_mpc_gradients" "jax_osqp_linear_mpc_gradients" "${ROOT}/benchmarks/linear_mpc_gradients/run_jax_osqp_linear_mpc_gradients.sh" \
  "VARIANTS=${PAPER_OSQP_VARIANTS}"
run_job "linear_mpc_gradients" "boxosqp_linear_mpc_gradients" "${ROOT}/benchmarks/linear_mpc_gradients/run_boxosqp_linear_mpc_gradients.sh"
run_job "linear_mpc_gradients" "qpax_linear_mpc_gradients" "${ROOT}/benchmarks/linear_mpc_gradients/run_qpax_linear_mpc_gradients.sh"
run_job "linear_mpc_gradients" "mpax_linear_mpc_gradients" "${ROOT}/benchmarks/linear_mpc_gradients/run_mpax_linear_mpc_gradients.sh"

run_job "nonlinear_mpc" "cartpole_quadratic_mpc" "${ROOT}/benchmarks/nonlinear_mpc/run_cartpole_quadratic_mpc.sh" \
  "CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "NONLINEAR_QDLDL_VARIANTS=${PAPER_NONLINEAR_QDLDL_VARIANTS}" \
  "LEVELSOLVE_MODES=${PAPER_LEVELSOLVE_MODES}"
run_job "nonlinear_mpc" "cartpole_quadratic_mpc_mpax" "${ROOT}/benchmarks/nonlinear_mpc/run_mpax_cartpole_quadratic_mpc.sh" \
  "CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "MPAX_CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "cartpole_quadratic_mpc_mpx" "${ROOT}/benchmarks/nonlinear_mpc/run_mpx_cartpole_quadratic_mpc.sh" \
  "CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "MPX_CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "cartpole_quadratic_mpc_turbompc" "${ROOT}/benchmarks/nonlinear_mpc/run_turbompc_cartpole_quadratic_mpc.sh" \
  "CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "TURBOMPC_CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "cartpole_quadratic_mpc_turbompc_autodiff" "${ROOT}/benchmarks/nonlinear_mpc/run_turbompc_autodiff_cartpole_quadratic_mpc.sh" \
  "CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "TURBOMPC_AUTODIFF_CARTPOLE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "crazyflie_sqp_rollout" "${ROOT}/benchmarks/nonlinear_mpc/run_crazyflie_sqp_rollout.sh" \
  "CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "NONLINEAR_QDLDL_VARIANTS=${PAPER_NONLINEAR_QDLDL_VARIANTS}" \
  "LEVELSOLVE_MODES=${PAPER_LEVELSOLVE_MODES}"
run_job "nonlinear_mpc" "crazyflie_sqp_rollout_mpax" "${ROOT}/benchmarks/nonlinear_mpc/run_mpax_crazyflie_sqp_rollout.sh" \
  "CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "MPAX_CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "crazyflie_sqp_rollout_mpx" "${ROOT}/benchmarks/nonlinear_mpc/run_mpx_crazyflie_sqp_rollout.sh" \
  "CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "MPX_CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "crazyflie_sqp_rollout_turbompc" "${ROOT}/benchmarks/nonlinear_mpc/run_turbompc_crazyflie_sqp_rollout.sh" \
  "CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "TURBOMPC_CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "crazyflie_sqp_rollout_turbompc_autodiff" "${ROOT}/benchmarks/nonlinear_mpc/run_turbompc_autodiff_crazyflie_sqp_rollout.sh" \
  "CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "TURBOMPC_AUTODIFF_CRAZYFLIE_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "humanoid_mpc" "${ROOT}/benchmarks/nonlinear_mpc/run_humanoid_mpc.sh" \
  "HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "NONLINEAR_QDLDL_VARIANTS=${PAPER_NONLINEAR_QDLDL_VARIANTS}" \
  "LEVELSOLVE_MODES=${PAPER_LEVELSOLVE_MODES}"
run_job "nonlinear_mpc" "humanoid_mpc_mpax" "${ROOT}/benchmarks/nonlinear_mpc/run_mpax_humanoid_mpc.sh" \
  "HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "MPAX_HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "humanoid_mpc_mpx" "${ROOT}/benchmarks/nonlinear_mpc/run_mpx_humanoid_mpc.sh" \
  "HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "MPX_HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "humanoid_mpc_turbompc" "${ROOT}/benchmarks/nonlinear_mpc/run_turbompc_humanoid_mpc.sh" \
  "HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "TURBOMPC_HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"
run_job "nonlinear_mpc" "humanoid_mpc_turbompc_autodiff" "${ROOT}/benchmarks/nonlinear_mpc/run_turbompc_autodiff_humanoid_mpc.sh" \
  "HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}" \
  "TURBOMPC_AUTODIFF_HUMANOID_BATCH_SIZES=${PAPER_NONLINEAR_BATCH_SIZES}"

run_job "constraint_sqp_iteration_sweep" "cartpole_constraint_sqp_sweep" "${ROOT}/benchmarks/nonlinear_mpc/run_constraint_sqp_iteration_sweep.sh" \
  "CONSTRAINT_SWEEP_NAME=cartpole_constraint_sqp_iteration_sweep" \
  "CONSTRAINT_SWEEP_SYSTEMS=cartpole" \
  "CONSTRAINT_SWEEP_SOLVERS=${SWEEP_SOLVERS}" \
  "CONSTRAINT_SWEEP_SQP_ITERATIONS=${SWEEP_CARTPOLE_SQP_ITERATIONS}" \
  "CONSTRAINT_SWEEP_JAX_OSQP_SQP_ITERATIONS=${SWEEP_CARTPOLE_JAX_OSQP_SQP_ITERATIONS}" \
  "CONSTRAINT_SWEEP_MPX_SQP_ITERATIONS=${SWEEP_CARTPOLE_MPX_SQP_ITERATIONS}" \
  "CONSTRAINT_SWEEP_BATCH_SIZE=${SWEEP_BATCH_SIZE}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_HORIZON_STEPS=${SWEEP_CRAZYFLIE_HORIZON_STEPS}"
run_job "constraint_sqp_iteration_sweep" "crazyflie_obstacle_constraint_sqp_sweep" "${ROOT}/benchmarks/nonlinear_mpc/run_constraint_sqp_iteration_sweep.sh" \
  "CONSTRAINT_SWEEP_NAME=crazyflie_obstacle_constraint_sqp_iteration_sweep" \
  "CONSTRAINT_SWEEP_SYSTEMS=crazyflie_obstacle" \
  "CONSTRAINT_SWEEP_SOLVERS=${SWEEP_SOLVERS}" \
  "CONSTRAINT_SWEEP_SQP_ITERATIONS=${SWEEP_CRAZYFLIE_OBSTACLE_SQP_ITERATIONS}" \
  "CONSTRAINT_SWEEP_JAX_OSQP_SQP_ITERATIONS=${SWEEP_CRAZYFLIE_OBSTACLE_JAX_OSQP_SQP_ITERATIONS}" \
  "CONSTRAINT_SWEEP_MPX_SQP_ITERATIONS=${SWEEP_CRAZYFLIE_OBSTACLE_MPX_SQP_ITERATIONS}" \
  "CONSTRAINT_SWEEP_BATCH_SIZE=${SWEEP_BATCH_SIZE}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_HORIZON_STEPS=${SWEEP_CRAZYFLIE_HORIZON_STEPS}"
run_job "constraint_sqp_iteration_sweep" "crazyflie_obstacle_rollout_paths" "${ROOT}/benchmarks/nonlinear_mpc/run_crazyflie_obstacle_rollout_paths.sh" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_HORIZON_STEPS=${SWEEP_CRAZYFLIE_HORIZON_STEPS}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_SIM_TIME=${SWEEP_CRAZYFLIE_SIM_TIME}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_CONTROL_DT=${SWEEP_CRAZYFLIE_CONTROL_DT}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER=${SWEEP_CRAZYFLIE_OBSTACLE_OSQP_MAX_ITER}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE=${SWEEP_CRAZYFLIE_OBSTACLE_CONSTRAINT_SCALE}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_SCALING=${SWEEP_CRAZYFLIE_OBSTACLE_OSQP_SCALING}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN=${SWEEP_CRAZYFLIE_OBSTACLE_LINE_SEARCH_STEP_MIN}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION=${SWEEP_CRAZYFLIE_OBSTACLE_TRAJECTORY_INITIALIZATION}" \
  "CONSTRAINT_SWEEP_CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC=${SWEEP_CRAZYFLIE_OBSTACLE_OSQP_RHO_IS_VEC}"

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "Completed H100 paper benchmark sequential run."
  echo "Results: ${RESULTS_ROOT}"
  echo "Status: ${STATUS_CSV}"
  exit 0
fi

echo "Completed H100 paper benchmark sequential run with ${failures} failed job(s)."
echo "Results: ${RESULTS_ROOT}"
echo "Status: ${STATUS_CSV}"
exit 1
