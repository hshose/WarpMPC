#!/bin/bash
### Job Parameters
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=cpphysquad
#SBATCH --output=hpclogs/%A.log

set -uo pipefail

if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/cartpole_tuning" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_h100_cartpole_physical_quadratic_tuning.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/examples/cartpole_tuning"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV="${VENV:-}"
if [[ -n "${VENV}" ]]; then
  PYTHON="${PYTHON:-${VENV}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi
BASHRC="${BASHRC:-${HOME}/.bashrc}"
if [[ -f "${BASHRC}" ]]; then
  source "${BASHRC}"
fi
if [[ -n "${VENV}" && -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${ROOT}/results/cartpole_physical_quadratic_tuning_h100_${STAMP}"
LOG_DIR="${RESULTS_DIR}/logs"
EPISODES="${EPISODES:-10}"
EXPERIMENTS_PER_EPISODE="${EXPERIMENTS_PER_EPISODE:-500}"
ROLLOUTS_PER_EXPERIMENT="${ROLLOUTS_PER_EXPERIMENT:-500}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${ROOT}/hpclogs"
cd "${ROOT}" || exit 1

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export JAX_COMPILATION_CACHE_DIR="${ROOT}/.jax_cache"

echo "Cartpole physical-parameter pure-quadratic MPC TuRBO tuning H100 run"
echo "Host: $(uname -n)"
echo "Root: ${ROOT}"
echo "Python: ${PYTHON}"
echo "Results: ${RESULTS_DIR}"
echo "Episodes: ${EPISODES}"
echo "Experiments per episode: ${EXPERIMENTS_PER_EPISODE}"
echo "Rollouts per experiment: ${ROLLOUTS_PER_EXPERIMENT}"
echo "Started: $(date -Is)"

"${PYTHON}" examples/cartpole_tuning/tune_cartpole_physical_quadratic_mpc.py \
  --output-dir "${RESULTS_DIR}" \
  --dtype float32 \
  --episodes "${EPISODES}" \
  --experiments-per-episode "${EXPERIMENTS_PER_EPISODE}" \
  --rollouts-per-experiment "${ROLLOUTS_PER_EXPERIMENT}" \
  --horizon-steps 100 \
  --dt-start 0.1 \
  --dt-growth 1.0 \
  --sim-time 10.0 \
  --control-dt 0.1 \
  --rollout-steps 100 \
  --integrator-substeps 1 \
  --sqp-iterations 10 \
  --osqp-max-iter 50 \
  --rho 0.1 \
  --sigma 1e-6 \
  --alpha 1.6 \
  --qdldl-backend warp \
  --qdldl-factor-backend warp \
  --qdldl-solve-backend warp \
  --level-scheduled-solve \
  --level-scheduled-solve-threshold 2 \
  --segment-budget 384 \
  --segment-strategy optimal \
  --line-search-step-min 0.1 \
  --enable-rail-constraint \
  --noise-scale 0 \
  --process-noise-scale "0,0,0,0" \
  --input-noise-scale "0" \
  --simulation-parameter-mode nominal \
  --input-disturbance-bound 0 \
  --turbo-candidate-pool 8192 \
  --turbo-max-gp-points 400 \
  --max-experiment-plots 1 \
  2>&1 | tee "${LOG_DIR}/tune_cartpole_physical_quadratic_mpc.log"

code="${PIPESTATUS[0]}"
echo "Finished: $(date -Is), exit=${code}"
exit "${code}"
