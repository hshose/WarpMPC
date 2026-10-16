#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=mpc-sensitivity
#SBATCH --output=hpclogs/%A.log
#SBATCH --error=hpclogs/%A.log
set -euo pipefail
if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks/linear_mpc_gradients" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks/linear_mpc_gradients"
VENV="${VENV:-}"
PYTHON="${PYTHON:-${VENV:+${VENV}/bin/}python}"
BASHRC="${BASHRC:-${HOME}/.bashrc}"
if [[ -f "${BASHRC}" ]]; then source "${BASHRC}"; fi
if [[ -f "${VENV}/bin/activate" ]]; then source "${VENV}/bin/activate"; fi
RUN_ROOT="${1:-${ROOT}/results/sensitivity_accuracy_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${RUN_ROOT}/linear_mpc_gradients/sensitivity_accuracy"
WORKERS="${SENSITIVITY_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=multi_output_fusion}"
export MPLCONFIGDIR="${RESULTS}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
echo "Host: $(hostname); job: ${SLURM_JOB_ID:-manual}; results: ${RESULTS}"
SCRIPT="${ROOT}/benchmarks/linear_mpc_gradients"
GPU_ARGS=(--require-gpu)
if [[ "${SENSITIVITY_REQUIRE_GPU:-1}" == 0 ]]; then GPU_ARGS=(); fi
PENALTY_ARGS=(--fixed-only)
if [[ "${SENSITIVITY_INCLUDE_LEGACY:-0}" == 1 ]]; then PENALTY_ARGS=(--include-legacy); fi
PLOT_ARGS=(--no-tex)
if [[ "${SENSITIVITY_USE_TEX:-0}" == 1 ]]; then PLOT_ARGS=(); fi

# Compute the converged reference once; forward states are shared across refinements.
"${PYTHON}" "${SCRIPT}/benchmark_sensitivity_accuracy.py" \
  --output-dir "${RESULTS}/reference" --reference-only --objective u0-jacobian \
  --batch-size "${SENSITIVITY_BATCH_SIZE:-1000}" --horizon "${SENSITIVITY_HORIZON:-40}" \
  --refinements "${SENSITIVITY_REFINEMENTS:-1,5,10,20,25,50,100,200,250,400,500}" \
  --forward-iterations "${SENSITIVITY_FORWARD_ITERATIONS:-25,100,200,800,2000}" \
  --workers "${WORKERS}"
"${PYTHON}" "${SCRIPT}/analyze_active_sets.py" \
  --results-dir "${RESULTS}/reference" --reference-only --workers "${WORKERS}"
"${PYTHON}" "${SCRIPT}/benchmark_precision_grid.py" \
  --reference-dir "${RESULTS}/reference" --output-dir "${RESULTS}/precision_grid" \
  "${PENALTY_ARGS[@]}" "${GPU_ARGS[@]}"
"${PYTHON}" "${SCRIPT}/plot_precision_grid.py" \
  --results-dir "${RESULTS}/precision_grid" \
  --output-dir "${RESULTS}/precision_grid/figures" \
  --workers "${SENSITIVITY_PLOT_WORKERS:-4}" "${PLOT_ARGS[@]}"
echo "COMPLETE: ${RESULTS}/precision_grid/figures/INDEX.md"
