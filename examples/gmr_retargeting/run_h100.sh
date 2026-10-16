#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --job-name=gmr-retarget
#SBATCH --output=hpclogs/%A.log

# Defaults use independent motion chunks and adaptive SQP stopping.
set -euo pipefail
# Positional settings also work on sites that strip exported environment variables.
ROOT="${ROOT:-${1:-}}"
DATA="${DATA:-${2:-}}"
PYTHON="${PYTHON:-${3:-}}"
if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/gmr_retargeting" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_h100.sh" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
VENV="${VENV:-}"
PYTHON="${PYTHON:-${VENV:+${VENV}/bin/}python}"
SQP_ITERATIONS="${SQP_ITERATIONS:-11}"
INITIAL_SQP_ITERATIONS="${INITIAL_SQP_ITERATIONS:-11}"
ADMM_ITERATIONS="${ADMM_ITERATIONS:-50}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
SOURCE_ARGS=()
BENCHMARK_ARGS=()
SOLVER_ARGS=(--scheduler "${SCHEDULER:-independent}")
if [[ "${ADAPTIVE_STOPPING:-1}" == 1 ]]; then
  SOLVER_ARGS+=(--adaptive-stopping)
fi
if [[ "${SYNTHETIC:-0}" == 1 ]]; then
  SOURCE_ARGS+=(--synthetic)
  BENCHMARK_ARGS+=(--allow-synthetic)
  if [[ "${STRESS:-0}" == 1 ]]; then
    SOURCE_ARGS+=(--synthetic-stress)
  fi
else
  : "${DATA:?Set DATA to prepared AMASS motion, or explicitly set SYNTHETIC=1}"
  SOURCE_ARGS+=(--data "${DATA}")
fi
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/gmr_h100_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZES="${BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${ROOT}/cache/warp-gmr}"
cd "${ROOT}"
mkdir -p "${RESULTS_DIR}" hpclogs
nvidia-smi
"${PYTHON}" -c 'import jax; assert "H100" in jax.devices()[0].device_kind, jax.devices()'
"${PYTHON}" examples/gmr_retargeting/validate.py \
  "${SOURCE_ARGS[@]}" --frames "${FRAMES:-262144}" \
  --chunk-size "${CHUNK_SIZE}" --initial-sqp-iterations "${INITIAL_SQP_ITERATIONS}" \
  --sqp-iterations "${SQP_ITERATIONS}" --admm-iterations "${ADMM_ITERATIONS}" \
  "${SOLVER_ARGS[@]}" \
  --rho "${RHO:-0.1}" --output-dir "${RESULTS_DIR}/reference" \
  2>&1 | tee "${RESULTS_DIR}/accuracy.log"
"${PYTHON}" examples/gmr_retargeting/benchmark.py \
  --reference-dir "${RESULTS_DIR}/reference" --output-dir "${RESULTS_DIR}/throughput" \
  --batch-sizes "${BATCH_SIZES}" --sqp-iterations "${SQP_ITERATIONS}" \
  --admm-iterations "${ADMM_ITERATIONS}" --rho "${RHO:-0.1}" \
  "${SOLVER_ARGS[@]}" \
  --chunk-size "${CHUNK_SIZE}" --initial-sqp-iterations "${INITIAL_SQP_ITERATIONS}" \
  --warmups "${WARMUPS:-1}" --repeats "${REPEATS:-3}" --timeout "${BATCH_TIMEOUT:-43200}" "${BENCHMARK_ARGS[@]}" \
  2>&1 | tee "${RESULTS_DIR}/throughput.log"
# Plot locally by default: the compute nodes may lack the paper's TeX fonts.
if [[ "${PLOT:-0}" == 1 ]]; then
  PLOT_ENV=()
  if [[ "${PLOT_CLEAN_LD_LIBRARY_PATH:-0}" == 1 ]]; then
    # System TeX can conflict with compiler-module zlib libraries. This only
    # changes the plotting subprocess; the solver keeps its CUDA environment.
    PLOT_ENV=(env -u LD_LIBRARY_PATH)
  fi
  "${PLOT_ENV[@]}" "${PYTHON}" examples/gmr_retargeting/plot.py \
    --reference-dir "${RESULTS_DIR}/reference" --results-dir "${RESULTS_DIR}/throughput" \
    --output "${RESULTS_DIR}/gmr_retargeting_throughput.pdf"
fi
