#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=cgfact
#SBATCH --output=hpclogs/%A.log
#SBATCH --error=hpclogs/%A.log

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
source "${SCRIPT_DIR}/common.sh" "${1:-}" "jaxopt_cg_factorization"

run_benchmark "jaxopt_cg_factorization" \
  "${PYTHON}" benchmarks/qdldl_factorization/benchmark_jaxopt_cg_factorization.py \
    --batch-sizes "${QDLDL_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --repeat "${REPEAT}" \
    --max-device-gb "${MAX_DEVICE_GB}" \
    --cg-maxiter "${MAX_ITER}" \
    --csv-path "${RESULTS_ROOT}/throughput_mpc_jaxopt_solve_cg.csv" \
    --plot-path "${RESULTS_ROOT}/throughput_mpc_jaxopt_solve_cg.png"
