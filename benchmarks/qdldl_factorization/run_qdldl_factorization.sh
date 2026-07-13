#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=qdldl
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
source "${SCRIPT_DIR}/common.sh" "${1:-}" "qdldl_factorization"

run_benchmark "qdldl_factorization" \
  "${PYTHON}" benchmarks/qdldl_factorization/benchmark_qdldl_factorization.py \
    --batch-sizes "${QDLDL_BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --repeat "${REPEAT}" \
    --variants "${QDLDL_VARIANTS}" \
    --segment-budget "${SEGMENT_BUDGET}" \
    --segment-strategy "${SEGMENT_STRATEGY}" \
    --max-device-gb "${MAX_DEVICE_GB}" \
    --csv-path "${RESULTS_ROOT}/throughput_mpc_qdldl_variants.csv" \
    --plot-path "${RESULTS_ROOT}/throughput_mpc_qdldl_variants.png" \
    --factor-plot-path "${RESULTS_ROOT}/factor_throughput_mpc_qdldl_variants.png" \
    --solve-plot-path "${RESULTS_ROOT}/solve_throughput_mpc_qdldl_variants.png" \
    --speedup-plot-path "${RESULTS_ROOT}/speedup_mpc_qdldl_variants.png"
