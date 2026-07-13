#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=boxgrad
#SBATCH --output=hpclogs/%A.log
#SBATCH --error=hpclogs/%A.log

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
source "${SCRIPT_DIR}/common.sh" "${1:-}" "boxosqp_linear_mpc_gradients"

run_benchmark "boxosqp_linear_mpc_gradients" \
  "${PYTHON}" benchmarks/linear_mpc_gradients/benchmark_boxosqp_linear_mpc_gradients.py \
    --horizon "${HORIZON}" \
    --batch-sizes "${BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --modes "grad" \
    --eq-qp-solve cg \
    --max-iter "${MAX_ITER}" \
    --repeat "${REPEAT}" \
    --max-device-gb "${MAX_DEVICE_GB}" \
    --csv-path "${RESULTS_ROOT}/throughput_mpc_boxosqp_gradients.csv" \
    --plot-path "${RESULTS_ROOT}/throughput_mpc_boxosqp_gradients.png" \
    --skip-horizon-sweep
