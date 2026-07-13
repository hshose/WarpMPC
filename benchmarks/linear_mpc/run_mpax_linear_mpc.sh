#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=mpaxmpc
#SBATCH --output=hpclogs/%A.log
#SBATCH --error=hpclogs/%A.log

if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks/linear_mpc" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks/linear_mpc"
MAX_ITER="${MAX_ITER:-1000}"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "mpax_linear_mpc"

run_benchmark "mpax_linear_mpc" \
  "${PYTHON}" benchmarks/linear_mpc/benchmark_mpax_linear_mpc.py \
    --horizon "${HORIZON}" \
    --batch-sizes "${BATCH_SIZES}" \
    --dtypes "${DTYPES}" \
    --iteration-limit "${MAX_ITER}" \
    --eps-abs "${EPS_ABS}" \
    --eps-rel "${EPS_REL}" \
    --repeat "${REPEAT}" \
    --max-device-gb "${MAX_DEVICE_GB}" \
    --csv-path "${RESULTS_ROOT}/throughput_mpc_mpax.csv" \
    --plot-path "${RESULTS_ROOT}/throughput_mpc_mpax.png"
