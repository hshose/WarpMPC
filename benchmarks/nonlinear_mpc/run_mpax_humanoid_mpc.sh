#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=mpaxhuman
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

export QP_SOLVERS="${QP_SOLVERS:-mpax}"
export LEVELSOLVE_MODES="${LEVELSOLVE_MODES:-regular}"
export HUMANOID_BATCH_SIZES="${HUMANOID_BATCH_SIZES:-${MPAX_HUMANOID_BATCH_SIZES:-512,2048,10000,20000,50000,100000,200000,300000}}"
export HUMANOID_OUTPUT_NAME="${HUMANOID_OUTPUT_NAME:-humanoid_mpc_mpax}"
export MAX_ITER="${MAX_ITER:-25}"

exec "${SCRIPT_DIR}/run_humanoid_mpc.sh" "$@"
