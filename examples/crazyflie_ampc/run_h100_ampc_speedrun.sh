#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --job-name=ampc-speedrun
#SBATCH --output=ampc-speedrun-%j.log
set -euo pipefail

# CLAIX uses --export=NONE; configure through positional arguments.
# [SOURCE_ROOT] [run|smoke|validate] [WORK_ROOT] [PYTHON] [CACHE_ROOT]
AMPC_SOURCE_ROOT="${1:-${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}}"
AMPC_SOURCE_ROOT="$(cd "${AMPC_SOURCE_ROOT}" && pwd)"
AMPC_MODE="${2:-run}"
AMPC_WORK_ROOT="${3:-${AMPC_SOURCE_ROOT}/results/ampc_speedrun}"
AMPC_PYTHON="${4:-${PYTHON:-${VENV:+${VENV}/bin/}python}}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  AMPC_CACHE_ROOT="${5:-${AMPC_WORK_ROOT}/cache}"
else
  AMPC_CACHE_ROOT="${5:-${AMPC_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/warpmpc/ampc_speedrun}}"
fi
case "${AMPC_MODE}" in
  run|smoke|validate) ;;
  *) echo "Unknown AMPC mode: ${AMPC_MODE}" >&2; exit 2 ;;
esac
if [[ ! -f "${AMPC_SOURCE_ROOT}/examples/crazyflie_ampc/speedrun.py" ]]; then
  echo "Invalid source directory: ${AMPC_SOURCE_ROOT}" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
unset XLA_FLAGS

run_speedrun() {
  if [[ "${AMPC_MODE}" == validate ]]; then
    "${AMPC_PYTHON}" examples/crazyflie_ampc/validate_speedrun.py --collector \
      --cache-dir "${AMPC_CACHE_ROOT}"
  else
    local extra_args=()
    if [[ "${AMPC_MODE}" == smoke ]]; then
      extra_args=(--collect-batch-size 64 --rollout-steps 4 --train-epochs 2 --train-batch-size 128)
    fi
    "${AMPC_PYTHON}" examples/crazyflie_ampc/speedrun.py --cache-dir "${AMPC_CACHE_ROOT}" \
      "${extra_args[@]}"
  fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cd "${AMPC_SOURCE_ROOT}"
  export PYTHONPATH="${AMPC_SOURCE_ROOT}"
  run_speedrun
  exit
fi

case_dir="${AMPC_WORK_ROOT}/job-${SLURM_JOB_ID}"
mkdir -p "${AMPC_WORK_ROOT}"
mkdir "${case_dir}"
mkdir "${case_dir}/code"
rsync -a --exclude=.git --exclude=.worktrees --exclude=resources --exclude=results \
  --exclude=studies --exclude=cache --exclude=.jax_cache --exclude=__pycache__ \
  "${AMPC_SOURCE_ROOT}/" "${case_dir}/code/"
cd "${case_dir}/code"
export PYTHONPATH="${case_dir}/code" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
AMPC_LOG_NAME=run
[[ "${AMPC_MODE}" != validate ]] || AMPC_LOG_NAME=validation
run_speedrun 2>&1 | tee "${case_dir}/${AMPC_LOG_NAME}.log"
