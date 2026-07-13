#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=cudss
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
source "${SCRIPT_DIR}/common.sh" "${1:-}" "cudss_factorization"

if [[ "${SKIP_CUDA_MODULE_LOAD:-0}" != "1" ]] && command -v module >/dev/null 2>&1; then
  module load "${CUDSS_CUDA_MODULE:-CUDA/12.8.0}"
fi

if declare -F use_cudss >/dev/null 2>&1; then
  use_cudss
fi

if [[ -z "${CUDSS_DIR:-}" && -n "${CUDSS_HOME:-}" ]]; then
  CUDSS_DIR="${CUDSS_HOME}"
  export CUDSS_DIR
fi

if [[ -z "${CUDSS_DIR:-}" ]]; then
  CUDSS_DIR="$("${PYTHON}" - <<'PY'
import importlib.util
import pathlib
import sys

for module_name in ("nvidia.cu13", "nvidia.cu12"):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.submodule_search_locations is None:
        continue
    root = pathlib.Path(next(iter(spec.submodule_search_locations)))
    if (root / "include" / "cudss.h").exists() and list((root / "lib").glob("libcudss.so*")):
        print(root)
        sys.exit(0)
sys.exit(1)
PY
)"
  export CUDSS_DIR
fi

CUDSS_LIB_DIR=""
if [[ -n "${CUDSS_DIR:-}" && -d "${CUDSS_DIR}/lib" ]]; then
  CUDSS_LIB_DIR="${CUDSS_DIR}/lib"
  export CUDSS_HOME="${CUDSS_HOME:-${CUDSS_DIR}}"
  export CUDSS_DIR
fi
if [[ -n "${CUDSS_DIR:-}" && -d "${CUDSS_DIR}/include" ]]; then
  export CPATH="${CUDSS_DIR}/include${CPATH:+:${CPATH}}"
fi

CUDA_LIB_DIRS=()
while IFS= read -r lib_dir; do
  if [[ -n "${CUDSS_LIB_DIR}" ]] && compgen -G "${lib_dir}/libcudss.so*" >/dev/null; then
    continue
  fi
  CUDA_LIB_DIRS+=("${lib_dir}")
done < <(find "${VENV}/lib" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | sort -V)
if (( ${#CUDA_LIB_DIRS[@]} > 0 )); then
  CUDA_LIBS="$(
    IFS=:
    echo "${CUDA_LIB_DIRS[*]}"
  )"
  export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${CUDSS_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDSS_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

DTYPE_BATCH_ARGS=()
if [[ -n "${CUDSS_DTYPE_BATCH_SIZES:-}" ]]; then
  DTYPE_BATCH_ARGS+=(--dtype-batch-sizes "${CUDSS_DTYPE_BATCH_SIZES}")
fi

run_benchmark "cudss_factorization" \
  "${PYTHON}" benchmarks/qdldl_factorization/benchmark_cudss_factorization.py \
    --batch-sizes "${CUDSS_BATCH_SIZES}" \
    "${DTYPE_BATCH_ARGS[@]}" \
    --dtypes "${DTYPES}" \
    --repeat "${REPEAT}" \
    --max-device-gb "${MAX_DEVICE_GB}" \
    --csv-path "${RESULTS_ROOT}/throughput_mpc_cudss.csv" \
    --plot-path "${RESULTS_ROOT}/throughput_mpc_cudss.png"
