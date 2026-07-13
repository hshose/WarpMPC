#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=npsadi
#SBATCH --output=hpclogs/%A.log
#SBATCH --error=hpclogs/%A.log

DTYPES="${DTYPES:-float32}"
NUMPYSADI_PROVIDERS="${NUMPYSADI_PROVIDERS:-numpysadi jaxadi}"
RUN_CUSADI="${RUN_CUSADI:-0}"
CUSADI_ROOT="${CUSADI_ROOT:-}"
CUSADI_CMAKE_JOBS="${CUSADI_CMAKE_JOBS:-${SLURM_CPUS_PER_TASK:-24}}"
CUSADI_CUDA_ARCHITECTURES="${CUSADI_CUDA_ARCHITECTURES:-native}"
CUSADI_CUDA_COMPILER="${CUSADI_CUDA_COMPILER:-}"
CUSADI_PYTHON="${CUSADI_PYTHON:-}"
if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/benchmarks/numpysadi_synthetic" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/benchmarks/numpysadi_synthetic"
source "${SCRIPT_DIR}/common.sh" "${1:-}" "numpysadi_synthetic"

SIZES=(100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000)
read -r -a PROVIDERS <<< "${NUMPYSADI_PROVIDERS}"

cusadi_python_has_deps() {
  "$1" -c "import casadi, numpy, torch" >/dev/null 2>&1
}

select_cusadi_python() {
  if [[ -n "${CUSADI_PYTHON}" ]]; then
    return
  fi

  local candidate
  for candidate in "${PYTHON}" python3.12 python3.11; do
    if command -v "${candidate}" >/dev/null 2>&1 && cusadi_python_has_deps "${candidate}"; then
      CUSADI_PYTHON="$(command -v "${candidate}")"
      return
    fi
  done

  CUSADI_PYTHON="${PYTHON}"
}

CUSADI_CUDA_COMPILER_ARGS=()
if [[ -z "${CUSADI_CUDA_COMPILER}" ]]; then
  CUSADI_CUDA_COMPILER="$(
    find "${VENV}/lib" -path '*/site-packages/nvidia/cu*/bin/nvcc' -type f -executable 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
fi
if [[ -n "${CUSADI_CUDA_COMPILER}" ]]; then
  CUSADI_CUDA_COMPILER_ARGS=(--cuda-compiler "${CUSADI_CUDA_COMPILER}")
fi

for dtype in ${DTYPES//,/ }; do
  RESULTS="${RESULTS_ROOT}/numpysadi_synthetic_${dtype}.jsonl"
  PLOT="${RESULTS_ROOT}/numpysadi_synthetic_${dtype}.png"

  run_benchmark "numpysadi_synthetic_${dtype}" \
    "${PYTHON}" benchmarks/numpysadi_synthetic/benchmark_numpysadi_synthetic.py \
      --sizes "${SIZES[@]}" \
      --providers "${PROVIDERS[@]}" \
      --repeats 5 \
      --compile-timeout-seconds 1800 \
      --dtype "${dtype}" \
      --output "${RESULTS}"

  if [[ "${RUN_CUSADI}" == "1" ]]; then
    if [[ -z "${CUSADI_ROOT}" ]]; then
      echo "RUN_CUSADI=1 requires CUSADI_ROOT to point at an external CusADi checkout." >&2
      exit 1
    fi
    if [[ ! -f "${CUSADI_ROOT}/src/__init__.py" ]]; then
      echo "CusADi source package not found under: ${CUSADI_ROOT}" >&2
      echo "Set CUSADI_ROOT to an external CusADi checkout." >&2
      exit 1
    fi
    select_cusadi_python
    if ! cusadi_python_has_deps "${CUSADI_PYTHON}"; then
      echo "CusADi Python lacks required modules (casadi, numpy, torch): ${CUSADI_PYTHON}" >&2
      echo "Install torch in the benchmark venv or set CUSADI_PYTHON to a compatible Python." >&2
      exit 1
    fi
    echo "CusADi Python: ${CUSADI_PYTHON}"

    run_benchmark "cusadi_synthetic_${dtype}" \
      "${CUSADI_PYTHON}" numpysadi/benchmarks/benchmark_cusadi_synthetic.py \
        --sizes "${SIZES[@]}" \
        --batch-size 10000 \
        --repeats 5 \
        --compile-timeout-seconds 1800 \
        --output "${RESULTS}" \
        --cusadi-root "${CUSADI_ROOT}" \
        --cmake-jobs "${CUSADI_CMAKE_JOBS}" \
        --cuda-architectures "${CUSADI_CUDA_ARCHITECTURES}" \
        "${CUSADI_CUDA_COMPILER_ARGS[@]}"
  fi

  run_benchmark "plot_numpysadi_synthetic_${dtype}" \
    "${PYTHON}" benchmarks/numpysadi_synthetic/plot_numpysadi_synthetic.py \
      "${RESULTS}" \
      --output "${PLOT}" \
      --title "numpysadi synthetic benchmark (${dtype}, batch size 10000)"
done
