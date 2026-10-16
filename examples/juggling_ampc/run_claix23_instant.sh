#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=juggling-instant
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
# Positional arguments survive CLAIX's sbatch --export=NONE wrapper.
study_root="${1:?Private case directory}"
study_seconds="${2:-5}"
study_protocol="${3:?arm15 or calibrated}"
study_dependencies="${4:?Immutable installed dependencies}"
case "$study_protocol" in arm15) study_arm=1.5;; calibrated) study_arm=1;; *) exit 2;; esac
case "$study_seconds" in 5|10|20) ;; *) exit 2;; esac
study_source="$study_root/source"
study_python="$study_dependencies/venv/bin/python"
mkdir -p "$study_root/results" "$study_root/cache/"{jax,warp,cuda,xdg,tmp,matplotlib}
# Grow on demand with CUDA virtual-memory pooling; never reserve a JAX arena.
# BFC's large contiguous scratch request can fail despite free physical VRAM.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=0
unset XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_MEM_FRACTION
export JAX_COMPILATION_CACHE_DIR="$study_root/cache/jax" WARP_CACHE_PATH="$study_root/cache/warp"
export CUDA_CACHE_PATH="$study_root/cache/cuda" XDG_CACHE_HOME="$study_root/cache/xdg"
export TMPDIR="$study_root/cache/tmp" MPLCONFIGDIR="$study_root/cache/matplotlib"
export PYTHONPATH="$study_source:$study_source/external"
export PATH="$study_dependencies/venv/bin:$PATH"
export LD_LIBRARY_PATH="$("$study_python" -c 'import casadi,pathlib; print(pathlib.Path(casadi.__file__).parent)')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
cd "$study_source"
nvidia-smi -q > "$study_root/results/gpu-before.txt"
scontrol show job "$SLURM_JOB_ID" > "$study_root/results/allocation.txt"
"$study_python" -m pip freeze --all > "$study_root/results/packages.txt"
if [[ ! -f "$study_root/gpu_factors/metadata.json" ]]; then
 "$study_python" examples/juggling_ampc/build_gpu_factors.py \
  examples/juggling_ampc/reference/config/cpp_ball_spec.json "$study_root/gpu_factors"
fi
trap 'nvidia-smi -q > "$study_root/results/gpu-after.txt" || true' EXIT
/usr/bin/time -v -o "$study_root/results/process-resources.txt" \
 "$study_python" -u examples/juggling_ampc/run_gpu_pipeline.py \
 --factors "$study_root/gpu_factors" \
 --source "$study_source/examples/juggling_ampc/reference" \
 --calibration "$study_source/examples/juggling_ampc/calibration/model.json" \
 --eigen "$study_dependencies/eigen/include/eigen3" \
 --collection-multiplier 1 --collection-arm-multiplier "$study_arm" \
 --validation-multiplier 1 --validation-arm-multiplier 1 \
 --batch 60000 --seconds "$study_seconds" --eval-batch 1024 \
 --dataset-gib 3 --epochs 100 --train-batch 8192 \
 --output "$study_root/results"
