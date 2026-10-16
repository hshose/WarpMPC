#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ROOT
export PYTHON="$ROOT/venv/bin/python"
export PYTHONPATH="$ROOT/external${PYTHONPATH:+:$PYTHONPATH}"
# Allocate JAX GPU memory on demand, as requested for CLAIX.
# Grow on demand with CUDA virtual-memory pooling; never reserve a JAX arena.
# BFC's large contiguous scratch request can fail despite free physical VRAM.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=0
unset XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_MEM_FRACTION
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_COMPILATION_CACHE_DIR="$ROOT/results/jax_cache"
export WARP_CACHE_PATH="${TMPDIR:-/tmp}/juggling-warp-${SLURM_JOB_ID:-local}"
export FRANKA_SOURCE="$ROOT/live_source"
export CALIBRATION="$FRANKA_SOURCE/runs/20261006_sim_quality/model.json"
export REFERENCE="$ROOT/results/juggling_reference"
export OBSERVER="$ROOT/results/observer_claix/libjuggling_batch_observer.so"
export EIGEN_DIR="$FRANKA_SOURCE/build/eigen_install/include/eigen3"
export LD_LIBRARY_PATH="$($PYTHON -c 'import casadi,pathlib; print(pathlib.Path(casadi.__file__).parent)')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ROOT/venv/bin:$PATH"
cd "$ROOT"
