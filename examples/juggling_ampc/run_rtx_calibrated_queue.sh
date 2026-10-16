#!/usr/bin/env bash
# Detached sequential calibrated collection -> training -> 30 s evaluation.
set -uo pipefail
queue_root="${1:?Queue root}"
dependencies="${2:?Tested RTX dependency root}"
python="${3:?Python executable}"
mkdir -p "$queue_root/cache"
exec 9>"$queue_root/queue.lock"
flock -n 9 || exit 2
printf '%s\n' "$$" > "$queue_root/queue.pid"
export PYTHONPATH="$dependencies/python-deps:$dependencies/python-deps/cmeel.prefix/lib/python3.14/site-packages"
export LD_LIBRARY_PATH="$dependencies/python-deps/cmeel.prefix/lib:$dependencies/python-deps/casadi"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=0
unset XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_MEM_FRACTION
export WARP_CACHE_PATH="$queue_root/cache/warp" JAX_COMPILATION_CACHE_DIR="$queue_root/cache/jax"
export CUDA_CACHE_PATH="$queue_root/cache/cuda" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=24
queue_failed=0
for seconds in 5 10 20; do
    case_root="$queue_root/calibrated_${seconds}s"
    if [[ -f "$case_root/finished.txt" && -f "$case_root/exit_code.txt" ]] && [[ "$(cat "$case_root/exit_code.txt")" == 0 ]]; then
        echo "$(date -Is) SKIP completed calibrated_${seconds}s"
        continue
    fi
    export PYTHONPATH="$case_root/source:$case_root/source/external:$dependencies/python-deps:$dependencies/python-deps/cmeel.prefix/lib/python3.14/site-packages"
    date -Is > "$case_root/started.txt"
    echo "$(date -Is) START calibrated_${seconds}s"
    "$python" -u "$case_root/source/examples/juggling_ampc/run_gpu_pipeline.py" \
        --factors "$case_root/gpu_factors" \
        --source "$case_root/source/examples/juggling_ampc/reference" \
        --calibration "$case_root/source/examples/juggling_ampc/calibration/model.json" \
        --eigen "$dependencies/python-deps/cmeel.prefix/include/eigen3" \
        --collection-multiplier 1 --collection-arm-multiplier 1 \
        --validation-multiplier 1 --validation-arm-multiplier 1 \
        --batch 60000 --seconds "$seconds" --eval-batch 1024 --dataset-gib 3 \
        --epochs 100 --train-batch 8192 --output "$case_root/artifacts" \
        > "$case_root/run.log" 2>&1 &
    child=$!
    printf '%s\n' "$child" > "$case_root/process.pid"
    wait "$child"
    result=$?
    printf '%s\n' "$result" > "$case_root/exit_code.txt"
    date -Is > "$case_root/finished.txt"
    echo "$(date -Is) END calibrated_${seconds}s exit=$result"
    if ((result!=0)); then queue_failed=1; fi
 done
printf '%s\n' "$queue_failed" > "$queue_root/exit_code.txt"
date -Is > "$queue_root/finished.txt"
exit "$queue_failed"
