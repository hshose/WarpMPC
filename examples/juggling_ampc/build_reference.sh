#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${FRANKA_SOURCE:?Set FRANKA_SOURCE to the native controller SDK checkout}"
OUT="${1:?output directory}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
PYTHON="${FRANKA_PYTHON:-$SOURCE/.venv/bin/python}"
cd "$HERE/reference"
OPENBLAS_NUM_THREADS=1 "$PYTHON" -m optimization.native_export config/cpp_ball_spec.json "$OUT/factors"
"${CMAKE:-$SOURCE/.venv/bin/cmake}" -S optimization/native -B "$OUT/controller" \
 -DFACTOR_DIR="$OUT/factors" \
 -Dcasadi_DIR="$($PYTHON -c 'import casadi,pathlib; print(pathlib.Path(casadi.__file__).parent/"cmake")')" \
 -Dosqp_DIR="$SOURCE/build/osqp_install/lib/cmake/osqp" \
 -DEIGEN_DIR="${EIGEN_DIR:-$SOURCE/resources/franka_ampc/ext/eigen}"
"${CMAKE:-$SOURCE/.venv/bin/cmake}" --build "$OUT/controller" -j "${BUILD_JOBS:-8}"
