#!/usr/bin/env bash
set -euo pipefail
cpp_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$cpp_project_root"
cpp_python="${CPP_PYTHON:-$cpp_project_root/.venv/bin/python}"
cpp_cmake="${CPP_CMAKE:-$cpp_project_root/.venv/bin/cmake}"
cpp_cc="${CC:-clang}"
cpp_cxx="${CXX:-clang++}"
cpp_jobs="${CPP_BUILD_JOBS:-32}"
cpp_casadi_dir="$($cpp_python -c 'import casadi,pathlib; print(pathlib.Path(casadi.__file__).parent/"cmake")')"
OPENBLAS_NUM_THREADS=1 "$cpp_python" -m cpp.export
"$cpp_cmake" -S resources/OSQP -B build/osqp_native \
  -DCMAKE_C_COMPILER="$cpp_cc" -DCMAKE_CXX_COMPILER="$cpp_cxx" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG -march=native' \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DOSQP_USE_LONG=OFF \
  -DOSQP_BUILD_SHARED_LIB=OFF -DOSQP_BUILD_STATIC_LIB=ON -DOSQP_BUILD_DEMO_EXE=OFF \
  -DOSQP_ENABLE_DERIVATIVES=OFF -DOSQP_CODEGEN=OFF -DOSQP_ENABLE_PRINTING=OFF \
  -DOSQP_ENABLE_INTERRUPT=OFF -DCMAKE_INSTALL_PREFIX="$cpp_project_root/build/osqp_install"
"$cpp_cmake" --build build/osqp_native -j "$cpp_jobs"
"$cpp_cmake" --install build/osqp_native
"$cpp_cmake" -S cpp -B build/controller \
  -DCMAKE_C_COMPILER="$cpp_cc" -DCMAKE_CXX_COMPILER="$cpp_cxx" \
  -Dcasadi_DIR="$cpp_casadi_dir" -Dosqp_DIR="$cpp_project_root/build/osqp_install/lib/cmake/osqp" \
  -DJUGGLING_BUILD_FACTORS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"$cpp_cmake" --build build/controller -j "$cpp_jobs"
