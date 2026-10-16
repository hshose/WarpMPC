#!/usr/bin/env bash
# Optional user-directory build of the Panda-era SDK pinned by the supplied repo.
# No robot is connected, and no system packages or configuration are changed.
set -euo pipefail
cpp_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$cpp_project_root"
cpp_cmake="${CPP_CMAKE:-$cpp_project_root/.venv/bin/cmake}"
cpp_cc="${CC:-clang}"
cpp_cxx="${CXX:-clang++}"
cpp_jobs="${CPP_BUILD_JOBS:-32}"
if [ ! -d build/poco_src/.git ]; then
  git clone --depth 1 --branch poco-1.12.5-release https://github.com/pocoproject/poco.git build/poco_src
fi
"$cpp_cmake" -S build/poco_src -B build/poco \
  -DCMAKE_C_COMPILER="$cpp_cc" -DCMAKE_CXX_COMPILER="$cpp_cxx" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$cpp_project_root/build/poco_install" \
  -DENABLE_XML=OFF -DENABLE_JSON=OFF -DENABLE_UTIL=OFF -DENABLE_NETSSL=OFF \
  -DENABLE_CRYPTO=OFF -DENABLE_JWT=OFF -DENABLE_ENCODINGS=OFF -DENABLE_MONGODB=OFF \
  -DENABLE_DATA=OFF -DENABLE_DATA_SQLITE=OFF -DENABLE_DATA_MYSQL=OFF \
  -DENABLE_DATA_POSTGRESQL=OFF -DENABLE_DATA_ODBC=OFF -DENABLE_REDIS=OFF \
  -DENABLE_PROMETHEUS=OFF -DENABLE_ZIP=OFF -DENABLE_PAGECOMPILER=OFF \
  -DENABLE_PAGECOMPILER_FILE2PAGE=OFF -DENABLE_ACTIVERECORD=OFF \
  -DENABLE_ACTIVERECORD_COMPILER=OFF -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF
"$cpp_cmake" --build build/poco -j "$cpp_jobs"
"$cpp_cmake" --install build/poco
"$cpp_cmake" -S resources/franka_ampc/ext/eigen -B build/eigen \
  -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX="$cpp_project_root/build/eigen_install"
"$cpp_cmake" --install build/eigen
if [ ! -f build/libfranka_092/CMakeLists.txt ]; then
  mkdir -p build/libfranka_092
  git -C resources/libfranka archive 0.9.2 | tar -x -C build/libfranka_092
fi
if [ ! -d build/libfranka_092/common/.git ]; then
  git clone https://github.com/frankaemika/libfranka-common.git build/libfranka_092/common
  git -C build/libfranka_092/common checkout e6aa0fc210d93fe618bfd8956829a264d5476ba8
fi
"$cpp_cmake" -S build/libfranka_092 -B build/franka_092 \
  -DCMAKE_CXX_COMPILER="$cpp_cxx" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPoco_DIR="$cpp_project_root/build/poco_install/lib/cmake/Poco" \
  -DEigen3_DIR="$cpp_project_root/build/eigen_install/share/eigen3/cmake" \
  -DCMAKE_INSTALL_PREFIX="$cpp_project_root/build/franka_install"
"$cpp_cmake" --build build/franka_092 -j "$cpp_jobs"
"$cpp_cmake" --install build/franka_092
"$cpp_cmake" -S cpp -B build/controller -DJUGGLING_BUILD_FRANKA=ON \
  -DFranka_DIR="$cpp_project_root/build/franka_install/lib/cmake/Franka" \
  -DPoco_DIR="$cpp_project_root/build/poco_install/lib/cmake/Poco" \
  -DEigen3_DIR="$cpp_project_root/build/eigen_install/share/eigen3/cmake"
"$cpp_cmake" --build build/controller -j "$cpp_jobs"
# An invocation with no arguments is a verified non-connecting usage check.
build/controller/juggling_franka
