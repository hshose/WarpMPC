#!/usr/bin/env bash
# Build only; never starts Vicon/FCI. Uses existing optimized factors/libfranka.
set -euo pipefail
tracking_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$tracking_root"
tracking_cmake="$tracking_root/.venv/bin/cmake"
tracking_factors="${FACTOR_DIR:-$tracking_root/build/variants/mesh24_prune_cad_mount_m90/factors}"
tracking_sdk="${VICON_SDK_DIR:-$tracking_root/resources/vicon-datastream-sdk}"
if [ ! -f "$tracking_sdk/CMakeLists.txt" ]; then
  tracking_sdk="${VICON_SDK_DIR:-$tracking_root/../acoustic/crazyflie_experiments/ext/libmotioncapture/deps/vicon-datastream-sdk}"
fi
if [ ! -f "$tracking_sdk/CMakeLists.txt" ]; then
  echo "Set VICON_SDK_DIR to the DataStream SDK sources from libmotioncapture/deps." >&2
  exit 2
fi
"$tracking_cmake" -S cpp/vicon -B build/vicon -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DVICON_SDK_DIR="$tracking_sdk" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"$tracking_cmake" --build build/vicon -j32
"$tracking_cmake" -S cpp/session -B build/session -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DFACTOR_DIR="$tracking_factors" \
  -DFranka_DIR="$tracking_root/build/franka_install/lib/cmake/Franka" \
  -DPoco_DIR="$tracking_root/build/poco_install/lib/cmake/Poco" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"$tracking_cmake" --build build/session -j32
"$tracking_cmake" --build "$(dirname "$tracking_factors")/controller" -j32
build/session/juggling_franka_session --describe
