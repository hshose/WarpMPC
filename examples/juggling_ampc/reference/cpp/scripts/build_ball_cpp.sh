#!/usr/bin/env bash
# Compilation only. Never connects to the robot or Vicon.
set -euo pipefail
cpp_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$cpp_project"
unset JUGGLING_RUNTIME_VALUES
cpp_profile=""; cpp_spec="$cpp_project/config/cpp_ball_spec.json"; cpp_reuse=""
while (($#)); do
  case "$1" in
    --profile-dir) cpp_profile="$2"; shift 2;;
    --spec) cpp_spec="$2"; shift 2;;
    --reuse-factors) cpp_reuse="$2"; shift 2;;
    *) echo "Unknown build option: $1" >&2; exit 2;;
  esac
done
[[ -n "$cpp_profile" ]] || { echo '--profile-dir is required' >&2; exit 2; }
mkdir -p "$cpp_project/build"
exec 9>"$cpp_project/build/cpp_experiment.lock"
flock -n 9 || { echo 'Another compilation or experiment is active.' >&2; exit 2; }
mkdir -p "$cpp_profile"
cpp_profile="$(cd "$cpp_profile" && pwd)"
[[ ! -e "$cpp_profile/manifest.sha256" ]] || { echo 'Use a new profile directory; complete profiles are immutable.' >&2; exit 2; }
if [[ -x "$cpp_project/toolchain/usr/bin/clang++-21" ]]; then
  export CC="$cpp_project/toolchain/usr/bin/clang-21" CXX="$cpp_project/toolchain/usr/bin/clang++-21"
else
  export CC="${CC:-clang}" CXX="${CXX:-clang++}"
fi
export OPENBLAS_NUM_THREADS=1 JUGGLING_EVAL_THREADS=4
cp "$cpp_spec" "$cpp_profile/spec.json"
cpp_home_args=()
if [[ -f "$cpp_project/config/cpp_juggling_home.json" ]]; then
  cp "$cpp_project/config/cpp_juggling_home.json" "$cpp_profile/juggling_home.json"
  cpp_home_args=(--home-config "$cpp_profile/juggling_home.json")
fi
if [[ -n "$cpp_reuse" ]]; then
  [[ -d "$cpp_profile/factors" ]] || cp -a "$cpp_reuse" "$cpp_profile/factors"
  "$cpp_project/.venv/bin/python" - "$cpp_profile" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert json.loads((p/'spec.json').read_text()) == json.loads((p/'factors/metadata.json').read_text())['spec'], 'Reuse factors/spec mismatch'
PY
else
  "$cpp_project/.venv/bin/python" -m optimization.native_export "$cpp_profile/spec.json" "$cpp_profile/factors"
fi
# Numerical tuning of a compatible parametric profile requires no compiler.
if [[ -f "$cpp_project/config/cpp_active_profile" ]]; then
  IFS= read -r cpp_previous < "$cpp_project/config/cpp_active_profile"
  if "$cpp_project/.venv/bin/python" -m optimization.runtime_profile "$cpp_profile" --source "$cpp_previous"; then
    echo "Prepared and checked runtime tuning (no compilation): $cpp_profile"
    exit 0
  else
    cpp_reuse_status=$?
    [[ "$cpp_reuse_status" == 20 ]] || exit "$cpp_reuse_status"
  fi
fi
"$cpp_project/.venv/bin/python" -m optimization.build_native "$cpp_profile/factors" "$cpp_profile/build/controller" --jobs 32
cpp_cmake="$cpp_project/.venv/bin/cmake"
"$cpp_cmake" -S cpp/session -B "$cpp_profile/build/session" -DCMAKE_CXX_COMPILER="$CXX" \
  -DFACTOR_DIR="$cpp_profile/factors" -DFranka_DIR="$cpp_project/build/franka_install/lib/cmake/Franka" \
  -DPoco_DIR="$cpp_project/build/poco_install/lib/cmake/Poco" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"$cpp_cmake" --build "$cpp_profile/build/session" -j32
cpp_sdk="${VICON_SDK_DIR:-$cpp_project/resources/vicon-datastream-sdk}"
cpp_boost="$(find "$cpp_project/.venv/lib" -path '*/include/boost/property_tree/json_parser.hpp' -print -quit)"
[[ -n "$cpp_boost" ]] && cpp_boost="${cpp_boost%/boost/property_tree/json_parser.hpp}"
"$cpp_cmake" -S cpp/vicon -B "$cpp_profile/build/vicon" -DCMAKE_CXX_COMPILER="$CXX" \
  -DVICON_SDK_DIR="$cpp_sdk" -DCMAKE_PREFIX_PATH="${cpp_boost%/include}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"$cpp_cmake" --build "$cpp_profile/build/vicon" -j32
cpp_casadi="$(find "$cpp_project/.venv/lib" -path '*/casadi/cmake/casadi-config.cmake' -print -quit)"
[[ -n "$cpp_casadi" ]] || cpp_casadi="$(find "$cpp_project/.venv/lib" -path '*/casadi/cmake/casadiConfig.cmake' -print -quit)"
cpp_boost="$(find "$cpp_project/.venv/lib" -path '*/include/boost/property_tree/json_parser.hpp' -print -quit)"
[[ -n "$cpp_boost" ]] && cpp_boost="${cpp_boost%/boost/property_tree/json_parser.hpp}"
"$cpp_cmake" -S cpp/experiment -B "$cpp_profile/build/runtime" -DCMAKE_CXX_COMPILER="$CXX" \
  -DFACTOR_DIR="$cpp_profile/factors" -DCONTROLLER_DIR="$cpp_profile/build/controller" \
  -Dcasadi_DIR="$(dirname "$cpp_casadi")" -Dosqp_DIR="$cpp_project/build/osqp_install/lib/cmake/osqp" \
  -DBOOST_INCLUDE_DIR="$cpp_boost" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"$cpp_cmake" --build "$cpp_profile/build/runtime" -j32
if [[ -f "$cpp_profile/factors/runtime_values.txt" ]]; then
  export JUGGLING_RUNTIME_VALUES="$cpp_profile/factors/runtime_values.txt"
fi
"$cpp_profile/build/controller/juggling_session_check" > "$cpp_profile/session_checks.json"
"$cpp_profile/build/runtime/juggling_ball_estimator_check" > "$cpp_profile/estimator_checks.json"
"$cpp_profile/build/runtime/juggling_ball_cpp" --describe > "$cpp_profile/runtime_description.json"
"$cpp_profile/build/session/juggling_franka_session" --describe > "$cpp_profile/native_description.json"
"$cpp_profile/build/runtime/juggling_ball_cpp" --self-check --factors "$cpp_profile/factors" "${cpp_home_args[@]}" > "$cpp_profile/runtime_checks.json"
"$cpp_project/.venv/bin/python" -m optimization.runtime_profile "$cpp_profile" --stamp
(cd "$cpp_profile"; find factors build -type f \( -name '*.so' -o -name 'model_config.hpp' -o -name 'metadata.json' -o -name 'runtime_values.txt' -o -name 'juggling_ball_cpp' -o -name 'juggling_franka_session' -o -name 'juggling_vicon' \) -print0 | sort -z | xargs -0 sha256sum; sha256sum spec.json runtime_source.sha256; if [[ -f juggling_home.json ]]; then sha256sum juggling_home.json; fi) > "$cpp_profile/manifest.sha256"
echo "Built and checked C++ profile: $cpp_profile"
