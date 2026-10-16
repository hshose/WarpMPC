#!/usr/bin/env bash
# The running hardware experiment contains C++ processes only; no Python.
set -euo pipefail
cpp_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cpp_profile=""; cpp_output=""; cpp_args=()
while (($#)); do
  case "$1" in
    --profile-dir) cpp_profile="$2"; shift 2;;
    --output) cpp_output="$2"; shift 2;;
    *) cpp_args+=("$1"); shift;;
  esac
done
if [[ -z "$cpp_profile" ]]; then
  [[ -f "$cpp_project/config/cpp_active_profile" ]] || { echo 'Build and activate a C++ profile first.' >&2; exit 2; }
  IFS= read -r cpp_profile < "$cpp_project/config/cpp_active_profile"
fi
cpp_profile="$(cd "$cpp_profile" && pwd)"
[[ -n "$cpp_output" ]] || cpp_output="$cpp_project/runs/real_ball_cpp_$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$cpp_project/build" "$(dirname "$cpp_output")"
exec 9>"$cpp_project/build/cpp_experiment.lock"
flock -n 9 || { echo 'Another compilation or experiment is active.' >&2; exit 2; }
(cd "$cpp_profile"; sha256sum --quiet -c manifest.sha256)
cpp_home_args=()
[[ ! -f "$cpp_profile/juggling_home.json" ]] || cpp_home_args=(--home-config "$cpp_profile/juggling_home.json")
unset JUGGLING_RUNTIME_VALUES
if [[ -f "$cpp_profile/factors/runtime_values.txt" ]]; then
  export JUGGLING_RUNTIME_VALUES="$cpp_profile/factors/runtime_values.txt"
fi
export OPENBLAS_NUM_THREADS=1 JUGGLING_EVAL_THREADS=4
echo "Recording this C++ trial to: $cpp_output"
exec "$cpp_profile/build/runtime/juggling_ball_cpp" --execute \
  --factors "$cpp_profile/factors" --robot-binary "$cpp_profile/build/session/juggling_franka_session" \
  --vicon-binary "$cpp_profile/build/vicon/juggling_vicon" \
  --calibration "$cpp_project/config/paddle_hand_eye.json" --timing-calibration "$cpp_project/config/vicon_timing.json" \
  --profile-dir "$cpp_profile" --output "$cpp_output" "${cpp_home_args[@]}" "${cpp_args[@]}"
