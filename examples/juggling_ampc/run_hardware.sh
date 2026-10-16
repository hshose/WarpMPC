#!/usr/bin/env bash
set -euo pipefail
PROFILE="${1:?compiled AMPC profile directory}"
shift
PROFILE="$(cd "$PROFILE" && pwd)"
# Research policies may be tested even when their simulated uptime trails MPC.
SOURCE="${FRANKA_SOURCE:?Set FRANKA_SOURCE to the workstation Franka checkout}"
export LD_LIBRARY_PATH="$SOURCE/build/poco_install/lib:$SOURCE/build/franka_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>"$SOURCE/build/cpp_experiment.lock"
flock -n 9 || { echo 'Another experiment or build holds the robot project lock.' >&2; exit 2; }
export JUGGLING_RUNTIME_VALUES="$PROFILE/factors/runtime_values.txt"
export OPENBLAS_NUM_THREADS=1 JUGGLING_EVAL_THREADS=4
OUTPUT="${OUTPUT:-$PROFILE/runs/ampc_$(date +%Y%m%d_%H%M%S_%N)}"
mkdir -p "$(dirname "$OUTPUT")"
exec "$PROFILE/build/runtime/juggling_ball_cpp" --execute \
 --factors "$PROFILE/factors" --robot-binary "$PROFILE/build/session/juggling_franka_session" \
 --vicon-binary "$PROFILE/build/vicon/juggling_vicon" \
 --calibration "$PROFILE/config/paddle_hand_eye.json" --timing-calibration "$PROFILE/config/vicon_timing.json" \
 --home-config "$PROFILE/juggling_home.json" --profile-dir "$PROFILE" --output "$OUTPUT" "$@"
