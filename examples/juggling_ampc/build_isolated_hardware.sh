#!/usr/bin/env bash
# Rebuild both exported policies with a private SDK, source tree, and lock.
# This script never starts the robot.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${1:?Pass the existing trained-policy bundle}"
SDK_SOURCE="${FRANKA_SOURCE:?Set FRANKA_SOURCE to the SDK checkout to copy}"
if [[ -n "${2:-}" ]]; then
 OUT="$2"
 [[ ! -e "$OUT" ]] || { echo 'Use a new isolated bundle directory.' >&2; exit 2; }
 mkdir -p "$OUT"
else
 OUT="$(mktemp -d /tmp/juggling-ampc-hardware.XXXXXX)"
fi
OUT="$(cd "$OUT" && pwd)"
printf 'Isolated hardware bundle: %s\n' "$OUT"
mkdir -p "$OUT/sdk/build" "$OUT/sdk/resources" "$OUT/sdk/config" "$OUT/profiles"
cp -a --reflink=auto "$BUNDLE/source" "$OUT/source"
cp -a --reflink=auto "$BUNDLE/profiles" "$OUT/seed_profiles"
cp -a --reflink=auto "$SDK_SOURCE/.venv" "$OUT/sdk/.venv"
for DEPENDENCY in poco_install franka_install osqp_install eigen_install; do
 cp -aL --reflink=auto "$SDK_SOURCE/build/$DEPENDENCY" "$OUT/sdk/build/"
done
cp -aL --reflink=auto "$SDK_SOURCE/resources/vicon-datastream-sdk" "$OUT/sdk/resources/"
# Preserve the captured controller sources, with the current build/launch helpers.
cp "$HERE/build_hardware.sh" "$HERE/run_hardware.sh" "$OUT/source/examples/juggling_ampc/"
export FRANKA_SOURCE="$OUT/sdk"
export FRANKA_PYTHON="$FRANKA_SOURCE/.venv/bin/python"
export CMAKE="$($FRANKA_PYTHON -c 'import cmake,pathlib; print(pathlib.Path(cmake.__file__).parent/"data/bin/cmake")')"
export EIGEN_DIR="$FRANKA_SOURCE/build/eigen_install/include/eigen3"
export BOOST_INCLUDE_DIR="$($FRANKA_PYTHON -c 'import importlib.metadata as m; print(m.distribution("cmeel-boost").locate_file("cmeel.prefix/include"))')"
export VICON_SDK_DIR="$FRANKA_SOURCE/resources/vicon-datastream-sdk"
export OPENBLAS_NUM_THREADS=1
for POLICY_NAME in arm15_20s comparison_20s; do
 export BASE_PROFILE="$OUT/seed_profiles/$POLICY_NAME"
 cp -a "$BASE_PROFILE/config/." "$FRANKA_SOURCE/config/"
 bash "$OUT/source/examples/juggling_ampc/build_hardware.sh" \
  "$OUT/source/results/trained_policies/$POLICY_NAME/policy/eigen" \
  "$OUT/profiles/$POLICY_NAME" > "$OUT/build_$POLICY_NAME.log" 2>&1
 printf 'Built %s\n' "$OUT/profiles/$POLICY_NAME/run.sh"
done
printf 'Run one policy from %s/profiles\n' "$OUT"
