#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${FRANKA_SOURCE:?Set FRANKA_SOURCE to the workstation Franka checkout for SDK dependencies}"
export LD_LIBRARY_PATH="$SOURCE/build/poco_install/lib:$SOURCE/build/franka_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BASE="${BASE_PROFILE:?Set BASE_PROFILE to the matching optimization controller profile}"
POLICY="${1:?trained policy export directory}"
OUT="${2:?new output profile directory}"
POLICY="$(cd "$POLICY" && pwd)"
CHECKPOINT="${3:-$(dirname "$POLICY")/policy.npz}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing policy checkpoint: $CHECKPOINT" >&2; exit 2; }
PYTHON="${FRANKA_PYTHON:-$SOURCE/.venv/bin/python}"
"$PYTHON" - "$BASE/spec.json" "$POLICY" "${SYNTHETIC_COMPILE_TEST:-0}" <<'PYCODE'
import json,sys
from pathlib import Path
base=json.loads(Path(sys.argv[1]).read_text());policy=Path(sys.argv[2]);meta=json.loads((policy/'policy_manifest.json').read_text())
if sys.argv[3]!='1':
    # Parametric export is an implementation detail; every physical/controller
    # setting must still exactly match the deployed profile.
    semantic=lambda spec:{k:v for k,v in spec.items() if k!='runtime_parameters'}
    if not isinstance(meta.get('mpc_spec'),dict) or semantic(meta['mpc_spec'])!=semantic(base):raise SystemExit('Policy and hardware MPC specifications differ')
    if meta.get('hidden_layers')!=[256,256,256]:raise SystemExit('Expected three hidden layers of 256 neurons')
    # Simulation performance is a research result, not a build/run prerequisite.
    # The actual compiled policy mapping is exercised below without robot access.
PYCODE
[[ ! -e "$OUT" ]] || { echo 'Use a new output profile directory.' >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cp -aL "$BASE/factors" "$OUT/factors"
cp -a "$POLICY" "$OUT/policy"
cp "$BASE/spec.json" "$OUT/spec.json"
cp "$BASE/juggling_home.json" "$OUT/juggling_home.json"
mkdir -p "$OUT/config"
cp "$SOURCE/config/paddle_hand_eye.json" "$SOURCE/config/vicon_timing.json" "$OUT/config/"
CMAKE="${CMAKE:-$SOURCE/.venv/bin/cmake}"
PYTHON="${FRANKA_PYTHON:-$SOURCE/.venv/bin/python}"
CASADI="$($PYTHON -c 'import casadi,pathlib; print(pathlib.Path(casadi.__file__).parent/"cmake")')"
EIGEN="${EIGEN_DIR:-$SOURCE/build/eigen_install/include/eigen3}"
BOOST="${BOOST_INCLUDE_DIR:-$($PYTHON -c 'import importlib.metadata as m; print(m.distribution("cmeel-boost").locate_file("cmeel.prefix/include"))')}"
OSQP="$SOURCE/build/osqp_install/lib/cmake/osqp"
"$CMAKE" -S "$HERE/hardware" -B "$OUT/build/controller" -DFACTOR_DIR="$OUT/factors" -DPOLICY_DIR="$OUT/policy" -DEIGEN_DIR="$EIGEN" -Dcasadi_DIR="$CASADI" -Dosqp_DIR="$OSQP"
"$CMAKE" --build "$OUT/build/controller" -j "${BUILD_JOBS:-8}"
"$CMAKE" -S "$HERE/reference/cpp/experiment" -B "$OUT/build/runtime" -DFACTOR_DIR="$OUT/factors" -DCONTROLLER_DIR="$OUT/build/controller" -DEIGEN_DIR="$EIGEN" -DBOOST_INCLUDE_DIR="$BOOST" -Dcasadi_DIR="$CASADI" -Dosqp_DIR="$OSQP"
"$CMAKE" --build "$OUT/build/runtime" --target juggling_ball_cpp -j "${BUILD_JOBS:-8}"
"$CMAKE" -S "$HERE/reference/cpp/session" -B "$OUT/build/session" -DFACTOR_DIR="$OUT/factors" -DFranka_DIR="$SOURCE/build/franka_install/lib/cmake/Franka" -DPoco_DIR="$SOURCE/build/poco_install/lib/cmake/Poco"
"$CMAKE" --build "$OUT/build/session" --target juggling_franka_session -j "${BUILD_JOBS:-8}"
"$CMAKE" -S "$HERE/reference/cpp/vicon" -B "$OUT/build/vicon" -DVICON_SDK_DIR="${VICON_SDK_DIR:-$SOURCE/resources/vicon-datastream-sdk}" -DCMAKE_PREFIX_PATH="${BOOST%/include}"
"$CMAKE" --build "$OUT/build/vicon" -j "${BUILD_JOBS:-8}"
"$OUT/build/runtime/juggling_ball_cpp" --describe > "$OUT/runtime_description.json"
"$OUT/build/session/juggling_franka_session" --describe > "$OUT/native_description.json"
"$PYTHON" "$HERE/check_hardware_policy.py" "$CHECKPOINT" "$OUT"
# The copied supervisor records this file in each run; it is provenance only.
(cd "$OUT" && sha256sum policy/ampc_policy.hpp factors/model_config.hpp spec.json > manifest.sha256)
# An explicit per-profile launcher makes selecting the two trained policies
# unambiguous. Creating this file never starts the hardware application.
"$PYTHON" - "$OUT" "$HERE/run_hardware.sh" "$SOURCE" <<'PYCODE'
import shlex,sys
from pathlib import Path
out=Path(sys.argv[1])
launcher='#!/usr/bin/env bash\nset -euo pipefail\n'
# Use the SDK and lock belonging to this build, even if the caller's shell
# still exports FRANKA_SOURCE for a different checkout.
launcher+='export FRANKA_SOURCE='+shlex.quote(sys.argv[3])+'\n'
launcher+='exec '+shlex.quote(sys.argv[2])+' '+shlex.quote(str(out))+' "$@"\n'
(out/'run.sh').write_text(launcher)
(out/'run.sh').chmod(0o755)
PYCODE
echo "$OUT"
