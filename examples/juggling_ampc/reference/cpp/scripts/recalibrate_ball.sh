#!/usr/bin/env bash
# Three physical trials, then offline fitting/building and atomic activation.
# Only C++ executables run while the robot experiment is active.
set -euo pipefail
recal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
recal_python="${JUGGLING_PYTHON:-$recal_root/.venv/bin/python}"
recal_defaults="$recal_root/config/cpp_active_profile"
recal_profile=""
recal_output=""
recal_seconds=120
recal_extra=()
while (($#)); do
  case "$1" in
    --profile-dir) recal_profile="${2:?Missing profile directory}"; shift 2 ;;
    --output) recal_output="${2:?Missing output directory}"; shift 2 ;;
    --seconds) recal_seconds="${2:?Missing trial duration}"; shift 2 ;;
    --) shift; recal_extra=("$@"); break ;;
    --help|-h)
      echo "Usage: bash cpp/scripts/recalibrate_ball.sh [--profile-dir DIR] [--output NEW_DIR] [--seconds 120] [-- extra run_ball_cpp.sh options]"
      echo "Runs exactly three manually released-ball attempts with the current model, fits gravity and paddle normal, prepares runtime tuning (or builds -j32 if structurally changed), and activates checked defaults."
      echo "Robot faults stop the sequence. Insufficient data or rejected fits preserve existing defaults."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! -t 0 ]]; then
  echo "Recalibration requires an interactive terminal to refixture the ball between attempts." >&2
  exit 2
fi
for recal_option in "${recal_extra[@]}"; do
  case "$recal_option" in
    --output|--profile-dir|--seconds|--describe|--help)
      echo "Set $recal_option before '--'; trial identity and model must stay fixed." >&2; exit 2 ;;
  esac
done
cd "$recal_root"
# This lock protects a multi-attempt calibration transaction. The runner and
# builder additionally share the project-wide experiment/compilation lock.
exec 8>"$recal_root/config/.cpp_recalibration.lock"
if ! flock -n 8; then
  echo "A recalibration workflow is already active." >&2; exit 2
fi
recal_expected=""
if [[ -f "$recal_defaults" ]]; then recal_expected="$(cat "$recal_defaults")"; fi
if [[ -z "$recal_profile" ]]; then recal_profile="$recal_expected"; fi
if [[ -z "$recal_profile" || ! -f "$recal_profile/factors/metadata.json" ]]; then
  echo "No built C++ profile. Run cpp/scripts/build_ball_cpp.sh first or pass --profile-dir." >&2
  exit 2
fi
recal_profile="$(cd "$recal_profile" && pwd)"
if [[ -z "$recal_output" ]]; then
  recal_output="$recal_root/runs/recalibration_cpp_$(date +%Y%m%d_%H%M%S_%N)"
fi
mkdir -p "$(dirname "$recal_output")"
mkdir "$recal_output"
recal_output="$(cd "$recal_output" && pwd)"
printf '%s\n' "$recal_profile" > "$recal_output/previous_profile.txt"
cp "$recal_profile/factors/metadata.json" "$recal_output/previous_model_metadata.json"
printf 'Recording three calibration attempts to: %s\n' "$recal_output"
recal_trials=()
for recal_attempt in 1 2 3; do
  recal_trial="$recal_output/attempt_0$recal_attempt"
  recal_trials+=("$recal_trial")
  printf '\nAttempt %s/3: refixture the ball, keep it held, and clear the robot workspace.\n' "$recal_attempt"
  read -r -p "Press Enter when ready. Wait for RELEASE BALL NOW before releasing. " recal_ready
  recal_status=0
  bash "$recal_root/cpp/scripts/run_ball_cpp.sh" \
    --profile-dir "$recal_profile" --output "$recal_trial" --seconds "$recal_seconds" \
    "${recal_extra[@]}" || recal_status=$?
  printf '%s\n' "$recal_status" > "$recal_output/attempt_0${recal_attempt}_exit_status.txt"
  # The C++ runner has exited and joined/stopped its robot process before this
  # offline check. Do not retry a reflex, communication fault or failed stop.
  "$recal_python" - "$recal_trial" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
try:
    native = json.loads((p/'robot/session.json').read_text())
except (OSError, ValueError) as error:
    raise SystemExit(f'CALIBRATION STOPPED: no complete native stop record: {error}')
if (not native.get('control_completed') or native.get('fault', 1) != 0
        or not native.get('log_ok') or native.get('error')):
    raise SystemExit('CALIBRATION STOPPED: native hardware fault/stop failure. Resolve it before starting a new batch. Defaults unchanged.')
if not (p/'release.json').is_file():
    raise SystemExit('CALIBRATION STOPPED: no ball release was recorded. Defaults unchanged.')
print('Attempt recorded; native robot stop completed.', flush=True)
PY
done
printf '\nThree attempts finished. Robot stopped; fitting gravity and paddle normal offline.\n'
"$recal_python" -u -m identification.recalibrate "${recal_trials[@]}" --output "$recal_output/fit"
recal_new_profile="$recal_root/build/cpp_profiles/recalibrated_$(date +%Y%m%d_%H%M%S_%N)"
bash "$recal_root/cpp/scripts/build_ball_cpp.sh" \
  --spec "$recal_output/fit/candidate_spec.json" --profile-dir "$recal_new_profile"
"$recal_python" -m identification.recalibrate activate \
  --profile-dir "$recal_new_profile" --fit-output "$recal_output/fit" \
  --defaults "$recal_defaults" --expected-active "$recal_expected"
printf '\nNew gravity and paddle normal are now the defaults for C++ juggling.\n'
printf 'Evidence and rollback profile: %s\n' "$recal_output"
printf 'Next experiment: bash %s/cpp/scripts/run_ball_cpp.sh\n' "$recal_root"
