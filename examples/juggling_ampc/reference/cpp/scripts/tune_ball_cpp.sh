#!/usr/bin/env bash
# Prepare, check, and activate numerical tuning. Never starts robot motion.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
spec="$root/config/cpp_ball_spec.json"
if (($#)); then
  [[ $# == 2 && $1 == --spec ]] || { echo 'Usage: tune_ball_cpp.sh [--spec FILE]'; exit 2; }
  spec="$2"
fi
IFS= read -r previous < "$root/config/cpp_active_profile"
profile="$root/build/cpp_profiles/tuned_$(date +%Y%m%d_%H%M%S_%N)"
bash "$root/cpp/scripts/build_ball_cpp.sh" --spec "$spec" --profile-dir "$profile"
# Reacquire between preparation and activation; reject concurrent replacement.
exec 9>"$root/build/cpp_experiment.lock"
flock -n 9 || { echo 'An experiment/build is active; prepared profile was not activated.'; exit 2; }
"$root/.venv/bin/python" - "$profile" "$previous" <<'PY'
import json,os,subprocess,sys,tempfile
from pathlib import Path
profile=Path(sys.argv[1]);previous=sys.argv[2];pointer=Path('config/cpp_active_profile')
assert pointer.read_text().strip()==previous,'Active profile changed; prepared profile was not activated'
subprocess.run(['sha256sum','--quiet','-c','manifest.sha256'],cwd=profile,check=True)
record=dict(profile=str(profile),previous_profile=previous,physical_robot_used=False,
            reused_binaries=(profile/'runtime_reuse.json').exists())
(profile/'activation.json').write_text(json.dumps(record,indent=2)+'\n')
fd,path=tempfile.mkstemp(prefix='.cpp_active.',dir=pointer.parent)
with os.fdopen(fd,'w') as f:f.write(str(profile)+'\n');f.flush();os.fsync(f.fileno())
os.replace(path,pointer)
print('Tuning checked and active: '+str(profile))
PY
