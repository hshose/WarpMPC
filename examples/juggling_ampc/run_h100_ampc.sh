#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=juggling-ampc
# Submit with submit_h100_ampc.py to stage a private source tree and log path.
set -euo pipefail
study_root="${1:?Private case directory prepared by submit_h100_ampc.py}"
exec bash "${study_root}/source/examples/juggling_ampc/run_claix23_instant.sh" "$@"
