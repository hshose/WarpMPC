#!/usr/bin/env bash
# Run on CLAIX23 after uploading a source archive. Each duration gets its own
# source tree, empirical model, Newton source, caches, logs, and artifacts.
set -euo pipefail
ARCHIVE="${1:?Pass the fresh source tar.gz}"
DEPENDENCIES="${2:?Pass the installed dependency directory}"
SWEEP="${3:-${PWD}/results/juggling_fast_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SWEEP"
for DURATION in 5 10 20; do
 JOB_ROOT="$SWEEP/${DURATION}s"
 SOURCE_ROOT="$JOB_ROOT/source"
 mkdir -p "$SOURCE_ROOT" "$JOB_ROOT/artifacts"
 tar -xzf "$ARCHIVE" -C "$SOURCE_ROOT"
 cp -a "$DEPENDENCIES/live_source" "$SOURCE_ROOT/live_source"
 cp -a "$DEPENDENCIES/external" "$SOURCE_ROOT/external"
 mkdir -p "$SOURCE_ROOT/results/juggling_reference"
 cp -a "$DEPENDENCIES/results/juggling_reference/gpu_factors" "$SOURCE_ROOT/results/juggling_reference/gpu_factors"
 # Installed libraries are shared; every editable source tree is private.
 ln -s "$DEPENDENCIES/venv" "$SOURCE_ROOT/venv"
 ln -s "$JOB_ROOT/artifacts" "$SOURCE_ROOT/artifacts"
 if [[ -d "${WARP_CACHE_SEED:-}" ]]; then cp -a "$WARP_CACHE_SEED" "$SOURCE_ROOT/.warp_cache"; fi
 JOB_ID=$(sbatch --parsable --chdir="$SOURCE_ROOT" \
  --job-name="juggle_fast_${DURATION}s" --output="$JOB_ROOT/artifacts/slurm-%j.log" \
  --export="ALL,ROOT=$SOURCE_ROOT,SECONDS_SIM=$DURATION" \
  "$SOURCE_ROOT/examples/juggling_ampc/run_claix23_fast.sh")
 printf '%s\t%s\t%s\n' "$JOB_ID" "$DURATION" "$JOB_ROOT" | tee -a "$SWEEP/submitted.tsv"
done
