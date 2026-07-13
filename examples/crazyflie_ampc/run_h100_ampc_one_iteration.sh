#!/bin/bash
### Job Parameters
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=2:00:00
#SBATCH --partition=c23g
#SBATCH --job-name=cfampc1
#SBATCH --output=hpclogs/%A.log

set -uo pipefail

if [[ -n "${ROOT:-}" ]]; then
  ROOT="$(cd "${ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/crazyflie_ampc" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_h100_ampc_one_iteration.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${ROOT}/examples/crazyflie_ampc"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV="${VENV:-}"
if [[ -n "${VENV}" ]]; then
  PYTHON="${PYTHON:-${VENV}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi
BASHRC="${BASHRC:-${HOME}/.bashrc}"
if [[ -f "${BASHRC}" ]]; then
  source "${BASHRC}"
fi
if [[ -n "${VENV}" && -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/crazyflie_ampc_h100_one_iter_${STAMP}}"
LOG_DIR="${RESULTS_DIR}/logs"

DTYPE="${DTYPE:-float32}"
DAGGER_ITERATIONS=1
DAGGER_MIXING="${DAGGER_MIXING:-0}"
COLLECT_BATCH_SIZE="${COLLECT_BATCH_SIZE:-100000}"
DATASET_KEEP_FRACTION="${DATASET_KEEP_FRACTION:-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
CONTROL_DT="${CONTROL_DT:-0.005}"
INTEGRATOR_SUBSTEPS="${INTEGRATOR_SUBSTEPS:-5}"
HORIZON_STEPS="${HORIZON_STEPS:-40}"
HIDDEN_SIZES="${HIDDEN_SIZES:-32,32,32}"
PREDICTION_TARGET="${PREDICTION_TARGET:-first_action}"
ACTIVATION="${ACTIVATION:-leaky_relu}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-100000}"
SQP_ITERATIONS="${SQP_ITERATIONS:-1}"
OSQP_MAX_ITER="${OSQP_MAX_ITER:-25}"
SEGMENT_BUDGET="${SEGMENT_BUDGET:-256}"
SEGMENT_STRATEGY="${SEGMENT_STRATEGY:-optimal}"
QDLDL_BACKEND="${QDLDL_BACKEND:-warp}"
QDLDL_FACTOR_BACKEND="${QDLDL_FACTOR_BACKEND:-warp}"
QDLDL_SOLVE_BACKEND="${QDLDL_SOLVE_BACKEND:-warp}"
LEVEL_SCHEDULED_SOLVE_THRESHOLD="${LEVEL_SCHEDULED_SOLVE_THRESHOLD:-1}"
LINE_SEARCH_STEP_MIN="${LINE_SEARCH_STEP_MIN:-0.1}"

CODE_EXPORT_BACKEND="${CODE_EXPORT_BACKEND:-simple}"
CODE_EXPORT_PRECISION="${CODE_EXPORT_PRECISION:-float32}"
CODE_EXPORT_QUANTIZE="${CODE_EXPORT_QUANTIZE:-none}"
CODE_EXPORT_TEST_COUNT="${CODE_EXPORT_TEST_COUNT:-64}"
CODE_EXPORT_SEED="${CODE_EXPORT_SEED:-2026}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${ROOT}/hpclogs"
cd "${ROOT}" || exit 1

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export JAX_COMPILATION_CACHE_DIR="${ROOT}/.jax_cache"

echo "Crazyflie AMPC H100 one-iteration run"
echo "Host: $(uname -n)"
echo "Root: ${ROOT}"
echo "Python: ${PYTHON}"
echo "Results: ${RESULTS_DIR}"
echo "DAgger iterations: ${DAGGER_ITERATIONS}"
echo "DAgger mixing: ${DAGGER_MIXING}"
echo "Collect batch size: ${COLLECT_BATCH_SIZE}"
echo "Dataset keep fraction: ${DATASET_KEEP_FRACTION}"
echo "Train epochs: ${TRAIN_EPOCHS}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "SQP iterations: ${SQP_ITERATIONS}"
echo "QDLDL backend: ${QDLDL_BACKEND}"
echo "QDLDL factor backend: ${QDLDL_FACTOR_BACKEND}"
echo "QDLDL solve backend: ${QDLDL_SOLVE_BACKEND}"
echo "QDLDL levelsolve threshold: ${LEVEL_SCHEDULED_SOLVE_THRESHOLD}"
echo "Prediction target: ${PREDICTION_TARGET}"
echo "Code export backend: ${CODE_EXPORT_BACKEND}"
echo "Code export quantization: ${CODE_EXPORT_QUANTIZE}"
echo "Started: $(date -Is)"

"${PYTHON}" examples/crazyflie_ampc/train_ampc.py \
  --output-dir "${RESULTS_DIR}" \
  --dtype "${DTYPE}" \
  --dagger-iterations "${DAGGER_ITERATIONS}" \
  --collect-batch-size "${COLLECT_BATCH_SIZE}" \
  --dataset-keep-fraction "${DATASET_KEEP_FRACTION}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --control-dt "${CONTROL_DT}" \
  --integrator-substeps "${INTEGRATOR_SUBSTEPS}" \
  --horizon-steps "${HORIZON_STEPS}" \
  --dagger-mixing "${DAGGER_MIXING}" \
  --hidden-sizes "${HIDDEN_SIZES}" \
  --prediction-target "${PREDICTION_TARGET}" \
  --activation "${ACTIVATION}" \
  --learning-rate "${LEARNING_RATE}" \
  --train-epochs "${TRAIN_EPOCHS}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --sqp-iterations "${SQP_ITERATIONS}" \
  --osqp-max-iter "${OSQP_MAX_ITER}" \
  --segment-budget "${SEGMENT_BUDGET}" \
  --segment-strategy "${SEGMENT_STRATEGY}" \
  --level-scheduled-solve \
  --level-scheduled-solve-threshold "${LEVEL_SCHEDULED_SOLVE_THRESHOLD}" \
  --qdldl-backend "${QDLDL_BACKEND}" \
  --qdldl-factor-backend "${QDLDL_FACTOR_BACKEND}" \
  --qdldl-solve-backend "${QDLDL_SOLVE_BACKEND}" \
  --line-search-step-min "${LINE_SEARCH_STEP_MIN}" \
  2>&1 | tee "${LOG_DIR}/train_ampc.log"

code="${PIPESTATUS[0]}"
export_elapsed_s=""
if [[ "${code}" -eq 0 ]]; then
  echo "Training finished successfully; exporting standalone controller code..."
  export_start_s="$(date +%s)"
  "${PYTHON}" examples/crazyflie_ampc/export_ampc_controller.py \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${RESULTS_DIR}/code_export" \
    --backend "${CODE_EXPORT_BACKEND}" \
    --precision "${CODE_EXPORT_PRECISION}" \
    --quantize "${CODE_EXPORT_QUANTIZE}" \
    --test-source initial_distribution \
    --test-count "${CODE_EXPORT_TEST_COUNT}" \
    --seed "${CODE_EXPORT_SEED}" \
    --generate-example-main \
    2>&1 | tee "${LOG_DIR}/export_ampc_controller.log"
  export_code="${PIPESTATUS[0]}"
  export_end_s="$(date +%s)"
  export_elapsed_s="$((export_end_s - export_start_s))"
  if [[ "${export_code}" -ne 0 ]]; then
    code="${export_code}"
  fi
else
  echo "Skipping code export because training failed with exit=${code}"
fi

"${PYTHON}" - "${RESULTS_DIR}" "${export_elapsed_s}" <<'PY'
from __future__ import annotations

import json
import math
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
export_elapsed_arg = sys.argv[2]
summary_path = results_dir / "summary.json"
print("")
print("One-iteration timing summary")
if not summary_path.exists():
    print(f"  summary: missing ({summary_path})")
    raise SystemExit(0)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
history = summary.get("history") or []
timing_totals = summary.get("timing_totals") or {}
first_iter = history[0] if history else {}
timing = first_iter.get("timing") or {}
collection = first_iter.get("collection") or {}
training = first_iter.get("training") or {}


def pick(*values):
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def fmt(value):
    return "n/a" if value is None else f"{value:.2f}s"


compile_s = pick(timing.get("warmup_compilations_s"), timing_totals.get("warmup_compilations_s"))
dataset_s = pick(
    timing.get("dataset_generation_iter_s"),
    collection.get("dataset_generation_elapsed_s"),
    timing_totals.get("dataset_generation_s"),
)
sqp_collect_s = pick(timing.get("sqp_collection_iter_s"), collection.get("elapsed_s"))
training_s = pick(timing.get("training_iter_s"), training.get("elapsed_s"), timing_totals.get("training_s"))
wall_s = pick(timing.get("wall_elapsed_s"), timing_totals.get("wall_elapsed_s"))
export_s = float(export_elapsed_arg) if export_elapsed_arg else None

print(f"  compile/warmup:      {fmt(compile_s)}")
print(f"  dataset generation:  {fmt(dataset_s)}")
print(f"  sqp collection only: {fmt(sqp_collect_s)}")
print(f"  model training:      {fmt(training_s)}")
print(f"  code export:         {fmt(export_s)}")
print(f"  train wall elapsed:  {fmt(wall_s)}")
print(f"  summary:             {summary_path}")
PY

echo "Finished: $(date -Is), exit=${code}"
exit "${code}"
