#!/usr/bin/env python3
"""Batch-size sweeps for the MPPI/CEM nonlinear MPC baselines."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_sampling_mpc.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.nonlinear_mpc.sampling_mpc import (
    BATCH_SIZES,
    CHUNK_SIZES,
    REFERENCE_SQP_ITERATIONS,
    SIM_STEPS,
    default_settings,
)
from benchmarks.vram_cases import record_vram_case


PREFIXES = {
    "cartpole": "cartpole_quadratic_mpc",
    "crazyflie": "crazyflie_sqp_rollout",
}
FIELDNAMES = [
    "system",
    "method",
    "qp_solver",
    "mode",
    "status",
    "batch_size",
    "dtype",
    "seed",
    "chunk_size",
    "padded_instances",
    "horizon_steps",
    "sim_steps",
    "rollout_steps",
    "control_dt",
    "elapsed_s",
    "solve_s",
    "total_closed_loop_steps",
    "closed_loop_steps_per_s",
    "compile_s",
    "warmup_s",
    "finite_rate",
    "closed_loop_cost_mean",
    "final_state_cost_mean",
    "final_position_rms",
    "final_state_rms",
    "return_mean",
    "return_std",
    "return_p10",
    "return_median",
    "return_p90",
    "rollout_success_rate",
    "rail_violation_rate",
    "rollout_rail_violation_mean",
    "rollout_rail_violation_max",
    "max_action_violation",
    "samples",
    "iterations",
    "initial_iterations",
    "warm_sigma",
    "first_action_candidate_rollouts",
    "later_action_candidate_rollouts",
    "knots",
    "sigma",
    "temperature",
    "elite_fraction",
    "smoothing",
    "min_std",
    "penalty",
    "noise_decay",
    "noise_correlation",
    "control_scale",
    "feedback",
    "shift",
    "device_kind",
    "hostname",
    "slurm_job_id",
    "jax_version",
    "source_hash",
    "sqp_equivalent_iterations_per_s",
    "reference_sqp_iterations",
    "sampling_protocol",
    "repeat_times_s",
    "summary_json",
    "error",
    "wall_s",
    "log_path",
]


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parse_ints(text: str) -> list[int]:
    values = [_positive_int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("batch sizes must be nonempty and unique")
    return values


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


@record_vram_case
def run_batch(args: argparse.Namespace, *, batch_size: int) -> dict[str, object]:
    prefix = f"{PREFIXES[args.system]}_{args.method}_float32_batch_{batch_size}"
    summary_json = args.output_dir / f"{prefix}.json"
    log_path = args.output_dir / f"{prefix}.log"
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--system", args.system,
        "--method", args.method,
        "--batch-size", str(batch_size),
        "--seed", str(args.seed),
        "--chunk-size", str(args.chunk_size),
        "--repeats", str(args.repeats),
        "--sim-steps", str(args.sim_steps),
        "--reference-sqp-iterations", str(args.reference_sqp_iterations),
        "--summary-json", str(summary_json),
        "--require-gpu",
    ]
    if args.require_h100:
        cmd.append("--require-h100")
    if args.resume:
        cmd.append("--resume")
    previous_summary = summary_json.read_bytes() if summary_json.exists() else None
    print(f"Running solver={args.method} dtype=float32 batch={batch_size}: {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_s = time.perf_counter() - start
    log_path.write_text(proc.stdout)
    row: dict[str, object] = {
        "system": args.system,
        "method": args.method,
        "qp_solver": args.method,
        "mode": args.method.upper(),
        "dtype": "float32",
        "batch_size": batch_size,
        "wall_s": wall_s,
        "summary_json": str(summary_json),
        "log_path": str(log_path),
    }
    result = None
    if summary_json.exists():
        current_summary = summary_json.read_bytes()
        # Successful resume may leave identical data. After a failed process,
        # retain only diagnostics written by this attempt, never a stale success.
        if proc.returncode == 0 or current_summary != previous_summary:
            try:
                result = json.loads(current_summary)
            except json.JSONDecodeError:
                pass
    if result is not None:
        row.update(result)
        row.update(result.get("settings", {}))
    if proc.returncode != 0 or result is None:
        if result is None or row.get("status") == "ok":
            row["status"] = "error"
        row.setdefault("error", f"evaluator exited with code {proc.returncode}; see {log_path}")
        summary_json.write_text(json.dumps(dict(row, command=cmd), indent=2) + "\n")
    if row["status"] == "ok":
        print(f"{batch_size}: {row['closed_loop_steps_per_s']:.3f} control updates/s", flush=True)
    else:
        print(row.get("error", "Measurement failed"), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("cartpole", "crazyflie"), required=True)
    parser.add_argument("--method", choices=("mppi", "cem"), required=True)
    parser.add_argument("--batch-sizes", type=_parse_ints, default=list(BATCH_SIZES))
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--csv-path", type=pathlib.Path, required=True)
    parser.add_argument("--chunk-size", type=_positive_int)
    parser.add_argument("--sim-steps", type=_positive_int)
    parser.add_argument("--reference-sqp-iterations", type=_positive_int)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.chunk_size = args.chunk_size or CHUNK_SIZES[args.system]
    args.sim_steps = args.sim_steps or SIM_STEPS[args.system]
    args.reference_sqp_iterations = args.reference_sqp_iterations or REFERENCE_SQP_ITERATIONS[args.system]
    # Scalar labels used by the same external VRAM monitor as other baselines.
    args.qp_solver = args.method
    args.mode = args.method.upper()
    args.dtype = "float32"
    if args.dry_run:
        print(json.dumps({
            "system": args.system,
            "method": args.method,
            "settings": asdict(default_settings(args.system, args.method)),
            "batch_sizes": args.batch_sizes,
            "chunk_size": args.chunk_size,
            "sim_steps": args.sim_steps,
            "reference_sqp_iterations": args.reference_sqp_iterations,
            "repeats": args.repeats,
            "seed": args.seed,
            "output_dir": str(args.output_dir),
            "csv_path": str(args.csv_path),
        }, indent=2))
        return
    args.output_dir = args.output_dir.resolve()
    args.csv_path = args.csv_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_size in args.batch_sizes:
        rows.append(run_batch(args, batch_size=batch_size))
        _write_csv(args.csv_path, rows)
    failures = [row["batch_size"] for row in rows if row["status"] != "ok"]
    if failures:
        raise RuntimeError(f"Sampling sweep failed at batches {failures}; inspect {args.output_dir}")


if __name__ == "__main__":
    main()
