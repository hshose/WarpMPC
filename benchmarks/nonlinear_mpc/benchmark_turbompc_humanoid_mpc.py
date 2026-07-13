#!/usr/bin/env python3
"""Batch-size sweep for the TurboMPC humanoid MPC baseline."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_turbompc_humanoid_mpc.py"


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip().replace("_", "")) for part in text.split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


FIELDNAMES = [
    "batch_size",
    "status",
    "dtype",
    "qp_solver",
    "model_name",
    "mode",
    "turbompc_forward_backend",
    "turbompc_backward_backend",
    "horizon_nodes",
    "sim_steps",
    "standing_reference_s",
    "max_iter",
    "turbompc_eps_abs",
    "turbompc_eps_rel",
    "rho",
    "sigma",
    "alpha",
    "total_rti_steps",
    "elapsed_s",
    "rti_steps_per_s",
    "mean_step",
    "max_violation",
    "mean_qp_prim",
    "mean_qp_dual",
    "dense_block_dim",
    "dense_inequality_dim",
    "n_variables",
    "n_constraints",
    "nnz_p",
    "nnz_a",
    "problem_build_s",
    "setup_s",
    "warmup_compile_and_run_s",
    "wall_s",
    "plot_path",
    "step_throughput_plot_path",
    "step_throughput_csv_path",
    "summary_json",
    "log_path",
    "error",
]


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def _plot_summary(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    group = sorted(ok_rows, key=lambda row: int(row["batch_size"]))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(
        [int(row["batch_size"]) for row in group],
        [float(row["rti_steps_per_s"]) for row in group],
        marker="o",
        label="TurboMPC",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("SQP-RTI steps / s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_batch(args: argparse.Namespace, *, batch_size: int, dtype: str) -> dict[str, object]:
    prefix = f"humanoid_mpc_turbompc_{dtype}_batch_{batch_size}"
    plot_path = args.output_dir / f"{prefix}.png"
    step_throughput_plot_path = args.output_dir / f"{prefix}_throughput.png"
    step_throughput_csv_path = args.output_dir / f"{prefix}_throughput.csv"
    summary_json = args.output_dir / f"{prefix}.json"
    log_path = args.output_dir / f"{prefix}.log"
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--batch-size",
        str(batch_size),
        "--horizon-nodes",
        str(args.horizon_nodes),
        "--sim-steps",
        str(args.sim_steps),
        "--control-dt",
        str(args.control_dt),
        "--standing-reference-s",
        str(args.standing_reference_s),
        "--phi-vel",
        str(args.phi_vel),
        "--model-name",
        args.model_name,
        "--parameters",
        str(args.parameters),
        "--reference-velocity-x",
        str(args.reference_velocity_x),
        "--reference-velocity-y",
        str(args.reference_velocity_y),
        "--reference-yaw-rate",
        str(args.reference_yaw_rate),
        "--dtype",
        dtype,
        "--max-iter",
        str(args.max_iter),
        "--turbompc-eps-abs",
        str(args.turbompc_eps_abs),
        "--turbompc-eps-rel",
        str(args.turbompc_eps_rel),
        "--rho",
        str(args.rho),
        "--sigma",
        str(args.sigma),
        "--alpha",
        str(args.alpha),
        "--line-search-step-min",
        str(args.line_search_step_min),
        "--turbompc-forward-backend",
        args.turbompc_forward_backend,
        "--turbompc-backward-backend",
        args.turbompc_backward_backend,
        "--seed",
        str(args.seed + batch_size),
        "--plot-path",
        str(plot_path),
        "--throughput-plot-path",
        str(step_throughput_plot_path),
        "--throughput-csv-path",
        str(step_throughput_csv_path),
        "--summary-json",
        str(summary_json),
    ]
    if args.reference_body_height is not None:
        cmd += ["--reference-body-height", str(args.reference_body_height)]
    if args.skip_plots:
        cmd.append("--skip-plots")

    print(f"Running solver=turbompc dtype={dtype} batch={batch_size}: {' '.join(cmd)}")
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
        "batch_size": batch_size,
        "dtype": dtype,
        "qp_solver": "turbompc",
        "model_name": args.model_name,
        "mode": args.turbompc_forward_backend,
        "turbompc_forward_backend": args.turbompc_forward_backend,
        "turbompc_backward_backend": args.turbompc_backward_backend,
        "wall_s": wall_s,
        "plot_path": "" if args.skip_plots else str(plot_path),
        "step_throughput_plot_path": "" if args.skip_plots else str(step_throughput_plot_path),
        "step_throughput_csv_path": "" if args.skip_plots else str(step_throughput_csv_path),
        "summary_json": str(summary_json),
        "log_path": str(log_path),
    }
    if proc.returncode == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text())
        row.update(summary)
        row["status"] = "ok"
        row["error"] = ""
        print(
            f"solver=turbompc dtype={dtype} batch={batch_size}: "
            f"{float(summary['rti_steps_per_s']):.3g} SQP-RTI steps/s"
        )
    else:
        row["status"] = "failed"
        row["error"] = proc.stdout[-4000:]
        print(f"solver=turbompc dtype={dtype} batch={batch_size}: failed, see {log_path}")
    return row


def main() -> None:
    from benchmarks.problems.humanoid_mpc import (
        HUMANOID_MODEL_NAME,
        HUMANOID_MPC_STANDING_REFERENCE_S,
        HUMANOID_PARAMETERS_PATH,
        HUMANOID_PHI_VEL,
        load_humanoid_mpc_parameters,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000,50000,100000,200000,300000")
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--horizon-nodes", type=int, default=None)
    parser.add_argument("--sim-steps", type=int, default=20)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--standing-reference-s", type=float, default=HUMANOID_MPC_STANDING_REFERENCE_S)
    parser.add_argument("--phi-vel", type=float, default=HUMANOID_PHI_VEL)
    parser.add_argument("--model-name", default=HUMANOID_MODEL_NAME)
    parser.add_argument("--parameters", type=pathlib.Path, default=HUMANOID_PARAMETERS_PATH)
    parser.add_argument("--reference-velocity-x", type=float, default=0.0)
    parser.add_argument("--reference-velocity-y", type=float, default=0.0)
    parser.add_argument("--reference-yaw-rate", type=float, default=0.0)
    parser.add_argument("--reference-body-height", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--turbompc-eps-abs", type=float, default=1e-3)
    parser.add_argument("--turbompc-eps-rel", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--line-search-step-min", type=float, default=None)
    parser.add_argument("--turbompc-forward-backend", default="admm_fused_cudss")
    parser.add_argument("--turbompc-backward-backend", default="direct_cudss_ffi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/humanoid_mpc_turbompc"))
    parser.add_argument("--csv-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/humanoid_mpc_turbompc.csv"))
    parser.add_argument("--plot-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/humanoid_mpc_turbompc_summary.png"))
    args = parser.parse_args()
    loaded_params = load_humanoid_mpc_parameters(args.parameters)
    if args.horizon_nodes is None:
        args.horizon_nodes = loaded_params.n_nodes
    if args.max_iter is None:
        args.max_iter = loaded_params.osqp_max_iter
    if args.rho is None:
        args.rho = loaded_params.osqp_rho
    if args.alpha is None:
        args.alpha = loaded_params.osqp_alpha
    if args.line_search_step_min is None:
        args.line_search_step_min = loaded_params.line_search_step_min
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    batch_sizes = _parse_ints(args.batch_sizes)
    for dtype in _parse_strings(args.dtypes):
        for batch_size in batch_sizes:
            row = run_batch(args, batch_size=batch_size, dtype=dtype)
            rows.append(row)
            _write_csv(args.csv_path, rows)
            _plot_summary(args.plot_path, rows)
            if row.get("status") != "ok":
                remaining = [batch for batch in batch_sizes if batch > batch_size]
                if remaining:
                    print(
                        f"Skipping larger batch sizes for solver=turbompc dtype={dtype} "
                        f"mode={row['mode']} after failed batch={batch_size}: "
                        f"{','.join(str(batch) for batch in remaining)}"
                    )
                break
    print(f"Wrote {args.csv_path}")
    print(f"Wrote {args.plot_path}")


if __name__ == "__main__":
    main()
