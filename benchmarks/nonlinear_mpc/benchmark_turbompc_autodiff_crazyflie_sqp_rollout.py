#!/usr/bin/env python3
"""Batch-size sweep for the TurboMPC-autodiff Crazyflie SQP-RTI baseline."""

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
EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_turbompc_autodiff_crazyflie_sqp.py"


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip().replace("_", "")) for part in text.split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


FIELDNAMES = [
    "batch_size",
    "status",
    "repeat",
    "dtype",
    "qp_solver",
    "linearization_backend",
    "mode",
    "turbompc_forward_backend",
    "turbompc_backward_backend",
    "horizon_steps",
    "sim_steps",
    "max_iter",
    "turbompc_eps_abs",
    "turbompc_eps_rel",
    "rho",
    "sigma",
    "alpha",
    "total_rti_steps",
    "problem_build_s",
    "setup_s",
    "warmup_compile_and_run_s",
    "compile_s",
    "elapsed_s",
    "solve_s",
    "rti_steps_per_s",
    "mean_step",
    "final_position_rms",
    "final_state_rms",
    "max_violation",
    "mean_qp_prim",
    "mean_qp_dual",
    "dense_block_dim",
    "dense_inequality_dim",
    "n_variables",
    "n_constraints",
    "nnz_p",
    "nnz_a",
    "wall_s",
    "output_npz",
    "plot_path",
    "position_plot_path",
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
        label="TurboMPC autodiff",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("SQP-RTI steps / s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_batch(
    args: argparse.Namespace,
    *,
    batch_size: int,
    dtype: str,
    repeat_index: int,
) -> dict[str, object]:
    prefix = f"crazyflie_sqp_rollout_turbompc_autodiff_{dtype}_batch_{batch_size}_run_{repeat_index + 1}"
    output_npz = args.output_dir / f"{prefix}.npz"
    plot_path = args.output_dir / f"{prefix}_states.png"
    position_plot_path = args.output_dir / f"{prefix}_positions.png"
    summary_json = args.output_dir / f"{prefix}.json"
    log_path = args.output_dir / f"{prefix}.log"
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--batch-size",
        str(batch_size),
        "--horizon-steps",
        str(args.horizon_steps),
        "--sim-time",
        str(args.sim_time),
        "--control-dt",
        str(args.control_dt),
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
        "--position-plot-path",
        str(position_plot_path),
        "--summary-json",
        str(summary_json),
    ]
    if args.sim_steps is not None:
        cmd.extend(["--sim-steps", str(args.sim_steps)])
    if args.skip_output_npz:
        pass
    else:
        cmd.extend(["--output-npz", str(output_npz)])
    if args.skip_plots:
        cmd.append("--skip-plots")

    print(
        f"Running repeat={repeat_index + 1} solver=turbompc_autodiff dtype={dtype} "
        f"batch={batch_size}: {' '.join(cmd)}"
    )
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
        "repeat": repeat_index + 1,
        "dtype": dtype,
        "qp_solver": "turbompc_autodiff",
        "linearization_backend": "turbompc_jax_autodiff",
        "mode": args.turbompc_forward_backend,
        "turbompc_forward_backend": args.turbompc_forward_backend,
        "turbompc_backward_backend": args.turbompc_backward_backend,
        "wall_s": wall_s,
        "output_npz": "" if args.skip_output_npz else str(output_npz),
        "plot_path": "" if args.skip_plots else str(plot_path),
        "position_plot_path": "" if args.skip_plots else str(position_plot_path),
        "summary_json": str(summary_json),
        "log_path": str(log_path),
    }
    if proc.returncode == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text())
        row.update(summary)
        row["status"] = "ok"
        row["error"] = ""
        print(
            f"repeat={repeat_index + 1} solver=turbompc_autodiff dtype={dtype} batch={batch_size}: "
            f"{float(summary['rti_steps_per_s']):.3g} SQP-RTI steps/s"
        )
    else:
        row["status"] = "failed"
        row["error"] = proc.stdout[-4000:]
        print(f"solver=turbompc_autodiff dtype={dtype} batch={batch_size}: failed, see {log_path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000,50000,100000,200000,300000")
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--sim-time", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--turbompc-eps-abs", type=float, default=1e-3)
    parser.add_argument("--turbompc-eps-rel", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
    parser.add_argument("--turbompc-forward-backend", default="admm_fused_cudss")
    parser.add_argument("--turbompc-backward-backend", default="direct_cudss_ffi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-output-npz", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_rollout_turbompc_autodiff"))
    parser.add_argument("--csv-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_rollout_turbompc_autodiff.csv"))
    parser.add_argument("--plot-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_rollout_turbompc_autodiff_summary.png"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    batch_sizes = _parse_ints(args.batch_sizes)
    for dtype in _parse_strings(args.dtypes):
        for batch_size in batch_sizes:
            batch_failed = False
            for repeat_index in range(args.repeat):
                row = run_batch(
                    args,
                    batch_size=batch_size,
                    dtype=dtype,
                    repeat_index=repeat_index,
                )
                rows.append(row)
                _write_csv(args.csv_path, rows)
                _plot_summary(args.plot_path, rows)
                if row.get("status") != "ok":
                    batch_failed = True
                    break
            if batch_failed:
                remaining = [batch for batch in batch_sizes if batch > batch_size]
                if remaining:
                    print(
                        f"Skipping larger batch sizes for solver=turbompc_autodiff dtype={dtype} "
                        f"mode={row['mode']} after failed batch={batch_size}: "
                        f"{','.join(str(batch) for batch in remaining)}"
                    )
                break
    print(f"Wrote {args.csv_path}")
    print(f"Wrote {args.plot_path}")


if __name__ == "__main__":
    main()
