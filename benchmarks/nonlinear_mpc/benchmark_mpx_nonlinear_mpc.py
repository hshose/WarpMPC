#!/usr/bin/env python3
"""Batch-size sweeps for MPX relaxed-barrier nonlinear MPC baselines."""

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
EVALUATORS = {
    "cartpole": ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_mpx_cartpole_quadratic_mpc.py",
    "crazyflie": ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_mpx_crazyflie_sqp.py",
    "humanoid": ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_mpx_humanoid_mpc.py",
}
PREFIXES = {
    "cartpole": "cartpole_quadratic_mpc_mpx",
    "crazyflie": "crazyflie_sqp_rollout_mpx",
    "humanoid": "humanoid_mpc_mpx",
}


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


FIELDNAMES = [
    "system",
    "batch_size",
    "status",
    "repeat",
    "dtype",
    "qp_solver",
    "mode",
    "mpx_solver_mode",
    "mpx_equality_weight",
    "mpx_barrier_alpha",
    "mpx_barrier_sigma",
    "mpx_num_alpha",
    "mpx_limited_memory",
    "horizon_steps",
    "horizon_nodes",
    "rollout_steps",
    "sim_steps",
    "sqp_iterations",
    "control_dt",
    "sim_time",
    "total_closed_loop_steps",
    "total_sqp_iterations",
    "total_rti_steps",
    "elapsed_s",
    "solve_s",
    "closed_loop_steps_per_s",
    "sqp_iterations_per_s",
    "rti_steps_per_s",
    "return_mean",
    "return_p10",
    "return_median",
    "return_p90",
    "rollout_success_rate",
    "rail_violation_rate",
    "rollout_rail_violation_mean",
    "rollout_rail_violation_max",
    "final_position_rms",
    "final_state_rms",
    "final_state_cost_mean",
    "final_state_cost_median",
    "final_state_cost_max",
    "mean_step",
    "max_violation",
    "max_action_violation",
    "max_rail_constraint_violation",
    "max_dynamics_defect",
    "max_initial_defect",
    "finite_rate",
    "mean_qp_prim",
    "mean_qp_dual",
    "line_search_accept_rate",
    "sqp_finite_rate",
    "n_variables",
    "n_constraints",
    "nnz_p",
    "nnz_a",
    "problem_build_s",
    "compile_setup_s",
    "solver_setup_s",
    "initialization_s",
    "rollout_compile_s",
    "warmup_compile_and_run_s",
    "compile_s",
    "total_compile_s",
    "wall_s",
    "plot_path",
    "position_plot_path",
    "step_throughput_plot_path",
    "step_throughput_csv_path",
    "output_npz",
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


def _throughput_key(system: str) -> str:
    if system == "cartpole":
        return "sqp_iterations_per_s"
    return "rti_steps_per_s"


def _plot_summary(path: pathlib.Path, rows: list[dict[str, object]], *, system: str) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    throughput_key = _throughput_key(system)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers = ["o", "s", "^", "D", "v", "P"]
    keys = sorted({(str(row["dtype"]), str(row["mode"])) for row in ok_rows})
    for idx, (dtype, mode) in enumerate(keys):
        group = sorted(
            [row for row in ok_rows if str(row["dtype"]) == dtype and str(row["mode"]) == mode],
            key=lambda row: int(row["batch_size"]),
        )
        ax.plot(
            [int(row["batch_size"]) for row in group],
            [float(row[throughput_key]) for row in group],
            marker=markers[idx % len(markers)],
            label=f"{dtype} {mode}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("MPX iterations / s" if system == "cartpole" else "MPX-RTI steps / s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _common_mpx_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "--mpx-solver-mode",
        args.mpx_solver_mode,
        "--mpx-equality-weight",
        str(args.mpx_equality_weight),
        "--mpx-barrier-alpha",
        str(args.mpx_barrier_alpha),
        "--mpx-barrier-sigma",
        str(args.mpx_barrier_sigma),
        "--mpx-num-alpha",
        str(args.mpx_num_alpha),
    ]
    if not args.mpx_limited_memory:
        cmd.append("--no-mpx-limited-memory")
    return cmd


def _cartpole_args(args: argparse.Namespace, plot_path: pathlib.Path) -> list[str]:
    cmd = [
        "--horizon-steps",
        str(args.horizon_steps),
        "--dt-start",
        str(args.dt_start),
        "--dt-growth",
        str(args.dt_growth),
        "--sim-time",
        str(args.sim_time),
        "--control-dt",
        str(args.control_dt),
        "--integrator-substeps",
        str(args.integrator_substeps),
        "--sqp-iterations",
        str(args.sqp_iterations),
        "--noise-scale",
        str(args.noise_scale),
        "--process-noise-scale",
        args.process_noise_scale,
        "--input-noise-scale",
        args.input_noise_scale,
        "--plot-path",
        str(plot_path),
        "--plot-samples",
        str(args.plot_samples),
    ]
    if args.rollout_steps is not None:
        cmd.extend(["--rollout-steps", str(args.rollout_steps)])
    if args.enable_rail_constraint:
        cmd.append("--enable-rail-constraint")
    if args.skip_plots:
        cmd.append("--skip-state-plot")
    return cmd


def _crazyflie_args(
    args: argparse.Namespace,
    *,
    plot_path: pathlib.Path,
    position_plot_path: pathlib.Path,
    output_npz: pathlib.Path,
) -> list[str]:
    cmd = [
        "--horizon-steps",
        str(args.horizon_steps),
        "--sim-time",
        str(args.sim_time),
        "--control-dt",
        str(args.control_dt),
        "--sqp-iterations",
        str(args.sqp_iterations),
        "--plot-path",
        str(plot_path),
        "--position-plot-path",
        str(position_plot_path),
    ]
    if args.sim_steps is not None:
        cmd.extend(["--sim-steps", str(args.sim_steps)])
    if args.skip_output_npz:
        cmd.append("--skip-output-npz")
    else:
        cmd.extend(["--output-npz", str(output_npz)])
    if args.skip_plots:
        cmd.append("--skip-plots")
    return cmd


def _humanoid_args(
    args: argparse.Namespace,
    *,
    plot_path: pathlib.Path,
    throughput_plot_path: pathlib.Path,
    throughput_csv_path: pathlib.Path,
) -> list[str]:
    cmd = [
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
        "--plot-path",
        str(plot_path),
        "--throughput-plot-path",
        str(throughput_plot_path),
        "--throughput-csv-path",
        str(throughput_csv_path),
    ]
    if args.horizon_nodes is not None:
        cmd.extend(["--horizon-nodes", str(args.horizon_nodes)])
    if args.reference_body_height is not None:
        cmd.extend(["--reference-body-height", str(args.reference_body_height)])
    if args.skip_plots:
        cmd.append("--skip-plots")
    return cmd


def run_batch(
    args: argparse.Namespace,
    *,
    batch_size: int,
    dtype: str,
    repeat_index: int,
) -> dict[str, object]:
    mode = f"mpx:{args.mpx_solver_mode}:penalty"
    mode_slug = mode.replace(":", "_").replace("+", "_")
    prefix = f"{PREFIXES[args.system]}_{dtype}_{mode_slug}_batch_{batch_size}_run_{repeat_index + 1}"
    plot_path = args.output_dir / f"{prefix}.png"
    position_plot_path = args.output_dir / f"{prefix}_positions.png"
    throughput_plot_path = args.output_dir / f"{prefix}_throughput.png"
    throughput_csv_path = args.output_dir / f"{prefix}_throughput.csv"
    output_npz = args.output_dir / f"{prefix}.npz"
    summary_json = args.output_dir / f"{prefix}.json"
    log_path = args.output_dir / f"{prefix}.log"
    cmd = [
        sys.executable,
        str(EVALUATORS[args.system]),
        "--batch-size",
        str(batch_size),
        "--dtype",
        dtype,
        "--seed",
        str(args.seed + batch_size + 1009 * repeat_index),
        "--summary-json",
        str(summary_json),
    ]
    cmd.extend(_common_mpx_args(args))
    if args.system == "cartpole":
        cmd.extend(_cartpole_args(args, plot_path))
    elif args.system == "crazyflie":
        cmd.extend(
            _crazyflie_args(
                args,
                plot_path=plot_path,
                position_plot_path=position_plot_path,
                output_npz=output_npz,
            )
        )
    else:
        cmd.extend(
            _humanoid_args(
                args,
                plot_path=plot_path,
                throughput_plot_path=throughput_plot_path,
                throughput_csv_path=throughput_csv_path,
            )
        )

    print(
        f"Running system={args.system} repeat={repeat_index + 1} dtype={dtype} "
        f"mode={mode} batch={batch_size}: {' '.join(cmd)}",
        flush=True,
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
        "system": args.system,
        "batch_size": batch_size,
        "repeat": repeat_index + 1,
        "dtype": dtype,
        "qp_solver": "mpx",
        "mode": mode,
        "mpx_solver_mode": args.mpx_solver_mode,
        "mpx_equality_weight": args.mpx_equality_weight,
        "mpx_barrier_alpha": args.mpx_barrier_alpha,
        "mpx_barrier_sigma": args.mpx_barrier_sigma,
        "mpx_num_alpha": args.mpx_num_alpha,
        "mpx_limited_memory": args.mpx_limited_memory,
        "wall_s": wall_s,
        "plot_path": "" if args.skip_plots else str(plot_path),
        "position_plot_path": "" if args.system != "crazyflie" or args.skip_plots else str(position_plot_path),
        "step_throughput_plot_path": "" if args.system != "humanoid" or args.skip_plots else str(throughput_plot_path),
        "step_throughput_csv_path": str(throughput_csv_path) if args.system == "humanoid" else "",
        "output_npz": str(output_npz) if args.system == "crazyflie" and not args.skip_output_npz else "",
        "summary_json": str(summary_json),
        "log_path": str(log_path),
    }
    if proc.returncode == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text())
        row.update(summary)
        row["system"] = args.system
        row["repeat"] = repeat_index + 1
        row["dtype"] = dtype
        row["qp_solver"] = "mpx"
        row["mode"] = mode
        row["status"] = "ok"
        row["error"] = ""
        throughput_key = _throughput_key(args.system)
        print(
            f"system={args.system} dtype={dtype} mode={mode} batch={batch_size}: "
            f"{float(summary[throughput_key]):.3g} {throughput_key}",
            flush=True,
        )
    else:
        row["status"] = "failed"
        row["error"] = proc.stdout[-4000:]
        print(f"system={args.system} dtype={dtype} mode={mode} batch={batch_size}: failed, see {log_path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=tuple(EVALUATORS), required=True)
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000,50000,100000,200000,300000")
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mpx-solver-mode", choices=("primal_dual", "fddp", "ilqr"), default="primal_dual")
    parser.add_argument("--mpx-equality-weight", type=float, default=1.0e4)
    parser.add_argument("--mpx-barrier-alpha", type=float, default=0.1)
    parser.add_argument("--mpx-barrier-sigma", type=float, default=1.0)
    parser.add_argument("--mpx-num-alpha", type=int, default=11)
    parser.add_argument("--mpx-limited-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--horizon-nodes", type=int, default=None)
    parser.add_argument("--dt-start", type=float, default=0.1)
    parser.add_argument("--dt-growth", type=float, default=1.0)
    parser.add_argument("--sim-time", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--integrator-substeps", type=int, default=1)
    parser.add_argument("--sqp-iterations", type=int, default=1)
    parser.add_argument("--enable-rail-constraint", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--process-noise-scale", default="0,0,0,0")
    parser.add_argument("--input-noise-scale", default="0")
    parser.add_argument("--plot-samples", type=int, default=2048)
    parser.add_argument("--skip-output-npz", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--standing-reference-s", type=float, default=0.0)
    parser.add_argument("--phi-vel", type=float, default=1.0 / 0.6)
    parser.add_argument("--model-name", default="no_rotors")
    parser.add_argument(
        "--parameters",
        type=pathlib.Path,
        default=ROOT / "benchmarks" / "problems" / "humanoid_mpc_parameters.yaml",
    )
    parser.add_argument("--reference-velocity-x", type=float, default=0.0)
    parser.add_argument("--reference-velocity-y", type=float, default=0.0)
    parser.add_argument("--reference-yaw-rate", type=float, default=0.0)
    parser.add_argument("--reference-body-height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/mpx"),
    )
    parser.add_argument(
        "--csv-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/mpx.csv"),
    )
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/mpx_summary.png"),
    )
    args = parser.parse_args()
    if args.control_dt is None:
        args.control_dt = 0.1 if args.system == "cartpole" else 0.01
    if args.system == "humanoid" and args.sim_steps is None:
        args.sim_steps = 20
    if args.system == "crazyflie" and args.sim_steps is None:
        args.sim_steps = int(np.ceil(args.sim_time / args.control_dt))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    batch_sizes = _parse_ints(args.batch_sizes)
    dtypes = _parse_strings(args.dtypes)
    for dtype in dtypes:
        if dtype not in {"float32", "float64"}:
            raise ValueError(f"unsupported dtype: {dtype}")
        for batch_size in batch_sizes:
            batch_failed = False
            for repeat_index in range(args.repeat):
                row = run_batch(args, batch_size=batch_size, dtype=dtype, repeat_index=repeat_index)
                rows.append(row)
                _write_csv(args.csv_path, rows)
                _plot_summary(args.plot_path, rows, system=args.system)
                if row.get("status") != "ok":
                    batch_failed = True
                    break
            if batch_failed:
                remaining = [batch for batch in batch_sizes if batch > batch_size]
                if remaining:
                    print(f"Skipping larger batch sizes after failure: {remaining}")
                break


if __name__ == "__main__":
    main()
