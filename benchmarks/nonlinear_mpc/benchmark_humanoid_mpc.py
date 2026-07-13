#!/usr/bin/env python3
"""Batch-size sweep for the humanoid sparse SQP MPC benchmark."""

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
EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_humanoid_mpc.py"


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_backend_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in _parse_strings(text):
        if ":" in item:
            factor_backend, solve_backend = (part.strip() for part in item.split(":", 1))
        else:
            factor_backend = solve_backend = item
        if factor_backend not in {"jax", "warp"} or solve_backend not in {"jax", "warp"}:
            raise ValueError(f"unsupported QDLDL backend pair: {item!r}")
        pairs.append((factor_backend, solve_backend))
    return pairs


QDLDL_VARIANTS: dict[str, dict[str, object]] = {
    "baseline": {
        "qdldl_factor_backend": "jax",
        "qdldl_solve_backend": "jax",
        "transpose_work": False,
        "segmented": False,
        "level_scheduled_solve": False,
    },
    "transpose+segmented": {
        "qdldl_factor_backend": "jax",
        "qdldl_solve_backend": "jax",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": False,
    },
    "transpose+segmented+levelsolve": {
        "qdldl_factor_backend": "jax",
        "qdldl_solve_backend": "jax",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
    "factor-warp+solve-jax:transpose+segmented": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "jax",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": False,
    },
    "factor-warp+solve-jax:transpose+segmented+levelsolve": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "jax",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
    "factor-warp+solve-warp:transpose": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "warp",
        "transpose_work": True,
        "segmented": False,
        "level_scheduled_solve": False,
    },
    "factor-warp+solve-warp:transpose+levelsolve": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "warp",
        "transpose_work": True,
        "segmented": False,
        "level_scheduled_solve": True,
    },
    "factor-warp+solve-warp:transpose+segmented": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "warp",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": False,
    },
    "factor-warp+solve-warp:transpose+segmented+levelsolve": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "warp",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
}


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "status",
        "dtype",
        "qp_solver",
        "qdldl_backend",
        "qdldl_factor_backend",
        "qdldl_solve_backend",
        "model_name",
        "mode",
        "group_repeated_stages",
        "level_scheduled_solve",
        "level_scheduled_solve_threshold",
        "qdldl_variant",
        "transpose_work",
        "segmented",
        "segment_budget",
        "segment_strategy",
        "horizon_nodes",
        "sim_steps",
        "standing_reference_s",
        "mpax_iteration_limit",
        "mpax_eps_abs",
        "mpax_eps_rel",
        "mpax_termination_evaluation_frequency",
        "mpax_l_inf_ruiz_iterations",
        "mpax_pock_chambolle_alpha",
        "mpax_regularization",
        "mpax_unroll",
        "total_rti_steps",
        "elapsed_s",
        "rti_steps_per_s",
        "mean_step",
        "max_violation",
        "mean_qp_prim",
        "mean_qp_dual",
        "n_variables",
        "n_constraints",
        "nnz_p",
        "nnz_a",
        "wall_s",
        "plot_path",
        "step_throughput_plot_path",
        "step_throughput_csv_path",
        "summary_json",
        "log_path",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_summary(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    markers = ["o", "s", "^", "D", "v", "P"]
    keys = sorted({(str(row["dtype"]), str(row["mode"])) for row in ok_rows})
    for idx, (dtype, mode) in enumerate(keys):
        group = sorted(
            [
                row
                for row in ok_rows
                if str(row["dtype"]) == dtype and str(row["mode"]) == mode
            ],
            key=lambda row: int(row["batch_size"]),
        )
        ax.plot(
            [int(row["batch_size"]) for row in group],
            [float(row["rti_steps_per_s"]) for row in group],
            marker=markers[idx % len(markers)],
            label=f"{dtype} {mode}",
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
    qp_solver: str,
    qdldl_factor_backend: str,
    qdldl_solve_backend: str,
    group_repeated_stages: bool,
    level_scheduled_solve: bool,
    transpose_work: bool,
    segmented: bool,
    qdldl_variant: str | None = None,
) -> dict[str, object]:
    base_mode = "grouped" if group_repeated_stages else "ungrouped"
    if qp_solver == "mpax":
        mode = f"mpax:{base_mode}"
    elif qdldl_variant is not None:
        mode = f"{qdldl_variant}:{base_mode}"
    else:
        mode = f"factor-{qdldl_factor_backend}+solve-{qdldl_solve_backend}:{base_mode}"
    if level_scheduled_solve:
        if "levelsolve" not in mode:
            mode = f"{mode}+levelsolve"
        mode = f"{mode}+threshold{args.level_scheduled_solve_threshold}"
    mode_slug = mode.replace(":", "_").replace("+", "_")
    prefix = f"humanoid_mpc_{qp_solver}_{dtype}_{mode_slug}_batch_{batch_size}"
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
        "--qp-solver",
        qp_solver,
        "--qdldl-backend",
        qdldl_factor_backend,
        "--qdldl-factor-backend",
        qdldl_factor_backend,
        "--qdldl-solve-backend",
        qdldl_solve_backend,
        "--max-iter",
        str(args.max_iter),
        "--mpax-iteration-limit",
        str(args.mpax_iteration_limit),
        "--mpax-eps-abs",
        str(args.mpax_eps_abs),
        "--mpax-eps-rel",
        str(args.mpax_eps_rel),
        "--mpax-termination-evaluation-frequency",
        str(args.mpax_termination_evaluation_frequency),
        "--mpax-l-inf-ruiz-iterations",
        str(args.mpax_l_inf_ruiz_iterations),
        "--mpax-pock-chambolle-alpha",
        str(args.mpax_pock_chambolle_alpha),
        "--mpax-regularization",
        str(args.mpax_regularization),
        "--rho",
        str(args.rho),
        "--sigma",
        str(args.sigma),
        "--alpha",
        str(args.alpha),
        "--scaling",
        str(args.scaling),
        "--segment-budget",
        str(args.segment_budget),
        "--segment-strategy",
        args.segment_strategy,
        "--line-search-step-min",
        str(args.line_search_step_min),
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
    if not group_repeated_stages:
        cmd.append("--no-group-repeated-stages")
    if level_scheduled_solve:
        cmd.append("--level-scheduled-solve")
        cmd.extend([
            "--level-scheduled-solve-threshold",
            str(args.level_scheduled_solve_threshold),
        ])
    if args.mpax_unroll:
        cmd.append("--mpax-unroll")
    if not transpose_work:
        cmd.append("--no-transpose-work")
    if not segmented:
        cmd.append("--no-segmented")
    if args.skip_plots:
        cmd.append("--skip-plots")

    print(f"Running solver={qp_solver} dtype={dtype} mode={mode} batch={batch_size}: {' '.join(cmd)}")
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
        "qp_solver": qp_solver,
        "qdldl_backend": qdldl_factor_backend,
        "qdldl_factor_backend": qdldl_factor_backend,
        "qdldl_solve_backend": qdldl_solve_backend,
        "model_name": args.model_name,
        "mode": mode,
        "group_repeated_stages": group_repeated_stages,
        "level_scheduled_solve": level_scheduled_solve,
        "level_scheduled_solve_threshold": (
            args.level_scheduled_solve_threshold if level_scheduled_solve else ""
        ),
        "qdldl_variant": qdldl_variant or "",
        "transpose_work": transpose_work,
        "segmented": segmented,
        "segment_budget": args.segment_budget,
        "segment_strategy": args.segment_strategy,
        "wall_s": wall_s,
        "plot_path": "" if args.skip_plots else str(plot_path),
        "step_throughput_plot_path": "" if args.skip_plots else str(step_throughput_plot_path),
        "step_throughput_csv_path": str(step_throughput_csv_path),
        "summary_json": str(summary_json),
        "log_path": str(log_path),
    }
    if proc.returncode == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text())
        row.update(summary)
        row["dtype"] = dtype
        row["qp_solver"] = qp_solver
        row["qdldl_backend"] = qdldl_factor_backend
        row["qdldl_factor_backend"] = qdldl_factor_backend
        row["qdldl_solve_backend"] = qdldl_solve_backend
        row["mode"] = mode
        row["group_repeated_stages"] = group_repeated_stages
        row["level_scheduled_solve"] = level_scheduled_solve
        row["level_scheduled_solve_threshold"] = (
            args.level_scheduled_solve_threshold if level_scheduled_solve else ""
        )
        row["qdldl_variant"] = qdldl_variant or ""
        row["transpose_work"] = transpose_work
        row["segmented"] = segmented
        row["segment_budget"] = args.segment_budget
        row["segment_strategy"] = args.segment_strategy
        if args.skip_plots:
            row["plot_path"] = ""
            row["step_throughput_plot_path"] = ""
        row["status"] = "ok"
        row["error"] = ""
        print(
            f"solver={qp_solver} dtype={dtype} mode={mode} batch={batch_size}: "
            f"{float(summary['rti_steps_per_s']):.3g} SQP-RTI steps/s"
        )
    else:
        row["status"] = "failed"
        row["error"] = proc.stdout[-4000:]
        print(f"solver={qp_solver} dtype={dtype} mode={mode} batch={batch_size}: failed, see {log_path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000,50000,100000,200000,300000")
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--qp-solvers", default="jax_osqp")
    parser.add_argument("--qdldl-backend-pairs", default="jax:jax")
    parser.add_argument("--qdldl-variants", default=None)
    parser.add_argument("--group-modes", default="grouped")
    parser.add_argument("--levelsolve-modes", default="regular,levelsolve")
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument(
        "--horizon-nodes",
        type=int,
        default=None,
        help="Override the humanoid horizon node count from the parameter YAML.",
    )
    parser.add_argument("--sim-steps", type=int, default=20)
    parser.add_argument("--control-dt", type=float, default=0.01)
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
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--mpax-iteration-limit", type=int, default=None)
    parser.add_argument("--mpax-eps-abs", type=float, default=1e-3)
    parser.add_argument("--mpax-eps-rel", type=float, default=1e-3)
    parser.add_argument("--mpax-termination-evaluation-frequency", type=int, default=100)
    parser.add_argument("--mpax-l-inf-ruiz-iterations", type=int, default=10)
    parser.add_argument("--mpax-pock-chambolle-alpha", type=float, default=1.0)
    parser.add_argument("--mpax-regularization", type=float, default=0.0)
    parser.add_argument("--mpax-unroll", action="store_true")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--scaling", type=int, default=None)
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-budget", type=int, default=96)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--line-search-step-min", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/humanoid_mpc"),
    )
    parser.add_argument(
        "--csv-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/humanoid_mpc.csv"),
    )
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/humanoid_mpc_summary.png"),
    )
    args = parser.parse_args()
    from benchmarks.problems.humanoid_mpc import load_humanoid_mpc_parameters

    loaded_params = load_humanoid_mpc_parameters(args.parameters)
    if args.horizon_nodes is None:
        args.horizon_nodes = loaded_params.n_nodes
    if args.max_iter is None:
        args.max_iter = loaded_params.osqp_max_iter
    if args.rho is None:
        args.rho = loaded_params.osqp_rho
    if args.alpha is None:
        args.alpha = loaded_params.osqp_alpha
    if args.scaling is None:
        args.scaling = loaded_params.osqp_scaling
    if args.line_search_step_min is None:
        args.line_search_step_min = loaded_params.line_search_step_min
    if args.mpax_iteration_limit is None:
        args.mpax_iteration_limit = args.max_iter
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    batch_sizes = _parse_ints(args.batch_sizes)
    qp_solvers = _parse_strings(args.qp_solvers)
    for qp_solver in qp_solvers:
        if qp_solver not in {"jax_osqp", "mpax"}:
            raise ValueError(f"unsupported QP solver: {qp_solver}")
    backend_pairs = _parse_backend_pairs(args.qdldl_backend_pairs)
    qdldl_variant_names = _parse_strings(args.qdldl_variants) if args.qdldl_variants else []
    unknown_qdldl_variants = sorted(set(qdldl_variant_names) - set(QDLDL_VARIANTS))
    if unknown_qdldl_variants:
        raise ValueError(f"unknown QDLDL variants: {unknown_qdldl_variants}")
    levelsolve_modes = _parse_strings(args.levelsolve_modes)
    for levelsolve_mode in levelsolve_modes:
        if levelsolve_mode not in {"regular", "levelsolve"}:
            raise ValueError(f"unsupported levelsolve mode: {levelsolve_mode}")
    for dtype in _parse_strings(args.dtypes):
        if dtype not in {"float32", "float64"}:
            raise ValueError(f"unsupported dtype: {dtype}")
        for qp_solver in qp_solvers:
            if qp_solver == "jax_osqp" and qdldl_variant_names:
                solver_variants = [
                    (
                        qdldl_variant,
                        str(QDLDL_VARIANTS[qdldl_variant]["qdldl_factor_backend"]),
                        str(QDLDL_VARIANTS[qdldl_variant]["qdldl_solve_backend"]),
                        bool(QDLDL_VARIANTS[qdldl_variant]["transpose_work"]),
                        bool(QDLDL_VARIANTS[qdldl_variant]["segmented"]),
                        bool(QDLDL_VARIANTS[qdldl_variant]["level_scheduled_solve"]),
                    )
                    for qdldl_variant in qdldl_variant_names
                ]
            elif qp_solver == "jax_osqp":
                solver_variants = [
                    (
                        None,
                        qdldl_factor_backend,
                        qdldl_solve_backend,
                        args.transpose_work,
                        args.segmented,
                        levelsolve_mode == "levelsolve",
                    )
                    for qdldl_factor_backend, qdldl_solve_backend in backend_pairs
                    for levelsolve_mode in levelsolve_modes
                ]
            else:
                solver_variants = [(None, "jax", "jax", args.transpose_work, args.segmented, False)]
            for (
                qdldl_variant,
                qdldl_factor_backend,
                qdldl_solve_backend,
                transpose_work,
                segmented,
                level_scheduled_solve,
            ) in solver_variants:
                for mode in _parse_strings(args.group_modes):
                    if mode not in {"grouped", "ungrouped"}:
                        raise ValueError(f"unsupported group mode: {mode}")
                    for batch_size in batch_sizes:
                        row = run_batch(
                            args,
                            batch_size=batch_size,
                            dtype=dtype,
                            qp_solver=qp_solver,
                            qdldl_factor_backend=qdldl_factor_backend,
                            qdldl_solve_backend=qdldl_solve_backend,
                            group_repeated_stages=mode == "grouped",
                            level_scheduled_solve=level_scheduled_solve,
                            transpose_work=transpose_work,
                            segmented=segmented,
                            qdldl_variant=qdldl_variant,
                        )
                        rows.append(row)
                        _write_csv(args.csv_path, rows)
                        _plot_summary(args.plot_path, rows)
                        if row.get("status") != "ok":
                            remaining = [batch for batch in batch_sizes if batch > batch_size]
                            if remaining:
                                print(
                                    f"Skipping larger batch sizes for solver={qp_solver} dtype={dtype} "
                                    f"mode={row['mode']} after failed batch={batch_size}: "
                                    f"{','.join(str(batch) for batch in remaining)}"
                                )
                            break
    print(f"Wrote {args.csv_path}")
    print(f"Wrote {args.plot_path}")


if __name__ == "__main__":
    main()
