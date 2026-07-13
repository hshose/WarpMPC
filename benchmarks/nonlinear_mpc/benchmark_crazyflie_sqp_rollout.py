#!/usr/bin/env python3
"""Batch-size sweep for Crazyflie closed-loop SQP-RTI rollouts."""

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
EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_crazyflie_sqp.py"


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_backend_pairs(
    text: str | None,
    fallback_backends: list[str],
) -> list[tuple[str, str]]:
    if text is None:
        text = ",".join(f"{backend}:{backend}" for backend in fallback_backends)
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
        "repeat",
        "dtype",
        "qp_solver",
        "qdldl_backend",
        "qdldl_factor_backend",
        "qdldl_solve_backend",
        "mode",
        "group_repeated_stages",
        "level_scheduled_solve",
        "level_scheduled_solve_threshold",
        "qdldl_variant",
        "transpose_work",
        "segmented",
        "segment_budget",
        "segment_strategy",
        "horizon_steps",
        "sim_steps",
        "mpax_iteration_limit",
        "mpax_eps_abs",
        "mpax_eps_rel",
        "mpax_termination_evaluation_frequency",
        "mpax_l_inf_ruiz_iterations",
        "mpax_pock_chambolle_alpha",
        "mpax_regularization",
        "mpax_unroll",
        "total_rti_steps",
        "problem_build_s",
        "plan_build_s",
        "compile_setup_s",
        "setup_s",
        "warmup_compile_and_run_s",
        "compile_s",
        "total_compile_s",
        "elapsed_s",
        "solve_s",
        "rti_steps_per_s",
        "mean_step",
        "final_position_rms",
        "final_state_rms",
        "max_violation",
        "mean_qp_prim",
        "mean_qp_dual",
        "wall_s",
        "output_npz",
        "plot_path",
        "position_plot_path",
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
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    keys = sorted({(str(row["dtype"]), str(row["mode"])) for row in ok_rows})
    for idx, (dtype, mode) in enumerate(keys):
        group_rows = sorted(
            [
                row
                for row in ok_rows
                if str(row["dtype"]) == dtype and str(row["mode"]) == mode
            ],
            key=lambda row: int(row["batch_size"]),
        )
        batch_sizes = np.asarray([int(row["batch_size"]) for row in group_rows])
        throughput = np.asarray([float(row["rti_steps_per_s"]) for row in group_rows])
        final_position = np.asarray(
            [float(row["final_position_rms"]) for row in group_rows]
        )
        max_violation = np.asarray([float(row["max_violation"]) for row in group_rows])
        label = f"{dtype} {mode}"
        marker = markers[idx % len(markers)]
        axes[0].plot(batch_sizes, throughput, marker=marker, label=label)
        axes[1].plot(batch_sizes, final_position, marker=marker, label=label)
        axes[2].plot(batch_sizes, max_violation, marker=marker, label=label)
    axes[0].set_ylabel("SQP-RTI steps / s")
    axes[1].set_ylabel("final position RMS")
    axes[2].set_ylabel("max constraint violation")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
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
    repeat_index: int,
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
    prefix = (
        f"crazyflie_sqp_rollout_{qp_solver}_{dtype}_{mode_slug}_batch_{batch_size}"
        f"_run_{repeat_index + 1}"
    )
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
        "--position-plot-path",
        str(position_plot_path),
        "--summary-json",
        str(summary_json),
    ]
    if args.sim_steps is not None:
        cmd.extend(["--sim-steps", str(args.sim_steps)])
    if args.skip_output_npz:
        cmd.append("--skip-output-npz")
    else:
        cmd.extend(["--output-npz", str(output_npz)])
    if not group_repeated_stages:
        cmd.append("--no-group-repeated-stages")
    if level_scheduled_solve:
        cmd.append("--level-scheduled-solve")
        cmd.extend([
            "--level-scheduled-solve-threshold",
            str(args.level_scheduled_solve_threshold),
        ])
    if args.skip_plots:
        cmd.append("--skip-plots")
    if args.mpax_unroll:
        cmd.append("--mpax-unroll")
    if not transpose_work:
        cmd.append("--no-transpose-work")
    if not segmented:
        cmd.append("--no-segmented")

    print(
        f"Running repeat={repeat_index + 1} solver={qp_solver} dtype={dtype} mode={mode} "
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
        "qp_solver": qp_solver,
        "qdldl_backend": qdldl_factor_backend,
        "qdldl_factor_backend": qdldl_factor_backend,
        "qdldl_solve_backend": qdldl_solve_backend,
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
        "output_npz": "" if args.skip_output_npz else str(output_npz),
        "plot_path": str(plot_path),
        "position_plot_path": str(position_plot_path),
        "summary_json": str(summary_json),
        "log_path": str(log_path),
    }
    if proc.returncode == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text())
        row.update(summary)
        row["repeat"] = repeat_index + 1
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
        row["status"] = "ok"
        row["error"] = ""
        print(
            f"repeat={repeat_index + 1} solver={qp_solver} dtype={dtype} mode={mode} batch={batch_size}: "
            f"compile={float(summary['compile_s']):.3f}s, "
            f"solve={float(summary['solve_s']):.3f}s, "
            f"{float(summary['rti_steps_per_s']):.3g} SQP_RTI steps/s, "
            f"elapsed={float(summary['elapsed_s']):.3f}s"
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
    parser.add_argument("--qdldl-backends", default="jax")
    parser.add_argument("--qdldl-backend-pairs", default=None)
    parser.add_argument("--qdldl-variants", default=None)
    parser.add_argument("--group-modes", default="grouped")
    parser.add_argument("--levelsolve-modes", default="regular,levelsolve")
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--sim-time", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--mpax-iteration-limit", type=int, default=None)
    parser.add_argument("--mpax-eps-abs", type=float, default=1e-3)
    parser.add_argument("--mpax-eps-rel", type=float, default=1e-3)
    parser.add_argument("--mpax-termination-evaluation-frequency", type=int, default=100)
    parser.add_argument("--mpax-l-inf-ruiz-iterations", type=int, default=10)
    parser.add_argument("--mpax-pock-chambolle-alpha", type=float, default=1.0)
    parser.add_argument("--mpax-regularization", type=float, default=0.0)
    parser.add_argument("--mpax-unroll", action="store_true")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-budget", type=int, default=256)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-output-npz",
        action="store_true",
        help="Do not write per-batch raw rollout arrays.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_rollout"),
    )
    parser.add_argument(
        "--csv-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_rollout.csv"),
    )
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_rollout_summary.png"),
    )
    args = parser.parse_args()
    if args.mpax_iteration_limit is None:
        args.mpax_iteration_limit = args.max_iter
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    batch_sizes = _parse_ints(args.batch_sizes)
    dtypes = _parse_strings(args.dtypes)
    qp_solvers = _parse_strings(args.qp_solvers)
    for qp_solver in qp_solvers:
        if qp_solver not in {"jax_osqp", "mpax"}:
            raise ValueError(f"unsupported QP solver: {qp_solver}")
    qdldl_backends = _parse_strings(args.qdldl_backends)
    backend_pairs = _parse_backend_pairs(args.qdldl_backend_pairs, qdldl_backends)
    qdldl_variant_names = _parse_strings(args.qdldl_variants) if args.qdldl_variants else []
    unknown_qdldl_variants = sorted(set(qdldl_variant_names) - set(QDLDL_VARIANTS))
    if unknown_qdldl_variants:
        raise ValueError(f"unknown QDLDL variants: {unknown_qdldl_variants}")
    modes = _parse_strings(args.group_modes)
    levelsolve_modes = _parse_strings(args.levelsolve_modes)
    for levelsolve_mode in levelsolve_modes:
        if levelsolve_mode not in {"regular", "levelsolve"}:
            raise ValueError(f"unsupported levelsolve mode: {levelsolve_mode}")
    for dtype in dtypes:
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
                for mode in modes:
                    if mode not in {"grouped", "ungrouped"}:
                        raise ValueError(f"unsupported group mode: {mode}")
                    group_repeated_stages = mode == "grouped"
                    for batch_size in batch_sizes:
                        batch_failed = False
                        for repeat_index in range(args.repeat):
                            row = run_batch(
                                args,
                                batch_size=batch_size,
                                dtype=dtype,
                                qp_solver=qp_solver,
                                qdldl_factor_backend=qdldl_factor_backend,
                                qdldl_solve_backend=qdldl_solve_backend,
                                group_repeated_stages=group_repeated_stages,
                                level_scheduled_solve=level_scheduled_solve,
                                transpose_work=transpose_work,
                                segmented=segmented,
                                repeat_index=repeat_index,
                                qdldl_variant=qdldl_variant,
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
                                    f"Skipping larger batch sizes for solver={qp_solver} dtype={dtype} "
                                    f"mode={row['mode']} after failed batch={batch_size}: "
                                    f"{','.join(str(batch) for batch in remaining)}"
                                )
                            break
    print(f"Wrote {args.csv_path}")
    print(f"Wrote {args.plot_path}")


if __name__ == "__main__":
    main()
