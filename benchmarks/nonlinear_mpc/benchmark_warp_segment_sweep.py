#!/usr/bin/env python3
"""Warp QDLDL segmentation budget sweep for nonlinear MPC benchmarks."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
import statistics
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_BUDGETS = "16,24,32,48,64,96,128,192,256,384,512"


@dataclasses.dataclass(frozen=True)
class ProblemConfig:
    script: pathlib.Path
    title: str
    throughput_key: str
    throughput_label: str


PROBLEMS = {
    "cartpole": ProblemConfig(
        script=ROOT / "benchmarks" / "nonlinear_mpc" / "benchmark_cartpole_quadratic_mpc.py",
        title="cartpole quadratic MPC",
        throughput_key="sqp_iterations_per_s",
        throughput_label="SQP iterations / s",
    ),
    "crazyflie": ProblemConfig(
        script=ROOT / "benchmarks" / "nonlinear_mpc" / "benchmark_crazyflie_sqp_rollout.py",
        title="Crazyflie SQP rollout",
        throughput_key="rti_steps_per_s",
        throughput_label="SQP-RTI steps / s",
    ),
    "humanoid": ProblemConfig(
        script=ROOT / "benchmarks" / "nonlinear_mpc" / "benchmark_humanoid_mpc.py",
        title="humanoid MPC",
        throughput_key="rti_steps_per_s",
        throughput_label="SQP-RTI steps / s",
    ),
}


FIELDNAMES = [
    "problem",
    "label",
    "batch_size",
    "dtype",
    "repeat",
    "qp_solver",
    "qdldl_backend",
    "qdldl_factor_backend",
    "qdldl_solve_backend",
    "mode",
    "group_repeated_stages",
    "level_scheduled_solve",
    "qdldl_variant",
    "transpose_work",
    "segmented",
    "segment_budget",
    "segment_strategy",
    "status",
    "throughput_metric",
    "throughput",
    "speedup_vs_no_segmentation",
    "elapsed_s",
    "solve_s",
    "rti_steps_per_s",
    "sqp_iterations_per_s",
    "closed_loop_steps_per_s",
    "total_rti_steps",
    "total_sqp_iterations",
    "n_variables",
    "n_constraints",
    "nnz_p",
    "nnz_a",
    "compile_setup_s",
    "warmup_compile_and_run_s",
    "compile_s",
    "total_compile_s",
    "solver_setup_s",
    "initialization_s",
    "rollout_compile_s",
    "wall_s",
    "return_mean",
    "rollout_success_rate",
    "final_position_rms",
    "final_state_rms",
    "max_violation",
    "mean_qp_prim",
    "mean_qp_dual",
    "raw_output_dir",
    "raw_benchmark_csv",
    "raw_benchmark_plot",
    "raw_invocation_log",
    "summary_json",
    "log_path",
    "returncode",
    "error",
]


def _parse_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _point_paths(
    output_dir: pathlib.Path,
    *,
    segmented: bool,
    budget: int | None,
) -> tuple[str, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    label = f"segment_budget_{budget}" if segmented else "no_segmentation"
    raw_dir = output_dir / "raw" / label
    return (
        label,
        raw_dir,
        raw_dir / "benchmark.csv",
        raw_dir / "summary.png",
        raw_dir / "invocation.log",
    )


def _run_point(
    args: argparse.Namespace,
    benchmark_args: list[str],
    config: ProblemConfig,
    *,
    segmented: bool,
    budget: int | None,
) -> list[dict[str, object]]:
    label, raw_dir, raw_csv, raw_plot, invocation_log = _point_paths(
        args.output_dir,
        segmented=segmented,
        budget=budget,
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    segment_budget = budget if budget is not None else args.baseline_segment_budget
    cmd = [
        args.benchmark_python,
        str(config.script),
        *benchmark_args,
        "--batch-sizes",
        str(args.batch_size),
        "--dtypes",
        args.dtypes,
        "--qp-solvers",
        args.qp_solvers,
        "--qdldl-backend-pairs",
        args.qdldl_backend_pairs,
        "--group-modes",
        args.group_modes,
        "--levelsolve-modes",
        args.levelsolve_modes,
        "--segment-budget",
        str(segment_budget),
        "--segment-strategy",
        args.segment_strategy,
        "--output-dir",
        str(raw_dir),
        "--csv-path",
        str(raw_csv),
        "--plot-path",
        str(raw_plot),
    ]
    if not segmented:
        cmd.append("--no-segmented")

    print(f"=== {args.problem} {label} ===", flush=True)
    print("Command:", " ".join(cmd), flush=True)
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
    invocation_log.write_text(proc.stdout)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)

    raw_rows = _read_csv(raw_csv)
    if not raw_rows:
        return [
            {
                "problem": args.problem,
                "label": label,
                "batch_size": args.batch_size,
                "segmented": str(segmented).lower(),
                "segment_budget": "" if budget is None else budget,
                "segment_strategy": args.segment_strategy,
                "status": "failed",
                "throughput_metric": config.throughput_key,
                "raw_output_dir": str(raw_dir),
                "raw_benchmark_csv": str(raw_csv),
                "raw_benchmark_plot": str(raw_plot),
                "raw_invocation_log": str(invocation_log),
                "returncode": proc.returncode,
                "wall_s": wall_s,
                "error": _tail(proc.stdout),
            }
        ]

    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        throughput = _float_or_none(raw.get(config.throughput_key))
        status = raw.get("status") or ("ok" if throughput is not None else "failed")
        row: dict[str, object] = {
            "problem": args.problem,
            "label": label,
            "batch_size": raw.get("batch_size", args.batch_size),
            "dtype": raw.get("dtype", ""),
            "repeat": raw.get("repeat", ""),
            "qp_solver": raw.get("qp_solver", ""),
            "qdldl_backend": raw.get("qdldl_backend", ""),
            "qdldl_factor_backend": raw.get("qdldl_factor_backend", ""),
            "qdldl_solve_backend": raw.get("qdldl_solve_backend", ""),
            "mode": raw.get("mode", ""),
            "group_repeated_stages": raw.get("group_repeated_stages", ""),
            "level_scheduled_solve": raw.get("level_scheduled_solve", ""),
            "qdldl_variant": raw.get("qdldl_variant", ""),
            "transpose_work": raw.get("transpose_work", ""),
            "segmented": str(segmented).lower(),
            "segment_budget": "" if budget is None else budget,
            "segment_strategy": args.segment_strategy,
            "status": status,
            "throughput_metric": config.throughput_key,
            "throughput": "" if throughput is None else throughput,
            "raw_output_dir": str(raw_dir),
            "raw_benchmark_csv": str(raw_csv),
            "raw_benchmark_plot": str(raw_plot),
            "raw_invocation_log": str(invocation_log),
            "summary_json": raw.get("summary_json", ""),
            "log_path": raw.get("log_path", ""),
            "returncode": proc.returncode,
            "wall_s": raw.get("wall_s", wall_s),
            "error": raw.get("error", ""),
        }
        for key in FIELDNAMES:
            if key not in row and key in raw:
                row[key] = raw[key]
        rows.append(row)
    return rows


def _attach_speedups(rows: list[dict[str, object]]) -> None:
    baselines: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok" or _is_true(row.get("segmented")):
            continue
        throughput = _float_or_none(row.get("throughput"))
        if throughput is None:
            continue
        dtype = str(row.get("dtype", ""))
        baselines.setdefault(dtype, []).append(throughput)

    baseline_means = {
        dtype: statistics.fmean(values)
        for dtype, values in baselines.items()
        if values
    }
    for row in rows:
        row["speedup_vs_no_segmentation"] = ""
        if row.get("status") != "ok":
            continue
        throughput = _float_or_none(row.get("throughput"))
        baseline = baseline_means.get(str(row.get("dtype", "")))
        if throughput is None or baseline is None or baseline <= 0.0:
            continue
        row["speedup_vs_no_segmentation"] = throughput / baseline


def _plot(
    path: pathlib.Path,
    rows: list[dict[str, object]],
    config: ProblemConfig,
    budgets: list[int],
) -> bool:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return False

    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    dtypes = sorted({str(row.get("dtype", "")) for row in ok_rows})
    for idx, dtype in enumerate(dtypes):
        color = colors[idx % len(colors)]
        baseline_values = [
            _float_or_none(row.get("throughput"))
            for row in ok_rows
            if str(row.get("dtype", "")) == dtype and not _is_true(row.get("segmented"))
        ]
        baseline_values = [value for value in baseline_values if value is not None]
        if baseline_values:
            baseline = statistics.fmean(baseline_values)
            ax.axhline(
                baseline,
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
                label=f"{dtype} no segmentation",
            )

        grouped: dict[int, list[float]] = {}
        for row in ok_rows:
            if str(row.get("dtype", "")) != dtype or not _is_true(row.get("segmented")):
                continue
            budget_text = row.get("segment_budget")
            throughput = _float_or_none(row.get("throughput"))
            if budget_text in (None, "") or throughput is None:
                continue
            grouped.setdefault(int(budget_text), []).append(throughput)
        xs = sorted(grouped)
        if xs:
            ys = [statistics.fmean(grouped[x]) for x in xs]
            ax.plot(
                xs,
                ys,
                color=color,
                marker="o",
                linewidth=1.8,
                label=f"{dtype} segmented",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(budget) for budget in budgets], rotation=35, ha="right")
    ax.set_xlabel("segmentation budget")
    ax.set_ylabel(config.throughput_label)
    ax.set_title(f"{config.title}, batch={rows[0].get('batch_size', '')}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_outputs(
    csv_path: pathlib.Path,
    plot_path: pathlib.Path,
    rows: list[dict[str, object]],
    config: ProblemConfig,
    budgets: list[int],
) -> None:
    _attach_speedups(rows)
    _write_csv(csv_path, rows)
    plotted = _plot(plot_path, rows, config, budgets)
    print(f"Wrote {csv_path}", flush=True)
    if plotted:
        print(f"Wrote {plot_path}", flush=True)
    else:
        print(f"No successful rows yet; skipped {plot_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-segmentation and segmented Warp backend sweeps for nonlinear MPC.",
    )
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--benchmark-python", default=sys.executable)
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--qp-solvers", default="jax_osqp")
    parser.add_argument("--qdldl-backend-pairs", default="warp:warp")
    parser.add_argument("--group-modes", default="grouped")
    parser.add_argument("--levelsolve-modes", default="regular")
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/warp_segment_sweep"),
    )
    parser.add_argument(
        "--csv-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/warp_segment_sweep.csv"),
    )
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/warp_segment_sweep.png"),
    )
    args, benchmark_args = parser.parse_known_args()
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]

    budgets = _parse_ints(args.budgets)
    args.baseline_segment_budget = budgets[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PROBLEMS[args.problem]

    rows: list[dict[str, object]] = []
    point_rows = _run_point(
        args,
        benchmark_args,
        config,
        segmented=False,
        budget=None,
    )
    rows.extend(point_rows)
    _write_outputs(args.csv_path, args.plot_path, rows, config, budgets)
    if any(row.get("status") != "ok" for row in point_rows) and not args.continue_on_failure:
        raise SystemExit(1)

    for budget in budgets:
        point_rows = _run_point(
            args,
            benchmark_args,
            config,
            segmented=True,
            budget=budget,
        )
        rows.extend(point_rows)
        _write_outputs(args.csv_path, args.plot_path, rows, config, budgets)
        if any(row.get("status") != "ok" for row in point_rows) and not args.continue_on_failure:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
