#!/usr/bin/env python3
"""Compare fixed-pattern JAX QDLDL layout/schedule variants."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np
import qdldl
import scipy.sparse as sp

from benchmarks.problems import make_mpc_kkt, sample_kkt_values, structural_density
from warpmpc.jax_qdldl import (
    build_qdldl_plan,
    compile_qdldl_variant,
)


VARIANTS = {
    "baseline": {},
    "transpose": {"transpose_work": True},
    "segmented": {"segmented": True},
    "transpose+segmented": {"transpose_work": True, "segmented": True},
    "transpose+segmented+levelsolve": {
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
    "factor-warp+solve-jax:transpose+segmented": {
        "factor_backend": "warp",
        "solve_backend": "jax",
        "transpose_work": True,
        "segmented": True,
    },
    "factor-warp+solve-jax:transpose+segmented+levelsolve": {
        "factor_backend": "warp",
        "solve_backend": "jax",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
    "factor-warp+solve-warp:transpose": {
        "factor_backend": "warp",
        "solve_backend": "warp",
        "transpose_work": True,
    },
    "factor-warp+solve-warp:transpose+segmented": {
        "factor_backend": "warp",
        "solve_backend": "warp",
        "transpose_work": True,
        "segmented": True,
    },
    "factor-warp+solve-warp:transpose+segmented+levelsolve": {
        "factor_backend": "warp",
        "solve_backend": "warp",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
}


MARKERS = {
    "baseline": "o",
    "transpose": "s",
    "segmented": "D",
    "transpose+segmented": "X",
    "transpose+segmented+levelsolve": ">",
    "factor-warp+solve-jax:transpose+segmented": "P",
    "factor-warp+solve-jax:transpose+segmented+levelsolve": "*",
    "factor-warp+solve-warp:transpose": "h",
    "factor-warp+solve-warp:transpose+segmented": "^",
    "factor-warp+solve-warp:transpose+segmented+levelsolve": "v",
}


@dataclass(frozen=True)
class CpuQdldlTiming:
    factor_s_per_system: float
    solve_s_per_system: float
    update_solve_s_per_system: float

    @property
    def factor_systems_per_s(self) -> float:
        return 1.0 / self.factor_s_per_system

    @property
    def solve_systems_per_s(self) -> float:
        return 1.0 / self.solve_s_per_system

    @property
    def update_solve_systems_per_s(self) -> float:
        return 1.0 / self.update_solve_s_per_system


@dataclass(frozen=True)
class VariantResult:
    variant: str
    dtype: str
    batch_size: int
    estimated_bytes: int
    factor_s: float
    solve_s: float
    factor_solve_s: float
    baseline_factor_solve_s: float | None
    cpu_timing: CpuQdldlTiming | None = None

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.factor_solve_s

    @property
    def factor_systems_per_s(self) -> float:
        return self.batch_size / self.factor_s

    @property
    def solve_systems_per_s(self) -> float:
        return self.batch_size / self.solve_s

    @property
    def speedup_vs_baseline(self) -> float | None:
        if self.baseline_factor_solve_s is None:
            return None
        return self.baseline_factor_solve_s / self.factor_solve_s

    @property
    def cpu_throughput(self) -> float | None:
        return None if self.cpu_timing is None else self.cpu_timing.update_solve_systems_per_s


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_batches(text: str) -> list[int]:
    return [int(part) for part in _parse_csv_list(text)]


def _parse_dtypes(text: str) -> list[np.dtype]:
    return [np.dtype(part) for part in _parse_csv_list(text)]


def _estimate_bytes(batch: int, n: int, nnz_a: int, nnz_l: int, dtype: np.dtype) -> int:
    item = np.dtype(dtype).itemsize
    arrays = nnz_a + nnz_l + 5 * n
    return batch * arrays * item


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def _compile_and_warm(fn, *args):
    jax.block_until_ready(args)
    compiled = fn.lower(*args).compile()
    out = compiled(*args)
    jax.block_until_ready(out)
    return compiled, out


def _time_compiled(compiled, *args, repeat: int) -> tuple[float, object]:
    start = time.perf_counter()
    for _ in range(repeat):
        out = compiled(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - start) / repeat, out


def _benchmark_cpu_qdldl(
    plan,
    values: np.ndarray,
    rhs: np.ndarray,
    repeat: int,
) -> CpuQdldlTiming:
    matrices = [
        sp.csc_matrix((np.asarray(row), plan.a_indices, plan.a_indptr), shape=(plan.n, plan.n))
        for row in values
    ]
    solver = qdldl.Solver(matrices[0], upper=True)
    solver.solve(rhs[0])

    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for matrix in matrices:
            solver.update(matrix, upper=True)
            count += 1
    factor_s_per_system = (time.perf_counter() - start) / count

    solvers = [qdldl.Solver(matrix, upper=True) for matrix in matrices]
    for solver_i, rhs_i in zip(solvers, rhs, strict=True):
        solver_i.solve(rhs_i)
    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for solver_i, rhs_i in zip(solvers, rhs, strict=True):
            solver_i.solve(rhs_i)
            count += 1
    solve_s_per_system = (time.perf_counter() - start) / count

    solver = qdldl.Solver(matrices[0], upper=True)
    solver.solve(rhs[0])
    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for i, matrix in enumerate(matrices):
            solver.update(matrix, upper=True)
            solver.solve(rhs[i])
            count += 1
    update_solve_s_per_system = (time.perf_counter() - start) / count
    return CpuQdldlTiming(
        factor_s_per_system=factor_s_per_system,
        solve_s_per_system=solve_s_per_system,
        update_solve_s_per_system=update_solve_s_per_system,
    )


def _values_for_layout(plan, values: np.ndarray, layout: str) -> np.ndarray:
    if layout == "original_batch":
        return values
    if layout == "original_symbolic":
        return values.T
    raise ValueError(f"unknown values layout {layout}")


def _rhs_for_layout(rhs: np.ndarray, layout: str) -> np.ndarray:
    if layout == "batch":
        return rhs
    if layout == "symbolic":
        return rhs.T
    raise ValueError(f"unknown rhs layout {layout}")


def _to_batch(array, layout: str) -> np.ndarray:
    out = jax.device_get(array)
    return out.T if layout == "symbolic" else out


def _check_variant(plan, compiled, values: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    values_in = _values_for_layout(plan, values, compiled.values_layout)
    rhs_in = _rhs_for_layout(rhs, compiled.rhs_layout)
    x, lx, d = compiled.factor_and_solve(jnp.asarray(values_in), jnp.asarray(rhs_in))
    x = _to_batch(x, compiled.rhs_layout)
    lx = _to_batch(lx, compiled.rhs_layout)
    d = _to_batch(d, compiled.rhs_layout)

    solver = qdldl.Solver(
        sp.csc_matrix((values[0], plan.a_indices, plan.a_indptr), shape=(plan.n, plan.n)),
        upper=True,
    )

    max_lx = 0.0
    max_d = 0.0
    max_x = 0.0
    for i in range(values.shape[0]):
        matrix = sp.csc_matrix(
            (values[i], plan.a_indices, plan.a_indptr), shape=(plan.n, plan.n)
        )
        solver.update(matrix, upper=True)
        cpu_l, cpu_d, _ = solver.factors()
        cpu_x = solver.solve(rhs[i])
        max_lx = max(max_lx, float(np.max(np.abs(cpu_l.data - lx[i]))))
        max_d = max(max_d, float(np.max(np.abs(cpu_d - d[i]))))
        max_x = max(max_x, float(np.max(np.abs(cpu_x - x[i]))))
    return {"max_abs_lx": max_lx, "max_abs_d": max_d, "max_abs_x": max_x}


def _write_csv(path: pathlib.Path, rows: list[VariantResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "dtype",
                "batch_size",
                "estimated_gib",
                "factor_s",
                "solve_s",
                "factor_solve_s",
                "factor_systems_per_s",
                "solve_systems_per_s",
                "systems_per_s",
                "speedup_vs_baseline",
                "cpu_qdldl_factor_s_per_system",
                "cpu_qdldl_solve_s_per_system",
                "cpu_qdldl_update_solve_s_per_system",
                "cpu_qdldl_factor_systems_per_s",
                "cpu_qdldl_solve_systems_per_s",
                "cpu_qdldl_update_solve_systems_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            cpu = row.cpu_timing
            writer.writerow(
                {
                    "variant": row.variant,
                    "dtype": row.dtype,
                    "batch_size": row.batch_size,
                    "estimated_gib": row.estimated_bytes / 1024**3,
                    "factor_s": row.factor_s,
                    "solve_s": row.solve_s,
                    "factor_solve_s": row.factor_solve_s,
                    "factor_systems_per_s": row.factor_systems_per_s,
                    "solve_systems_per_s": row.solve_systems_per_s,
                    "systems_per_s": row.systems_per_s,
                    "speedup_vs_baseline": row.speedup_vs_baseline,
                    "cpu_qdldl_factor_s_per_system": None if cpu is None else cpu.factor_s_per_system,
                    "cpu_qdldl_solve_s_per_system": None if cpu is None else cpu.solve_s_per_system,
                    "cpu_qdldl_update_solve_s_per_system": None if cpu is None else cpu.update_solve_s_per_system,
                    "cpu_qdldl_factor_systems_per_s": None if cpu is None else cpu.factor_systems_per_s,
                    "cpu_qdldl_solve_systems_per_s": None if cpu is None else cpu.solve_systems_per_s,
                    "cpu_qdldl_update_solve_systems_per_s": row.cpu_throughput,
                }
            )


def _plot(path: pathlib.Path, rows: list[VariantResult], cpu_throughput: float | None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 5), squeeze=False)
    for ax, dtype in zip(axes[0], dtypes, strict=True):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        for variant in VARIANTS:
            variant_rows = [row for row in dtype_rows if row.variant == variant]
            if not variant_rows:
                continue
            variant_rows.sort(key=lambda row: row.batch_size)
            ax.plot(
                [row.batch_size for row in variant_rows],
                [row.systems_per_s for row in variant_rows],
                marker=MARKERS.get(variant, "o"),
                markersize=7 if variant != "transpose+segmented" else 10,
                linewidth=1.8,
                label=variant,
            )
        if cpu_throughput is not None:
            ax.axhline(
                cpu_throughput,
                linestyle="--",
                color="black",
                linewidth=1.8,
                label="CPU qdldl update+solve",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput [systems/s]")
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("QDLDL variant throughput on fixed MPC KKT pattern")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_component(
    path: pathlib.Path,
    rows: list[VariantResult],
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 5), squeeze=False)
    for ax, dtype in zip(axes[0], dtypes, strict=True):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        for variant in VARIANTS:
            variant_rows = [row for row in dtype_rows if row.variant == variant]
            if not variant_rows:
                continue
            variant_rows.sort(key=lambda row: row.batch_size)
            ax.plot(
                [row.batch_size for row in variant_rows],
                [getattr(row, metric) for row in variant_rows],
                marker=MARKERS.get(variant, "o"),
                markersize=7 if variant != "transpose+segmented" else 10,
                linewidth=1.8,
                label=variant,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_speedups(path: pathlib.Path, rows: list[VariantResult]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 5), squeeze=False)
    for ax, dtype in zip(axes[0], dtypes, strict=True):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        variants = [variant for variant in VARIANTS if any(row.variant == variant for row in dtype_rows)]
        best_rows: list[VariantResult] = []
        for variant in variants:
            variant_rows = [row for row in dtype_rows if row.variant == variant]
            best_rows.append(max(variant_rows, key=lambda row: row.systems_per_s))

        xs = np.arange(len(best_rows))
        speeds = [
            1.0 if row.speedup_vs_baseline is None else row.speedup_vs_baseline
            for row in best_rows
        ]
        bars = ax.bar(xs, speeds, alpha=0.8)
        for bar, row, speed in zip(bars, best_rows, speeds, strict=True):
            bar.set_label(row.variant)
            ax.plot(
                bar.get_x() + bar.get_width() / 2,
                speed,
                marker=MARKERS.get(row.variant, "o"),
                color="black",
                markersize=7 if row.variant != "transpose+segmented" else 10,
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                speed,
                f"{speed:.2f}x\nb={row.batch_size}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([row.variant for row in best_rows], rotation=35, ha="right")
        ax.set_ylabel("Speedup vs baseline at same batch")
        ax.set_title(dtype)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Best measured speedup over baseline by QDLDL variant")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--nu", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=512)
    parser.add_argument("--cpu-repeat", type=int, default=3)
    parser.add_argument("--segment-budget", type=int, default=64)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variation", type=float, default=0.25)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--plot-path", default="results/qdldl_factorization/throughput_mpc_qdldl_variants.png")
    parser.add_argument(
        "--factor-plot-path",
        default="results/qdldl_factorization/factor_throughput_mpc_qdldl_variants.png",
    )
    parser.add_argument(
        "--solve-plot-path",
        default="results/qdldl_factorization/solve_throughput_mpc_qdldl_variants.png",
    )
    parser.add_argument("--speedup-plot-path", default="results/qdldl_factorization/speedup_mpc_qdldl_variants.png")
    parser.add_argument("--csv-path", default="results/qdldl_factorization/throughput_mpc_qdldl_variants.csv")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-cpu", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(_parse_csv_list(args.variants)) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    variant_names = _parse_csv_list(args.variants)
    if "baseline" in variant_names:
        variant_names = ["baseline"] + [name for name in variant_names if name != "baseline"]
    else:
        variant_names = ["baseline"] + variant_names

    problem = make_mpc_kkt(
        nx=args.nx,
        nu=args.nu,
        horizon=args.horizon,
        rho=args.rho,
        sigma=args.sigma,
        extra_kkt_density=0.0,
        seed=args.seed,
    )
    kkt = problem.kkt_upper
    plan = build_qdldl_plan(kkt, upper=True)
    print("Devices:", jax.devices())
    print(
        "KKT:",
        f"dim={kkt.shape[0]}",
        f"upper_nnz={kkt.nnz}",
        f"upper_density={100 * structural_density(kkt):.2f}%",
        f"nnz_L={plan.nnz_l}",
        f"max_row_nnz={plan.max_row_nnz}",
        f"max_col_nnz={plan.max_col_nnz}",
    )

    results: list[VariantResult] = []
    baseline_times: dict[tuple[str, int], float] = {}
    max_bytes = int(args.max_device_gb * 1024**3)
    cpu_timing = None
    if not args.no_cpu:
        cpu_values = sample_kkt_values(
            kkt,
            batch_size=args.cpu_samples,
            seed=args.seed + 50,
            variation=args.variation,
            dtype=np.float64,
        )
        cpu_rhs = np.random.default_rng(args.seed + 51).standard_normal(
            (args.cpu_samples, plan.n)
        )
        cpu_timing = _benchmark_cpu_qdldl(
            plan, cpu_values, cpu_rhs, repeat=args.cpu_repeat
        )
        print(
            "CPU qdldl:",
            f"samples={args.cpu_samples}",
            f"factor={cpu_timing.factor_systems_per_s:.2f}/s",
            f"solve={cpu_timing.solve_systems_per_s:.2f}/s",
            f"update+solve={cpu_timing.update_solve_systems_per_s:.2f}/s",
        )

    for dtype in _parse_dtypes(args.dtypes):
        jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
        for variant_name in variant_names:
            options = dict(VARIANTS[variant_name])
            if options.get("segmented"):
                options["segment_budget"] = args.segment_budget
                options["segment_strategy"] = args.segment_strategy
            compiled = compile_qdldl_variant(plan, dtype=dtype, **options)
            print(
                f"Variant {variant_name} dtype={dtype.name}:",
                f"values_layout={compiled.values_layout}",
                f"rhs_layout={compiled.rhs_layout}",
            )

            if not args.no_check:
                check_values = sample_kkt_values(
                    kkt,
                    batch_size=2,
                    seed=args.seed + 10,
                    variation=args.variation,
                    dtype=dtype,
                )
                check_rhs = np.random.default_rng(args.seed + 11).standard_normal(
                    (2, plan.n)
                ).astype(dtype)
                errs = _check_variant(plan, compiled, check_values, check_rhs)
                print(
                    "  check:",
                    f"max_abs_lx={errs['max_abs_lx']:.3e}",
                    f"max_abs_d={errs['max_abs_d']:.3e}",
                    f"max_abs_x={errs['max_abs_x']:.3e}",
                )

            for batch in _parse_batches(args.batch_sizes):
                estimate = _estimate_bytes(batch, plan.n, plan.nnz_a, plan.nnz_l, dtype)
                if estimate > max_bytes:
                    print(f"  {batch}: skipped over {_format_gb(estimate)}")
                    break

                values = sample_kkt_values(
                    kkt,
                    batch_size=batch,
                    seed=args.seed + batch,
                    variation=args.variation,
                    dtype=dtype,
                )
                rhs = np.random.default_rng(args.seed + 1000 + batch).standard_normal(
                    (batch, plan.n)
                ).astype(dtype)
                values_in = _values_for_layout(plan, values, compiled.values_layout)
                rhs_in = _rhs_for_layout(rhs, compiled.rhs_layout)
                values_jax = jnp.asarray(values_in)
                rhs_jax = jnp.asarray(rhs_in)

                try:
                    factor_exe, factor_out = _compile_and_warm(compiled.factor, values_jax)
                    lx, _, dinv = factor_out
                    solve_exe, _ = _compile_and_warm(compiled.solve, lx, dinv, rhs_jax)
                    factor_solve_exe, _ = _compile_and_warm(
                        compiled.factor_and_solve, values_jax, rhs_jax
                    )
                    factor_s, factor_out = _time_compiled(
                        factor_exe, values_jax, repeat=args.repeat
                    )
                    lx, _, dinv = factor_out
                    solve_s, _ = _time_compiled(
                        solve_exe, lx, dinv, rhs_jax, repeat=args.repeat
                    )
                    factor_solve_s, _ = _time_compiled(
                        factor_solve_exe, values_jax, rhs_jax, repeat=args.repeat
                    )
                except Exception as exc:
                    print(f"  {batch}: failed {type(exc).__name__}: {exc}")
                    break

                key = (dtype.name, batch)
                if variant_name == "baseline":
                    baseline_times[key] = factor_solve_s
                baseline_time = baseline_times.get(key)
                row = VariantResult(
                    variant=variant_name,
                    dtype=dtype.name,
                    batch_size=batch,
                    estimated_bytes=estimate,
                    factor_s=factor_s,
                    solve_s=solve_s,
                    factor_solve_s=factor_solve_s,
                    baseline_factor_solve_s=baseline_time,
                    cpu_timing=cpu_timing,
                )
                results.append(row)
                speed = row.speedup_vs_baseline
                speed_text = "-" if speed is None else f"{speed:.3f}x"
                print(
                    f"  {batch}: factor={factor_s:.6f}s solve={solve_s:.6f}s "
                    f"total={factor_solve_s:.6f}s throughput={row.systems_per_s:.2f}/s "
                    f"speedup={speed_text}"
                )

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    factor_plot_path = pathlib.Path(args.factor_plot_path)
    solve_plot_path = pathlib.Path(args.solve_plot_path)
    speedup_plot_path = pathlib.Path(args.speedup_plot_path)
    for output_path in (csv_path, plot_path, factor_plot_path, solve_plot_path, speedup_plot_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, results)
    _plot(plot_path, results, None if cpu_timing is None else cpu_timing.update_solve_systems_per_s)
    _plot_component(
        factor_plot_path,
        results,
        metric="factor_systems_per_s",
        title="QDLDL factorization-only throughput on fixed MPC KKT pattern",
        ylabel="Factorizations/s",
    )
    _plot_component(
        solve_plot_path,
        results,
        metric="solve_systems_per_s",
        title="QDLDL backsolve-only throughput on fixed MPC KKT pattern",
        ylabel="Backsolves/s",
    )
    _plot_speedups(speedup_plot_path, results)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {factor_plot_path}")
    print(f"Wrote {solve_plot_path}")
    print(f"Wrote {speedup_plot_path}")


if __name__ == "__main__":
    main()
