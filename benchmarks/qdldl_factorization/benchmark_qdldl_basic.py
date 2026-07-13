#!/usr/bin/env python3
"""Benchmark fixed-pattern batched JAX QDLDL against CPU qdldl."""

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
    compile_qdldl,
    verify_against_qdldl,
)


@dataclass(frozen=True)
class GpuResult:
    dtype: str
    batch_size: int
    estimated_bytes: int
    factor_s: float
    solve_s: float
    factor_solve_s: float

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.factor_solve_s


def _parse_batches(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_dtypes(text: str) -> list[np.dtype]:
    return [np.dtype(part.strip()) for part in text.split(",") if part.strip()]


def _estimate_bytes(batch: int, n: int, nnz_a: int, nnz_l: int, dtype: np.dtype) -> int:
    item = np.dtype(dtype).itemsize
    arrays = nnz_a + nnz_l + 5 * n
    return batch * arrays * item


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def _compile_and_warm(fn, *args):
    jax.block_until_ready(args)
    lowered = fn.lower(*args)
    compiled = lowered.compile()
    out = compiled(*args)
    jax.block_until_ready(out)
    return compiled, out


def _time_compiled(compiled, *args, repeat: int) -> tuple[float, object]:
    start = time.perf_counter()
    for _ in range(repeat):
        out = compiled(*args)
        jax.block_until_ready(out)
    elapsed = (time.perf_counter() - start) / repeat
    return elapsed, out


def _benchmark_cpu_qdldl(
    plan,
    values: np.ndarray,
    rhs: np.ndarray,
    repeat: int,
) -> float:
    matrices = [
        sp.csc_matrix((np.asarray(row), plan.a_indices, plan.a_indptr), shape=(plan.n, plan.n))
        for row in values
    ]
    solver = qdldl.Solver(matrices[0], upper=True)
    solver.solve(rhs[0])

    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for i, matrix in enumerate(matrices):
            if i == 0:
                solver.update(matrix, upper=True)
            else:
                solver.update(matrix, upper=True)
            solver.solve(rhs[i])
            count += 1
    return count / (time.perf_counter() - start)


def _write_csv(path: pathlib.Path, rows: list[GpuResult], cpu_throughput: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dtype",
                "batch_size",
                "estimated_gib",
                "factor_s",
                "solve_s",
                "factor_solve_s",
                "systems_per_s",
                "cpu_qdldl_update_solve_systems_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dtype": row.dtype,
                    "batch_size": row.batch_size,
                    "estimated_gib": row.estimated_bytes / 1024**3,
                    "factor_s": row.factor_s,
                    "solve_s": row.solve_s,
                    "factor_solve_s": row.factor_solve_s,
                    "systems_per_s": row.systems_per_s,
                    "cpu_qdldl_update_solve_systems_per_s": cpu_throughput,
                }
            )


def _plot(path: pathlib.Path, rows: list[GpuResult], cpu_throughput: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for dtype in sorted({row.dtype for row in rows}):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        dtype_rows.sort(key=lambda row: row.batch_size)
        ax.plot(
            [row.batch_size for row in dtype_rows],
            [row.systems_per_s for row in dtype_rows],
            marker="o",
            linewidth=2,
            label=f"JAX GPU {dtype}",
        )

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
    ax.set_title("Fixed-pattern MPC KKT QDLDL throughput")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
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
    parser.add_argument("--batch-sizes", default="1,8,32,128,512,2048,10000,20000,50000,100000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=512)
    parser.add_argument("--cpu-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variation", type=float, default=0.25)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--plot-path", default="results/qdldl_factorization/throughput_mpc_qdldl.png")
    parser.add_argument("--csv-path", default="results/qdldl_factorization/throughput_mpc_qdldl.csv")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

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
    print("Devices:", jax.devices())
    print(
        "KKT:",
        f"dim={kkt.shape[0]}",
        f"nvar={problem.n_variables}",
        f"m={problem.n_constraints}",
        f"upper_nnz={kkt.nnz}",
        f"upper_density={100 * structural_density(kkt):.2f}%",
    )

    t0 = time.perf_counter()
    plan = build_qdldl_plan(kkt, upper=True)
    print(
        "Plan:",
        f"nnz_A={plan.nnz_a}",
        f"nnz_L={plan.nnz_l}",
        f"max_row_nnz={plan.max_row_nnz}",
        f"max_col_nnz={plan.max_col_nnz}",
        f"symbolic_time={time.perf_counter() - t0:.3f}s",
    )

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
    cpu_throughput = _benchmark_cpu_qdldl(
        plan, cpu_values, cpu_rhs, repeat=args.cpu_repeat
    )
    print(
        "CPU qdldl:",
        "mode=update+solve",
        f"samples={args.cpu_samples}",
        f"systems/s={cpu_throughput:.2f}",
    )

    results: list[GpuResult] = []
    max_bytes = int(args.max_device_gb * 1024**3)
    batches = _parse_batches(args.batch_sizes)
    for dtype in _parse_dtypes(args.dtypes):
        jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
        print(f"Benchmark dtype={dtype.name}: fixed MPC structural pattern")
        compiled = compile_qdldl(plan, dtype=dtype)

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
            t_check = time.perf_counter()
            errors = verify_against_qdldl(
                plan,
                compiled=compiled,
                a_values=check_values,
                rhs=check_rhs,
                rtol=1e-4 if dtype == np.float32 else 1e-10,
                atol=1e-4 if dtype == np.float32 else 1e-10,
            )
            print(
                "Check:",
                f"max_abs_lx={errors['max_abs_lx']:.3e}",
                f"max_abs_d={errors['max_abs_d']:.3e}",
                f"max_abs_x={errors['max_abs_x']:.3e}",
                f"time={time.perf_counter() - t_check:.3f}s",
            )

        print("batch, estimated_arrays, factor_s, solve_s, factor+solve_s, systems/s")
        for batch in batches:
            estimate = _estimate_bytes(batch, plan.n, plan.nnz_a, plan.nnz_l, dtype)
            if estimate > max_bytes:
                print(f"{batch}, {_format_gb(estimate)}, skipped_over_limit, -, -, -")
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
            values_jax = jnp.asarray(values)
            rhs_jax = jnp.asarray(rhs)

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
                print(
                    f"{batch}, {_format_gb(estimate)}, failed, "
                    f"{type(exc).__name__}: {exc}"
                )
                break

            row = GpuResult(
                dtype=dtype.name,
                batch_size=batch,
                estimated_bytes=estimate,
                factor_s=factor_s,
                solve_s=solve_s,
                factor_solve_s=factor_solve_s,
            )
            results.append(row)
            print(
                f"{batch}, {_format_gb(estimate)}, {factor_s:.6f}, "
                f"{solve_s:.6f}, {factor_solve_s:.6f}, {row.systems_per_s:.2f}"
            )

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, results, cpu_throughput)
    _plot(plot_path, results, cpu_throughput)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
