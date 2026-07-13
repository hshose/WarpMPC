#!/usr/bin/env python3
"""Benchmark dense JAX LU solves on the same fixed-pattern KKT batches as QDLDL."""

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
import jax.scipy as jsp
import numpy as np
import qdldl
import scipy.sparse as sp

from benchmarks.problems import make_mpc_kkt, sample_kkt_values, structural_density


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
class DenseLuPlan:
    n: int
    rows: np.ndarray
    cols: np.ndarray
    offdiag: np.ndarray


@dataclass(frozen=True)
class GpuResult:
    solver: str
    dtype: str
    batch_size: int
    estimated_bytes: int
    factor_s: float
    solve_s: float
    factor_solve_s: float

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.factor_solve_s

    @property
    def factor_systems_per_s(self) -> float:
        return self.batch_size / self.factor_s

    @property
    def solve_systems_per_s(self) -> float:
        return self.batch_size / self.solve_s


def _parse_batches(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_dtypes(text: str) -> list[np.dtype]:
    return [np.dtype(part.strip()) for part in text.split(",") if part.strip()]


def _upper_rows_cols(kkt_upper: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(kkt_upper.indices, dtype=np.int32)
    cols = np.repeat(np.arange(kkt_upper.shape[1], dtype=np.int32), np.diff(kkt_upper.indptr))
    return rows, np.asarray(cols, dtype=np.int32)


def _build_dense_lu_plan(kkt_upper: sp.csc_matrix) -> DenseLuPlan:
    rows, cols = _upper_rows_cols(kkt_upper)
    return DenseLuPlan(
        n=kkt_upper.shape[0],
        rows=rows,
        cols=cols,
        offdiag=np.asarray(rows != cols),
    )


def _estimate_bytes(batch: int, n: int, dtype: np.dtype) -> int:
    item = np.dtype(dtype).itemsize
    dense_matrices_and_work = 4 * n * n
    vectors_and_pivots = 8 * n
    return batch * (dense_matrices_and_work + vectors_and_pivots) * item


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


def _make_kernels(plan: DenseLuPlan):
    rows = jnp.asarray(plan.rows)
    cols = jnp.asarray(plan.cols)
    offdiag = jnp.asarray(plan.offdiag)
    rows_off = jnp.asarray(plan.rows[plan.offdiag])
    cols_off = jnp.asarray(plan.cols[plan.offdiag])

    def assemble(values):
        mats = jnp.zeros((values.shape[0], plan.n, plan.n), dtype=values.dtype)
        mats = mats.at[:, rows, cols].set(values)
        mats = mats.at[:, cols_off, rows_off].set(values[:, offdiag])
        return mats

    @jax.jit
    def factor(values):
        mats = assemble(values)
        return jax.vmap(lambda matrix: jsp.linalg.lu_factor(matrix, check_finite=False))(mats)

    @jax.jit
    def solve(lu, piv, rhs):
        return jax.vmap(lambda lu_i, piv_i, rhs_i: jsp.linalg.lu_solve((lu_i, piv_i), rhs_i, check_finite=False))(
            lu, piv, rhs
        )

    @jax.jit
    def factor_and_solve(values, rhs):
        lu, piv = factor(values)
        return solve(lu, piv, rhs)

    return factor, solve, factor_and_solve


def _benchmark_cpu_qdldl(kkt_upper: sp.csc_matrix, values: np.ndarray, rhs: np.ndarray, repeat: int) -> CpuQdldlTiming:
    matrices = [
        sp.csc_matrix((np.asarray(row), kkt_upper.indices, kkt_upper.indptr), shape=kkt_upper.shape)
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


def _verify_against_qdldl(
    kkt_upper: sp.csc_matrix,
    factor_and_solve_exe,
    values: np.ndarray,
    rhs: np.ndarray,
) -> dict[str, float]:
    x_gpu = np.asarray(jax.device_get(factor_and_solve_exe(jnp.asarray(values), jnp.asarray(rhs))))
    max_abs_x = 0.0
    max_residual = 0.0
    for i in range(values.shape[0]):
        matrix = sp.csc_matrix((values[i], kkt_upper.indices, kkt_upper.indptr), shape=kkt_upper.shape)
        solver = qdldl.Solver(matrix, upper=True)
        x_cpu = solver.solve(rhs[i])
        full = matrix + matrix.T - sp.diags(matrix.diagonal(), format="csc")
        max_abs_x = max(max_abs_x, float(np.max(np.abs(x_cpu - x_gpu[i]))))
        max_residual = max(max_residual, float(np.max(np.abs(full.dot(x_gpu[i]) - rhs[i]))))
    return {"max_abs_x": max_abs_x, "max_residual": max_residual}


def _write_csv(path: pathlib.Path, rows: list[GpuResult], cpu_timing: CpuQdldlTiming) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver",
                "dtype",
                "batch_size",
                "estimated_gib",
                "factor_s",
                "solve_s",
                "factor_solve_s",
                "factor_systems_per_s",
                "solve_systems_per_s",
                "systems_per_s",
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
            writer.writerow(
                {
                    "solver": row.solver,
                    "dtype": row.dtype,
                    "batch_size": row.batch_size,
                    "estimated_gib": row.estimated_bytes / 1024**3,
                    "factor_s": row.factor_s,
                    "solve_s": row.solve_s,
                    "factor_solve_s": row.factor_solve_s,
                    "factor_systems_per_s": row.factor_systems_per_s,
                    "solve_systems_per_s": row.solve_systems_per_s,
                    "systems_per_s": row.systems_per_s,
                    "cpu_qdldl_factor_s_per_system": cpu_timing.factor_s_per_system,
                    "cpu_qdldl_solve_s_per_system": cpu_timing.solve_s_per_system,
                    "cpu_qdldl_update_solve_s_per_system": cpu_timing.update_solve_s_per_system,
                    "cpu_qdldl_factor_systems_per_s": cpu_timing.factor_systems_per_s,
                    "cpu_qdldl_solve_systems_per_s": cpu_timing.solve_systems_per_s,
                    "cpu_qdldl_update_solve_systems_per_s": cpu_timing.update_solve_systems_per_s,
                }
            )


def _plot(path: pathlib.Path, rows: list[GpuResult], cpu_throughput: float) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
            marker="p",
            linewidth=1.8,
            label=f"dense LU {dtype}",
        )
    ax.axhline(cpu_throughput, linestyle="--", color="black", linewidth=1.5, label="CPU qdldl update+solve")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput [systems/s]")
    ax.set_title("Dense JAX LU fixed-pattern MPC KKT throughput")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
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
    parser.add_argument("--batch-sizes", default="1,8,32,128,512,2048,10000,20000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=128)
    parser.add_argument("--cpu-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variation", type=float, default=0.25)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--csv-path", default="results/qdldl_factorization/throughput_mpc_dense_lu_qdldl.csv")
    parser.add_argument("--plot-path", default="results/qdldl_factorization/throughput_mpc_dense_lu_qdldl.png")
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
    plan = _build_dense_lu_plan(kkt)
    print("Devices:", jax.devices())
    print(
        "KKT:",
        f"dim={kkt.shape[0]}",
        f"nvar={problem.n_variables}",
        f"m={problem.n_constraints}",
        f"upper_nnz={kkt.nnz}",
        f"upper_density={100 * structural_density(kkt):.2f}%",
    )

    cpu_values = sample_kkt_values(
        kkt,
        batch_size=args.cpu_samples,
        seed=args.seed + 50,
        variation=args.variation,
        dtype=np.float64,
    )
    cpu_rhs = np.random.default_rng(args.seed + 51).standard_normal((args.cpu_samples, plan.n))
    cpu_timing = _benchmark_cpu_qdldl(kkt, cpu_values, cpu_rhs, args.cpu_repeat)
    print(
        "CPU qdldl:",
        f"factor={cpu_timing.factor_systems_per_s:.2f}/s",
        f"solve={cpu_timing.solve_systems_per_s:.2f}/s",
        f"update+solve={cpu_timing.update_solve_systems_per_s:.2f}/s",
    )

    factor, solve, factor_and_solve = _make_kernels(plan)
    rows: list[GpuResult] = []
    max_bytes = int(args.max_device_gb * 1024**3)
    for dtype in _parse_dtypes(args.dtypes):
        jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
        print(f"Benchmark dtype={dtype.name}: dense LU on fixed MPC structural pattern")
        if not args.no_check:
            check_values = sample_kkt_values(kkt, 2, seed=args.seed + 10, variation=args.variation, dtype=dtype)
            check_rhs = np.random.default_rng(args.seed + 11).standard_normal((2, plan.n)).astype(dtype)
            factor_solve_exe, _ = _compile_and_warm(
                factor_and_solve,
                jnp.asarray(check_values),
                jnp.asarray(check_rhs),
            )
            errors = _verify_against_qdldl(kkt, factor_solve_exe, check_values, check_rhs)
            print(
                "Check:",
                f"max_abs_x={errors['max_abs_x']:.3e}",
                f"max_residual={errors['max_residual']:.3e}",
            )

        for batch in _parse_batches(args.batch_sizes):
            estimate = _estimate_bytes(batch, plan.n, dtype)
            if estimate > max_bytes:
                print(f"{batch}, {_format_gb(estimate)}, skipped_over_limit, -, -, -")
                break
            values = sample_kkt_values(kkt, batch, seed=args.seed + batch, variation=args.variation, dtype=dtype)
            rhs = np.random.default_rng(args.seed + 1000 + batch).standard_normal((batch, plan.n)).astype(dtype)
            values_jax = jnp.asarray(values)
            rhs_jax = jnp.asarray(rhs)
            try:
                factor_exe, factor_out = _compile_and_warm(factor, values_jax)
                lu, piv = factor_out
                solve_exe, _ = _compile_and_warm(solve, lu, piv, rhs_jax)
                factor_solve_exe, _ = _compile_and_warm(factor_and_solve, values_jax, rhs_jax)
                factor_s, factor_out = _time_compiled(factor_exe, values_jax, repeat=args.repeat)
                lu, piv = factor_out
                solve_s, _ = _time_compiled(solve_exe, lu, piv, rhs_jax, repeat=args.repeat)
                factor_solve_s, _ = _time_compiled(factor_solve_exe, values_jax, rhs_jax, repeat=args.repeat)
            except Exception as exc:
                print(f"{batch}, {_format_gb(estimate)}, failed, {type(exc).__name__}: {exc}")
                break
            row = GpuResult(
                solver="dense_lu",
                dtype=dtype.name,
                batch_size=batch,
                estimated_bytes=estimate,
                factor_s=factor_s,
                solve_s=solve_s,
                factor_solve_s=factor_solve_s,
            )
            rows.append(row)
            print(
                f"{batch}, {_format_gb(estimate)}, {factor_s:.6f}, "
                f"{solve_s:.6f}, {factor_solve_s:.6f}, {row.systems_per_s:.2f}"
            )

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows, cpu_timing)
    _plot(plot_path, rows, cpu_timing.update_solve_systems_per_s)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
