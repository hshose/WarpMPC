#!/usr/bin/env python3
"""Benchmark JAXopt solve_cg on fixed-pattern MPC KKT normal equations."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
JAXOPT_RESOURCE = ROOT / "resources" / "jaxopt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if JAXOPT_RESOURCE.exists() and str(JAXOPT_RESOURCE) not in sys.path:
    sys.path.insert(0, str(JAXOPT_RESOURCE))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np
import qdldl
import scipy.sparse as sp
from jaxopt.linear_solve import solve_cg

from benchmarks.problems import make_mpc_kkt, sample_kkt_values, structural_density


@dataclass(frozen=True)
class CgResult:
    dtype: str
    batch_size: int
    estimated_bytes: int
    solve_s: float
    cg_maxiter: int
    cg_tol: float
    ridge: float
    max_abs_x_vs_qdldl: float | None
    max_rel_residual: float | None

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.solve_s


def _parse_batches(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_dtypes(text: str) -> list[np.dtype]:
    return [np.dtype(part.strip()) for part in text.split(",") if part.strip()]


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def _estimate_bytes(batch: int, n: int, nnz_a: int, dtype: np.dtype) -> int:
    item = np.dtype(dtype).itemsize
    # The normal-equation CG loop keeps several batched vectors alive in
    # addition to the sparse value batch. Keep this estimate intentionally
    # conservative so oversized batches are skipped before XLA allocation.
    arrays = nnz_a + 12 * n
    return batch * arrays * item


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


def _upper_kkt_from_values(kkt_upper: sp.spmatrix, values: np.ndarray) -> sp.csc_matrix:
    return sp.csc_matrix((values, kkt_upper.indices, kkt_upper.indptr), shape=kkt_upper.shape)


def _full_kkt_from_upper(kkt_upper: sp.spmatrix, values: np.ndarray) -> sp.csc_matrix:
    upper = sp.csc_matrix((values, kkt_upper.indices, kkt_upper.indptr), shape=kkt_upper.shape)
    return upper + upper.T - sp.diags(upper.diagonal(), format="csc")


def _benchmark_cpu_qdldl(kkt_upper: sp.spmatrix, values: np.ndarray, rhs: np.ndarray, repeat: int) -> float:
    matrices = [_upper_kkt_from_values(kkt_upper, row) for row in values]
    solver = qdldl.Solver(matrices[0], upper=True)
    solver.solve(rhs[0])

    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for matrix, vector in zip(matrices, rhs, strict=True):
            solver.update(matrix, upper=True)
            solver.solve(vector)
            count += 1
    return count / (time.perf_counter() - start)


def _make_batched_cg_solver(
    kkt_upper: sp.spmatrix,
    *,
    dtype: np.dtype,
    maxiter: int,
    tol: float,
    ridge: float,
):
    kkt_upper = sp.csc_matrix(kkt_upper)
    n = kkt_upper.shape[0]
    rows = jnp.asarray(kkt_upper.indices, dtype=jnp.int32)
    cols_np = np.repeat(np.arange(n, dtype=np.int32), np.diff(kkt_upper.indptr))
    cols = jnp.asarray(cols_np, dtype=jnp.int32)
    offdiag_np = kkt_upper.indices != cols_np
    off_idx = jnp.asarray(np.nonzero(offdiag_np)[0], dtype=jnp.int32)
    off_rows = jnp.asarray(kkt_upper.indices[offdiag_np], dtype=jnp.int32)
    off_cols = jnp.asarray(cols_np[offdiag_np], dtype=jnp.int32)
    ridge_arg = None if ridge == 0.0 else float(ridge)

    def kkt_matvec(values, x):
        y = jnp.zeros((n,), dtype=x.dtype)
        y = y.at[rows].add(values * x[cols])
        y = y.at[off_cols].add(values[off_idx] * x[off_rows])
        return y

    def solve_one(values, rhs):
        values = jnp.asarray(values, dtype=dtype)
        rhs = jnp.asarray(rhs, dtype=dtype)
        normal_rhs = kkt_matvec(values, rhs)

        def normal_matvec(x):
            return kkt_matvec(values, kkt_matvec(values, x))

        init = jnp.zeros_like(rhs)
        return solve_cg(
            normal_matvec,
            normal_rhs,
            ridge=ridge_arg,
            init=init,
            maxiter=maxiter,
            tol=tol,
        )

    return jax.jit(jax.vmap(solve_one, in_axes=(0, 0)))


def _check_against_qdldl(
    kkt_upper: sp.spmatrix,
    solver,
    values: np.ndarray,
    rhs: np.ndarray,
) -> tuple[float, float]:
    values_jax = jnp.asarray(values)
    rhs_jax = jnp.asarray(rhs)
    cg_x = np.asarray(jax.device_get(solver(values_jax, rhs_jax)))

    max_abs_x = 0.0
    max_rel_residual = 0.0
    for value_row, rhs_row, cg_row in zip(values, rhs, cg_x, strict=True):
        upper = _upper_kkt_from_values(kkt_upper, value_row)
        matrix = _full_kkt_from_upper(kkt_upper, value_row)
        qdldl_x = qdldl.Solver(upper, upper=True).solve(rhs_row)
        max_abs_x = max(max_abs_x, float(np.max(np.abs(cg_row - qdldl_x))))
        residual = matrix @ cg_row - rhs_row
        denom = max(1.0, float(np.linalg.norm(rhs_row)))
        max_rel_residual = max(max_rel_residual, float(np.linalg.norm(residual) / denom))
    return max_abs_x, max_rel_residual


def _write_csv(path: pathlib.Path, rows: list[CgResult], cpu_throughput: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dtype",
                "batch_size",
                "estimated_gib",
                "solve_s",
                "systems_per_s",
                "cg_maxiter",
                "cg_tol",
                "ridge",
                "max_abs_x_vs_qdldl",
                "max_rel_residual",
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
                    "solve_s": row.solve_s,
                    "systems_per_s": row.systems_per_s,
                    "cg_maxiter": row.cg_maxiter,
                    "cg_tol": row.cg_tol,
                    "ridge": row.ridge,
                    "max_abs_x_vs_qdldl": row.max_abs_x_vs_qdldl,
                    "max_rel_residual": row.max_rel_residual,
                    "cpu_qdldl_update_solve_systems_per_s": cpu_throughput,
                }
            )


def _plot(path: pathlib.Path, rows: list[CgResult], cpu_throughput: float) -> None:
    if not rows:
        return
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
            label=f"JAXopt solve_cg normal eq {dtype}",
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
    ax.set_title("Fixed-pattern MPC KKT normal-equation CG throughput")
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
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000,50000,100000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=512)
    parser.add_argument("--cpu-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variation", type=float, default=0.25)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--cg-maxiter", type=int, default=25)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--plot-path", default="results/qdldl_factorization/throughput_mpc_jaxopt_solve_cg.png")
    parser.add_argument("--csv-path", default="results/qdldl_factorization/throughput_mpc_jaxopt_solve_cg.csv")
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
        "cg_operator=K^T K",
        f"cg_maxiter={args.cg_maxiter}",
        f"cg_tol={args.cg_tol}",
        f"ridge={args.ridge}",
    )

    cpu_values = sample_kkt_values(
        kkt,
        batch_size=args.cpu_samples,
        seed=args.seed + 50,
        variation=args.variation,
        dtype=np.float64,
    )
    cpu_rhs = np.random.default_rng(args.seed + 51).standard_normal(
        (args.cpu_samples, kkt.shape[0])
    )
    cpu_throughput = _benchmark_cpu_qdldl(kkt, cpu_values, cpu_rhs, repeat=args.cpu_repeat)
    print(
        "CPU qdldl:",
        "mode=update+solve",
        f"samples={args.cpu_samples}",
        f"systems/s={cpu_throughput:.2f}",
    )

    results: list[CgResult] = []
    max_bytes = int(args.max_device_gb * 1024**3)
    batches = _parse_batches(args.batch_sizes)
    for dtype in _parse_dtypes(args.dtypes):
        jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
        print(f"Benchmark dtype={dtype.name}: fixed MPC structural pattern")
        solver = _make_batched_cg_solver(
            kkt,
            dtype=dtype,
            maxiter=args.cg_maxiter,
            tol=args.cg_tol,
            ridge=args.ridge,
        )

        max_abs_x = None
        max_rel_residual = None
        if not args.no_check:
            check_values = sample_kkt_values(
                kkt,
                batch_size=2,
                seed=args.seed + 10,
                variation=args.variation,
                dtype=dtype,
            )
            check_rhs = np.random.default_rng(args.seed + 11).standard_normal(
                (2, kkt.shape[0])
            ).astype(dtype)
            t_check = time.perf_counter()
            max_abs_x, max_rel_residual = _check_against_qdldl(
                kkt, solver, check_values, check_rhs
            )
            print(
                "Check:",
                f"max_abs_x_vs_qdldl={max_abs_x:.3e}",
                f"max_rel_residual={max_rel_residual:.3e}",
                f"time={time.perf_counter() - t_check:.3f}s",
            )

        print("batch, estimated_arrays, solve_s, systems/s")
        for batch in batches:
            estimate = _estimate_bytes(batch, kkt.shape[0], kkt.nnz, dtype)
            if estimate > max_bytes:
                print(f"{batch}, {_format_gb(estimate)}, skipped_over_limit, -")
                break

            values = sample_kkt_values(
                kkt,
                batch_size=batch,
                seed=args.seed + batch,
                variation=args.variation,
                dtype=dtype,
            )
            rhs = np.random.default_rng(args.seed + 1000 + batch).standard_normal(
                (batch, kkt.shape[0])
            ).astype(dtype)
            values_jax = jnp.asarray(values)
            rhs_jax = jnp.asarray(rhs)

            try:
                solve_exe, _ = _compile_and_warm(solver, values_jax, rhs_jax)
                solve_s, _ = _time_compiled(solve_exe, values_jax, rhs_jax, repeat=args.repeat)
            except Exception as exc:
                print(f"{batch}, {_format_gb(estimate)}, failed, {type(exc).__name__}: {exc}")
                break

            row = CgResult(
                dtype=dtype.name,
                batch_size=batch,
                estimated_bytes=estimate,
                solve_s=solve_s,
                cg_maxiter=args.cg_maxiter,
                cg_tol=args.cg_tol,
                ridge=args.ridge,
                max_abs_x_vs_qdldl=max_abs_x,
                max_rel_residual=max_rel_residual,
            )
            results.append(row)
            print(f"{batch}, {_format_gb(estimate)}, {solve_s:.6f}, {row.systems_per_s:.2f}")

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
