#!/usr/bin/env python3
"""Benchmark JaxProxQP on fixed-pattern batched quadcopter MPC QPs."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
JAXPROXQP_RESOURCE = ROOT / "resources" / "jaxproxqp" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if JAXPROXQP_RESOURCE.exists() and str(JAXPROXQP_RESOURCE) not in sys.path:
    sys.path.insert(0, str(JAXPROXQP_RESOURCE))

from benchmarks.jax_cache import configure_jax_compilation_cache
from jax import config as jax_config

configure_jax_compilation_cache()
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import osqp

from benchmarks.problems import batched_problem_data, make_linear_mpc
from warpmpc.jax_osqp import OSQPSettings
from jaxproxqp.jaxproxqp import JaxProxQP
from jaxproxqp.qp_problems import QPModel
from jaxproxqp.settings import Settings


@dataclass(frozen=True)
class JaxProxQPProblem:
    h_matrix: np.ndarray
    a_eq: np.ndarray
    c_ineq: np.ndarray
    eq_rows: np.ndarray
    ineq_rows: np.ndarray
    l_box: np.ndarray
    u_box: np.ndarray
    n: int
    n_eq: int
    n_ineq: int


@dataclass(frozen=True)
class BatchResult:
    solver: str
    mode: str
    dtype: str
    batch_size: int
    horizon: int
    n: int
    m: int
    n_eq: int
    n_ineq: int
    max_iter: int
    max_iter_in: int
    solver_tol: float
    regularization: float
    elapsed_s: float
    mean_outer_iterations: float
    max_outer_iterations: int
    mean_inner_iterations: float
    max_inner_iterations: int
    max_primal_residual: float
    max_dual_residual: float
    max_duality_gap: float
    max_abs_x_vs_cpu: float | None
    max_abs_obj_vs_cpu: float | None
    cpu_throughput: float | None

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.elapsed_s


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_dtypes(text: str) -> list[np.dtype]:
    return [np.dtype(part.strip()) for part in text.split(",") if part.strip()]


def _osqp_settings_dict(settings: OSQPSettings) -> dict[str, object]:
    return {
        "verbose": False,
        "scaling": settings.scaling,
        "adaptive_rho": settings.adaptive_rho,
        "rho_is_vec": settings.rho_is_vec,
        "check_termination": settings.check_termination,
        "max_iter": settings.max_iter,
        "polish": settings.polishing,
        "warm_starting": settings.warm_starting,
        "rho": settings.rho,
        "sigma": settings.sigma,
        "alpha": settings.alpha,
        "eps_abs": settings.eps_abs,
        "eps_rel": settings.eps_rel,
    }


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


def _benchmark_cpu_osqp(problem, settings: OSQPSettings, q, l, u, repeat: int) -> float:
    solver = osqp.OSQP()
    solver.setup(problem.p_matrix, problem.q, problem.a_matrix, problem.l, problem.u, **_osqp_settings_dict(settings))
    count = 0
    start = time.perf_counter()
    for _ in range(repeat):
        for i in range(q.shape[0]):
            solver.update(
                q=np.asarray(q[i], dtype=np.float64),
                l=np.asarray(l[i], dtype=np.float64),
                u=np.asarray(u[i], dtype=np.float64),
            )
            solver.solve()
            count += 1
    return count / (time.perf_counter() - start)


def _jaxproxqp_problem(problem, dtype: np.dtype, regularization: float, eq_tol: float) -> JaxProxQPProblem:
    a_dense = np.asarray(problem.a_matrix.toarray(), dtype=dtype)
    h_matrix = np.asarray(problem.p_matrix.toarray(), dtype=dtype)
    h_matrix = h_matrix + np.asarray(regularization, dtype=dtype) * np.eye(h_matrix.shape[0], dtype=dtype)

    l = np.asarray(problem.l, dtype=np.float64)
    u = np.asarray(problem.u, dtype=np.float64)
    finite_l = np.isfinite(l)
    finite_u = np.isfinite(u)
    equality = finite_l & finite_u & (np.abs(u - l) <= eq_tol)
    inequality = ~equality

    n = h_matrix.shape[0]
    return JaxProxQPProblem(
        h_matrix=h_matrix,
        a_eq=a_dense[equality],
        c_ineq=a_dense[inequality],
        eq_rows=np.flatnonzero(equality).astype(np.int32),
        ineq_rows=np.flatnonzero(inequality).astype(np.int32),
        l_box=np.full(n, -1e9, dtype=dtype),
        u_box=np.full(n, 1e9, dtype=dtype),
        n=n,
        n_eq=int(np.count_nonzero(equality)),
        n_ineq=int(np.count_nonzero(inequality)),
    )


def _jaxproxqp_bounds(
    qp: JaxProxQPProblem,
    l: np.ndarray,
    u: np.ndarray,
    dtype: np.dtype,
    bound_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(u[:, qp.eq_rows], dtype=dtype)
    c_l = np.asarray(l[:, qp.ineq_rows], dtype=dtype)
    c_u = np.asarray(u[:, qp.ineq_rows], dtype=dtype)
    c_l = np.where(np.isfinite(c_l), c_l, np.asarray(-bound_value, dtype=dtype))
    c_u = np.where(np.isfinite(c_u), c_u, np.asarray(bound_value, dtype=dtype))
    return b, c_l.astype(dtype, copy=False), c_u.astype(dtype, copy=False)


def _settings(dtype: np.dtype, tol: float, max_iter: int, max_iter_in: int) -> Settings:
    settings = Settings.default_float32() if dtype == np.dtype(np.float32) else Settings.default_float64()
    settings.eps_abs = tol
    settings.eps_in_min = min(tol, settings.eps_in_min)
    settings.pri_res_thresh_abs = tol
    settings.dua_res_thresh_abs = tol
    settings.dua_gap_thresh_abs = tol
    settings.max_iter = max_iter
    settings.max_iter_in = max_iter_in
    settings.verbose = False
    return settings


def _make_solve_fn(qp: JaxProxQPProblem, settings: Settings):
    h_matrix = jnp.asarray(qp.h_matrix)
    a_eq = jnp.asarray(qp.a_eq)
    c_ineq = jnp.asarray(qp.c_ineq)
    l_box = jnp.asarray(qp.l_box)
    u_box = jnp.asarray(qp.u_box)

    def solve_one(g, b, c_l, c_u):
        model = QPModel.create(
            H=h_matrix,
            g=g,
            A=a_eq,
            b=b,
            C=c_ineq,
            l=c_l,
            u=c_u,
            l_box=l_box,
            u_box=u_box,
        )
        sol = JaxProxQP(model, settings).solve()
        return (
            sol.x,
            sol.obj_value,
            sol.pri_res,
            sol.dua_res,
            sol.duality_gap,
            sol.info.iter_ext,
            sol.info.iter_inner,
            sol.info.mu_updates,
        )

    return jax.jit(jax.vmap(solve_one, in_axes=(0, 0, 0, 0)))


def _check_against_cpu(
    problem,
    settings: OSQPSettings,
    compiled_solve,
    qp: JaxProxQPProblem,
    q,
    l,
    u,
    dtype,
    bound_value,
):
    b, c_l, c_u = _jaxproxqp_bounds(qp, l, u, dtype, bound_value)
    x, obj, *_ = jax.device_get(compiled_solve(jnp.asarray(q), jnp.asarray(b), jnp.asarray(c_l), jnp.asarray(c_u)))
    solver = osqp.OSQP()
    solver.setup(problem.p_matrix, problem.q, problem.a_matrix, problem.l, problem.u, **_osqp_settings_dict(settings))
    max_x = 0.0
    max_obj = 0.0
    for i in range(q.shape[0]):
        solver.update(
            q=np.asarray(q[i], dtype=np.float64),
            l=np.asarray(l[i], dtype=np.float64),
            u=np.asarray(u[i], dtype=np.float64),
        )
        res = solver.solve()
        max_x = max(max_x, float(np.max(np.abs(res.x - x[i]))))
        max_obj = max(max_obj, float(abs(res.info.obj_val - obj[i])))
    return max_x, max_obj


def _estimate_bytes(batch: int, qp: JaxProxQPProblem, dtype: np.dtype) -> int:
    item = np.dtype(dtype).itemsize
    kkt_dim = qp.n + qp.n_eq + qp.n_ineq + qp.n
    dense_constants = qp.n * qp.n + qp.n_eq * qp.n + qp.n_ineq * qp.n
    per_batch = kkt_dim * kkt_dim + 8 * (qp.n + qp.n_eq + qp.n_ineq)
    return item * (dense_constants + batch * per_batch)


def _write_csv(path: pathlib.Path, rows: list[BatchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver",
                "mode",
                "dtype",
                "batch_size",
                "horizon",
                "n",
                "m",
                "n_eq",
                "n_ineq",
                "max_iter",
                "max_iter_in",
                "solver_tol",
                "regularization",
                "elapsed_s",
                "systems_per_s",
                "mean_outer_iterations",
                "max_outer_iterations",
                "mean_inner_iterations",
                "max_inner_iterations",
                "max_primal_residual",
                "max_dual_residual",
                "max_duality_gap",
                "max_abs_x_vs_cpu",
                "max_abs_obj_vs_cpu",
                "cpu_osqp_systems_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "solver": row.solver,
                    "mode": row.mode,
                    "dtype": row.dtype,
                    "batch_size": row.batch_size,
                    "horizon": row.horizon,
                    "n": row.n,
                    "m": row.m,
                    "n_eq": row.n_eq,
                    "n_ineq": row.n_ineq,
                    "max_iter": row.max_iter,
                    "max_iter_in": row.max_iter_in,
                    "solver_tol": row.solver_tol,
                    "regularization": row.regularization,
                    "elapsed_s": row.elapsed_s,
                    "systems_per_s": row.systems_per_s,
                    "mean_outer_iterations": row.mean_outer_iterations,
                    "max_outer_iterations": row.max_outer_iterations,
                    "mean_inner_iterations": row.mean_inner_iterations,
                    "max_inner_iterations": row.max_inner_iterations,
                    "max_primal_residual": row.max_primal_residual,
                    "max_dual_residual": row.max_dual_residual,
                    "max_duality_gap": row.max_duality_gap,
                    "max_abs_x_vs_cpu": row.max_abs_x_vs_cpu,
                    "max_abs_obj_vs_cpu": row.max_abs_obj_vs_cpu,
                    "cpu_osqp_systems_per_s": row.cpu_throughput,
                }
            )


def _plot(path: pathlib.Path, rows: list[BatchResult]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 4.8), squeeze=False, sharey=True)
    for ax, dtype in zip(axes[0], dtypes, strict=True):
        dtype_rows = sorted([row for row in rows if row.dtype == dtype], key=lambda row: row.batch_size)
        ax.plot(
            [row.batch_size for row in dtype_rows],
            [row.systems_per_s for row in dtype_rows],
            marker="v",
            linewidth=1.8,
            label="JaxProxQP",
        )
        cpu = next((row.cpu_throughput for row in dtype_rows if row.cpu_throughput), None)
        if cpu is not None:
            ax.axhline(cpu, linestyle="--", color="black", linewidth=1.5, label="CPU OSQP")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Solved MPC problems/s")
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("JaxProxQP fixed-pattern quadcopter MPC throughput")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--batch-sizes", default="512,2048")
    parser.add_argument("--dtypes", default="float64")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--max-iter-in", type=int, default=15)
    parser.add_argument("--solver-tol", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--eq-tol", type=float, default=1e-9)
    parser.add_argument("--bound-value", type=float, default=1e9)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu-samples", type=int, default=128)
    parser.add_argument("--cpu-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--csv-path", default="results/linear_mpc/throughput_mpc_jaxproxqp.csv")
    parser.add_argument("--plot-path", default="results/linear_mpc/throughput_mpc_jaxproxqp.png")
    parser.add_argument("--no-cpu", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    problem = make_linear_mpc(args.horizon)
    cpu_settings = OSQPSettings(
        max_iter=4000,
        eps_abs=args.solver_tol,
        eps_rel=args.solver_tol,
        scaling=0,
        adaptive_rho=False,
        rho_is_vec=False,
        check_termination=25,
        warm_starting=False,
        polishing=False,
    )
    print("Devices:", jax.devices())
    print(
        "Problem:",
        f"N={args.horizon}",
        f"n={problem.q.size}",
        f"m={problem.l.size}",
        f"max_iter={args.max_iter}",
        f"max_iter_in={args.max_iter_in}",
        f"solver_tol={args.solver_tol:g}",
    )

    cpu_throughput = None
    if not args.no_cpu:
        _, _, q_cpu, l_cpu, u_cpu = batched_problem_data(
            problem, args.cpu_samples, np.float64, seed=args.seed, x0_variation=0.15
        )
        cpu_throughput = _benchmark_cpu_osqp(problem, cpu_settings, q_cpu, l_cpu, u_cpu, args.cpu_repeat)
        print(f"CPU OSQP throughput: {cpu_throughput:.2f}/s")

    rows: list[BatchResult] = []
    max_bytes = int(args.max_device_gb * 1024**3)
    for dtype in _parse_dtypes(args.dtypes):
        if dtype != np.dtype(np.float64):
            print(f"JaxProxQP dtype={dtype.name}: skipped; this benchmark is float64-only")
            continue
        qp = _jaxproxqp_problem(problem, dtype, args.regularization, args.eq_tol)
        settings = _settings(dtype, args.solver_tol, args.max_iter, args.max_iter_in)
        solve_fn = _make_solve_fn(qp, settings)
        print(
            f"JaxProxQP dtype={dtype.name}: n={qp.n} n_eq={qp.n_eq} n_ineq={qp.n_ineq} "
            f"kkt_dim={qp.n + qp.n_eq + qp.n_ineq + qp.n} regularization={args.regularization:g}"
        )

        max_x = None
        max_obj = None
        if not args.no_check:
            _, _, q_check, l_check, u_check = batched_problem_data(
                problem, 2, dtype, seed=args.seed + 10, x0_variation=0.15
            )
            b_check, c_l_check, c_u_check = _jaxproxqp_bounds(qp, l_check, u_check, dtype, args.bound_value)
            check_solve, _ = _compile_and_warm(
                solve_fn,
                jnp.asarray(q_check),
                jnp.asarray(b_check),
                jnp.asarray(c_l_check),
                jnp.asarray(c_u_check),
            )
            max_x, max_obj = _check_against_cpu(
                problem,
                cpu_settings,
                check_solve,
                qp,
                q_check,
                l_check,
                u_check,
                dtype,
                args.bound_value,
            )
            print(f"  check max_abs_x={max_x:.3e} max_abs_obj={max_obj:.3e}")

        for batch in _parse_ints(args.batch_sizes):
            estimate = _estimate_bytes(batch, qp, dtype)
            if estimate > max_bytes:
                print(f"  batch={batch}: skipped estimated {estimate / 1024**3:.2f} GiB")
                continue
            _, _, q, l, u = batched_problem_data(
                problem, batch, dtype, seed=args.seed + batch, x0_variation=0.15
            )
            b, c_l, c_u = _jaxproxqp_bounds(qp, l, u, dtype, args.bound_value)
            try:
                q_jax = jnp.asarray(q)
                b_jax = jnp.asarray(b)
                c_l_jax = jnp.asarray(c_l)
                c_u_jax = jnp.asarray(c_u)
                compiled, _ = _compile_and_warm(solve_fn, q_jax, b_jax, c_l_jax, c_u_jax)
                elapsed, out = _time_compiled(
                    compiled,
                    q_jax,
                    b_jax,
                    c_l_jax,
                    c_u_jax,
                    repeat=args.repeat,
                )
            except Exception as exc:
                print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                continue
            _, _, pri_res, dua_res, duality_gap, iter_ext, iter_inner, _ = jax.device_get(out)
            row = BatchResult(
                solver="jaxproxqp",
                mode="solve",
                dtype=dtype.name,
                batch_size=batch,
                horizon=args.horizon,
                n=qp.n,
                m=problem.l.size,
                n_eq=qp.n_eq,
                n_ineq=qp.n_ineq,
                max_iter=args.max_iter,
                max_iter_in=args.max_iter_in,
                solver_tol=args.solver_tol,
                regularization=args.regularization,
                elapsed_s=elapsed,
                mean_outer_iterations=float(np.mean(iter_ext)),
                max_outer_iterations=int(np.max(iter_ext)),
                mean_inner_iterations=float(np.mean(iter_inner)),
                max_inner_iterations=int(np.max(iter_inner)),
                max_primal_residual=float(np.max(pri_res)),
                max_dual_residual=float(np.max(dua_res)),
                max_duality_gap=float(np.max(duality_gap)),
                max_abs_x_vs_cpu=max_x,
                max_abs_obj_vs_cpu=max_obj,
                cpu_throughput=cpu_throughput,
            )
            rows.append(row)
            print(
                f"  batch={batch}: elapsed={elapsed:.6f}s throughput={row.systems_per_s:.2f}/s "
                f"outer_mean={row.mean_outer_iterations:.1f} inner_mean={row.mean_inner_iterations:.1f} "
                f"pri={row.max_primal_residual:.2e} dua={row.max_dual_residual:.2e}"
            )

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
