#!/usr/bin/env python3
"""Benchmark QPAX on fixed-pattern batched quadcopter MPC QPs."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
QPAX_RESOURCE = ROOT / "resources" / "qpax"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if QPAX_RESOURCE.exists() and str(QPAX_RESOURCE) not in sys.path:
    sys.path.insert(0, str(QPAX_RESOURCE))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np
import osqp
import qpax

from benchmarks.problems import batched_problem_data, make_linear_mpc
from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan


@dataclass(frozen=True)
class QPAXProblem:
    q_matrix: np.ndarray
    a_eq: np.ndarray
    g_ineq: np.ndarray
    eq_rows: np.ndarray
    upper_rows: np.ndarray
    lower_rows: np.ndarray
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
    solver_tol: float
    regularization: float
    elapsed_s: float
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


def _parse_modes(text: str) -> list[str]:
    modes = [part.strip() for part in text.split(",") if part.strip()]
    allowed = {"solve", "grad"}
    bad = sorted(set(modes) - allowed)
    if bad:
        raise ValueError(f"unknown modes: {bad}; expected subset of {sorted(allowed)}")
    return modes


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


def _qpax_problem(problem, dtype: np.dtype, regularization: float, eq_tol: float) -> QPAXProblem:
    a_dense = np.asarray(problem.a_matrix.toarray(), dtype=dtype)
    q_matrix = np.asarray(problem.p_matrix.toarray(), dtype=dtype)
    q_matrix = q_matrix + np.asarray(regularization, dtype=dtype) * np.eye(q_matrix.shape[0], dtype=dtype)

    l = np.asarray(problem.l, dtype=np.float64)
    u = np.asarray(problem.u, dtype=np.float64)
    finite_l = np.isfinite(l)
    finite_u = np.isfinite(u)
    equality = finite_l & finite_u & (np.abs(u - l) <= eq_tol)
    upper = finite_u & ~equality
    lower = finite_l & ~equality

    a_eq = a_dense[equality]
    g_ineq = np.concatenate([a_dense[upper], -a_dense[lower]], axis=0).astype(dtype, copy=False)
    return QPAXProblem(
        q_matrix=q_matrix,
        a_eq=a_eq,
        g_ineq=g_ineq,
        eq_rows=np.flatnonzero(equality).astype(np.int32),
        upper_rows=np.flatnonzero(upper).astype(np.int32),
        lower_rows=np.flatnonzero(lower).astype(np.int32),
        n=q_matrix.shape[0],
        n_eq=a_eq.shape[0],
        n_ineq=g_ineq.shape[0],
    )


def _qpax_bounds(qp: QPAXProblem, l: np.ndarray, u: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(u[:, qp.eq_rows], dtype=dtype)
    h = np.concatenate([u[:, qp.upper_rows], -l[:, qp.lower_rows]], axis=1).astype(dtype, copy=False)
    return b, h


def _make_solve_fn(qp: QPAXProblem, solver_tol: float, max_iter: int):
    q_matrix = jnp.asarray(qp.q_matrix)
    a_eq = jnp.asarray(qp.a_eq)
    g_ineq = jnp.asarray(qp.g_ineq)

    def solve_one(q, b, h):
        x, s, z, y, converged, pdip_iter = qpax.solve_qp(
            q_matrix,
            q,
            a_eq,
            b,
            g_ineq,
            h,
            solver_tol=solver_tol,
            max_iter=max_iter,
        )
        return x, converged, pdip_iter

    return jax.jit(jax.vmap(solve_one, in_axes=(0, 0, 0)))


def _make_grad_fn(qp: QPAXProblem, solver_tol: float, target_kappa: float, max_iter: int):
    q_matrix = jnp.asarray(qp.q_matrix)
    a_eq = jnp.asarray(qp.a_eq)
    g_ineq = jnp.asarray(qp.g_ineq)

    def objective(q, b, h, x_weights):
        def solve_one(q_i, b_i, h_i):
            return qpax.solve_qp_primal(
                q_matrix,
                q_i,
                a_eq,
                b_i,
                g_ineq,
                h_i,
                solver_tol=solver_tol,
                target_kappa=target_kappa,
                max_iter=max_iter,
            )

        x = jax.vmap(solve_one, in_axes=(0, 0, 0))(q, b, h)
        return jnp.sum(x * x_weights)

    return jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))


def _check_against_cpu(problem, settings: OSQPSettings, compiled_solve, qp: QPAXProblem, q, l, u, dtype):
    b, h = _qpax_bounds(qp, l, u, dtype)
    x, _, _ = jax.device_get(compiled_solve(jnp.asarray(q), jnp.asarray(b), jnp.asarray(h)))
    solver = osqp.OSQP()
    solver.setup(problem.p_matrix, problem.q, problem.a_matrix, problem.l, problem.u, **_osqp_settings_dict(settings))
    max_x = 0.0
    max_obj = 0.0
    p_matrix = problem.p_matrix
    for i in range(q.shape[0]):
        solver.update(
            q=np.asarray(q[i], dtype=np.float64),
            l=np.asarray(l[i], dtype=np.float64),
            u=np.asarray(u[i], dtype=np.float64),
        )
        res = solver.solve()
        obj = 0.5 * float(x[i] @ p_matrix.dot(x[i])) + float(q[i] @ x[i])
        max_x = max(max_x, float(np.max(np.abs(res.x - x[i]))))
        max_obj = max(max_obj, float(abs(res.info.obj_val - obj)))
    return max_x, max_obj


def _estimate_bytes(batch: int, qp: QPAXProblem, dtype: np.dtype, mode: str) -> int:
    item = np.dtype(dtype).itemsize
    dense_constants = qp.n * qp.n + qp.n_eq * qp.n + qp.n_ineq * qp.n
    per_batch = 5 * qp.n + 6 * qp.n_eq + 8 * qp.n_ineq
    if mode == "grad":
        per_batch += qp.n + qp.n_eq + qp.n_ineq
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
                "solver_tol",
                "regularization",
                "elapsed_s",
                "systems_per_s",
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
                    "solver_tol": row.solver_tol,
                    "regularization": row.regularization,
                    "elapsed_s": row.elapsed_s,
                    "systems_per_s": row.systems_per_s,
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

    modes = sorted({row.mode for row in rows})
    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(
        len(modes),
        len(dtypes),
        figsize=(7 * len(dtypes), 4.5 * len(modes)),
        squeeze=False,
        sharey="row",
    )
    for row_idx, mode in enumerate(modes):
        for ax, dtype in zip(axes[row_idx], dtypes, strict=True):
            dtype_rows = [row for row in rows if row.dtype == dtype and row.mode == mode]
            dtype_rows.sort(key=lambda row: row.batch_size)
            if dtype_rows:
                ax.plot(
                    [row.batch_size for row in dtype_rows],
                    [row.systems_per_s for row in dtype_rows],
                    marker="P" if mode == "solve" else "v",
                    linewidth=1.8,
                    label=f"QPAX {mode}",
                )
            cpu = next((row.cpu_throughput for row in dtype_rows if row.cpu_throughput), None)
            if mode == "solve" and cpu is not None:
                ax.axhline(cpu, linestyle="--", color="black", linewidth=1.5, label="CPU OSQP")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Solved MPC problems/s" if mode == "solve" else "Solved MPC gradients/s")
            ax.set_title(f"{dtype}, {mode}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("QPAX fixed-pattern quadcopter MPC throughput")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--batch-sizes", default="512,2048")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--modes", default="grad")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--solver-tol", type=float, default=-1.0)
    parser.add_argument("--target-kappa", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=128)
    parser.add_argument("--cpu-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eq-tol", type=float, default=1e-9)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--csv-path", default="results/linear_mpc_gradients/throughput_mpc_qpax_gradients.csv")
    parser.add_argument("--plot-path", default="results/linear_mpc_gradients/throughput_mpc_qpax_gradients.png")
    parser.add_argument("--no-cpu", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    settings = OSQPSettings(
        max_iter=args.max_iter,
        scaling=0,
        adaptive_rho=False,
        rho_is_vec=False,
        check_termination=0,
        warm_starting=False,
        polishing=False,
    )
    problem = make_linear_mpc(args.horizon)
    plan = build_osqp_plan(problem.p_matrix, problem.a_matrix, problem.l, problem.u, settings)
    print("Devices:", jax.devices())
    print(
        "Problem:",
        f"N={args.horizon}",
        f"n={plan.n}",
        f"m={plan.m}",
        f"max_iter={args.max_iter}",
        f"solver_tol={args.solver_tol:g}",
    )

    cpu_throughput = None
    if not args.no_cpu:
        _, _, q_cpu, l_cpu, u_cpu = batched_problem_data(
            problem, args.cpu_samples, np.float64, seed=args.seed, x0_variation=0.15
        )
        cpu_throughput = _benchmark_cpu_osqp(problem, settings, q_cpu, l_cpu, u_cpu, args.cpu_repeat)
        print(f"CPU OSQP fixed-rho throughput: {cpu_throughput:.2f}/s")

    rows: list[BatchResult] = []
    max_bytes = int(args.max_device_gb * 1024**3)
    modes = _parse_modes(args.modes)
    for dtype in _parse_dtypes(args.dtypes):
        jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
        qp = _qpax_problem(problem, dtype, args.regularization, args.eq_tol)
        solve_fn = _make_solve_fn(qp, args.solver_tol, args.max_iter)
        grad_fn = _make_grad_fn(qp, args.solver_tol, args.target_kappa, args.max_iter)
        print(
            f"QPAX dtype={dtype.name}: n={qp.n} n_eq={qp.n_eq} n_ineq={qp.n_ineq} "
            f"regularization={args.regularization:g}"
        )

        max_x = None
        max_obj = None
        if not args.no_check:
            _, _, q_check, l_check, u_check = batched_problem_data(
                problem, 2, dtype, seed=args.seed + 10, x0_variation=0.15
            )
            b_check, h_check = _qpax_bounds(qp, l_check, u_check, dtype)
            check_solve, _ = _compile_and_warm(
                solve_fn,
                jnp.asarray(q_check),
                jnp.asarray(b_check),
                jnp.asarray(h_check),
            )
            max_x, max_obj = _check_against_cpu(problem, settings, check_solve, qp, q_check, l_check, u_check, dtype)
            print(f"  check max_abs_x={max_x:.3e} max_abs_obj={max_obj:.3e}")

        for mode in modes:
            print(f"Benchmark dtype={dtype.name} mode={mode}")
            for batch in _parse_ints(args.batch_sizes):
                estimate = _estimate_bytes(batch, qp, dtype, mode)
                if estimate > max_bytes:
                    print(f"  batch={batch}: skipped estimated {estimate / 1024**3:.2f} GiB")
                    continue
                _, _, q, l, u = batched_problem_data(
                    problem, batch, dtype, seed=args.seed + batch, x0_variation=0.15
                )
                b, h = _qpax_bounds(qp, l, u, dtype)
                q_jax = jnp.asarray(q)
                b_jax = jnp.asarray(b)
                h_jax = jnp.asarray(h)
                try:
                    if mode == "solve":
                        compiled, _ = _compile_and_warm(solve_fn, q_jax, b_jax, h_jax)
                        elapsed, _ = _time_compiled(compiled, q_jax, b_jax, h_jax, repeat=args.repeat)
                    else:
                        rng = np.random.default_rng(args.seed + 1000 + batch)
                        x_weights = jnp.asarray(rng.standard_normal((batch, qp.n)).astype(dtype))
                        compiled, _ = _compile_and_warm(grad_fn, q_jax, b_jax, h_jax, x_weights)
                        elapsed, _ = _time_compiled(compiled, q_jax, b_jax, h_jax, x_weights, repeat=args.repeat)
                except Exception as exc:
                    print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                    continue
                row = BatchResult(
                    solver="qpax",
                    mode=mode,
                    dtype=dtype.name,
                    batch_size=batch,
                    horizon=args.horizon,
                    n=plan.n,
                    m=plan.m,
                    n_eq=qp.n_eq,
                    n_ineq=qp.n_ineq,
                    max_iter=args.max_iter,
                    solver_tol=args.solver_tol,
                    regularization=args.regularization,
                    elapsed_s=elapsed,
                    max_abs_x_vs_cpu=max_x,
                    max_abs_obj_vs_cpu=max_obj,
                    cpu_throughput=cpu_throughput,
                )
                rows.append(row)
                print(f"  batch={batch}: elapsed={elapsed:.6f}s throughput={row.systems_per_s:.2f}/s")

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
