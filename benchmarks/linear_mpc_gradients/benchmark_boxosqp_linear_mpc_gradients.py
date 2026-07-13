#!/usr/bin/env python3
"""Benchmark JAXopt BoxOSQP on fixed-pattern batched quadcopter MPC QPs."""

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
import osqp
from jaxopt import BoxOSQP
from jaxopt.tree_util import tree_map

from benchmarks.problems import batched_problem_data, make_linear_mpc
from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan


class DTypeStableOSQPLUSolver:
    """JAXopt LU solver variant that keeps float32 factorizations in float32."""

    @staticmethod
    def _lu_factor_dense(q_matrix, a_matrix, sigma, rho_bar):
        dtype = q_matrix.dtype
        sigma = jnp.asarray(sigma, dtype=dtype)
        rho_bar = jnp.asarray(rho_bar, dtype=dtype)
        dense = (
            q_matrix
            + sigma * jnp.eye(q_matrix.shape[0], dtype=dtype)
            + rho_bar * (a_matrix.T @ a_matrix)
        )
        return jax.scipy.linalg.lu_factor(dense)

    def _lu_factor_pytree(self, params_q, params_a, sigma, rho_bar):
        lu_factor_dense = lambda q_matrix, a_matrix: self._lu_factor_dense(
            q_matrix, a_matrix, sigma, rho_bar
        )
        return tree_map(lu_factor_dense, params_q, params_a)

    def init_state(self, init_params, params_q, params_a, sigma, rho_bar):
        lu_factors = self._lu_factor_pytree(params_q, params_a, sigma, rho_bar)
        return (params_q, params_a, sigma), lu_factors

    def update_stepsize(self, solver_state, rho_bar):
        (params_q, params_a, sigma), _ = solver_state
        lu_factors = self._lu_factor_pytree(params_q, params_a, sigma, rho_bar)
        return (params_q, params_a, sigma), lu_factors

    def run(self, b, osqp_state):
        _, lu_factors = osqp_state.solver_state

        def lu_solve(rhs, factors):
            lu, _ = factors
            rhs = jnp.asarray(rhs, dtype=lu.dtype)
            return jax.scipy.linalg.lu_solve(factors, rhs, check_finite=False)

        sol = tree_map(lu_solve, b, lu_factors)
        return sol, osqp_state.solver_state


class FixedRhoBoxOSQP(BoxOSQP):
    """BoxOSQP with rho adaptation disabled to match the fixed-rho OSQP benchmarks."""

    def __post_init__(self):
        super().__post_init__()
        if self.eq_qp_solve.lower() == "lu":
            self._eq_qp_solve_impl = DTypeStableOSQPLUSolver()

    def init_state(self, init_params, params_obj, params_eq, params_ineq):
        state = super().init_state(init_params, params_obj, params_eq, params_ineq)
        dtype = params_obj[1].dtype
        return state._replace(
            error=jnp.asarray(state.error, dtype=dtype),
            rho_bar=jnp.asarray(state.rho_bar, dtype=dtype),
        )

    def _update_stepsize(self, rho_bar, solver_state, primal_residuals, dual_residuals, Q, c, A, x, y):
        return rho_bar, solver_state


@dataclass(frozen=True)
class BatchResult:
    solver: str
    mode: str
    dtype: str
    batch_size: int
    horizon: int
    n: int
    m: int
    eq_qp_solve: str
    implicit_diff: bool
    max_iter: int
    elapsed_s: float
    max_abs_x_vs_cpu: float | None
    max_abs_obj_vs_cpu: float | None
    cpu_throughput: float | None

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.elapsed_s


@dataclass(frozen=True)
class HorizonResult:
    solver: str
    dtype: str
    horizon: int
    batch_size: int
    n: int
    m: int
    kkt_dim: int
    kkt_nnz: int
    eq_qp_solve: str
    implicit_diff: bool
    max_iter: int
    elapsed_s: float

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


def _make_solver(args) -> FixedRhoBoxOSQP:
    return FixedRhoBoxOSQP(
        eq_qp_solve=args.eq_qp_solve,
        sigma=args.sigma,
        momentum=args.alpha,
        rho_start=args.rho,
        rho_min=args.rho,
        rho_max=args.rho,
        maxiter=args.max_iter,
        tol=-1.0,
        termination_check_frequency=args.max_iter + 1,
        stepsize_updates_frequency=args.max_iter + 1,
        check_primal_dual_infeasability=False,
        implicit_diff=args.implicit_diff,
        jit=True,
        unroll="auto",
    )


def _make_solve_fn(solver: FixedRhoBoxOSQP, q_matrix: np.ndarray, a_matrix: np.ndarray):
    q_jax = jnp.asarray(q_matrix)
    a_jax = jnp.asarray(a_matrix)

    def solve_one(q, l, u):
        out = solver.run(
            init_params=None,
            params_obj=(q_jax, q),
            params_eq=a_jax,
            params_ineq=(l, u),
        )
        x, z = out.params.primal
        y = out.params.dual_eq
        return x, z, y, out.state.error

    return jax.jit(jax.vmap(solve_one, in_axes=(0, 0, 0)))


def _make_grad_fn(solver: FixedRhoBoxOSQP, q_matrix: np.ndarray, a_matrix: np.ndarray):
    solve_batch = _make_solve_fn(solver, q_matrix, a_matrix)

    def objective(q, l, u, x_weights, y_weights):
        x, _, y, _ = solve_batch(q, l, u)
        return jnp.sum(x * x_weights) + 0.01 * jnp.sum(y * y_weights)

    return jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))


def _check_against_cpu(problem, settings: OSQPSettings, compiled_solve, q, l, u) -> tuple[float, float]:
    x, _, _, _ = jax.device_get(compiled_solve(jnp.asarray(q), jnp.asarray(l), jnp.asarray(u)))
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


def _estimate_bytes(batch: int, n: int, m: int, dtype: np.dtype, *, mode: str) -> int:
    item = np.dtype(dtype).itemsize
    dense_constants = n * n + m * n
    per_batch = 3 * m + 4 * n
    if mode == "grad":
        per_batch += 2 * m + n
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
                "eq_qp_solve",
                "implicit_diff",
                "max_iter",
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
                    "eq_qp_solve": row.eq_qp_solve,
                    "implicit_diff": row.implicit_diff,
                    "max_iter": row.max_iter,
                    "elapsed_s": row.elapsed_s,
                    "systems_per_s": row.systems_per_s,
                    "max_abs_x_vs_cpu": row.max_abs_x_vs_cpu,
                    "max_abs_obj_vs_cpu": row.max_abs_obj_vs_cpu,
                    "cpu_osqp_systems_per_s": row.cpu_throughput,
                }
            )


def _write_horizon_csv(path: pathlib.Path, rows: list[HorizonResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver",
                "dtype",
                "horizon",
                "batch_size",
                "n",
                "m",
                "kkt_dim",
                "kkt_nnz",
                "eq_qp_solve",
                "implicit_diff",
                "max_iter",
                "elapsed_s",
                "systems_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "solver": row.solver,
                    "dtype": row.dtype,
                    "horizon": row.horizon,
                    "batch_size": row.batch_size,
                    "n": row.n,
                    "m": row.m,
                    "kkt_dim": row.kkt_dim,
                    "kkt_nnz": row.kkt_nnz,
                    "eq_qp_solve": row.eq_qp_solve,
                    "implicit_diff": row.implicit_diff,
                    "max_iter": row.max_iter,
                    "elapsed_s": row.elapsed_s,
                    "systems_per_s": row.systems_per_s,
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
                    marker="s" if mode == "solve" else "^",
                    linewidth=1.8,
                    label=f"BoxOSQP {mode}",
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
    fig.suptitle("JAXopt BoxOSQP fixed-pattern quadcopter MPC throughput")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _kkt_upper_density_percent(row: HorizonResult) -> float:
    return 100.0 * row.kkt_nnz / (row.kkt_dim * (row.kkt_dim + 1) / 2)


def _plot_horizon(path: pathlib.Path, rows: list[HorizonResult]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 5), squeeze=False, sharey=True)
    for ax, dtype in zip(axes[0], dtypes, strict=True):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        dtype_rows.sort(key=lambda row: row.horizon)
        ax.plot(
            [row.horizon for row in dtype_rows],
            [row.systems_per_s for row in dtype_rows],
            marker="s",
            linewidth=1.8,
            label="BoxOSQP lu",
        )
        for row in dtype_rows:
            ax.annotate(
                f"K={row.kkt_dim}\n{_kkt_upper_density_percent(row):.2f}% nnz",
                (row.horizon, row.systems_per_s),
                textcoords="offset points",
                xytext=(4, 7),
                fontsize=7,
                alpha=0.85,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("MPC horizon N")
        ax.set_ylabel("Solved MPC problems/s")
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("JAXopt BoxOSQP horizon sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--modes", default="grad")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=128)
    parser.add_argument("--cpu-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--eq-qp-solve", choices=["cg", "cg+jacobi", "lu"], default="cg")
    parser.add_argument("--implicit-diff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv-path", default="results/linear_mpc_gradients/throughput_mpc_boxosqp_gradients.csv")
    parser.add_argument("--plot-path", default="results/linear_mpc_gradients/throughput_mpc_boxosqp_gradients.png")
    parser.add_argument("--horizon-sweep", default="5,10,20,40")
    parser.add_argument("--horizon-batch-sizes", default="512,2048,10000,20000")
    parser.add_argument("--horizon-csv-path", default="results/linear_mpc_gradients/horizon_sweep_mpc_boxosqp_gradients.csv")
    parser.add_argument("--horizon-plot-path", default="results/linear_mpc_gradients/horizon_sweep_mpc_boxosqp_gradients.png")
    parser.add_argument("--skip-horizon-sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-cpu", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    settings = OSQPSettings(
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
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
    q_dense_by_dtype = {}
    a_dense_by_dtype = {}
    print("Devices:", jax.devices())
    print(
        "Problem:",
        f"N={args.horizon}",
        f"n={plan.n}",
        f"m={plan.m}",
        f"max_iter={args.max_iter}",
        f"eq_qp_solve={args.eq_qp_solve}",
        f"implicit_diff={args.implicit_diff}",
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
        q_dense_by_dtype[dtype.name] = np.asarray(problem.p_matrix.toarray(), dtype=dtype)
        a_dense_by_dtype[dtype.name] = np.asarray(problem.a_matrix.toarray(), dtype=dtype)
        solver = _make_solver(args)
        solve_fn = _make_solve_fn(solver, q_dense_by_dtype[dtype.name], a_dense_by_dtype[dtype.name])
        grad_fn = _make_grad_fn(solver, q_dense_by_dtype[dtype.name], a_dense_by_dtype[dtype.name])

        max_x = None
        max_obj = None
        if not args.no_check:
            _, _, q_check, l_check, u_check = batched_problem_data(
                problem, 2, dtype, seed=args.seed + 10, x0_variation=0.15
            )
            check_solve, _ = _compile_and_warm(
                solve_fn,
                jnp.asarray(q_check),
                jnp.asarray(l_check),
                jnp.asarray(u_check),
            )
            max_x, max_obj = _check_against_cpu(problem, settings, check_solve, q_check, l_check, u_check)
            print(f"Check dtype={dtype.name}: max_abs_x={max_x:.3e} max_abs_obj={max_obj:.3e}")

        for mode in modes:
            print(f"Benchmark dtype={dtype.name} mode={mode}")
            for batch in _parse_ints(args.batch_sizes):
                estimate = _estimate_bytes(batch, plan.n, plan.m, dtype, mode=mode)
                if estimate > max_bytes:
                    print(f"  batch={batch}: skipped estimated {estimate / 1024**3:.2f} GiB")
                    break
                _, _, q, l, u = batched_problem_data(
                    problem, batch, dtype, seed=args.seed + batch, x0_variation=0.15
                )
                q_jax = jnp.asarray(q)
                l_jax = jnp.asarray(l)
                u_jax = jnp.asarray(u)
                try:
                    if mode == "solve":
                        compiled, _ = _compile_and_warm(solve_fn, q_jax, l_jax, u_jax)
                        elapsed, _ = _time_compiled(compiled, q_jax, l_jax, u_jax, repeat=args.repeat)
                    else:
                        rng = np.random.default_rng(args.seed + 1000 + batch)
                        x_weights = jnp.asarray(rng.standard_normal((batch, plan.n)).astype(dtype))
                        y_weights = jnp.asarray(rng.standard_normal((batch, plan.m)).astype(dtype))
                        compiled, _ = _compile_and_warm(grad_fn, q_jax, l_jax, u_jax, x_weights, y_weights)
                        elapsed, _ = _time_compiled(
                            compiled,
                            q_jax,
                            l_jax,
                            u_jax,
                            x_weights,
                            y_weights,
                            repeat=args.repeat,
                        )
                except Exception as exc:
                    print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                    break
                row = BatchResult(
                    solver="boxosqp",
                    mode=mode,
                    dtype=dtype.name,
                    batch_size=batch,
                    horizon=args.horizon,
                    n=plan.n,
                    m=plan.m,
                    eq_qp_solve=args.eq_qp_solve,
                    implicit_diff=args.implicit_diff,
                    max_iter=args.max_iter,
                    elapsed_s=elapsed,
                    max_abs_x_vs_cpu=max_x,
                    max_abs_obj_vs_cpu=max_obj,
                    cpu_throughput=cpu_throughput,
                )
                rows.append(row)
                print(f"  batch={batch}: elapsed={elapsed:.6f}s throughput={row.systems_per_s:.2f}/s")

    horizon_rows: list[HorizonResult] = []
    horizon_batches = sorted(_parse_ints(args.horizon_batch_sizes), reverse=True)
    if not args.skip_horizon_sweep:
        for dtype in _parse_dtypes(args.dtypes):
            jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
            for horizon in _parse_ints(args.horizon_sweep):
                horizon_problem = make_linear_mpc(horizon)
                horizon_plan = build_osqp_plan(
                    horizon_problem.p_matrix,
                    horizon_problem.a_matrix,
                    horizon_problem.l,
                    horizon_problem.u,
                    settings,
                )
                horizon_q_dense = np.asarray(horizon_problem.p_matrix.toarray(), dtype=dtype)
                horizon_a_dense = np.asarray(horizon_problem.a_matrix.toarray(), dtype=dtype)
                horizon_solver = _make_solver(args)
                horizon_solve = _make_solve_fn(horizon_solver, horizon_q_dense, horizon_a_dense)
                print(
                    f"Horizon N={horizon} dtype={dtype.name}: n={horizon_plan.n} m={horizon_plan.m} "
                    f"kkt_dim={horizon_plan.qdldl_plan.n} kkt_nnz={horizon_plan.nnz_kkt}"
                )
                for batch in horizon_batches:
                    estimate = _estimate_bytes(batch, horizon_plan.n, horizon_plan.m, dtype, mode="solve")
                    if estimate > max_bytes:
                        print(f"  batch={batch}: skipped estimated {estimate / 1024**3:.2f} GiB")
                        continue
                    _, _, q, l, u = batched_problem_data(
                        horizon_problem,
                        batch,
                        dtype,
                        seed=args.seed + 4000 + horizon + batch,
                        x0_variation=0.15,
                    )
                    q_jax = jnp.asarray(q)
                    l_jax = jnp.asarray(l)
                    u_jax = jnp.asarray(u)
                    try:
                        compiled, _ = _compile_and_warm(horizon_solve, q_jax, l_jax, u_jax)
                        elapsed, _ = _time_compiled(compiled, q_jax, l_jax, u_jax, repeat=args.repeat)
                    except Exception as exc:
                        print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                        continue
                    row = HorizonResult(
                        solver="boxosqp",
                        dtype=dtype.name,
                        horizon=horizon,
                        batch_size=batch,
                        n=horizon_plan.n,
                        m=horizon_plan.m,
                        kkt_dim=horizon_plan.qdldl_plan.n,
                        kkt_nnz=horizon_plan.nnz_kkt,
                        eq_qp_solve=args.eq_qp_solve,
                        implicit_diff=args.implicit_diff,
                        max_iter=args.max_iter,
                        elapsed_s=elapsed,
                    )
                    horizon_rows.append(row)
                    print(f"  batch={batch}: elapsed={elapsed:.6f}s throughput={row.systems_per_s:.2f}/s")
                    break

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    horizon_csv_path = pathlib.Path(args.horizon_csv_path)
    horizon_plot_path = pathlib.Path(args.horizon_plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_horizon_sweep:
        horizon_csv_path.parent.mkdir(parents=True, exist_ok=True)
        horizon_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    if not args.skip_horizon_sweep:
        _write_horizon_csv(horizon_csv_path, horizon_rows)
        _plot_horizon(horizon_plot_path, horizon_rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    if not args.skip_horizon_sweep:
        print(f"Wrote {horizon_csv_path}")
        print(f"Wrote {horizon_plot_path}")


if __name__ == "__main__":
    main()
