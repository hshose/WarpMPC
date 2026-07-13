#!/usr/bin/env python3
"""Benchmark MPAX on fixed-pattern batched quadcopter MPC QPs."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUNTIME_IMPORTED = False


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
    nnz_q: int
    nnz_a_eq: int
    nnz_g_ineq: int
    iteration_limit: int
    eps_abs: float
    eps_rel: float
    l_inf_ruiz_iterations: int
    regularization: float
    elapsed_s: float
    mean_iterations: float
    max_iterations: int
    status_values: str
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


def _configure_jax_for_dtype(dtype: np.dtype) -> None:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", np.dtype(dtype) == np.dtype(np.float64))
    from benchmarks.jax_cache import configure_jax_compilation_cache

    configure_jax_compilation_cache()


def _import_runtime() -> None:
    global _RUNTIME_IMPORTED
    global OSQPSettings, batched_problem_data, jax, jnp, make_batched_solve_fn
    global make_linear_mpc, make_mpax_linear_mpc_problem, make_solver, mpax_rhs, osqp

    if _RUNTIME_IMPORTED:
        return

    import jax as _jax
    import jax.numpy as _jnp
    import osqp as _osqp

    from benchmarks.mpax_linear_mpc_utils import (
        make_batched_solve_fn as _make_batched_solve_fn,
        make_mpax_linear_mpc_problem as _make_mpax_linear_mpc_problem,
        make_solver as _make_solver,
        mpax_rhs as _mpax_rhs,
    )
    from benchmarks.problems import batched_problem_data as _batched_problem_data
    from benchmarks.problems import make_linear_mpc as _make_linear_mpc
    from warpmpc.jax_osqp import OSQPSettings as _OSQPSettings

    jax = _jax
    jnp = _jnp
    osqp = _osqp
    make_batched_solve_fn = _make_batched_solve_fn
    make_mpax_linear_mpc_problem = _make_mpax_linear_mpc_problem
    make_solver = _make_solver
    mpax_rhs = _mpax_rhs
    batched_problem_data = _batched_problem_data
    make_linear_mpc = _make_linear_mpc
    OSQPSettings = _OSQPSettings
    _RUNTIME_IMPORTED = True


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


def _check_against_cpu(problem, settings: OSQPSettings, compiled_solve, q, rhs, l, u):
    x, obj, _, _ = jax.device_get(compiled_solve(jnp.asarray(q), jnp.asarray(rhs)))
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


def _estimate_bytes(batch: int, qp, dtype: np.dtype, iteration_limit: int, *, unroll: bool) -> int:
    item = np.dtype(dtype).itemsize
    sparse_constants = qp.nnz_q + qp.nnz_a_eq + qp.nnz_g_ineq
    per_batch = 8 * (qp.n + qp.n_constraints)
    if unroll:
        per_batch += iteration_limit * 4 * (qp.n + qp.n_constraints)
    return item * (sparse_constants + batch * per_batch)


def _status_values(status) -> str:
    values = np.asarray(jax.device_get(status)).astype(np.int64, copy=False)
    unique = np.unique(values)
    return ";".join(str(int(value)) for value in unique)


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
                "nnz_q",
                "nnz_a_eq",
                "nnz_g_ineq",
                "iteration_limit",
                "eps_abs",
                "eps_rel",
                "l_inf_ruiz_iterations",
                "regularization",
                "elapsed_s",
                "systems_per_s",
                "mean_iterations",
                "max_iterations",
                "status_values",
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
                    "nnz_q": row.nnz_q,
                    "nnz_a_eq": row.nnz_a_eq,
                    "nnz_g_ineq": row.nnz_g_ineq,
                    "iteration_limit": row.iteration_limit,
                    "eps_abs": row.eps_abs,
                    "eps_rel": row.eps_rel,
                    "l_inf_ruiz_iterations": row.l_inf_ruiz_iterations,
                    "regularization": row.regularization,
                    "elapsed_s": row.elapsed_s,
                    "systems_per_s": row.systems_per_s,
                    "mean_iterations": row.mean_iterations,
                    "max_iterations": row.max_iterations,
                    "status_values": row.status_values,
                    "max_abs_x_vs_cpu": row.max_abs_x_vs_cpu,
                    "max_abs_obj_vs_cpu": row.max_abs_obj_vs_cpu,
                    "cpu_osqp_systems_per_s": row.cpu_throughput,
                }
            )


def _optional_float(value: str) -> float | None:
    return None if value in {"", "None"} else float(value)


def _read_csv(path: pathlib.Path) -> list[BatchResult]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            BatchResult(
                solver=row["solver"],
                mode=row["mode"],
                dtype=row["dtype"],
                batch_size=int(row["batch_size"]),
                horizon=int(row["horizon"]),
                n=int(row["n"]),
                m=int(row["m"]),
                n_eq=int(row["n_eq"]),
                n_ineq=int(row["n_ineq"]),
                nnz_q=int(row["nnz_q"]),
                nnz_a_eq=int(row["nnz_a_eq"]),
                nnz_g_ineq=int(row["nnz_g_ineq"]),
                iteration_limit=int(row["iteration_limit"]),
                eps_abs=float(row["eps_abs"]),
                eps_rel=float(row["eps_rel"]),
                l_inf_ruiz_iterations=int(row["l_inf_ruiz_iterations"]),
                regularization=float(row["regularization"]),
                elapsed_s=float(row["elapsed_s"]),
                mean_iterations=float(row["mean_iterations"]),
                max_iterations=int(row["max_iterations"]),
                status_values=row["status_values"],
                max_abs_x_vs_cpu=_optional_float(row["max_abs_x_vs_cpu"]),
                max_abs_obj_vs_cpu=_optional_float(row["max_abs_obj_vs_cpu"]),
                cpu_throughput=_optional_float(row["cpu_osqp_systems_per_s"]),
            )
            for row in reader
        ]


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
            marker="^",
            linewidth=1.8,
            label="MPAX",
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
    fig.suptitle("MPAX fixed-pattern quadcopter MPC throughput")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_benchmark(args: argparse.Namespace) -> None:
    _import_runtime()
    dtypes = _parse_dtypes(args.dtypes)
    settings = OSQPSettings(
        max_iter=args.iteration_limit,
        eps_abs=args.eps_abs,
        eps_rel=args.eps_rel,
        scaling=0,
        adaptive_rho=False,
        rho_is_vec=False,
        check_termination=0,
        warm_starting=False,
        polishing=False,
    )
    problem = make_linear_mpc(args.horizon)
    print("Devices:", jax.devices())
    print(
        "Problem:",
        f"N={args.horizon}",
        f"n={problem.q.size}",
        f"m={problem.l.size}",
        f"iteration_limit={args.iteration_limit}",
        f"eps_abs={args.eps_abs:g}",
        f"eps_rel={args.eps_rel:g}",
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
    for dtype in dtypes:
        qp = make_mpax_linear_mpc_problem(
            problem,
            dtype,
            regularization=args.regularization,
            eq_tol=args.eq_tol,
        )
        solver = make_solver(
            eps_abs=args.eps_abs,
            eps_rel=args.eps_rel,
            iteration_limit=args.iteration_limit,
            termination_evaluation_frequency=args.termination_evaluation_frequency,
            l_inf_ruiz_iterations=args.l_inf_ruiz_iterations,
            pock_chambolle_alpha=args.pock_chambolle_alpha,
            unroll=args.unroll,
        )
        solve_fn = make_batched_solve_fn(qp, solver)
        print(
            f"MPAX dtype={dtype.name}: n={qp.n} n_eq={qp.n_eq} n_ineq={qp.n_ineq} "
            f"nnz_Q={qp.nnz_q} nnz_Aeq={qp.nnz_a_eq} nnz_G={qp.nnz_g_ineq} "
            f"regularization={args.regularization:g}"
        )

        max_x = None
        max_obj = None
        if not args.no_check:
            _, _, q_check, l_check, u_check = batched_problem_data(
                problem, 2, dtype, seed=args.seed + 10, x0_variation=0.15
            )
            rhs_check = mpax_rhs(qp, l_check, u_check, dtype)
            check_solve, _ = _compile_and_warm(
                solve_fn,
                jnp.asarray(q_check),
                jnp.asarray(rhs_check),
            )
            max_x, max_obj = _check_against_cpu(
                problem,
                settings,
                check_solve,
                q_check,
                rhs_check,
                l_check,
                u_check,
            )
            print(f"  check max_abs_x={max_x:.3e} max_abs_obj={max_obj:.3e}")

        for batch in _parse_ints(args.batch_sizes):
            estimate = _estimate_bytes(batch, qp, dtype, args.iteration_limit, unroll=args.unroll)
            if estimate > max_bytes:
                print(f"  batch={batch}: skipped estimated {estimate / 1024**3:.2f} GiB")
                continue
            _, _, q, l, u = batched_problem_data(
                problem, batch, dtype, seed=args.seed + batch, x0_variation=0.15
            )
            rhs = mpax_rhs(qp, l, u, dtype)
            q_jax = jnp.asarray(q)
            rhs_jax = jnp.asarray(rhs)
            try:
                compiled, _ = _compile_and_warm(solve_fn, q_jax, rhs_jax)
                elapsed, out = _time_compiled(compiled, q_jax, rhs_jax, repeat=args.repeat)
            except Exception as exc:
                print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                continue
            _, _, iterations, status = out
            iterations_np = np.asarray(jax.device_get(iterations))
            row = BatchResult(
                solver="mpax",
                mode="solve",
                dtype=dtype.name,
                batch_size=batch,
                horizon=args.horizon,
                n=qp.n,
                m=qp.m_original,
                n_eq=qp.n_eq,
                n_ineq=qp.n_ineq,
                nnz_q=qp.nnz_q,
                nnz_a_eq=qp.nnz_a_eq,
                nnz_g_ineq=qp.nnz_g_ineq,
                iteration_limit=args.iteration_limit,
                eps_abs=args.eps_abs,
                eps_rel=args.eps_rel,
                l_inf_ruiz_iterations=args.l_inf_ruiz_iterations,
                regularization=args.regularization,
                elapsed_s=elapsed,
                mean_iterations=float(np.mean(iterations_np)),
                max_iterations=int(np.max(iterations_np)),
                status_values=_status_values(status),
                max_abs_x_vs_cpu=max_x,
                max_abs_obj_vs_cpu=max_obj,
                cpu_throughput=cpu_throughput,
            )
            rows.append(row)
            print(
                f"  batch={batch}: elapsed={elapsed:.6f}s throughput={row.systems_per_s:.2f}/s "
                f"iters_mean={row.mean_iterations:.1f} status={row.status_values}"
            )

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


def _worker_cmd(args: argparse.Namespace, dtype: np.dtype, csv_path: pathlib.Path, plot_path: pathlib.Path) -> list[str]:
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--horizon",
        str(args.horizon),
        "--batch-sizes",
        args.batch_sizes,
        "--dtypes",
        dtype.name,
        "--iteration-limit",
        str(args.iteration_limit),
        "--eps-abs",
        str(args.eps_abs),
        "--eps-rel",
        str(args.eps_rel),
        "--termination-evaluation-frequency",
        str(args.termination_evaluation_frequency),
        "--l-inf-ruiz-iterations",
        str(args.l_inf_ruiz_iterations),
        "--pock-chambolle-alpha",
        str(args.pock_chambolle_alpha),
        "--regularization",
        str(args.regularization),
        "--eq-tol",
        str(args.eq_tol),
        "--repeat",
        str(args.repeat),
        "--cpu-samples",
        str(args.cpu_samples),
        "--cpu-repeat",
        str(args.cpu_repeat),
        "--seed",
        str(args.seed),
        "--max-device-gb",
        str(args.max_device_gb),
        "--csv-path",
        str(csv_path),
        "--plot-path",
        str(plot_path),
        "--isolated-dtype-worker",
    ]
    cmd.append("--unroll" if args.unroll else "--no-unroll")
    if args.no_cpu:
        cmd.append("--no-cpu")
    if args.no_check:
        cmd.append("--no-check")
    return cmd


def _run_isolated_dtype_workers(args: argparse.Namespace, dtypes: list[np.dtype]) -> None:
    rows: list[BatchResult] = []
    with tempfile.TemporaryDirectory(prefix="mpax_linear_mpc_") as tmp:
        tmpdir = pathlib.Path(tmp)
        for dtype in dtypes:
            csv_path = tmpdir / f"{dtype.name}.csv"
            plot_path = tmpdir / f"{dtype.name}.png"
            env = os.environ.copy()
            env["JAX_ENABLE_X64"] = "1" if dtype == np.dtype(np.float64) else "0"
            print(
                f"Launching isolated MPAX dtype={dtype.name} with JAX_ENABLE_X64={env['JAX_ENABLE_X64']}",
                flush=True,
            )
            subprocess.run(
                _worker_cmd(args, dtype, csv_path, plot_path),
                cwd=ROOT,
                env=env,
                check=True,
            )
            rows.extend(_read_csv(csv_path))

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--batch-sizes", default="512,2048")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--iteration-limit", type=int, default=25)
    parser.add_argument("--eps-abs", type=float, default=1e-3)
    parser.add_argument("--eps-rel", type=float, default=1e-3)
    parser.add_argument("--termination-evaluation-frequency", type=int, default=100)
    parser.add_argument("--l-inf-ruiz-iterations", type=int, default=10)
    parser.add_argument("--pock-chambolle-alpha", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--eq-tol", type=float, default=1e-9)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=128)
    parser.add_argument("--cpu-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--unroll", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--csv-path", default="results/linear_mpc/throughput_mpc_mpax.csv")
    parser.add_argument("--plot-path", default="results/linear_mpc/throughput_mpc_mpax.png")
    parser.add_argument("--no-cpu", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--isolated-dtype-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    dtypes = _parse_dtypes(args.dtypes)
    if not dtypes:
        raise ValueError("--dtypes must contain at least one dtype")
    if len(dtypes) > 1 and not args.isolated_dtype_worker:
        _run_isolated_dtype_workers(args, dtypes)
        return

    _configure_jax_for_dtype(dtypes[0])
    _run_benchmark(args)


if __name__ == "__main__":
    main()
