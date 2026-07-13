#!/usr/bin/env python3
"""Benchmark fixed-pattern JAX OSQP solves with adjoint gradients for linear MPC."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from collections import Counter
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np
import osqp

from benchmarks.problems import batched_problem_data, make_linear_mpc
from warpmpc.jax_osqp import (
    OSQPSettings,
    build_osqp_plan,
    compile_osqp,
)


VARIANTS = {
    "baseline": {},
    "transpose+segmented": {"transpose_work": True, "segmented": True},
    "transpose+segmented+levelsolve": {
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
    "factor-warp+solve-jax:transpose+segmented": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "jax",
        "transpose_work": True,
        "segmented": True,
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
    },
    "factor-warp+solve-warp:transpose+segmented": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "warp",
        "transpose_work": True,
        "segmented": True,
    },
    "factor-warp+solve-warp:transpose+segmented+levelsolve": {
        "qdldl_factor_backend": "warp",
        "qdldl_solve_backend": "warp",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
}

MARKERS = {
    "baseline": "o",
    "transpose+segmented": "X",
    "transpose+segmented+levelsolve": ">",
    "factor-warp+solve-jax:transpose+segmented": "P",
    "factor-warp+solve-jax:transpose+segmented+levelsolve": "*",
    "factor-warp+solve-warp:transpose": "h",
    "factor-warp+solve-warp:transpose+segmented": "^",
    "factor-warp+solve-warp:transpose+segmented+levelsolve": "v",
}


@dataclass(frozen=True)
class GradientBatchResult:
    variant: str
    dtype: str
    batch_size: int
    horizon: int
    n: int
    m: int
    kkt_dim: int
    kkt_nnz: int
    nnz_l: int
    adjoint_dim: int
    adjoint_nnz: int
    adjoint_nnz_l: int
    derivative_refinement_iters: int
    value_grad_s: float
    cpu_throughput: float | None
    cpu_status_counts: str | None

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.value_grad_s


@dataclass(frozen=True)
class GradientHorizonResult:
    variant: str
    dtype: str
    horizon: int
    batch_size: int
    n: int
    m: int
    kkt_dim: int
    kkt_nnz: int
    nnz_l: int
    adjoint_dim: int
    adjoint_nnz: int
    adjoint_nnz_l: int
    derivative_refinement_iters: int
    value_grad_s: float

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.value_grad_s


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


def _make_value_and_grad(compiled):
    def objective(p_values, a_values, q, l, u, dx_seed, dy_seed):
        x, y = compiled.solve_xy(p_values, a_values, q, l, u)
        return jnp.sum(x * dx_seed) + jnp.sum(y * dy_seed)

    return jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2, 3, 4)))


def _gradient_seeds(plan, batch: int, dtype: np.dtype, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dx = rng.standard_normal((batch, plan.n)).astype(dtype)
    dy = rng.standard_normal((batch, plan.m)).astype(dtype)
    return dx, dy


def _format_status_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))


def _benchmark_cpu_gradient(problem, settings: OSQPSettings, q, l, u, dx, dy, repeat: int) -> tuple[float, str]:
    solver = osqp.OSQP()
    solver.setup(problem.p_matrix, problem.q, problem.a_matrix, problem.l, problem.u, **_osqp_settings_dict(settings))
    status_counts: Counter[str] = Counter()
    count = 0
    start = time.perf_counter()
    for _ in range(repeat):
        for i in range(q.shape[0]):
            solver.update(
                q=np.asarray(q[i], dtype=np.float64),
                l=np.asarray(l[i], dtype=np.float64),
                u=np.asarray(u[i], dtype=np.float64),
            )
            result = solver.solve()
            status_counts[str(result.info.status)] += 1
            if result.info.status_val != 1:
                raise RuntimeError(
                    "CPU OSQP did not solve before derivative: "
                    f"{result.info.status}; status counts so far: {_format_status_counts(status_counts)}"
                )
            solver.adjoint_derivative_compute(
                dx=np.asarray(dx[i], dtype=np.float64),
                dy=np.asarray(dy[i], dtype=np.float64),
            )
            solver.adjoint_derivative_get_mat(as_dense=False, dP_as_triu=True)
            solver.adjoint_derivative_get_vec()
            count += 1
    return count / (time.perf_counter() - start), _format_status_counts(status_counts)


def _estimated_bytes(batch: int, plan, dtype: np.dtype) -> int:
    item = np.dtype(dtype).itemsize
    derivative = plan.derivative_plan
    forward = plan.n + 3 * plan.m + plan.qdldl_plan.nnz_l + plan.qdldl_plan.n
    adjoint = derivative.adjoint_dim * 4 + derivative.adjoint_upper.nnz + derivative.qdldl_plan.nnz_l
    return batch * (forward + adjoint) * item


def _run_one(compiled, plan, p_values, a_values, q, l, u, repeat: int, seed: int):
    dx, dy = _gradient_seeds(plan, q.shape[0], np.asarray(q).dtype, seed)
    p_jax = jnp.asarray(p_values)
    a_jax = jnp.asarray(a_values)
    q_jax = jnp.asarray(q)
    l_jax = jnp.asarray(l)
    u_jax = jnp.asarray(u)
    dx_jax = jnp.asarray(dx)
    dy_jax = jnp.asarray(dy)
    grad_fn = _make_value_and_grad(compiled)
    executable, _ = _compile_and_warm(grad_fn, p_jax, a_jax, q_jax, l_jax, u_jax, dx_jax, dy_jax)
    value_grad_s, out = _time_compiled(
        executable, p_jax, a_jax, q_jax, l_jax, u_jax, dx_jax, dy_jax, repeat=repeat
    )
    return value_grad_s, out


def _write_batch_csv(path: pathlib.Path, rows: list[GradientBatchResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "dtype",
                "batch_size",
                "horizon",
                "n",
                "m",
                "kkt_dim",
                "kkt_nnz",
                "nnz_l",
                "adjoint_dim",
                "adjoint_nnz",
                "adjoint_nnz_l",
                "derivative_refinement_iters",
                "value_grad_s",
                "systems_per_s",
                "cpu_gradient_systems_per_s",
                "cpu_osqp_status_counts",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row.variant,
                    "dtype": row.dtype,
                    "batch_size": row.batch_size,
                    "horizon": row.horizon,
                    "n": row.n,
                    "m": row.m,
                    "kkt_dim": row.kkt_dim,
                    "kkt_nnz": row.kkt_nnz,
                    "nnz_l": row.nnz_l,
                    "adjoint_dim": row.adjoint_dim,
                    "adjoint_nnz": row.adjoint_nnz,
                    "adjoint_nnz_l": row.adjoint_nnz_l,
                    "derivative_refinement_iters": row.derivative_refinement_iters,
                    "value_grad_s": row.value_grad_s,
                    "systems_per_s": row.systems_per_s,
                    "cpu_gradient_systems_per_s": row.cpu_throughput,
                    "cpu_osqp_status_counts": row.cpu_status_counts,
                }
            )


def _write_horizon_csv(path: pathlib.Path, rows: list[GradientHorizonResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "dtype",
                "horizon",
                "batch_size",
                "n",
                "m",
                "kkt_dim",
                "kkt_nnz",
                "nnz_l",
                "adjoint_dim",
                "adjoint_nnz",
                "adjoint_nnz_l",
                "derivative_refinement_iters",
                "value_grad_s",
                "systems_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row.variant,
                    "dtype": row.dtype,
                    "horizon": row.horizon,
                    "batch_size": row.batch_size,
                    "n": row.n,
                    "m": row.m,
                    "kkt_dim": row.kkt_dim,
                    "kkt_nnz": row.kkt_nnz,
                    "nnz_l": row.nnz_l,
                    "adjoint_dim": row.adjoint_dim,
                    "adjoint_nnz": row.adjoint_nnz,
                    "adjoint_nnz_l": row.adjoint_nnz_l,
                    "derivative_refinement_iters": row.derivative_refinement_iters,
                    "value_grad_s": row.value_grad_s,
                    "systems_per_s": row.systems_per_s,
                }
            )


def _plot_batch(path: pathlib.Path, rows: list[GradientBatchResult], cpu_throughput: float | None) -> None:
    if not rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 5), squeeze=False, sharey=True)
    for ax, dtype in zip(axes[0], dtypes, strict=True):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        for variant in VARIANTS:
            variant_rows = [row for row in dtype_rows if row.variant == variant]
            variant_rows.sort(key=lambda row: row.batch_size)
            ax.plot(
                [row.batch_size for row in variant_rows],
                [row.systems_per_s for row in variant_rows],
                marker=MARKERS[variant],
                markersize=10 if MARKERS[variant] == "*" else 7,
                linewidth=1.8,
                label=variant,
            )
        if cpu_throughput is not None:
            ax.axhline(cpu_throughput, linestyle="--", color="black", linewidth=1.6, label="CPU OSQP grad")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Solved MPC gradients/s")
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-pattern quadcopter OSQP value+gradient throughput")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_horizon(path: pathlib.Path, rows: list[GradientHorizonResult]) -> None:
    if not rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes = sorted({row.dtype for row in rows})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(7 * len(dtypes), 5), squeeze=False, sharey=True)
    for ax, dtype in zip(axes[0], dtypes, strict=True):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        for variant in ["baseline", "transpose+segmented"]:
            variant_rows = [row for row in dtype_rows if row.variant == variant]
            variant_rows.sort(key=lambda row: row.horizon)
            ax.plot(
                [row.horizon for row in variant_rows],
                [row.systems_per_s for row in variant_rows],
                marker=MARKERS[variant],
                markersize=10 if MARKERS[variant] == "*" else 7,
                linewidth=1.8,
                label=variant,
            )
        ax.set_yscale("log")
        ax.set_xlabel("MPC horizon N")
        ax.set_ylabel("Solved MPC gradients/s")
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-pattern OSQP gradient horizon sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--batch-sizes", default="128,512,2048,10000,20000")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu-samples", type=int, default=16)
    parser.add_argument("--cpu-repeat", type=int, default=1)
    parser.add_argument("--segment-budget", type=int, default=8)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--derivative-refinement-iters", type=int, default=100)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--horizon-sweep", default="5,10,20")
    parser.add_argument("--horizon-batch-sizes", default="128,512,2048,10000,20000")
    parser.add_argument("--csv-path", default="results/linear_mpc_gradients/throughput_mpc_osqp_gradients.csv")
    parser.add_argument("--plot-path", default="results/linear_mpc_gradients/throughput_mpc_osqp_gradients.png")
    parser.add_argument("--horizon-csv-path", default="results/linear_mpc_gradients/horizon_sweep_mpc_osqp_gradients.csv")
    parser.add_argument("--horizon-plot-path", default="results/linear_mpc_gradients/horizon_sweep_mpc_osqp_gradients.png")
    parser.add_argument("--skip-horizon-sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-cpu", action="store_true")
    args = parser.parse_args()
    variant_names = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    settings = OSQPSettings(
        max_iter=args.max_iter,
        scaling=0,
        adaptive_rho=False,
        rho_is_vec=True,
        check_termination=0,
        warm_starting=False,
        polishing=False,
    )
    problem = make_linear_mpc(args.horizon)
    plan = build_osqp_plan(problem.p_matrix, problem.a_matrix, problem.l, problem.u, settings, derivatives=True)
    derivative_plan = plan.derivative_plan
    print("Devices:", jax.devices())
    print(
        "Problem:",
        f"N={args.horizon}",
        f"n={plan.n}",
        f"m={plan.m}",
        f"kkt_dim={plan.qdldl_plan.n}",
        f"kkt_nnz={plan.nnz_kkt}",
        f"nnz_L={plan.qdldl_plan.nnz_l}",
        f"adjoint_dim={derivative_plan.adjoint_dim}",
        f"adjoint_nnz={derivative_plan.adjoint_upper.nnz}",
        f"adjoint_nnz_L={derivative_plan.qdldl_plan.nnz_l}",
        f"max_iter={settings.max_iter}",
        f"refinement={args.derivative_refinement_iters}",
    )

    cpu_throughput = None
    cpu_status_counts = None
    if not args.no_cpu:
        cpu_settings = OSQPSettings(
            max_iter=4000,
            scaling=0,
            adaptive_rho=False,
            rho_is_vec=True,
            check_termination=25,
            warm_starting=False,
            polishing=False,
        )
        _, _, q_cpu, l_cpu, u_cpu = batched_problem_data(
            problem, args.cpu_samples, np.float64, seed=args.seed, x0_variation=0.15
        )
        dx_cpu, dy_cpu = _gradient_seeds(plan, args.cpu_samples, np.float64, args.seed + 9000)
        try:
            cpu_throughput, cpu_status_counts = _benchmark_cpu_gradient(
                problem, cpu_settings, q_cpu, l_cpu, u_cpu, dx_cpu, dy_cpu, args.cpu_repeat
            )
            print(f"CPU OSQP solved-gradient throughput: {cpu_throughput:.2f}/s status=({cpu_status_counts})")
        except Exception as exc:
            print(f"CPU OSQP gradient baseline skipped: {type(exc).__name__}: {exc}")

    rows: list[GradientBatchResult] = []
    max_bytes = int(args.max_device_gb * 1024**3)
    for dtype in _parse_dtypes(args.dtypes):
        jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
        for variant in variant_names:
            options = VARIANTS[variant]
            options = dict(options)
            if options.get("segmented"):
                options["segment_budget"] = args.segment_budget
                options["segment_strategy"] = args.segment_strategy
            compiled = compile_osqp(
                plan,
                dtype=dtype,
                derivatives=True,
                derivative_refinement_iters=args.derivative_refinement_iters,
                **options,
            )
            print(f"Variant {variant} dtype={dtype.name}")
            for batch in _parse_ints(args.batch_sizes):
                estimate = _estimated_bytes(batch, plan, dtype)
                if estimate > max_bytes:
                    print(f"  batch={batch}: skipped estimated {estimate / 1024**3:.2f} GiB")
                    break
                p_values, a_values, q, l, u = batched_problem_data(
                    problem, batch, dtype, seed=args.seed + batch, x0_variation=0.15
                )
                try:
                    value_grad_s, _ = _run_one(
                        compiled, plan, p_values, a_values, q, l, u, repeat=args.repeat, seed=args.seed + 3000 + batch
                    )
                except Exception as exc:
                    print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                    break
                row = GradientBatchResult(
                    variant=variant,
                    dtype=dtype.name,
                    batch_size=batch,
                    horizon=args.horizon,
                    n=plan.n,
                    m=plan.m,
                    kkt_dim=plan.qdldl_plan.n,
                    kkt_nnz=plan.nnz_kkt,
                    nnz_l=plan.qdldl_plan.nnz_l,
                    adjoint_dim=derivative_plan.adjoint_dim,
                    adjoint_nnz=derivative_plan.adjoint_upper.nnz,
                    adjoint_nnz_l=derivative_plan.qdldl_plan.nnz_l,
                    derivative_refinement_iters=args.derivative_refinement_iters,
                    value_grad_s=value_grad_s,
                    cpu_throughput=cpu_throughput,
                    cpu_status_counts=cpu_status_counts,
                )
                rows.append(row)
                print(f"  batch={batch}: value+grad={value_grad_s:.6f}s throughput={row.systems_per_s:.2f}/s")

    horizon_rows: list[GradientHorizonResult] = []
    horizon_variants = {
        "baseline": VARIANTS["baseline"],
        "transpose+segmented": VARIANTS["transpose+segmented"],
    }
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
                    derivatives=True,
                )
                horizon_derivative_plan = horizon_plan.derivative_plan
                print(
                    f"Horizon N={horizon} dtype={dtype.name}: n={horizon_plan.n} m={horizon_plan.m} "
                    f"kkt_dim={horizon_plan.qdldl_plan.n} kkt_nnz={horizon_plan.nnz_kkt} "
                    f"nnz_L={horizon_plan.qdldl_plan.nnz_l} adjoint_dim={horizon_derivative_plan.adjoint_dim} "
                    f"adjoint_nnz={horizon_derivative_plan.adjoint_upper.nnz} "
                    f"adjoint_nnz_L={horizon_derivative_plan.qdldl_plan.nnz_l}"
                )
                for variant, options in horizon_variants.items():
                    options = dict(options)
                    if options.get("segmented"):
                        options["segment_budget"] = args.segment_budget
                        options["segment_strategy"] = args.segment_strategy
                    compiled = compile_osqp(
                        horizon_plan,
                        dtype=dtype,
                        derivatives=True,
                        derivative_refinement_iters=args.derivative_refinement_iters,
                        **options,
                    )
                    best_row = None
                    for batch in sorted(_parse_ints(args.horizon_batch_sizes), reverse=True):
                        estimate = _estimated_bytes(batch, horizon_plan, dtype)
                        if estimate > max_bytes:
                            continue
                        p_values, a_values, q, l, u = batched_problem_data(
                            horizon_problem,
                            batch,
                            dtype,
                            seed=args.seed + 6000 + horizon + batch,
                            x0_variation=0.15,
                        )
                        try:
                            value_grad_s, _ = _run_one(
                                compiled,
                                horizon_plan,
                                p_values,
                                a_values,
                                q,
                                l,
                                u,
                                repeat=args.repeat,
                                seed=args.seed + 7000 + horizon + batch,
                            )
                        except Exception as exc:
                            print(f"  {variant} batch={batch}: failed {type(exc).__name__}: {exc}")
                            break
                        best_row = GradientHorizonResult(
                            variant=variant,
                            dtype=dtype.name,
                            horizon=horizon,
                            batch_size=batch,
                            n=horizon_plan.n,
                            m=horizon_plan.m,
                            kkt_dim=horizon_plan.qdldl_plan.n,
                            kkt_nnz=horizon_plan.nnz_kkt,
                            nnz_l=horizon_plan.qdldl_plan.nnz_l,
                            adjoint_dim=horizon_derivative_plan.adjoint_dim,
                            adjoint_nnz=horizon_derivative_plan.adjoint_upper.nnz,
                            adjoint_nnz_l=horizon_derivative_plan.qdldl_plan.nnz_l,
                            derivative_refinement_iters=args.derivative_refinement_iters,
                            value_grad_s=value_grad_s,
                        )
                        print(
                            f"  {variant} batch={batch}: value+grad={value_grad_s:.6f}s "
                            f"throughput={best_row.systems_per_s:.2f}/s"
                        )
                        break
                    if best_row is not None:
                        horizon_rows.append(best_row)

    batch_csv = pathlib.Path(args.csv_path)
    horizon_csv = pathlib.Path(args.horizon_csv_path)
    batch_csv.parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_horizon_sweep:
        horizon_csv.parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.horizon_plot_path).parent.mkdir(parents=True, exist_ok=True)
    _write_batch_csv(batch_csv, rows)
    _plot_batch(pathlib.Path(args.plot_path), rows, cpu_throughput)
    if not args.skip_horizon_sweep:
        _write_horizon_csv(horizon_csv, horizon_rows)
        _plot_horizon(pathlib.Path(args.horizon_plot_path), horizon_rows)
    print(f"Wrote {batch_csv}")
    print(f"Wrote {args.plot_path}")
    if not args.skip_horizon_sweep:
        print(f"Wrote {horizon_csv}")
        print(f"Wrote {args.horizon_plot_path}")


if __name__ == "__main__":
    main()
