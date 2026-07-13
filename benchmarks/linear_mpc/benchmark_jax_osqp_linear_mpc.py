#!/usr/bin/env python3
"""Benchmark fixed-pattern JAX OSQP iterations on the linear quadcopter MPC problem."""

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
    "jax:transpose+segmented": {
        "qdldl_backend": "jax",
        "transpose_work": True,
        "segmented": True,
    },
    "jax:transpose+segmented+levelsolve": {
        "qdldl_backend": "jax",
        "transpose_work": True,
        "segmented": True,
        "level_scheduled_solve": True,
    },
    "warp:transpose+segmented": {
        "qdldl_backend": "warp",
        "transpose_work": True,
        "segmented": True,
    },
    "warp:transpose+segmented+levelsolve": {
        "qdldl_backend": "warp",
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

DEFAULT_VARIANTS = (
    "baseline",
    "transpose+segmented",
    "transpose+segmented+levelsolve",
    "factor-warp+solve-jax:transpose+segmented",
    "factor-warp+solve-jax:transpose+segmented+levelsolve",
    "factor-warp+solve-warp:transpose",
    "factor-warp+solve-warp:transpose+segmented",
    "factor-warp+solve-warp:transpose+segmented+levelsolve",
)

MARKERS = {
    "baseline": "o",
    "transpose+segmented": "X",
    "transpose+segmented+levelsolve": ">",
    "jax:transpose+segmented": "X",
    "jax:transpose+segmented+levelsolve": ">",
    "warp:transpose+segmented": "^",
    "warp:transpose+segmented+levelsolve": "^",
    "factor-warp+solve-jax:transpose+segmented": "P",
    "factor-warp+solve-jax:transpose+segmented+levelsolve": "*",
    "factor-warp+solve-warp:transpose": "h",
    "factor-warp+solve-warp:transpose+segmented": "^",
    "factor-warp+solve-warp:transpose+segmented+levelsolve": "v",
}


@dataclass(frozen=True)
class BatchResult:
    variant: str
    dtype: str
    batch_size: int
    horizon: int
    n: int
    m: int
    kkt_dim: int
    kkt_nnz: int
    nnz_l: int
    linear_solve_count: int
    lower_compile_s: float
    compile_s: float
    factor_s: float
    solve_s: float
    linear_solve_loop_s: float
    factor_solve_s: float
    max_abs_x_vs_cpu: float | None
    max_abs_obj_vs_cpu: float | None
    cpu_throughput: float | None

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.factor_solve_s

    @property
    def solve_systems_per_s(self) -> float:
        return self.batch_size / self.solve_s

    @property
    def linear_solve_loop_systems_per_s(self) -> float:
        return self.batch_size / self.linear_solve_loop_s

    @property
    def factor_linear_solve_loop_s(self) -> float:
        return self.factor_s + self.linear_solve_loop_s

    @property
    def factor_linear_solve_loop_systems_per_s(self) -> float:
        return self.batch_size / self.factor_linear_solve_loop_s

    @property
    def factor_systems_per_s(self) -> float:
        return self.batch_size / self.factor_s


@dataclass(frozen=True)
class HorizonResult:
    variant: str
    dtype: str
    horizon: int
    batch_size: int
    n: int
    m: int
    kkt_dim: int
    kkt_nnz: int
    nnz_l: int
    factor_solve_s: float

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.factor_solve_s


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
    }


def _compile_and_warm(fn, *args):
    start_total = time.perf_counter()
    jax.block_until_ready(args)
    start_compile = time.perf_counter()
    compiled = fn.lower(*args).compile()
    lower_compile_s = time.perf_counter() - start_compile
    out = compiled(*args)
    jax.block_until_ready(out)
    return compiled, out, lower_compile_s, time.perf_counter() - start_total


def _time_compiled(compiled, *args, repeat: int) -> tuple[float, object]:
    start = time.perf_counter()
    for _ in range(repeat):
        out = compiled(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - start) / repeat, out


def _benchmark_cpu(problem, settings: OSQPSettings, q, l, u, repeat: int) -> float:
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


def _check_against_cpu(problem, settings: OSQPSettings, compiled, p_values, a_values, q, l, u):
    x, _, _, _, _, obj = jax.device_get(
        compiled.solve(
            jnp.asarray(p_values),
            jnp.asarray(a_values),
            jnp.asarray(q),
            jnp.asarray(l),
            jnp.asarray(u),
        )
    )
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


def _write_batch_csv(path: pathlib.Path, rows: list[BatchResult]) -> None:
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
                "lower_compile_s",
                "compile_s",
                "factor_s",
                "solve_s",
                "linear_solve_loop_s",
                "factor_linear_solve_loop_s",
                "factor_solve_s",
                "factor_systems_per_s",
                "solve_systems_per_s",
                "linear_solve_loop_systems_per_s",
                "factor_linear_solve_loop_systems_per_s",
                "systems_per_s",
                "linear_solve_count",
                "max_abs_x_vs_cpu",
                "max_abs_obj_vs_cpu",
                "cpu_osqp_systems_per_s",
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
                    "lower_compile_s": row.lower_compile_s,
                    "compile_s": row.compile_s,
                    "factor_s": row.factor_s,
                    "solve_s": row.solve_s,
                    "linear_solve_loop_s": row.linear_solve_loop_s,
                    "factor_linear_solve_loop_s": row.factor_linear_solve_loop_s,
                    "factor_solve_s": row.factor_solve_s,
                    "factor_systems_per_s": row.factor_systems_per_s,
                    "solve_systems_per_s": row.solve_systems_per_s,
                    "linear_solve_loop_systems_per_s": row.linear_solve_loop_systems_per_s,
                    "factor_linear_solve_loop_systems_per_s": row.factor_linear_solve_loop_systems_per_s,
                    "systems_per_s": row.systems_per_s,
                    "linear_solve_count": row.linear_solve_count,
                    "max_abs_x_vs_cpu": row.max_abs_x_vs_cpu,
                    "max_abs_obj_vs_cpu": row.max_abs_obj_vs_cpu,
                    "cpu_osqp_systems_per_s": row.cpu_throughput,
                }
            )


def _write_horizon_csv(path: pathlib.Path, rows: list[HorizonResult]) -> None:
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
                "factor_solve_s",
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
                    "factor_solve_s": row.factor_solve_s,
                    "systems_per_s": row.systems_per_s,
                }
            )


def _plot_batch(path: pathlib.Path, rows: list[BatchResult], cpu_throughput: float | None) -> None:
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
            ax.axhline(cpu_throughput, linestyle="--", color="black", linewidth=1.6, label="CPU OSQP")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Solved MPC problems/s")
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-pattern quadcopter OSQP throughput")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_horizon(path: pathlib.Path, rows: list[HorizonResult]) -> None:
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
        ax.set_ylabel("Solved MPC problems/s")
        ax.set_title(dtype)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-pattern OSQP horizon sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _estimated_bytes(batch: int, plan, dtype: np.dtype) -> int:
    item = np.dtype(dtype).itemsize
    return batch * (plan.n + 3 * plan.m + plan.qdldl_plan.nnz_l + plan.qdldl_plan.n) * item


def _run_one(compiled, p_values, a_values, q, l, u, repeat: int):
    p_jax = jnp.asarray(p_values)
    a_jax = jnp.asarray(a_values)
    q_jax = jnp.asarray(q)
    l_jax = jnp.asarray(l)
    u_jax = jnp.asarray(u)
    lower_compile_s = 0.0
    compile_s = 0.0
    factor_exe, factor_out, lower_s, total_s = _compile_and_warm(compiled.factor, p_jax, a_jax)
    lower_compile_s += lower_s
    compile_s += total_s
    lx, dinv = factor_out
    solve_exe, _, lower_s, total_s = _compile_and_warm(
        compiled.solve_with_factor, lx, dinv, p_jax, a_jax, q_jax, l_jax, u_jax
    )
    lower_compile_s += lower_s
    compile_s += total_s
    rhs0 = jnp.concatenate([-q_jax, jnp.zeros_like(l_jax)], axis=1)
    linear_solve_loop_exe, _, lower_s, total_s = _compile_and_warm(compiled.linear_solve_loop, lx, dinv, rhs0)
    lower_compile_s += lower_s
    compile_s += total_s
    total_exe, _, lower_s, total_s = _compile_and_warm(compiled.solve, p_jax, a_jax, q_jax, l_jax, u_jax)
    lower_compile_s += lower_s
    compile_s += total_s
    factor_s, factor_out = _time_compiled(factor_exe, p_jax, a_jax, repeat=repeat)
    lx, dinv = factor_out
    solve_s, _ = _time_compiled(solve_exe, lx, dinv, p_jax, a_jax, q_jax, l_jax, u_jax, repeat=repeat)
    linear_solve_loop_s, _ = _time_compiled(linear_solve_loop_exe, lx, dinv, rhs0, repeat=repeat)
    total_s, _ = _time_compiled(total_exe, p_jax, a_jax, q_jax, l_jax, u_jax, repeat=repeat)
    return lower_compile_s, compile_s, factor_s, solve_s, linear_solve_loop_s, total_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=128)
    parser.add_argument("--cpu-repeat", type=int, default=1)
    parser.add_argument("--segment-budget", type=int, default=8)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-device-gb", type=float, default=20.0)
    parser.add_argument("--horizon-sweep", default="5,10,20,40")
    parser.add_argument("--horizon-batch-sizes", default="512,2048,10000,20000")
    parser.add_argument("--csv-path", default="results/linear_mpc/throughput_mpc_osqp.csv")
    parser.add_argument("--plot-path", default="results/linear_mpc/throughput_mpc_osqp.png")
    parser.add_argument("--horizon-csv-path", default="results/linear_mpc/horizon_sweep_mpc_osqp.csv")
    parser.add_argument("--horizon-plot-path", default="results/linear_mpc/horizon_sweep_mpc_osqp.png")
    parser.add_argument("--skip-horizon-sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-cpu", action="store_true")
    parser.add_argument("--no-check", action="store_true")
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
    plan = build_osqp_plan(problem.p_matrix, problem.a_matrix, problem.l, problem.u, settings)
    print("Devices:", jax.devices())
    print(
        "Problem:",
        f"N={args.horizon}",
        f"n={plan.n}",
        f"m={plan.m}",
        f"kkt_dim={plan.qdldl_plan.n}",
        f"kkt_nnz={plan.nnz_kkt}",
        f"nnz_L={plan.qdldl_plan.nnz_l}",
        f"max_iter={settings.max_iter}",
    )

    cpu_throughput = None
    if not args.no_cpu:
        _, _, q_cpu, l_cpu, u_cpu = batched_problem_data(
            problem, args.cpu_samples, np.float64, seed=args.seed, x0_variation=0.15
        )
        cpu_throughput = _benchmark_cpu(problem, settings, q_cpu, l_cpu, u_cpu, args.cpu_repeat)
        print(f"CPU OSQP fixed-iteration throughput: {cpu_throughput:.2f}/s")

    rows: list[BatchResult] = []
    max_bytes = int(args.max_device_gb * 1024**3)
    for dtype in _parse_dtypes(args.dtypes):
        jax.config.update("jax_enable_x64", dtype == np.dtype(np.float64))
        for variant in variant_names:
            options = dict(VARIANTS[variant])
            if options.get("segmented"):
                options["segment_budget"] = args.segment_budget
                options["segment_strategy"] = args.segment_strategy
            compiled = compile_osqp(plan, dtype=dtype, **options)
            print(f"Variant {variant} dtype={dtype.name}")

            max_x = None
            max_obj = None
            if not args.no_check:
                p_check, a_check, q_check, l_check, u_check = batched_problem_data(
                    problem, 2, dtype, seed=args.seed + 10, x0_variation=0.15
                )
                max_x, max_obj = _check_against_cpu(
                    problem, settings, compiled, p_check, a_check, q_check, l_check, u_check
                )
                print(f"  check max_abs_x={max_x:.3e} max_abs_obj={max_obj:.3e}")

            for batch in _parse_ints(args.batch_sizes):
                estimate = _estimated_bytes(batch, plan, dtype)
                if estimate > max_bytes:
                    print(f"  batch={batch}: skipped estimated {estimate / 1024**3:.2f} GiB")
                    break
                p_values, a_values, q, l, u = batched_problem_data(
                    problem, batch, dtype, seed=args.seed + batch, x0_variation=0.15
                )
                try:
                    lower_compile_s, compile_s, factor_s, solve_s, linear_solve_loop_s, total_s = _run_one(
                        compiled, p_values, a_values, q, l, u, repeat=args.repeat
                    )
                except Exception as exc:
                    print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                    break
                row = BatchResult(
                    variant=variant,
                    dtype=dtype.name,
                    batch_size=batch,
                    horizon=args.horizon,
                    n=plan.n,
                    m=plan.m,
                    kkt_dim=plan.qdldl_plan.n,
                    kkt_nnz=plan.nnz_kkt,
                    nnz_l=plan.qdldl_plan.nnz_l,
                    linear_solve_count=settings.max_iter,
                    lower_compile_s=lower_compile_s,
                    compile_s=compile_s,
                    factor_s=factor_s,
                    solve_s=solve_s,
                    linear_solve_loop_s=linear_solve_loop_s,
                    factor_solve_s=total_s,
                    max_abs_x_vs_cpu=max_x,
                    max_abs_obj_vs_cpu=max_obj,
                    cpu_throughput=cpu_throughput,
                )
                rows.append(row)
                print(
                    f"  batch={batch}: compile={compile_s:.3f}s lower_compile={lower_compile_s:.3f}s "
                    f"factor={factor_s:.6f}s solve={solve_s:.6f}s "
                    f"linear_solves={linear_solve_loop_s:.6f}s total={total_s:.6f}s "
                    f"throughput={row.systems_per_s:.2f}/s"
                )

    horizon_rows: list[HorizonResult] = []
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
                    horizon_problem.p_matrix, horizon_problem.a_matrix, horizon_problem.l, horizon_problem.u, settings
                )
                print(
                    f"Horizon N={horizon} dtype={dtype.name}: n={horizon_plan.n} m={horizon_plan.m} "
                    f"kkt_dim={horizon_plan.qdldl_plan.n} kkt_nnz={horizon_plan.nnz_kkt} "
                    f"nnz_L={horizon_plan.qdldl_plan.nnz_l}"
                )
                for variant, options in horizon_variants.items():
                    options = dict(options)
                    if options.get("segmented"):
                        options["segment_budget"] = args.segment_budget
                        options["segment_strategy"] = args.segment_strategy
                    compiled = compile_osqp(horizon_plan, dtype=dtype, **options)
                    best_row = None
                    for batch in sorted(_parse_ints(args.horizon_batch_sizes), reverse=True):
                        estimate = _estimated_bytes(batch, horizon_plan, dtype)
                        if estimate > max_bytes:
                            continue
                        p_values, a_values, q, l, u = batched_problem_data(
                            horizon_problem, batch, dtype, seed=args.seed + 2000 + horizon + batch, x0_variation=0.15
                        )
                        try:
                            _, _, _, _, _, total_s = _run_one(
                                compiled, p_values, a_values, q, l, u, repeat=args.repeat
                            )
                        except Exception as exc:
                            print(f"  {variant} batch={batch}: failed {type(exc).__name__}: {exc}")
                            break
                        best_row = HorizonResult(
                            variant=variant,
                            dtype=dtype.name,
                            horizon=horizon,
                            batch_size=batch,
                            n=horizon_plan.n,
                            m=horizon_plan.m,
                            kkt_dim=horizon_plan.qdldl_plan.n,
                            kkt_nnz=horizon_plan.nnz_kkt,
                            nnz_l=horizon_plan.qdldl_plan.nnz_l,
                            factor_solve_s=total_s,
                        )
                        print(f"  {variant} batch={batch}: total={total_s:.6f}s throughput={best_row.systems_per_s:.2f}/s")
                        break
                    if best_row is not None:
                        horizon_rows.append(best_row)

    batch_csv = pathlib.Path(args.csv_path)
    batch_csv.parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
    _write_batch_csv(batch_csv, rows)
    _plot_batch(pathlib.Path(args.plot_path), rows, cpu_throughput)
    print(f"Wrote {batch_csv}")
    print(f"Wrote {args.plot_path}")
    if not args.skip_horizon_sweep:
        horizon_csv = pathlib.Path(args.horizon_csv_path)
        horizon_csv.parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.horizon_plot_path).parent.mkdir(parents=True, exist_ok=True)
        _write_horizon_csv(horizon_csv, horizon_rows)
        _plot_horizon(pathlib.Path(args.horizon_plot_path), horizon_rows)
        print(f"Wrote {horizon_csv}")
        print(f"Wrote {args.horizon_plot_path}")


if __name__ == "__main__":
    main()
