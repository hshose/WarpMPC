#!/usr/bin/env python3
"""Minimal sparse MPC example with multiple reusable stage functions."""

from __future__ import annotations

import argparse
import pathlib
import sys

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = pathlib.Path(__file__).resolve().parent
for path in (ROOT, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    CasadiStageFunction,
    SparseMPCProblem,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)
from utils import (
    plot_sqp_error_distribution,
    plot_sqp_timeseries_samples,
    unpack_state_action,
)
from jax_sqp_minimal import (
    sample_separated_reference,
)


DEFAULT_PLOT_DIR = pathlib.Path("results/jax_sqp_stagewise_minimal")


def _split(z):
    return z[0], z[1]


def make_stagewise_problem() -> SparseMPCProblem:
    dt = 0.1

    z0 = ca.SX.sym("z0", 2)
    z1 = ca.SX.sym("z1", 2)
    p0 = ca.SX.sym("p0", 2)
    x0, u0 = _split(z0)
    ref0, x_initial = p0[0], p0[1]
    cruise_defect = z1[0] - x0 - dt * (u0 - 0.05 * x0**3)
    first_fn = ca.Function(
        "stagewise_first",
        [z0, z1, p0],
        [
            0.5 * (x0 - ref0) ** 2 + 0.05 * u0**2,
            ca.vertcat(cruise_defect, u0, x0),
            ca.vertcat(0.0, -1.0, x_initial),
            ca.vertcat(0.0, 1.0, x_initial),
        ],
    )

    z = ca.SX.sym("z", 2)
    zn = ca.SX.sym("zn", 2)
    p = ca.SX.sym("p", 1)
    x, u = _split(z)
    ref = p[0]
    cruise_fn = ca.Function(
        "stagewise_cruise",
        [z, zn, p],
        [
            0.4 * (x - ref) ** 2 + 0.03 * u**2,
            ca.vertcat(zn[0] - x - dt * (u - 0.05 * x**3), u),
            ca.vertcat(0.0, -1.0),
            ca.vertcat(0.0, 1.0),
        ],
    )

    pg = ca.SX.sym("pg", 2)
    glue_ref, corridor_center = pg[0], pg[1]
    glue_fn = ca.Function(
        "stagewise_glue",
        [z, zn, pg],
        [
            0.7 * (x - glue_ref) ** 2 + 0.05 * u**2,
            ca.vertcat(zn[0] - x - dt * (0.8 * u - 0.02 * x**3), u, x - corridor_center),
            ca.vertcat(0.0, -0.9, -0.15),
            ca.vertcat(0.0, 0.9, 0.15),
        ],
    )

    approach_fn = ca.Function(
        "stagewise_approach",
        [z, zn, p],
        [
            1.2 * (x - ref) ** 2 + 0.08 * u**2,
            ca.vertcat(zn[0] - x - dt * (0.6 * u - 0.15 * x + 0.02 * ca.sin(x)), u),
            ca.vertcat(0.0, -0.7),
            ca.vertcat(0.0, 0.7),
        ],
    )

    zt = ca.SX.sym("zt", 2)
    pt = ca.SX.sym("pt", 1)
    xt, ut = _split(zt)
    terminal_fn = ca.Function(
        "stagewise_terminal",
        [zt, pt],
        [
            2.0 * (xt - pt[0]) ** 2 + 0.01 * ut**2,
            ca.vertcat(xt),
            ca.vertcat(-2.0),
            ca.vertcat(2.0),
        ],
    )

    first = CasadiStageFunction.from_function(first_fn, has_next=True)
    cruise = CasadiStageFunction.from_function(cruise_fn, has_next=True)
    glue = CasadiStageFunction.from_function(glue_fn, has_next=True)
    approach = CasadiStageFunction.from_function(approach_fn, has_next=True)
    terminal = CasadiStageFunction.from_function(terminal_fn, has_next=False)

    return SparseMPCProblem.from_stage_functions(
        horizon=8,
        first=first,
        intermediate=(cruise, cruise, glue, approach, approach, approach, approach),
        terminal=terminal,
    )


def sample_stagewise_batch(
    *,
    batch_size: int,
    horizon: int,
    dtype: np.dtype,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if horizon != 8:
        raise ValueError("sample_stagewise_batch currently matches the fixed horizon=8 example")
    rng = np.random.default_rng(seed)
    x_initial = rng.uniform(-0.75, 0.75, size=(batch_size, 1))
    first_ref = sample_separated_reference(
        rng,
        x_initial,
        lower=-0.75,
        upper=0.75,
        min_separation=0.25,
    )
    terminal_ref = np.clip(
        x_initial + rng.uniform(-0.30, 0.30, size=(batch_size, 1)),
        -0.75,
        0.75,
    )
    t = np.linspace(0.0, 1.0, horizon + 1, dtype=np.float64)[None, :]
    smoothstep = 3.0 * t**2 - 2.0 * t**3
    refs = first_ref + (terminal_ref - first_ref) * smoothstep
    refs += (
        rng.uniform(-0.06, 0.06, size=(batch_size, 1))
        * np.sin(2.0 * np.pi * t)
        * t
        * (1.0 - t)
    )
    refs = refs.astype(dtype, copy=False)
    x_initial = x_initial.astype(dtype, copy=False)
    glue_ref = refs[:, 3:4]
    corridor_center = x_initial
    params = np.concatenate(
        [
            refs[:, :1],
            x_initial,
            refs[:, 1:3],
            glue_ref,
            corridor_center,
            refs[:, 4:],
        ],
        axis=1,
    )
    return refs, x_initial, params


def warm_start_stagewise_from_reference(
    refs: np.ndarray,
    x_initial: np.ndarray,
    *,
    dt: float = 0.1,
) -> np.ndarray:
    batch_size, nodes = refs.shape
    horizon = nodes - 1
    stages = np.zeros((batch_size, nodes, 2), dtype=refs.dtype)
    stages[:, 0, 0] = x_initial[:, 0]
    for stage_index in range(horizon):
        x_cur = stages[:, stage_index, 0]
        desired_next = refs[:, stage_index + 1]
        delta = (desired_next - x_cur) / dt
        if stage_index <= 2:
            u_limit = 1.0
            u_guess = delta + 0.05 * x_cur**3
            x_dot = lambda u: u - 0.05 * x_cur**3
        elif stage_index == 3:
            u_limit = 0.9
            u_guess = (delta + 0.02 * x_cur**3) / 0.8
            x_dot = lambda u: 0.8 * u - 0.02 * x_cur**3
        else:
            u_limit = 0.7
            u_guess = (delta + 0.15 * x_cur - 0.02 * np.sin(x_cur)) / 0.6
            x_dot = lambda u: 0.6 * u - 0.15 * x_cur + 0.02 * np.sin(x_cur)
        u = np.clip(u_guess, -u_limit, u_limit)
        stages[:, stage_index, 1] = u
        stages[:, stage_index + 1, 0] = x_cur + dt * x_dot(u)
    return stages.reshape((batch_size, nodes * 2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sqp-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--qdldl-backend", choices=("warp", "jax"), default="warp")
    parser.add_argument("--plot-dir", type=pathlib.Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--timeseries-samples", type=int, default=6)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.sqp_iterations < 1:
        parser.error("--sqp-iterations must be at least 1")
    if args.timeseries_samples < 1:
        parser.error("--timeseries-samples must be at least 1")
    return args


def main() -> None:
    args = _parse_args()
    problem = make_stagewise_problem()
    horizon = len(problem.stages) - 1
    settings = OSQPSettings(max_iter=25, scaling=10, warm_starting=True)
    plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
    solver = compile_sparse_mpc_sqp(
        problem,
        plan,
        dtype=np.float32,
        osqp_settings=settings,
        qdldl_backend=args.qdldl_backend,
        transpose_work=True,
        segmented=True,
        segment_budget=32,
        segment_strategy="optimal",
        level_scheduled_solve=True,
        level_scheduled_solve_threshold=2,
    )

    refs, x_initial, params = sample_stagewise_batch(
        batch_size=args.batch_size,
        horizon=horizon,
        dtype=np.dtype("float32"),
        seed=args.seed,
    )
    z0 = warm_start_stagewise_from_reference(refs, x_initial)
    z = jnp.asarray(z0)
    params_jax = jnp.asarray(params)
    state = solver.init_state(args.batch_size)
    iterates = [z0]
    result = None
    for _ in range(args.sqp_iterations):
        result, state = solver.step(z, params_jax, state=state)
        z = result.z_next
        iterates.append(np.asarray(jax.device_get(z)))
    assert result is not None
    z_next, prim_res, dual_res = jax.device_get(
        (result.z_next, result.solve.prim_res, result.solve.dual_res)
    )
    iterates_np = np.stack(iterates, axis=0)
    x_pred, u_pred = unpack_state_action(z_next, horizon)
    tracking_error = x_pred - refs
    initial_tracking_error = x_initial[:, 0] - refs[:, 0]

    print("Stage-wise SQP example:")
    print("  stages:", ", ".join(stage.name for stage in problem.stages))
    print(f"  variables={plan.n_variables}, constraints={plan.n_constraints}")
    print(f"  qdldl_variant={solver.osqp.qdldl.variant}")
    print("  terminal x first 5:", z_next[:5, -2])
    print(
        "  initial tracking error RMS:",
        float(np.sqrt(np.mean(initial_tracking_error**2))),
    )
    print(
        "  initial tracking error p95:",
        float(np.percentile(np.abs(initial_tracking_error), 95.0)),
    )
    print("  final RMS tracking error:", float(np.sqrt(np.mean(tracking_error**2))))
    print(
        "  final terminal error p95:",
        float(np.percentile(np.abs(tracking_error[:, -1]), 95.0)),
    )
    print("  final max |u|:", float(np.max(np.abs(u_pred))))
    print("  max primal residual:", float(np.max(prim_res)))
    print("  max dual residual:", float(np.max(dual_res)))
    if not args.skip_plots:
        error_path = plot_sqp_error_distribution(
            args.plot_dir,
            iterates_np,
            refs,
        )
        timeseries_path = plot_sqp_timeseries_samples(
            args.plot_dir,
            iterates_np,
            refs,
            sample_count=args.timeseries_samples,
        )
        print(f"  wrote error plot: {error_path}")
        print(f"  wrote timeseries plot: {timeseries_path}")


if __name__ == "__main__":
    main()
