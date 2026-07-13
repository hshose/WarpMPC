#!/usr/bin/env python3
"""Minimal fixed-pattern sparse MPC tracking example with Warp-backed SQP."""

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


DEFAULT_PLOT_DIR = pathlib.Path("results/jax_sqp_minimal")
MINIMAL_INPUT_LIMIT = 0.25
MINIMAL_INPUT_WEIGHT = 0.5


def make_stage_functions() -> tuple[
    CasadiStageFunction,
    CasadiStageFunction,
    CasadiStageFunction,
]:
    """Create first, reusable middle, and terminal CasADi stage functions.

    The stage variable is z_k = [x_k, u_k].  The nonterminal constraint contains
    a mildly nonlinear scalar dynamics defect and an input box constraint:

        x_{k+1} - x_k - u_k - 0.1 x_k^2 = 0
        -0.25 <= u_k <= 0.25

    The first-stage parameter is p_0 = [x_ref, x_initial], and it adds the
    initial-state equality x_0 = x_initial.  The reusable middle and terminal
    stage parameters are p_k = [x_ref].
    """

    z0 = ca.SX.sym("z0", 2)
    z1 = ca.SX.sym("z1", 2)
    p0 = ca.SX.sym("p0", 2)
    x0, u0 = z0[0], z0[1]
    first_ref, x_initial = p0[0], p0[1]
    first_cost = 0.5 * ((x0 - first_ref) ** 2 + MINIMAL_INPUT_WEIGHT * u0**2)
    first_g = ca.vertcat(z1[0] - x0 - u0 - 0.1 * x0**2, u0, x0)
    first_l = ca.vertcat(0.0, -MINIMAL_INPUT_LIMIT, x_initial)
    first_u = ca.vertcat(0.0, MINIMAL_INPUT_LIMIT, x_initial)
    first_fn = ca.Function(
        "minimal_first",
        [z0, z1, p0],
        [first_cost, first_g, first_l, first_u],
    )

    z = ca.SX.sym("z", 2)
    zn = ca.SX.sym("zn", 2)
    p = ca.SX.sym("p", 1)
    x, u = z[0], z[1]
    x_next = zn[0]
    x_ref = p[0]
    cost = 0.5 * ((x - x_ref) ** 2 + MINIMAL_INPUT_WEIGHT * u**2)
    g = ca.vertcat(x_next - x - u - 0.1 * x**2, u)
    lower = ca.vertcat(0.0, -MINIMAL_INPUT_LIMIT)
    upper = ca.vertcat(0.0, MINIMAL_INPUT_LIMIT)
    stage_fn = ca.Function("minimal_stage", [z, zn, p], [cost, g, lower, upper])

    zt = ca.SX.sym("zt", 2)
    pt = ca.SX.sym("pt", 1)
    xt, ut = zt[0], zt[1]
    terminal_ref = pt[0]
    terminal_cost = 2.0 * (xt - terminal_ref) ** 2 + 0.01 * ut**2
    terminal_g = ca.vertcat(xt)
    terminal_l = ca.vertcat(-5.0)
    terminal_u = ca.vertcat(5.0)
    terminal_fn = ca.Function(
        "minimal_terminal",
        [zt, pt],
        [terminal_cost, terminal_g, terminal_l, terminal_u],
    )

    first = CasadiStageFunction.from_function(first_fn, has_next=True)
    stage = CasadiStageFunction.from_function(stage_fn, has_next=True)
    terminal = CasadiStageFunction.from_function(terminal_fn, has_next=False)
    return first, stage, terminal


def sample_tracking_batch(
    *,
    batch_size: int,
    horizon: int,
    dtype: np.dtype,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample sensible scalar MPC initial states and smooth references.

    The initial state and reference are sampled independently over the full
    [-1, 1] tracking domain, with a minimum first-node separation so the MPC
    examples show visible initial tracking error.  The reference then follows a
    smooth trajectory from that first reference to a terminal target.
    """

    rng = np.random.default_rng(seed)
    x_initial = rng.uniform(-1.0, 1.0, size=(batch_size, 1))
    first_ref = sample_separated_reference(
        rng,
        x_initial,
        lower=-1.0,
        upper=1.0,
        min_separation=0.35,
    )
    terminal_ref = rng.uniform(-1.0, 1.0, size=(batch_size, 1))
    t = np.linspace(0.0, 1.0, horizon + 1, dtype=np.float64)[None, :]
    smoothstep = 3.0 * t**2 - 2.0 * t**3
    base = first_ref + (terminal_ref - first_ref) * smoothstep
    wiggle_amp = rng.uniform(-0.12, 0.12, size=(batch_size, 1))
    wiggle_phase = rng.uniform(0.0, 2.0 * np.pi, size=(batch_size, 1))
    wiggle = wiggle_amp * np.sin(2.0 * np.pi * t + wiggle_phase) * t * (1.0 - t)
    refs = np.clip(base + wiggle, -1.0, 1.0)
    refs = refs.astype(dtype, copy=False)
    x_initial = x_initial.astype(dtype, copy=False)
    params = params_from_reference(refs, x_initial)
    return refs, x_initial, params


def sample_separated_reference(
    rng: np.random.Generator,
    x_initial: np.ndarray,
    *,
    lower: float,
    upper: float,
    min_separation: float,
) -> np.ndarray:
    """Sample a reference point in [lower, upper] away from x_initial."""

    left_room = x_initial - lower
    right_room = upper - x_initial
    can_left = left_room >= min_separation
    can_right = right_room >= min_separation
    random_right = rng.random(size=x_initial.shape) < 0.5
    use_right = np.where(can_left & can_right, random_right, can_right)
    room = np.where(use_right, right_room, left_room)
    span = np.maximum(room - min_separation, 0.0)
    offset = min_separation + rng.random(size=x_initial.shape) * span
    sign = np.where(use_right, 1.0, -1.0)
    return x_initial + sign * offset


def params_from_reference(refs: np.ndarray, x_initial: np.ndarray) -> np.ndarray:
    """Pack stage parameters as [ref_0, x_initial, ref_1, ..., ref_N]."""

    if refs.ndim != 2:
        raise ValueError("refs must have shape (batch, horizon + 1)")
    if x_initial.shape != (refs.shape[0], 1):
        raise ValueError("x_initial must have shape (batch, 1)")
    return np.concatenate([refs[:, :1], x_initial, refs[:, 1:]], axis=1)


def warm_start_from_reference(refs: np.ndarray, x_initial: np.ndarray) -> np.ndarray:
    """Build a plausible open-loop [x, u] SQP warm start from the reference."""

    batch_size, nodes = refs.shape
    stages = np.zeros((batch_size, nodes, 2), dtype=refs.dtype)
    stages[:, :, 0] = refs
    stages[:, 0, 0] = x_initial[:, 0]
    x_cur = stages[:, :-1, 0]
    x_next = stages[:, 1:, 0]
    u_guess = x_next - x_cur - 0.1 * x_cur**2
    stages[:, :-1, 1] = np.clip(
        u_guess,
        -MINIMAL_INPUT_LIMIT,
        MINIMAL_INPUT_LIMIT,
    )
    return stages.reshape((batch_size, nodes * 2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--sqp-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
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
    horizon = args.horizon
    first, stage, terminal = make_stage_functions()
    problem = SparseMPCProblem.from_stage_functions(
        horizon=horizon,
        first=first,
        intermediate=stage,
        terminal=terminal,
    )

    settings = OSQPSettings(
        max_iter=25,
        scaling=10,
        warm_starting=True,
    )
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

    refs, x_initial, params = sample_tracking_batch(
        batch_size=args.batch_size,
        horizon=horizon,
        dtype=np.dtype("float32"),
        seed=args.seed,
    )
    z0 = warm_start_from_reference(refs, x_initial)

    linearization = solver.build_qp(jnp.asarray(z0), jnp.asarray(params))
    solver_state = solver.init_state(args.batch_size)
    z = jnp.asarray(z0)
    params_jax = jnp.asarray(params)
    iterates = [z0]
    results = []
    for _ in range(args.sqp_iterations):
        result, solver_state = solver.step(z, params_jax, state=solver_state)
        z = result.z_next
        results.append(result)
        iterates.append(np.asarray(jax.device_get(z)))

    final_result = results[-1]
    z_next, direction, step_length, prim_res, dual_res = jax.device_get(
        (
            final_result.z_next,
            final_result.solve.direction,
            final_result.line_search.step_length,
            final_result.solve.prim_res,
            final_result.solve.dual_res,
        )
    )
    iterates_np = np.stack(iterates, axis=0)
    x_pred, u_pred = unpack_state_action(z_next, horizon)
    tracking_error = x_pred - refs
    initial_tracking_error = x_initial[:, 0] - refs[:, 0]

    print("QP dimensions:")
    print(f"  variables={plan.n_variables}, constraints={plan.n_constraints}")
    print(f"  nnz(P)={plan.p_pattern.nnz}, nnz(A)={plan.a_pattern.nnz}")
    print(f"  qdldl_variant={solver.osqp.qdldl.variant}")
    print("Linearized QP value shapes:")
    print(f"  P values: {linearization.p_values.shape}")
    print(f"  A values: {linearization.a_values.shape}")
    print(f"{args.sqp_iterations} SQP iterations with filter line search:")
    print("  direction first 5:", direction[:5])
    print("  line-search step length first 5:", step_length[:5])
    print("  terminal x first 5:", z_next[:5, -2])
    print(
        "  constrained initial x max error:",
        float(np.max(np.abs(z_next[:, 0] - x_initial[:, 0]))),
    )
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
