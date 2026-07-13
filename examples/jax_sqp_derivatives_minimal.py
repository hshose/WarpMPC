#!/usr/bin/env python3
"""Minimal sparse MPC SQP example with OSQP adjoint derivatives enabled."""

from __future__ import annotations

import argparse
import pathlib
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = pathlib.Path(__file__).resolve().parent
for path in (ROOT, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    SparseMPCProblem,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)
from jax_sqp_minimal import (
    make_stage_functions,
    sample_tracking_batch,
    warm_start_from_reference,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--qdldl-backend", choices=("warp", "jax"), default="warp")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    horizon = 6
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
    plan = build_sparse_mpc_plan(
        problem,
        osqp_settings=settings,
        derivatives=True,
    )
    solver = compile_sparse_mpc_sqp(
        problem,
        plan,
        dtype=np.float64,
        osqp_settings=settings,
        qdldl_backend=args.qdldl_backend,
        transpose_work=True,
        segmented=True,
        segment_budget=32,
        segment_strategy="optimal",
        level_scheduled_solve=True,
        level_scheduled_solve_threshold=2,
        derivatives=True,
        derivative_refinement_iters=100,
    )

    batch_size = args.batch_size
    refs, x_initial, params_np = sample_tracking_batch(
        batch_size=batch_size,
        horizon=horizon,
        dtype=np.dtype("float64"),
        seed=11,
    )
    params = jnp.asarray(params_np, dtype=jnp.float64)
    z0 = jnp.asarray(warm_start_from_reference(refs, x_initial), dtype=jnp.float64)
    state = solver.init_state(batch_size)
    initial_tracking_error = x_initial[:, 0] - refs[:, 0]

    def loss(params_in):
        result, _ = solver.step(z0, params_in, state=state)
        target = params_in[:, -1]
        terminal_x = result.z_next[:, -2]
        return jnp.mean((terminal_x - target) ** 2)

    value, grad_params = jax.value_and_grad(loss)(params)
    result, _ = solver.step(z0, params, state=state)

    print("SQP derivatives example:")
    print(f"  variables={plan.n_variables}, constraints={plan.n_constraints}")
    print(f"  qdldl_variant={solver.osqp.qdldl.variant}")
    print(f"  adjoint_dim={plan.osqp_plan.derivative_plan.adjoint_dim}")
    print(f"  loss={float(value):.6g}")
    print(
        "  initial tracking error RMS:",
        float(np.sqrt(np.mean(initial_tracking_error**2))),
    )
    print("  grad wrt first-stage ref first 5:", np.asarray(grad_params[:5, 0]))
    print("  grad wrt terminal ref first 5:", np.asarray(grad_params[:5, -1]))
    print("  grad wrt initial x first 5:", np.asarray(grad_params[:5, 1]))
    print("  line-search step first 5:", np.asarray(result.line_search.step_length[:5]))


if __name__ == "__main__":
    main()
