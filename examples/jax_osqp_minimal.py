#!/usr/bin/env python3
"""Minimal batched bound-constrained QP solve with Warp-backed JAX OSQP."""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan, compile_osqp


def main() -> None:
    p_matrix = sp.csc_matrix([[4.0, 1.0], [1.0, 2.0]])
    a_matrix = sp.eye(2, format="csc")
    l = np.array([0.0, 0.0])
    u = np.array([1.0, 1.0])
    q = np.array([[-1.0, -1.0], [-2.0, -0.5]], dtype=np.float32)

    settings = OSQPSettings(max_iter=50, scaling=0, check_termination=0)
    plan = build_osqp_plan(p_matrix, a_matrix, l, u, settings)
    solver = compile_osqp(
        plan,
        dtype=np.float32,
        qdldl_backend="warp",
        qdldl_factor_backend="warp",
        qdldl_solve_backend="warp",
        transpose_work=True,
        segmented=True,
        segment_budget=8,
        segment_strategy="optimal",
        level_scheduled_solve=True,
        level_scheduled_solve_threshold=2,
    )

    batch_size = q.shape[0]
    p_values = jnp.asarray(np.broadcast_to(np.asarray(plan.p_upper.data, dtype=np.float32), (batch_size, plan.p_upper.nnz)))
    a_values = jnp.asarray(np.broadcast_to(np.asarray(plan.a_matrix.data, dtype=np.float32), (batch_size, plan.a_matrix.nnz)))
    l_batch = jnp.asarray(np.broadcast_to(l.astype(np.float32), q.shape))
    u_batch = jnp.asarray(np.broadcast_to(u.astype(np.float32), q.shape))
    x, _, _, prim_res, dual_res, obj_val = solver.solve(
        p_values,
        a_values,
        jnp.asarray(q),
        l_batch,
        u_batch,
    )

    x, prim_res, dual_res, obj_val = jax.device_get((x, prim_res, dual_res, obj_val))
    print(f"QDLDL variant: {solver.qdldl.variant}")
    print("x:", x)
    print("objective:", obj_val)
    print("max primal residual:", float(np.max(prim_res)))
    print("max dual residual:", float(np.max(dual_res)))


if __name__ == "__main__":
    main()
