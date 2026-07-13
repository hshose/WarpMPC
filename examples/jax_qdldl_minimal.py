#!/usr/bin/env python3
"""Minimal fixed-pattern batched QDLDL solve with the Warp backend."""

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

from warpmpc.jax_qdldl import build_qdldl_plan, compile_qdldl_variant


def main() -> None:
    matrix = sp.csc_matrix(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 0.25],
            [0.0, 0.0, 0.25, 1.0],
        ]
    )
    plan = build_qdldl_plan(matrix)
    solver = compile_qdldl_variant(
        plan,
        dtype=np.float32,
        backend="warp",
        factor_backend="warp",
        solve_backend="warp",
        transpose_work=True,
        segmented=True,
        segment_budget=4,
        segment_strategy="optimal",
        level_scheduled_solve=True,
        level_scheduled_solve_threshold=2,
    )

    values = jnp.asarray(np.broadcast_to(plan.a_data.astype(np.float32), (2, plan.nnz_a)))
    rhs = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=jnp.float32,
    )

    solution, _, _ = solver.factor_and_solve(values, rhs)
    solution = jax.device_get(solution)
    print(f"variant: {solver.variant}")
    print("solution:")
    print(solution)


if __name__ == "__main__":
    main()
