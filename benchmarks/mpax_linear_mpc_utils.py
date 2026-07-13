"""Shared MPAX helpers for the linear MPC benchmarks."""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
MPAX_RESOURCE = ROOT / "resources" / "MPAX"
if MPAX_RESOURCE.exists() and str(MPAX_RESOURCE) not in sys.path:
    sys.path.insert(0, str(MPAX_RESOURCE))

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from mpax import create_qp, raPDHG


@dataclass(frozen=True)
class MPAXLinearMPCProblem:
    base_problem: Any
    equality_rows: np.ndarray
    lower_rows: np.ndarray
    upper_rows: np.ndarray
    n: int
    m_original: int
    n_eq: int
    n_ineq: int
    n_constraints: int
    nnz_q: int
    nnz_a_eq: int
    nnz_g_ineq: int
    regularization: float


def make_mpax_linear_mpc_problem(
    problem,
    dtype: np.dtype,
    *,
    regularization: float = 0.0,
    eq_tol: float = 1e-9,
) -> MPAXLinearMPCProblem:
    """Convert the OSQP-style linear MPC QP to MPAX's sparse QP interface."""

    dtype = np.dtype(dtype)
    lower_bound = np.asarray(problem.l, dtype=np.float64)
    upper_bound = np.asarray(problem.u, dtype=np.float64)
    finite_lower = np.isfinite(lower_bound)
    finite_upper = np.isfinite(upper_bound)
    equality = finite_lower & finite_upper & (np.abs(upper_bound - lower_bound) <= eq_tol)
    lower = finite_lower & ~equality
    upper = finite_upper & ~equality

    equality_rows = np.flatnonzero(equality).astype(np.int32)
    lower_rows = np.flatnonzero(lower).astype(np.int32)
    upper_rows = np.flatnonzero(upper).astype(np.int32)

    q_matrix = problem.p_matrix.astype(dtype)
    if regularization:
        q_matrix = q_matrix + regularization * sp.eye(q_matrix.shape[0], dtype=dtype, format="csc")
    q_matrix = q_matrix.tocoo()

    a_eq = problem.a_matrix[equality_rows].astype(dtype).tocoo()
    g_ineq = sp.vstack(
        [
            problem.a_matrix[lower_rows],
            -problem.a_matrix[upper_rows],
        ],
        format="coo",
    ).astype(dtype)

    objective_vector = np.asarray(problem.q, dtype=dtype)
    b_eq = np.asarray(lower_bound[equality_rows], dtype=dtype)
    h_ineq = np.concatenate(
        [
            np.asarray(lower_bound[lower_rows], dtype=dtype),
            np.asarray(-upper_bound[upper_rows], dtype=dtype),
        ],
        axis=0,
    )
    variable_lower = np.full(problem.q.size, -np.inf, dtype=dtype)
    variable_upper = np.full(problem.q.size, np.inf, dtype=dtype)

    base_problem = create_qp(
        BCOO.from_scipy_sparse(q_matrix),
        jnp.asarray(objective_vector),
        BCOO.from_scipy_sparse(a_eq),
        jnp.asarray(b_eq),
        BCOO.from_scipy_sparse(g_ineq),
        jnp.asarray(h_ineq),
        jnp.asarray(variable_lower),
        jnp.asarray(variable_upper),
        use_sparse_matrix=True,
    )

    return MPAXLinearMPCProblem(
        base_problem=base_problem,
        equality_rows=equality_rows,
        lower_rows=lower_rows,
        upper_rows=upper_rows,
        n=problem.q.size,
        m_original=problem.l.size,
        n_eq=equality_rows.size,
        n_ineq=lower_rows.size + upper_rows.size,
        n_constraints=equality_rows.size + lower_rows.size + upper_rows.size,
        nnz_q=q_matrix.nnz,
        nnz_a_eq=a_eq.nnz,
        nnz_g_ineq=g_ineq.nnz,
        regularization=float(regularization),
    )


def mpax_rhs(qp: MPAXLinearMPCProblem, l: np.ndarray, u: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Build batched MPAX right-hand sides from OSQP row lower/upper bounds."""

    dtype = np.dtype(dtype)
    return np.concatenate(
        [
            np.asarray(l[:, qp.equality_rows], dtype=dtype),
            np.asarray(l[:, qp.lower_rows], dtype=dtype),
            np.asarray(-u[:, qp.upper_rows], dtype=dtype),
        ],
        axis=1,
    )


def make_solver(
    *,
    eps_abs: float,
    eps_rel: float,
    iteration_limit: int,
    termination_evaluation_frequency: int,
    l_inf_ruiz_iterations: int,
    pock_chambolle_alpha: float,
    unroll: bool,
) -> raPDHG:
    return raPDHG(
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        iteration_limit=iteration_limit,
        termination_evaluation_frequency=termination_evaluation_frequency,
        l_inf_ruiz_iterations=l_inf_ruiz_iterations,
        pock_chambolle_alpha=pock_chambolle_alpha,
        verbose=False,
        debug=False,
        jit=True,
        unroll=unroll,
        warm_start=False,
        feasibility_polishing=False,
    )


def make_batched_solve_fn(qp: MPAXLinearMPCProblem, solver: raPDHG):
    base_problem = qp.base_problem

    def solve_one(objective_vector, right_hand_side):
        problem = dataclasses.replace(
            base_problem,
            objective_vector=objective_vector,
            right_hand_side=right_hand_side,
        )
        result = solver.optimize(problem)
        return (
            result.primal_solution,
            result.primal_objective,
            result.iteration_count,
            result.termination_status,
        )

    return jax.jit(jax.vmap(solve_one, in_axes=(0, 0)))


def make_value_and_grad_fn(qp: MPAXLinearMPCProblem, solver: raPDHG):
    base_problem = qp.base_problem

    def objective(objective_vectors, right_hand_sides, primal_weights):
        def solve_one(objective_vector, right_hand_side, primal_weight):
            problem = dataclasses.replace(
                base_problem,
                objective_vector=objective_vector,
                right_hand_side=right_hand_side,
            )
            result = solver.optimize(problem)
            return jnp.sum(result.primal_solution * primal_weight)

        return jnp.sum(jax.vmap(solve_one)(objective_vectors, right_hand_sides, primal_weights))

    return jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
