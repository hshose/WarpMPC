"""Quadcopter MPC problem data adapted from the OSQP Python example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class QuadcopterMPC:
    p_matrix: sp.csc_matrix
    q: np.ndarray
    a_matrix: sp.csc_matrix
    l: np.ndarray
    u: np.ndarray
    ad: sp.csc_matrix
    bd: sp.csc_matrix
    xmin: np.ndarray
    xmax: np.ndarray
    umin: np.ndarray
    umax: np.ndarray
    x0: np.ndarray
    xr: np.ndarray
    nx: int
    nu: int
    horizon: int


def make_linear_mpc(horizon: int = 10, x0: np.ndarray | None = None) -> QuadcopterMPC:
    """Adapt the OSQP quadcopter MPC example into reusable problem data."""

    ad = sp.csc_matrix(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0992, 0.0, 0.0],
            [0.0, -0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, -0.0016, 0.0, 0.0, 0.0992, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0992],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0488, 0.0, 0.0, 0.9846, 0.0, 0.0],
            [0.0, -0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0488, 0.0, 0.0, 0.9846, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9846],
        ]
    )
    bd = sp.csc_matrix(
        [
            [0.0, -0.0726, 0.0, 0.0726],
            [-0.0726, 0.0, 0.0726, 0.0],
            [-0.0152, 0.0152, -0.0152, 0.0152],
            [-0.0, -0.0006, -0.0, 0.0006],
            [0.0006, 0.0, -0.0006, 0.0],
            [0.0106, 0.0106, 0.0106, 0.0106],
            [0.0, -1.4512, 0.0, 1.4512],
            [-1.4512, 0.0, 1.4512, 0.0],
            [-0.3049, 0.3049, -0.3049, 0.3049],
            [-0.0, -0.0236, 0.0, 0.0236],
            [0.0236, 0.0, -0.0236, 0.0],
            [0.2107, 0.2107, 0.2107, 0.2107],
        ]
    )
    nx, nu = bd.shape
    u0 = 10.5916
    umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
    umax = np.array([13.0, 13.0, 13.0, 13.0]) - u0
    xmin = np.array(
        [
            -np.pi / 6,
            -np.pi / 6,
            -np.inf,
            -np.inf,
            -np.inf,
            -1.0,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    xmax = np.array(
        [
            np.pi / 6,
            np.pi / 6,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ]
    )
    q_state = sp.diags([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    q_terminal = q_state
    r_input = 0.1 * sp.eye(nu)
    x0 = np.zeros(nx) if x0 is None else np.asarray(x0, dtype=float)
    xr = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n_horizon = int(horizon)

    p_matrix = sp.block_diag(
        [sp.kron(sp.eye(n_horizon), q_state), q_terminal, sp.kron(sp.eye(n_horizon), r_input)],
        format="csc",
    )
    q = np.hstack(
        [
            np.kron(np.ones(n_horizon), -q_state.dot(xr)),
            -q_terminal.dot(xr),
            np.zeros(n_horizon * nu),
        ]
    )
    ax = sp.kron(sp.eye(n_horizon + 1), -sp.eye(nx)) + sp.kron(
        sp.eye(n_horizon + 1, k=-1), ad
    )
    bu = sp.kron(sp.vstack([sp.csc_matrix((1, n_horizon)), sp.eye(n_horizon)]), bd)
    aeq = sp.hstack([ax, bu])
    leq = np.hstack([-x0, np.zeros(n_horizon * nx)])
    ueq = leq.copy()
    aineq = sp.eye((n_horizon + 1) * nx + n_horizon * nu)
    lineq = np.hstack([np.kron(np.ones(n_horizon + 1), xmin), np.kron(np.ones(n_horizon), umin)])
    uineq = np.hstack([np.kron(np.ones(n_horizon + 1), xmax), np.kron(np.ones(n_horizon), umax)])
    a_matrix = sp.vstack([aeq, aineq], format="csc")
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    return QuadcopterMPC(
        p_matrix=p_matrix,
        q=q,
        a_matrix=a_matrix,
        l=l,
        u=u,
        ad=ad,
        bd=bd,
        xmin=xmin,
        xmax=xmax,
        umin=umin,
        umax=umax,
        x0=x0,
        xr=xr,
        nx=nx,
        nu=nu,
        horizon=n_horizon,
    )


def batched_problem_data(
    problem: QuadcopterMPC,
    batch_size: int,
    dtype: np.dtype | str = np.float64,
    *,
    seed: int = 0,
    x0_variation: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-pattern batched quadcopter data with varied initial states."""

    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    p_values = np.asarray(problem.p_matrix.data, dtype=dtype)[None, :]
    a_values = np.asarray(problem.a_matrix.data, dtype=dtype)[None, :]
    q = np.broadcast_to(np.asarray(problem.q, dtype=dtype), (batch_size, problem.q.size)).copy()
    l = np.broadcast_to(np.asarray(problem.l, dtype=dtype), (batch_size, problem.l.size)).copy()
    u = np.broadcast_to(np.asarray(problem.u, dtype=dtype), (batch_size, problem.u.size)).copy()
    x0_batch = np.asarray(problem.x0, dtype=dtype)[None, :] + x0_variation * rng.standard_normal(
        (batch_size, problem.nx)
    ).astype(dtype)
    l[:, : problem.nx] = -x0_batch
    u[:, : problem.nx] = -x0_batch
    return p_values, a_values, q, l, u
