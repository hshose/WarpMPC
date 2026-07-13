"""Sparse OSQP-style KKT generators for fixed-pattern QDLDL benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class MPCProblem:
    """A sparse quadratic MPC problem and its direct OSQP KKT matrix."""

    p_matrix: sp.csc_matrix
    a_matrix: sp.csc_matrix
    kkt_upper: sp.csc_matrix
    n_variables: int
    n_constraints: int
    nx: int
    nu: int
    horizon: int


def structural_density(matrix: sp.spmatrix, triangular: bool = True) -> float:
    matrix = sp.csc_matrix(matrix)
    nrows, ncols = matrix.shape
    if triangular and nrows == ncols:
        denom = nrows * (nrows + 1) / 2
    else:
        denom = nrows * ncols
    return float(matrix.nnz / denom)


def _state_index(t: int, state: int, nx: int) -> int:
    return t * nx + state


def _input_index(t: int, control: int, nx: int, nu: int, horizon: int) -> int:
    return (horizon + 1) * nx + t * nu + control


def _sample_extra_positions(
    rng: np.random.Generator,
    existing: set[int],
    rows: int,
    cols: int,
    count: int,
) -> tuple[list[int], list[int]]:
    extra_rows: list[int] = []
    extra_cols: list[int] = []
    if count <= 0:
        return extra_rows, extra_cols

    max_entries = rows * cols
    if count > max_entries - len(existing):
        raise ValueError("requested extra density exceeds available A positions")

    while len(extra_rows) < count:
        draw = max(4 * (count - len(extra_rows)), 1024)
        rr = rng.integers(0, rows, size=draw)
        cc = rng.integers(0, cols, size=draw)
        for r, c in zip(rr, cc, strict=False):
            key = int(r) * cols + int(c)
            if key in existing:
                continue
            existing.add(key)
            extra_rows.append(int(r))
            extra_cols.append(int(c))
            if len(extra_rows) == count:
                break
    return extra_rows, extra_cols


def make_mpc_kkt(
    nx: int = 12,
    nu: int = 4,
    horizon: int = 24,
    sigma: float = 1e-6,
    rho: float = 0.1,
    extra_kkt_density: float = 0.0,
    seed: int = 1,
) -> MPCProblem:
    """Create an OSQP direct-solver KKT matrix for a sparse linear MPC QP.

    The KKT matrix is

        [[P + sigma I, A.T],
         [A,          -rho^{-1} I]]

    and the returned matrix stores only the upper triangle in CSC format.  Extra
    random entries, when requested, are added to the constraint matrix ``A`` so
    the KKT remains quasi-definite for any numeric values of those entries.
    """

    rng = np.random.default_rng(seed)
    n_variables = (horizon + 1) * nx + horizon * nu
    n_dyn = horizon * nx
    n_box = n_variables
    n_constraints = n_dyn + n_box

    q_diag = np.ones((horizon + 1) * nx)
    q_diag[-nx:] = 5.0
    r_diag = 0.1 * np.ones(horizon * nu)
    p_diag = np.concatenate([q_diag, r_diag])
    p_matrix = sp.diags(p_diag, format="csc")

    a_rows: list[int] = []
    a_cols: list[int] = []
    a_vals: list[float] = []

    for t in range(horizon):
        for s in range(nx):
            row = t * nx + s
            a_rows.append(row)
            a_cols.append(_state_index(t + 1, s, nx))
            a_vals.append(1.0)

            a_rows.append(row)
            a_cols.append(_state_index(t, s, nx))
            a_vals.append(-1.0)

            if s > 0:
                a_rows.append(row)
                a_cols.append(_state_index(t, s - 1, nx))
                a_vals.append(0.05)
            if s + 1 < nx:
                a_rows.append(row)
                a_cols.append(_state_index(t, s + 1, nx))
                a_vals.append(-0.03)

            for u in range(nu):
                if (s + u) % max(1, nx // nu) == 0 or u == s % nu:
                    a_rows.append(row)
                    a_cols.append(_input_index(t, u, nx, nu, horizon))
                    a_vals.append(-0.1 * (1.0 + 0.05 * rng.standard_normal()))

    for var in range(n_variables):
        a_rows.append(n_dyn + var)
        a_cols.append(var)
        a_vals.append(1.0)

    existing = {r * n_variables + c for r, c in zip(a_rows, a_cols, strict=False)}
    current_kkt_nnz = p_matrix.nnz + len(a_rows) + n_constraints
    kkt_dim = n_variables + n_constraints
    target_nnz = int(np.ceil(extra_kkt_density * kkt_dim * (kkt_dim + 1) / 2))
    extra_count = max(0, target_nnz - current_kkt_nnz)
    extra_rows, extra_cols = _sample_extra_positions(
        rng, existing, n_constraints, n_variables, extra_count
    )
    if extra_rows:
        a_rows.extend(extra_rows)
        a_cols.extend(extra_cols)
        a_vals.extend((0.05 * rng.standard_normal(len(extra_rows))).tolist())

    a_matrix = sp.coo_matrix(
        (a_vals, (a_rows, a_cols)), shape=(n_constraints, n_variables)
    ).tocsc()
    a_matrix.sum_duplicates()
    a_matrix.sort_indices()

    rho_inv = 1.0 / rho
    kkt = sp.bmat(
        [
            [p_matrix + sigma * sp.eye(n_variables, format="csc"), a_matrix.T],
            [a_matrix, -rho_inv * sp.eye(n_constraints, format="csc")],
        ],
        format="csc",
    )
    kkt_upper = sp.triu(kkt, format="csc")
    kkt_upper.sum_duplicates()
    kkt_upper.sort_indices()

    return MPCProblem(
        p_matrix=p_matrix,
        a_matrix=a_matrix,
        kkt_upper=kkt_upper,
        n_variables=n_variables,
        n_constraints=n_constraints,
        nx=nx,
        nu=nu,
        horizon=horizon,
    )


def sample_kkt_values(
    kkt_upper: sp.spmatrix,
    batch_size: int,
    seed: int = 0,
    variation: float = 0.25,
    dtype: np.dtype | str = np.float64,
) -> np.ndarray:
    """Sample batched numeric values without changing the sparsity pattern."""

    kkt_upper = sp.csc_matrix(kkt_upper)
    dtype = np.dtype(dtype)
    base = np.asarray(kkt_upper.data, dtype=dtype)
    rng = np.random.default_rng(seed)

    rows = kkt_upper.indices
    cols = np.repeat(np.arange(kkt_upper.shape[1]), np.diff(kkt_upper.indptr))
    diag = rows == cols

    values = np.empty((batch_size, base.size), dtype=dtype)
    if np.any(~diag):
        off_scale = 1.0 + variation * rng.standard_normal((batch_size, int(np.sum(~diag))))
        values[:, ~diag] = base[~diag][None, :] * off_scale
    if np.any(diag):
        low = max(1e-3, 1.0 - variation)
        diag_scale = rng.uniform(low, 1.0 + variation, size=(batch_size, int(np.sum(diag))))
        values[:, diag] = base[diag][None, :] * diag_scale
    return values
