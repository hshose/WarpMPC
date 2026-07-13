"""Fixed-pattern batched OSQP iterations backed by the JAX QDLDL kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax

from warpmpc.jax_qdldl.core import (
    CompiledQDLDL,
    QDLDLPlan,
    build_qdldl_plan,
    compile_qdldl_variant,
)
from .types import (
    CompiledOSQP,
    FixedOSQPPlan,
    OSQPDerivativePlan,
    OSQPSettings,
    OSQPWarmStart,
)


OSQP_INFTY = 1e30
OSQP_MIN_SCALING = 1e-4
OSQP_MAX_SCALING = 1e4


def _require_x64_for_float64(dtype: np.dtype) -> None:
    if dtype == np.dtype(np.float64) and not jax.config.jax_enable_x64:
        raise ValueError(
            "compile_osqp(..., dtype=float64) requires jax_enable_x64=True. "
            "Set it in the benchmark or example entry point before compiling."
        )


@dataclass(frozen=True)
class _ScalingData:
    c: float
    cinv: float
    d: np.ndarray
    dinv: np.ndarray
    e: np.ndarray
    einv: np.ndarray
    p_scale: np.ndarray
    a_scale: np.ndarray
    q_scale: np.ndarray
    y_scale: np.ndarray
    y_unscale: np.ndarray



def _as_batch(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(array, dtype=dtype)
    if array.ndim == 1:
        return array[None, :]
    return array


def _upper_with_diagonal(matrix: sp.spmatrix) -> sp.csc_matrix:
    upper = sp.triu(matrix, format="csc")
    upper.sum_duplicates()
    upper.sort_indices()
    n = upper.shape[0]
    present = np.zeros(n, dtype=bool)
    for col in range(n):
        rows = upper.indices[upper.indptr[col] : upper.indptr[col + 1]]
        present[col] = np.any(rows == col)
    if not np.all(present):
        missing = np.flatnonzero(~present)
        upper = (upper + sp.csc_matrix((np.zeros_like(missing, dtype=float), (missing, missing)), shape=upper.shape)).tocsc()
        upper.sum_duplicates()
        upper.sort_indices()
    return upper


def _build_kkt_and_maps(
    p_upper: sp.csc_matrix,
    a_matrix: sp.csc_matrix,
    settings: OSQPSettings,
    rho_inv_vec: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = p_upper.shape[0]
    m = a_matrix.shape[0]

    entries: dict[tuple[int, int], list[float | int | str | None]] = {}
    p_to_key: list[tuple[int, int] | None] = [None] * p_upper.nnz
    a_to_key: list[tuple[int, int] | None] = [None] * a_matrix.nnz
    rho_keys: list[tuple[int, int]] = []

    def ensure_entry(row: int, col: int, value: float) -> None:
        entries[(row, col)] = [float(value), None, None, None]

    for col in range(n):
        diag_seen = False
        for ptr in range(p_upper.indptr[col], p_upper.indptr[col + 1]):
            row = int(p_upper.indices[ptr])
            value = float(p_upper.data[ptr])
            if row == col:
                value += settings.sigma
                diag_seen = True
            ensure_entry(row, col, value)
            entries[(row, col)][1] = ptr
            p_to_key[ptr] = (row, col)
        if not diag_seen:
            ensure_entry(col, col, settings.sigma)

    for col in range(a_matrix.shape[1]):
        for ptr in range(a_matrix.indptr[col], a_matrix.indptr[col + 1]):
            row_a = int(a_matrix.indices[ptr])
            key = (col, n + row_a)
            ensure_entry(key[0], key[1], float(a_matrix.data[ptr]))
            entries[key][2] = ptr
            a_to_key[ptr] = key

    for constr in range(m):
        key = (n + constr, n + constr)
        ensure_entry(key[0], key[1], -float(rho_inv_vec[constr]))
        entries[key][3] = constr
        rho_keys.append(key)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    keys = sorted(entries, key=lambda key: (key[1], key[0]))
    key_to_pos = {}
    for pos, key in enumerate(keys):
        rows.append(key[0])
        cols.append(key[1])
        data.append(float(entries[key][0]))
        key_to_pos[key] = pos

    kkt = sp.csc_matrix((data, (rows, cols)), shape=(n + m, n + m))
    kkt.sum_duplicates()
    kkt.sort_indices()

    p_to_kkt = np.asarray([key_to_pos[key] for key in p_to_key], dtype=np.int32)
    a_to_kkt = np.asarray([key_to_pos[key] for key in a_to_key], dtype=np.int32)
    rho_to_kkt = np.asarray([key_to_pos[key] for key in rho_keys], dtype=np.int32)
    p_diag_sigma = np.zeros(p_upper.nnz, dtype=np.float64)
    for col in range(n):
        for ptr in range(p_upper.indptr[col], p_upper.indptr[col + 1]):
            if int(p_upper.indices[ptr]) == col:
                p_diag_sigma[ptr] = settings.sigma

    return kkt, p_to_kkt, p_diag_sigma, a_to_kkt, rho_to_kkt


def _bounds_type(l: np.ndarray, u: np.ndarray) -> np.ndarray:
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    loose = (l <= -OSQP_INFTY) & (u >= OSQP_INFTY)
    equality = np.abs(u - l) < 1e-4
    out = np.zeros(l.shape, dtype=np.int32)
    out[loose] = -1
    out[equality] = 1
    return out


def _rho_vectors(
    settings: OSQPSettings,
    m: int,
    l: np.ndarray | None,
    u: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if settings.rho_is_vec:
        if l is None or u is None:
            raise ValueError("l and u are required when settings.rho_is_vec=True")
        constr_type = _bounds_type(np.asarray(l)[:m], np.asarray(u)[:m])
        rho_vec = np.where(
            constr_type < 0,
            1e-6,
            np.where(constr_type > 0, 1e3 * settings.rho, settings.rho),
        )
    else:
        constr_type = np.zeros(m, dtype=np.int32)
        rho_vec = np.full(m, settings.rho, dtype=np.float64)
    rho_vec = rho_vec.astype(np.float64)
    return rho_vec, 1.0 / rho_vec, constr_type


def _p_full_pattern(p_upper: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    value_map: list[int] = []
    for col in range(p_upper.shape[1]):
        for ptr in range(p_upper.indptr[col], p_upper.indptr[col + 1]):
            row = int(p_upper.indices[ptr])
            rows.append(row)
            cols.append(col)
            value_map.append(ptr)
            if row != col:
                rows.append(col)
                cols.append(row)
                value_map.append(ptr)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(value_map, dtype=np.int32),
    )


def _csc_rows_cols(matrix: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    for col in range(matrix.shape[1]):
        for ptr in range(matrix.indptr[col], matrix.indptr[col + 1]):
            rows.append(int(matrix.indices[ptr]))
            cols.append(col)
    return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)


def _csc_col_norm_inf(matrix: sp.csc_matrix) -> np.ndarray:
    norm = np.zeros(matrix.shape[1], dtype=np.float64)
    for col in range(matrix.shape[1]):
        start = matrix.indptr[col]
        stop = matrix.indptr[col + 1]
        if stop > start:
            norm[col] = float(np.max(np.abs(matrix.data[start:stop])))
    return norm


def _csc_row_norm_inf(matrix: sp.csc_matrix) -> np.ndarray:
    norm = np.zeros(matrix.shape[0], dtype=np.float64)
    for col in range(matrix.shape[1]):
        for ptr in range(matrix.indptr[col], matrix.indptr[col + 1]):
            row = int(matrix.indices[ptr])
            norm[row] = max(norm[row], abs(float(matrix.data[ptr])))
    return norm


def _limit_scaling_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    values[values < OSQP_MIN_SCALING] = 1.0
    values[values > OSQP_MAX_SCALING] = OSQP_MAX_SCALING
    return values


def _limit_scaling_scalar(value: float) -> float:
    if value < OSQP_MIN_SCALING:
        return 1.0
    if value > OSQP_MAX_SCALING:
        return OSQP_MAX_SCALING
    return float(value)


def _nominal_scaling_q(q: np.ndarray | None, n: int) -> np.ndarray | None:
    if q is None:
        return None
    q_array = np.asarray(q, dtype=np.float64)
    if q_array.ndim == 2 and q_array.shape[0] == 1:
        q_array = q_array[0]
    if q_array.shape != (n,):
        raise ValueError("scaling_q must have shape (n,) or (1, n)")
    return q_array.copy()


def _compute_scaling(
    p_upper: sp.csc_matrix,
    a_matrix: sp.csc_matrix,
    iterations: int,
    *,
    q: np.ndarray | None = None,
) -> _ScalingData:
    n = p_upper.shape[0]
    m = a_matrix.shape[0]
    d = np.ones(n, dtype=np.float64)
    e = np.ones(m, dtype=np.float64)
    c = 1.0
    p_rows, p_cols = _csc_rows_cols(p_upper)
    a_rows, a_cols = _csc_rows_cols(a_matrix)

    if iterations <= 0:
        p_scale = np.ones(p_upper.nnz, dtype=np.float64)
        a_scale = np.ones(a_matrix.nnz, dtype=np.float64)
        return _ScalingData(
            c=1.0,
            cinv=1.0,
            d=d,
            dinv=d.copy(),
            e=e,
            einv=e.copy(),
            p_scale=p_scale,
            a_scale=a_scale,
            q_scale=d.copy(),
            y_scale=e.copy(),
            y_unscale=e.copy(),
        )

    p_scaled = p_upper.copy().astype(np.float64)
    a_scaled = a_matrix.copy().astype(np.float64)
    q_scaled = _nominal_scaling_q(q, n)

    for _ in range(iterations):
        p_col_norm = _csc_col_norm_inf(p_scaled)
        a_col_norm = _csc_col_norm_inf(a_scaled)
        d_temp = _limit_scaling_vector(np.maximum(p_col_norm, a_col_norm))
        e_temp = _limit_scaling_vector(_csc_row_norm_inf(a_scaled))

        d_temp = 1.0 / np.sqrt(d_temp)
        e_temp = 1.0 / np.sqrt(e_temp)

        p_scaled.data *= d_temp[p_rows] * d_temp[p_cols]
        a_scaled.data *= e_temp[a_rows] * d_temp[a_cols]
        if q_scaled is not None:
            q_scaled *= d_temp

        d *= d_temp
        e *= e_temp

        p_cost_measure = float(np.sum(_csc_col_norm_inf(p_scaled)) / n)
        q_cost_measure = 0.0 if q_scaled is None else float(np.max(np.abs(q_scaled)))
        q_cost_measure = _limit_scaling_scalar(q_cost_measure)
        cost_measure = _limit_scaling_scalar(max(p_cost_measure, q_cost_measure))
        gamma = 1.0 / cost_measure

        p_scaled.data *= gamma
        if q_scaled is not None:
            q_scaled *= gamma
        c *= gamma

    dinv = 1.0 / d
    einv = 1.0 / e
    cinv = 1.0 / c
    p_scale = c * d[p_rows] * d[p_cols]
    a_scale = e[a_rows] * d[a_cols]
    return _ScalingData(
        c=float(c),
        cinv=float(cinv),
        d=d,
        dinv=dinv,
        e=e,
        einv=einv,
        p_scale=p_scale,
        a_scale=a_scale,
        q_scale=c * d,
        y_scale=c * einv,
        y_unscale=cinv * e,
    )


def _full_symmetric_csc_pattern(
    p_upper: sp.csc_matrix,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    p_map_by_key: dict[tuple[int, int], int] = {}
    for col in range(p_upper.shape[1]):
        for ptr in range(p_upper.indptr[col], p_upper.indptr[col + 1]):
            row = int(p_upper.indices[ptr])
            rows.append(row)
            cols.append(col)
            p_map_by_key[(row, col)] = ptr
            if row != col:
                rows.append(col)
                cols.append(row)
                p_map_by_key[(col, row)] = ptr

    matrix = sp.csc_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)),
        shape=p_upper.shape,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    csc_rows, csc_cols = _csc_rows_cols(matrix)
    value_map = np.asarray(
        [p_map_by_key[(int(row), int(col))] for row, col in zip(csc_rows, csc_cols, strict=True)],
        dtype=np.int32,
    )
    return matrix, csc_rows, csc_cols, value_map


def _selected_a_csc_pattern(
    a_matrix: sp.csc_matrix,
    selected_rows: np.ndarray,
    *,
    row_offset: int = 0,
    sign: float = 1.0,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_rows = np.asarray(selected_rows, dtype=np.int32)
    row_to_selected = {int(row): pos for pos, row in enumerate(selected_rows)}
    rows: list[int] = []
    cols: list[int] = []
    key_to_a_ptr: dict[tuple[int, int], int] = {}
    for col in range(a_matrix.shape[1]):
        for ptr in range(a_matrix.indptr[col], a_matrix.indptr[col + 1]):
            row = int(a_matrix.indices[ptr])
            if row not in row_to_selected:
                continue
            selected_pos = int(row_to_selected[row])
            out_row = row_offset + selected_pos
            rows.append(out_row)
            cols.append(col)
            key_to_a_ptr[(out_row, col)] = ptr

    shape = (row_offset + selected_rows.size, a_matrix.shape[1])
    matrix = sp.csc_matrix((np.ones(len(rows), dtype=float), (rows, cols)), shape=shape)
    matrix.sum_duplicates()
    matrix.sort_indices()
    csc_rows, csc_cols = _csc_rows_cols(matrix)
    value_map = np.asarray(
        [key_to_a_ptr[(int(row), int(col))] for row, col in zip(csc_rows, csc_cols, strict=True)],
        dtype=np.int32,
    )
    signs = np.full(value_map.size, sign, dtype=np.float64)
    return matrix, csc_rows, csc_cols, value_map, signs


def _build_derivative_plan(
    p_upper: sp.csc_matrix,
    a_matrix: sp.csc_matrix,
    l: np.ndarray | None,
    u: np.ndarray | None,
) -> OSQPDerivativePlan | None:
    if l is None or u is None:
        return None

    n = p_upper.shape[0]
    m = a_matrix.shape[0]
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    infval = OSQP_INFTY * OSQP_MIN_SCALING

    is_ineq = l < u
    lower_indices = np.flatnonzero(is_ineq & (l > -infval)).astype(np.int32)
    upper_indices = np.flatnonzero(is_ineq & (u < infval)).astype(np.int32)
    eq_indices = np.flatnonzero(~is_ineq).astype(np.int32)
    n_ineq_l = int(lower_indices.size)
    n_ineq_u = int(upper_indices.size)
    n_ineq = n_ineq_l + n_ineq_u
    n_eq = int(eq_indices.size)
    half_dim = n + n_ineq + n_eq
    adjoint_dim = 2 * half_dim

    p_upper_rows, p_upper_cols = _csc_rows_cols(p_upper)
    _, p_full_rows, p_full_cols, p_full_value_map = _full_symmetric_csc_pattern(p_upper)

    _, lower_rows, lower_cols, lower_map, lower_sign = _selected_a_csc_pattern(
        a_matrix, lower_indices, row_offset=0, sign=-1.0
    )
    _, upper_rows, upper_cols, upper_map, upper_sign = _selected_a_csc_pattern(
        a_matrix, upper_indices, row_offset=n_ineq_l, sign=1.0
    )
    g_rows = np.concatenate([lower_rows, upper_rows]).astype(np.int32)
    g_cols = np.concatenate([lower_cols, upper_cols]).astype(np.int32)
    g_a_value_map = np.concatenate([lower_map, upper_map]).astype(np.int32)
    g_sign = np.concatenate([lower_sign, upper_sign]).astype(np.float64)

    # The concatenated lower/upper construction above is already CSC-column ordered
    # within each half, but combining the two halves can interleave rows.  Re-sort by
    # constructing a fresh CSC matrix and remapping rows/columns back to A entries.
    g_key_to_payload = {
        (int(row), int(col)): (int(a_ptr), float(sign_value))
        for row, col, a_ptr, sign_value in zip(g_rows, g_cols, g_a_value_map, g_sign, strict=True)
    }
    g_matrix = sp.csc_matrix(
        (np.ones(g_rows.size, dtype=float), (g_rows, g_cols)),
        shape=(n_ineq, n),
    )
    g_matrix.sum_duplicates()
    g_matrix.sort_indices()
    g_rows, g_cols = _csc_rows_cols(g_matrix)
    g_payload = [g_key_to_payload[(int(row), int(col))] for row, col in zip(g_rows, g_cols, strict=True)]
    g_a_value_map = np.asarray([payload[0] for payload in g_payload], dtype=np.int32)
    g_sign = np.asarray([payload[1] for payload in g_payload], dtype=np.float64)

    _, aeq_rows, aeq_cols, aeq_a_value_map, _ = _selected_a_csc_pattern(
        a_matrix, eq_indices, row_offset=0, sign=1.0
    )

    rows: list[int] = []
    cols: list[int] = []
    tags: dict[str, list[tuple[int, int]]] = {
        "diag_first": [],
        "diag_second": [],
        "p": [],
        "g": [],
        "aeq": [],
        "glambda": [],
        "slack": [],
        "aeq_t": [],
    }

    def add(tag: str, row: int, col: int) -> None:
        rows.append(row)
        cols.append(col)
        tags[tag].append((row, col))

    for i in range(half_dim):
        add("diag_first", i, i)
    for row, col in zip(p_full_rows, p_full_cols, strict=True):
        add("p", int(row), half_dim + int(col))
    for row, col in zip(g_rows, g_cols, strict=True):
        add("g", n + int(row), half_dim + int(col))
    for row, col in zip(aeq_rows, aeq_cols, strict=True):
        add("aeq", n + n_ineq + int(row), half_dim + int(col))
    for row, col in zip(g_rows, g_cols, strict=True):
        add("glambda", int(col), half_dim + n + int(row))
    for i in range(n_ineq):
        add("slack", n + i, half_dim + n + i)
    for row, col in zip(aeq_rows, aeq_cols, strict=True):
        add("aeq_t", int(col), half_dim + n + n_ineq + int(row))
    for i in range(half_dim):
        add("diag_second", half_dim + i, half_dim + i)

    adjoint_upper = sp.csc_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)),
        shape=(adjoint_dim, adjoint_dim),
    )
    adjoint_upper.sum_duplicates()
    adjoint_upper.sort_indices()

    key_to_pos: dict[tuple[int, int], int] = {}
    for col in range(adjoint_dim):
        for ptr in range(adjoint_upper.indptr[col], adjoint_upper.indptr[col + 1]):
            key_to_pos[(int(adjoint_upper.indices[ptr]), col)] = ptr

    def positions(tag: str) -> np.ndarray:
        return np.asarray([key_to_pos[key] for key in tags[tag]], dtype=np.int32)

    eps = 1e-6
    numeric_data = np.zeros_like(adjoint_upper.data)
    numeric_data[positions("diag_first")] = 1.0 + eps
    numeric_data[positions("diag_second")] = -eps
    numeric_data[positions("p")] = np.asarray(p_upper.data[p_full_value_map], dtype=float)
    numeric_data[positions("g")] = np.asarray(a_matrix.data[g_a_value_map] * g_sign, dtype=float)
    numeric_data[positions("aeq")] = np.asarray(a_matrix.data[aeq_a_value_map], dtype=float)
    numeric_data[positions("glambda")] = 0.0
    numeric_data[positions("slack")] = -1.0
    numeric_data[positions("aeq_t")] = np.asarray(a_matrix.data[aeq_a_value_map], dtype=float)
    adjoint_upper.data = numeric_data

    qdldl_plan = build_qdldl_plan(adjoint_upper, upper=True)
    return OSQPDerivativePlan(
        n=n,
        m=m,
        n_ineq_l=n_ineq_l,
        n_ineq_u=n_ineq_u,
        n_eq=n_eq,
        adjoint_dim=adjoint_dim,
        adjoint_upper=adjoint_upper,
        qdldl_plan=qdldl_plan,
        lower_indices=lower_indices,
        upper_indices=upper_indices,
        eq_indices=eq_indices,
        p_upper_rows=p_upper_rows,
        p_upper_cols=p_upper_cols,
        p_full_rows=p_full_rows,
        p_full_cols=p_full_cols,
        p_full_value_map=p_full_value_map,
        g_rows=g_rows,
        g_cols=g_cols,
        g_a_value_map=g_a_value_map,
        g_sign=g_sign,
        aeq_rows=aeq_rows,
        aeq_cols=aeq_cols,
        aeq_a_value_map=aeq_a_value_map,
        diag_first_pos=positions("diag_first"),
        diag_second_pos=positions("diag_second"),
        p_block_pos=positions("p"),
        g_block_pos=positions("g"),
        aeq_block_pos=positions("aeq"),
        glambda_block_pos=positions("glambda"),
        slack_diag_pos=positions("slack"),
        aeq_t_block_pos=positions("aeq_t"),
        eps=eps,
    )


def build_osqp_plan(
    p_matrix: sp.spmatrix,
    a_matrix: sp.spmatrix,
    l: np.ndarray | None = None,
    u: np.ndarray | None = None,
    settings: OSQPSettings | None = None,
    derivatives: bool = False,
    scaling_q: np.ndarray | None = None,
) -> FixedOSQPPlan:
    """Do all fixed-pattern OSQP direct-solver setup on the CPU."""

    settings = OSQPSettings() if settings is None else settings
    if settings.scaling < 0:
        raise ValueError("settings.scaling must be nonnegative")
    if settings.adaptive_rho:
        raise NotImplementedError("JAX OSQP currently expects adaptive_rho=False")
    if settings.polishing:
        raise NotImplementedError("JAX OSQP currently expects polishing=False")

    p_upper = _upper_with_diagonal(p_matrix)
    a_csc = sp.csc_matrix(a_matrix)
    a_csc.sum_duplicates()
    a_csc.sort_indices()
    if p_upper.shape[0] != p_upper.shape[1]:
        raise ValueError("P must be square")
    if a_csc.shape[1] != p_upper.shape[0]:
        raise ValueError("A column count must match P dimension")

    scaling = _compute_scaling(p_upper, a_csc, settings.scaling, q=scaling_q)
    scaled_l = None if l is None else np.asarray(l, dtype=np.float64) * scaling.e
    scaled_u = None if u is None else np.asarray(u, dtype=np.float64) * scaling.e
    rho_vec, rho_inv_vec, constr_type = _rho_vectors(
        settings, a_csc.shape[0], scaled_l, scaled_u
    )
    p_upper_for_kkt = p_upper.copy()
    a_csc_for_kkt = a_csc.copy()
    p_upper_for_kkt.data = (
        np.asarray(p_upper_for_kkt.data, dtype=np.float64) * scaling.p_scale
    )
    a_csc_for_kkt.data = (
        np.asarray(a_csc_for_kkt.data, dtype=np.float64) * scaling.a_scale
    )
    kkt, p_to_kkt, p_diag_sigma, a_to_kkt, rho_to_kkt = _build_kkt_and_maps(
        p_upper_for_kkt, a_csc_for_kkt, settings, rho_inv_vec
    )
    qdldl_plan = build_qdldl_plan(kkt, upper=True)
    p_rows, p_cols, p_value_map = _p_full_pattern(p_upper)
    derivative_plan = _build_derivative_plan(p_upper, a_csc, l, u) if derivatives else None

    a_rows: list[int] = []
    a_cols: list[int] = []
    for col in range(a_csc.shape[1]):
        for ptr in range(a_csc.indptr[col], a_csc.indptr[col + 1]):
            a_rows.append(int(a_csc.indices[ptr]))
            a_cols.append(col)

    return FixedOSQPPlan(
        n=p_upper.shape[0],
        m=a_csc.shape[0],
        p_upper=p_upper,
        a_matrix=a_csc,
        kkt_upper=kkt,
        qdldl_plan=qdldl_plan,
        p_to_kkt=p_to_kkt,
        p_diag_sigma=p_diag_sigma,
        a_to_kkt=a_to_kkt,
        rho_to_kkt=rho_to_kkt,
        a_rows=np.asarray(a_rows, dtype=np.int32),
        a_cols=np.asarray(a_cols, dtype=np.int32),
        p_full_rows=p_rows,
        p_full_cols=p_cols,
        p_full_value_map=p_value_map,
        rho_vec=rho_vec,
        rho_inv_vec=rho_inv_vec,
        constr_type=constr_type,
        scaling_c=scaling.c,
        scaling_cinv=scaling.cinv,
        scaling_d=scaling.d,
        scaling_dinv=scaling.dinv,
        scaling_e=scaling.e,
        scaling_einv=scaling.einv,
        p_scale=scaling.p_scale,
        a_scale=scaling.a_scale,
        q_scale=scaling.q_scale,
        y_scale=scaling.y_scale,
        y_unscale=scaling.y_unscale,
        derivative_plan=derivative_plan,
        settings=settings,
    )


def values_for_compiled_qdldl(compiled: CompiledQDLDL, kkt_values: jax.Array) -> jax.Array:
    if compiled.values_layout == "original_batch":
        return kkt_values
    if compiled.values_layout == "original_symbolic":
        return kkt_values.T
    raise ValueError(f"unknown QDLDL values layout {compiled.values_layout}")


def rhs_for_compiled_qdldl(compiled: CompiledQDLDL, rhs: jax.Array) -> jax.Array:
    if compiled.rhs_layout == "batch":
        return rhs
    if compiled.rhs_layout == "symbolic":
        return rhs.T
    raise ValueError(f"unknown QDLDL rhs layout {compiled.rhs_layout}")


def batch_from_compiled_qdldl(compiled: CompiledQDLDL, values: jax.Array) -> jax.Array:
    return values.T if compiled.rhs_layout == "symbolic" else values


def compile_osqp(
    plan: FixedOSQPPlan,
    dtype: np.dtype | str = np.float64,
    *,
    qdldl_backend: str = "jax",
    qdldl_factor_backend: str | None = None,
    qdldl_solve_backend: str | None = None,
    transpose_work: bool = False,
    segmented: bool = False,
    segment_budget: int = 64,
    segment_strategy: str = "optimal",
    level_scheduled_solve: bool = False,
    level_scheduled_solve_threshold: int = 1,
    derivatives: bool = False,
    derivative_refinement_iters: int = 100,
) -> CompiledOSQP:
    """Compile fixed-pattern batched OSQP iterations for one QDLDL variant."""

    dtype = np.dtype(dtype)
    _require_x64_for_float64(dtype)
    jdtype = jnp.dtype(dtype)
    settings = plan.settings
    qdldl = compile_qdldl_variant(
        plan.qdldl_plan,
        dtype=dtype,
        backend=qdldl_backend,
        factor_backend=qdldl_factor_backend,
        solve_backend=qdldl_solve_backend,
        transpose_work=transpose_work,
        segmented=segmented,
        segment_budget=segment_budget,
        segment_strategy=segment_strategy,
        level_scheduled_solve=level_scheduled_solve,
        level_scheduled_solve_threshold=level_scheduled_solve_threshold,
    )
    derivative_qdldl = None
    derivative_plan = plan.derivative_plan
    if derivatives:
        if derivative_plan is None:
            raise ValueError("call build_osqp_plan(..., derivatives=True) before compile_osqp(..., derivatives=True)")
        derivative_qdldl = compile_qdldl_variant(
            derivative_plan.qdldl_plan,
            dtype=dtype,
            backend=qdldl_backend,
            factor_backend=qdldl_factor_backend,
            solve_backend=qdldl_solve_backend,
            transpose_work=transpose_work,
            segmented=segmented,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
            level_scheduled_solve=level_scheduled_solve,
            level_scheduled_solve_threshold=level_scheduled_solve_threshold,
        )

    kkt_base = jnp.asarray(plan.kkt_upper.data, dtype=jdtype)
    p_to_kkt = jnp.asarray(plan.p_to_kkt, dtype=jnp.int32)
    p_diag_sigma = jnp.asarray(plan.p_diag_sigma, dtype=jdtype)
    a_to_kkt = jnp.asarray(plan.a_to_kkt, dtype=jnp.int32)
    p_scale = jnp.asarray(plan.p_scale, dtype=jdtype)
    a_scale = jnp.asarray(plan.a_scale, dtype=jdtype)
    q_scale = jnp.asarray(plan.q_scale, dtype=jdtype)
    scaling_d = jnp.asarray(plan.scaling_d, dtype=jdtype)
    scaling_dinv = jnp.asarray(plan.scaling_dinv, dtype=jdtype)
    scaling_e = jnp.asarray(plan.scaling_e, dtype=jdtype)
    scaling_einv = jnp.asarray(plan.scaling_einv, dtype=jdtype)
    y_scale = jnp.asarray(plan.y_scale, dtype=jdtype)
    y_unscale = jnp.asarray(plan.y_unscale, dtype=jdtype)
    a_rows = jnp.asarray(plan.a_rows, dtype=jnp.int32)
    a_cols = jnp.asarray(plan.a_cols, dtype=jnp.int32)
    p_full_rows = jnp.asarray(plan.p_full_rows, dtype=jnp.int32)
    p_full_cols = jnp.asarray(plan.p_full_cols, dtype=jnp.int32)
    p_full_value_map = jnp.asarray(plan.p_full_value_map, dtype=jnp.int32)
    rho_vec = jnp.asarray(plan.rho_vec, dtype=jdtype)
    rho_inv_vec = jnp.asarray(plan.rho_inv_vec, dtype=jdtype)
    sigma = jnp.asarray(settings.sigma, dtype=jdtype)
    alpha = jnp.asarray(settings.alpha, dtype=jdtype)
    n = plan.n
    m = plan.m

    def assemble_kkt_values(p_values: jax.Array, a_values: jax.Array) -> jax.Array:
        p_values = jnp.asarray(p_values, dtype=jdtype)
        a_values = jnp.asarray(a_values, dtype=jdtype)
        matrix_batch = max(p_values.shape[0], a_values.shape[0])
        values = jnp.broadcast_to(kkt_base, (matrix_batch, plan.nnz_kkt))
        p_update = p_values * p_scale[None, :] + p_diag_sigma[None, :]
        p_update = jnp.broadcast_to(p_update, (matrix_batch, plan.nnz_p))
        a_update = jnp.broadcast_to(
            a_values * a_scale[None, :], (matrix_batch, plan.nnz_a)
        )
        values = values.at[:, p_to_kkt].set(p_update)
        values = values.at[:, a_to_kkt].set(a_update)
        return values

    def a_matvec(a_values: jax.Array, x: jax.Array) -> jax.Array:
        a_values = jnp.asarray(a_values, dtype=jdtype)
        x = jnp.asarray(x, dtype=jdtype)
        updates = a_values[:, :] * x[:, a_cols]
        out = jnp.zeros((x.shape[0], m), dtype=jdtype)
        return out.at[:, a_rows].add(updates)

    def at_matvec(a_values: jax.Array, y: jax.Array) -> jax.Array:
        a_values = jnp.asarray(a_values, dtype=jdtype)
        y = jnp.asarray(y, dtype=jdtype)
        updates = a_values[:, :] * y[:, a_rows]
        out = jnp.zeros((y.shape[0], n), dtype=jdtype)
        return out.at[:, a_cols].add(updates)

    def p_matvec(p_values: jax.Array, x: jax.Array) -> jax.Array:
        p_values = jnp.asarray(p_values, dtype=jdtype)
        x = jnp.asarray(x, dtype=jdtype)
        updates = p_values[:, p_full_value_map] * x[:, p_full_cols]
        out = jnp.zeros((x.shape[0], n), dtype=jdtype)
        return out.at[:, p_full_rows].add(updates)

    @jax.jit
    def factor(p_values: jax.Array, a_values: jax.Array) -> tuple[jax.Array, jax.Array]:
        kkt_values = assemble_kkt_values(p_values, a_values)
        qdldl_values = values_for_compiled_qdldl(qdldl, kkt_values)
        lx, _, dinv = qdldl.factor(qdldl_values)
        return lx, dinv

    def _iterate(carry, _):
        x, z, y, p_values, a_values, q, l, u, lx, dinv = carry
        rhs_x = sigma * x - q
        rhs_z = z - rho_inv_vec[None, :] * y
        rhs = jnp.concatenate([rhs_x, rhs_z], axis=1)
        sol = qdldl.solve(lx, dinv, rhs_for_compiled_qdldl(qdldl, rhs))
        sol = batch_from_compiled_qdldl(qdldl, sol)
        xtilde = sol[:, :n]
        ztilde = rhs_z + rho_inv_vec[None, :] * sol[:, n:]

        x_next = alpha * xtilde + (1.0 - alpha) * x
        z_relaxed = alpha * ztilde + (1.0 - alpha) * z
        z_next = jnp.minimum(jnp.maximum(z_relaxed + rho_inv_vec[None, :] * y, l), u)
        y_next = y + rho_vec[None, :] * (z_relaxed - z_next)
        return (x_next, z_next, y_next, p_values, a_values, q, l, u, lx, dinv), None

    def _solve_with_factor_from_state(
        lx: jax.Array,
        dinv: jax.Array,
        p_values: jax.Array,
        a_values: jax.Array,
        q: jax.Array,
        l: jax.Array,
        u: jax.Array,
        x0: jax.Array,
        z0: jax.Array,
        y0: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        p_values = jnp.asarray(p_values, dtype=jdtype)
        a_values = jnp.asarray(a_values, dtype=jdtype)
        q = jnp.asarray(q, dtype=jdtype)
        l = jnp.asarray(l, dtype=jdtype)
        u = jnp.asarray(u, dtype=jdtype)
        x0 = jnp.asarray(x0, dtype=jdtype)
        z0 = jnp.asarray(z0, dtype=jdtype)
        y0 = jnp.asarray(y0, dtype=jdtype)
        q_scaled = q * q_scale[None, :]
        l_scaled = l * scaling_e[None, :]
        u_scaled = u * scaling_e[None, :]
        x0_scaled = x0 * scaling_dinv[None, :]
        z0_scaled = z0 * scaling_e[None, :]
        y0_scaled = y0 * y_scale[None, :]
        carry = (
            x0_scaled,
            z0_scaled,
            y0_scaled,
            p_values,
            a_values,
            q_scaled,
            l_scaled,
            u_scaled,
            lx,
            dinv,
        )
        carry, _ = lax.scan(_iterate, carry, None, length=settings.max_iter)
        x, z, y, _, _, _, _, _, _, _ = carry

        x = x * scaling_d[None, :]
        z = z * scaling_einv[None, :]
        y = y * y_unscale[None, :]
        ax = a_matvec(a_values, x)
        px = p_matvec(p_values, x)
        aty = at_matvec(a_values, y)
        prim_res = jnp.max(jnp.abs(ax - z), axis=1)
        dual_res = jnp.max(jnp.abs(px + q + aty), axis=1)
        obj_val = 0.5 * jnp.sum(x * px, axis=1) + jnp.sum(q * x, axis=1)
        return x, z, y, prim_res, dual_res, obj_val

    def init_warm_start(batch_size: int) -> OSQPWarmStart:
        return OSQPWarmStart(
            x=jnp.zeros((batch_size, n), dtype=jdtype),
            z=jnp.zeros((batch_size, m), dtype=jdtype),
            y=jnp.zeros((batch_size, m), dtype=jdtype),
        )

    @jax.jit
    def solve_with_factor(
        lx: jax.Array,
        dinv: jax.Array,
        p_values: jax.Array,
        a_values: jax.Array,
        q: jax.Array,
        l: jax.Array,
        u: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        q = jnp.asarray(q, dtype=jdtype)
        x0 = jnp.zeros((q.shape[0], n), dtype=jdtype)
        z0 = jnp.zeros((q.shape[0], m), dtype=jdtype)
        y0 = jnp.zeros((q.shape[0], m), dtype=jdtype)
        return _solve_with_factor_from_state(
            lx,
            dinv,
            p_values,
            a_values,
            q,
            l,
            u,
            x0,
            z0,
            y0,
        )

    @jax.jit
    def solve_with_factor_warm_start(
        lx: jax.Array,
        dinv: jax.Array,
        p_values: jax.Array,
        a_values: jax.Array,
        q: jax.Array,
        l: jax.Array,
        u: jax.Array,
        warm_start_x: jax.Array,
        warm_start_z: jax.Array,
        warm_start_y: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return _solve_with_factor_from_state(
            lx,
            dinv,
            p_values,
            a_values,
            q,
            l,
            u,
            warm_start_x,
            warm_start_z,
            warm_start_y,
        )

    @jax.jit
    def solve(
        p_values: jax.Array,
        a_values: jax.Array,
        q: jax.Array,
        l: jax.Array,
        u: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        lx, dinv = factor(p_values, a_values)
        return solve_with_factor(lx, dinv, p_values, a_values, q, l, u)

    @jax.jit
    def linear_solve_loop(
        lx: jax.Array,
        dinv: jax.Array,
        rhs: jax.Array,
    ) -> jax.Array:
        rhs_layout = rhs_for_compiled_qdldl(qdldl, jnp.asarray(rhs, dtype=jdtype))

        def solve_once(carry, _):
            return qdldl.solve(lx, dinv, carry), None

        sol, _ = lax.scan(solve_once, rhs_layout, None, length=settings.max_iter)
        return batch_from_compiled_qdldl(qdldl, sol)

    @jax.jit
    def solve_warm_start(
        p_values: jax.Array,
        a_values: jax.Array,
        q: jax.Array,
        l: jax.Array,
        u: jax.Array,
        warm_start_x: jax.Array,
        warm_start_z: jax.Array,
        warm_start_y: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        lx, dinv = factor(p_values, a_values)
        return solve_with_factor_warm_start(
            lx,
            dinv,
            p_values,
            a_values,
            q,
            l,
            u,
            warm_start_x,
            warm_start_z,
            warm_start_y,
        )

    adjoint_derivative_compute = None
    adjoint_derivative_get_mat = None
    adjoint_derivative_get_vec = None
    solve_primal = None
    solve_xy = None
    solve_callable = solve
    solve_warm_start_callable = solve_warm_start

    if derivatives:
        assert derivative_plan is not None
        assert derivative_qdldl is not None

        d_adj_base = jnp.asarray(derivative_plan.adjoint_upper.data, dtype=jdtype)
        d_adj_rows_np, d_adj_cols_np = _csc_rows_cols(derivative_plan.adjoint_upper)
        d_adj_rows = jnp.asarray(d_adj_rows_np, dtype=jnp.int32)
        d_adj_cols = jnp.asarray(d_adj_cols_np, dtype=jnp.int32)
        d_adj_offdiag = jnp.asarray(d_adj_rows_np != d_adj_cols_np)
        d_lower_indices = jnp.asarray(derivative_plan.lower_indices, dtype=jnp.int32)
        d_upper_indices = jnp.asarray(derivative_plan.upper_indices, dtype=jnp.int32)
        d_eq_indices = jnp.asarray(derivative_plan.eq_indices, dtype=jnp.int32)
        d_p_upper_rows = jnp.asarray(derivative_plan.p_upper_rows, dtype=jnp.int32)
        d_p_upper_cols = jnp.asarray(derivative_plan.p_upper_cols, dtype=jnp.int32)
        d_p_full_value_map = jnp.asarray(derivative_plan.p_full_value_map, dtype=jnp.int32)
        d_g_rows = jnp.asarray(derivative_plan.g_rows, dtype=jnp.int32)
        d_g_cols = jnp.asarray(derivative_plan.g_cols, dtype=jnp.int32)
        d_g_a_value_map = jnp.asarray(derivative_plan.g_a_value_map, dtype=jnp.int32)
        d_g_sign = jnp.asarray(derivative_plan.g_sign, dtype=jdtype)
        d_aeq_rows = jnp.asarray(derivative_plan.aeq_rows, dtype=jnp.int32)
        d_aeq_cols = jnp.asarray(derivative_plan.aeq_cols, dtype=jnp.int32)
        d_aeq_a_value_map = jnp.asarray(derivative_plan.aeq_a_value_map, dtype=jnp.int32)
        d_diag_first_pos = jnp.asarray(derivative_plan.diag_first_pos, dtype=jnp.int32)
        d_diag_second_pos = jnp.asarray(derivative_plan.diag_second_pos, dtype=jnp.int32)
        d_p_block_pos = jnp.asarray(derivative_plan.p_block_pos, dtype=jnp.int32)
        d_g_block_pos = jnp.asarray(derivative_plan.g_block_pos, dtype=jnp.int32)
        d_aeq_block_pos = jnp.asarray(derivative_plan.aeq_block_pos, dtype=jnp.int32)
        d_glambda_block_pos = jnp.asarray(derivative_plan.glambda_block_pos, dtype=jnp.int32)
        d_slack_diag_pos = jnp.asarray(derivative_plan.slack_diag_pos, dtype=jnp.int32)
        d_aeq_t_block_pos = jnp.asarray(derivative_plan.aeq_t_block_pos, dtype=jnp.int32)
        d_half_dim = derivative_plan.half_dim
        d_adjoint_dim = derivative_plan.adjoint_dim
        d_n_ineq_l = derivative_plan.n_ineq_l
        d_n_ineq_u = derivative_plan.n_ineq_u
        d_n_ineq = derivative_plan.n_ineq
        d_n_eq = derivative_plan.n_eq
        d_eps = jnp.asarray(derivative_plan.eps, dtype=jdtype)
        d_a_rows = jnp.asarray(plan.a_rows, dtype=jnp.int32)
        d_a_cols = jnp.asarray(plan.a_cols, dtype=jnp.int32)

        def _broadcast_first_dim(array: jax.Array, batch_size: int, width: int) -> jax.Array:
            return jnp.broadcast_to(array, (batch_size, width))

        def _adjoint_matvec(adj_values: jax.Array, vector: jax.Array) -> jax.Array:
            updates = adj_values * vector[:, d_adj_cols]
            out = jnp.zeros((vector.shape[0], d_adjoint_dim), dtype=jdtype)
            out = out.at[:, d_adj_rows].add(updates)
            transpose_updates = (
                adj_values
                * vector[:, d_adj_rows]
                * d_adj_offdiag.astype(jdtype)[None, :]
            )
            return out.at[:, d_adj_cols].add(transpose_updates)

        @jax.jit
        def _adjoint_derivative_compute_jit(
            p_values: jax.Array,
            a_values: jax.Array,
            l: jax.Array,
            u: jax.Array,
            x: jax.Array,
            y: jax.Array,
            dx: jax.Array,
            dy: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            p_values = jnp.asarray(p_values, dtype=jdtype)
            a_values = jnp.asarray(a_values, dtype=jdtype)
            l = jnp.asarray(l, dtype=jdtype)
            u = jnp.asarray(u, dtype=jdtype)
            x = jnp.asarray(x, dtype=jdtype)
            y = jnp.asarray(y, dtype=jdtype)
            dx = jnp.asarray(dx, dtype=jdtype)
            dy = jnp.asarray(dy, dtype=jdtype)
            batch_size = max(
                p_values.shape[0],
                a_values.shape[0],
                l.shape[0],
                u.shape[0],
                x.shape[0],
                y.shape[0],
                dx.shape[0],
                dy.shape[0],
            )
            p_values = _broadcast_first_dim(p_values, batch_size, plan.nnz_p)
            a_values = _broadcast_first_dim(a_values, batch_size, plan.nnz_a)
            l = _broadcast_first_dim(l, batch_size, m)
            u = _broadcast_first_dim(u, batch_size, m)
            x = _broadcast_first_dim(x, batch_size, n)
            y = _broadcast_first_dim(y, batch_size, m)
            dx = _broadcast_first_dim(dx, batch_size, n)
            dy = _broadcast_first_dim(dy, batch_size, m)

            y_u = jnp.maximum(y, 0.0)
            y_l = -jnp.minimum(y, 0.0)
            lambda_l = y_l[:, d_lower_indices]
            lambda_u = y_u[:, d_upper_indices]
            lambda_ineq = jnp.concatenate([lambda_l, lambda_u], axis=1)

            g_values = a_values[:, d_g_a_value_map] * d_g_sign[None, :]
            aeq_values = a_values[:, d_aeq_a_value_map]
            g_x_updates = g_values * x[:, d_g_cols]
            g_x = jnp.zeros((batch_size, d_n_ineq), dtype=jdtype)
            g_x = g_x.at[:, d_g_rows].add(g_x_updates)
            h = jnp.concatenate([-l[:, d_lower_indices], u[:, d_upper_indices]], axis=1)
            slacks = g_x - h

            adj_values_unperturbed = jnp.broadcast_to(
                d_adj_base, (batch_size, derivative_plan.adjoint_upper.nnz)
            )
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_diag_first_pos].set(1.0)
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_diag_second_pos].set(0.0)
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_p_block_pos].set(
                p_values[:, d_p_full_value_map]
            )
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_g_block_pos].set(g_values)
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_aeq_block_pos].set(aeq_values)
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_glambda_block_pos].set(
                lambda_ineq[:, d_g_rows] * g_values
            )
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_slack_diag_pos].set(slacks)
            adj_values_unperturbed = adj_values_unperturbed.at[:, d_aeq_t_block_pos].set(aeq_values)
            adj_values = adj_values_unperturbed.at[:, d_diag_first_pos].set(1.0 + d_eps)
            adj_values = adj_values.at[:, d_diag_second_pos].set(-d_eps)

            rhs = jnp.zeros((batch_size, d_adjoint_dim), dtype=jdtype)
            rhs = rhs.at[:, :n].set(-dx)
            rhs = rhs.at[:, n : n + d_n_ineq_l].set(-dy[:, d_lower_indices])
            rhs = rhs.at[:, n + d_n_ineq_l : n + d_n_ineq].set(-dy[:, d_upper_indices])
            d_nu = jnp.where(y[:, d_eq_indices] >= 0.0, dy[:, d_eq_indices], -dy[:, d_eq_indices])
            rhs = rhs.at[:, n + d_n_ineq : n + d_n_ineq + d_n_eq].set(-d_nu)

            adj_qdldl_values = values_for_compiled_qdldl(derivative_qdldl, adj_values)
            adj_lx, _, adj_dinv = derivative_qdldl.factor(adj_qdldl_values)
            sol = derivative_qdldl.solve(adj_lx, adj_dinv, rhs_for_compiled_qdldl(derivative_qdldl, rhs))
            sol = batch_from_compiled_qdldl(derivative_qdldl, sol)

            def refinement_step(sol_inner, _):
                residual = _adjoint_matvec(adj_values_unperturbed, sol_inner) - rhs
                correction = derivative_qdldl.solve(
                    adj_lx,
                    adj_dinv,
                    rhs_for_compiled_qdldl(derivative_qdldl, residual),
                )
                correction = batch_from_compiled_qdldl(derivative_qdldl, correction)
                return sol_inner - correction, None

            if derivative_refinement_iters:
                sol, _ = lax.scan(refinement_step, sol, None, length=derivative_refinement_iters)

            r_yl_raw = jnp.zeros((batch_size, m), dtype=jdtype)
            r_yu_raw = jnp.zeros((batch_size, m), dtype=jdtype)
            pos = d_half_dim + n
            r_yl_raw = r_yl_raw.at[:, d_lower_indices].set(-sol[:, pos : pos + d_n_ineq_l])
            pos += d_n_ineq_l
            r_yu_raw = r_yu_raw.at[:, d_upper_indices].set(sol[:, pos : pos + d_n_ineq_u])
            pos += d_n_ineq_u
            eq_sol = sol[:, pos : pos + d_n_eq]
            eq_y = y[:, d_eq_indices]
            eq_positive = eq_y >= 0.0
            r_yu_raw = r_yu_raw.at[:, d_eq_indices].set(jnp.where(eq_positive, eq_sol / eq_y, 0.0))
            r_yl_raw = r_yl_raw.at[:, d_eq_indices].set(jnp.where(eq_positive, 0.0, -eq_sol / eq_y))

            ryl = -(r_yl_raw * y_l)
            ryu = r_yu_raw * y_u
            return sol, y_l, y_u, ryl, ryu

        @jax.jit
        def _adjoint_derivative_get_vec_jit(
            derivative_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            sol, _, _, ryl, ryu = derivative_state
            rx = sol[:, d_half_dim : d_half_dim + n]
            return rx, ryl, -ryu

        @jax.jit
        def _adjoint_derivative_get_mat_jit(
            x: jax.Array,
            y: jax.Array,
            derivative_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            sol, y_l, y_u, ryl, ryu = derivative_state
            x = jnp.asarray(x, dtype=jdtype)
            y = jnp.asarray(y, dtype=jdtype)
            rx = sol[:, d_half_dim : d_half_dim + n]
            batch_size = max(x.shape[0], y.shape[0], rx.shape[0])
            x = _broadcast_first_dim(x, batch_size, n)
            y_l = _broadcast_first_dim(y_l, batch_size, m)
            y_u = _broadcast_first_dim(y_u, batch_size, m)
            ryl = _broadcast_first_dim(ryl, batch_size, m)
            ryu = _broadcast_first_dim(ryu, batch_size, m)
            rx = _broadcast_first_dim(rx, batch_size, n)

            d_p = 0.5 * (
                rx[:, d_p_upper_rows] * x[:, d_p_upper_cols]
                + rx[:, d_p_upper_cols] * x[:, d_p_upper_rows]
            )
            d_a = (
                (y_u[:, d_a_rows] - y_l[:, d_a_rows]) * rx[:, d_a_cols]
                + (ryu[:, d_a_rows] - ryl[:, d_a_rows]) * x[:, d_a_cols]
            )
            return d_p, d_a

        def _as_batched(value: jax.Array, width: int) -> jax.Array:
            array = jnp.asarray(value, dtype=jdtype)
            if array.ndim == 1:
                array = array[None, :]
            if array.shape[1] != width:
                raise ValueError(f"expected width {width}, got {array.shape[1]}")
            return array

        def _adjoint_derivative_compute_wrapper(
            p_values: jax.Array,
            a_values: jax.Array,
            l: jax.Array,
            u: jax.Array,
            x: jax.Array,
            y: jax.Array,
            dx: jax.Array | None = None,
            dy: jax.Array | None = None,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            p_values = _as_batched(p_values, plan.nnz_p)
            a_values = _as_batched(a_values, plan.nnz_a)
            l = _as_batched(l, m)
            u = _as_batched(u, m)
            x = _as_batched(x, n)
            y = _as_batched(y, m)
            dx = jnp.zeros_like(x) if dx is None else _as_batched(dx, n)
            dy = jnp.zeros_like(y) if dy is None else _as_batched(dy, m)
            return _adjoint_derivative_compute_jit(p_values, a_values, l, u, x, y, dx, dy)

        def _adjoint_derivative_get_mat_wrapper(
            x: jax.Array,
            y: jax.Array,
            derivative_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            return _adjoint_derivative_get_mat_jit(_as_batched(x, n), _as_batched(y, m), derivative_state)

        adjoint_derivative_compute = _adjoint_derivative_compute_wrapper
        adjoint_derivative_get_mat = _adjoint_derivative_get_mat_wrapper
        adjoint_derivative_get_vec = _adjoint_derivative_get_vec_jit

        def _match_broadcast_gradient(grad: jax.Array, reference: jax.Array) -> jax.Array:
            if reference.shape[0] == 1 and grad.shape[0] != 1:
                return jnp.sum(grad, axis=0, keepdims=True)
            return grad

        @jax.custom_vjp
        def _solve_primal_custom(
            p_values: jax.Array,
            a_values: jax.Array,
            q: jax.Array,
            l: jax.Array,
            u: jax.Array,
        ) -> jax.Array:
            return solve(p_values, a_values, q, l, u)[0]

        def _solve_primal_fwd(p_values, a_values, q, l, u):
            x_out, _, y_out, _, _, _ = solve(p_values, a_values, q, l, u)
            return x_out, (p_values, a_values, q, l, u, x_out, y_out)

        def _solve_primal_bwd(residual, x_bar):
            p_values, a_values, q, l, u, x_out, y_out = residual
            state = _adjoint_derivative_compute_jit(
                p_values,
                a_values,
                l,
                u,
                x_out,
                y_out,
                x_bar,
                jnp.zeros_like(y_out),
            )
            d_p, d_a = _adjoint_derivative_get_mat_jit(x_out, y_out, state)
            d_q, d_l, d_u = _adjoint_derivative_get_vec_jit(state)
            return (
                _match_broadcast_gradient(d_p, p_values),
                _match_broadcast_gradient(d_a, a_values),
                d_q,
                d_l,
                d_u,
            )

        _solve_primal_custom.defvjp(_solve_primal_fwd, _solve_primal_bwd)

        @jax.custom_vjp
        def _solve_xy_custom(
            p_values: jax.Array,
            a_values: jax.Array,
            q: jax.Array,
            l: jax.Array,
            u: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            x_out, _, y_out, _, _, _ = solve(p_values, a_values, q, l, u)
            return x_out, y_out

        def _solve_xy_fwd(p_values, a_values, q, l, u):
            x_out, _, y_out, _, _, _ = solve(p_values, a_values, q, l, u)
            return (x_out, y_out), (p_values, a_values, q, l, u, x_out, y_out)

        def _solve_xy_bwd(residual, cotangents):
            p_values, a_values, q, l, u, x_out, y_out = residual
            x_bar, y_bar = cotangents
            state = _adjoint_derivative_compute_jit(
                p_values,
                a_values,
                l,
                u,
                x_out,
                y_out,
                x_bar,
                y_bar,
            )
            d_p, d_a = _adjoint_derivative_get_mat_jit(x_out, y_out, state)
            d_q, d_l, d_u = _adjoint_derivative_get_vec_jit(state)
            return (
                _match_broadcast_gradient(d_p, p_values),
                _match_broadcast_gradient(d_a, a_values),
                d_q,
                d_l,
                d_u,
            )

        _solve_xy_custom.defvjp(_solve_xy_fwd, _solve_xy_bwd)
        solve_primal = _solve_primal_custom
        solve_xy = _solve_xy_custom

        @jax.custom_vjp
        def _solve_custom(
            p_values: jax.Array,
            a_values: jax.Array,
            q: jax.Array,
            l: jax.Array,
            u: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return solve(p_values, a_values, q, l, u)

        def _solve_custom_fwd(p_values, a_values, q, l, u):
            out = solve(p_values, a_values, q, l, u)
            x_out, _, y_out, _, _, _ = out
            return out, (p_values, a_values, q, l, u, x_out, y_out)

        def _solve_custom_bwd(residual, cotangents):
            p_values, a_values, q, l, u, x_out, y_out = residual
            x_bar, _, y_bar, _, _, obj_bar = cotangents
            px = p_matvec(p_values, x_out)
            dx_total = x_bar + obj_bar[:, None] * (px + q)
            state = _adjoint_derivative_compute_jit(
                p_values,
                a_values,
                l,
                u,
                x_out,
                y_out,
                dx_total,
                y_bar,
            )
            d_p, d_a = _adjoint_derivative_get_mat_jit(x_out, y_out, state)
            d_q, d_l, d_u = _adjoint_derivative_get_vec_jit(state)
            p_diag = d_p_upper_rows == d_p_upper_cols
            direct_p_scale = jnp.where(p_diag, 0.5, 1.0).astype(jdtype)
            d_p = d_p + obj_bar[:, None] * direct_p_scale[None, :] * (
                x_out[:, d_p_upper_rows] * x_out[:, d_p_upper_cols]
            )
            d_q = d_q + obj_bar[:, None] * x_out
            return (
                _match_broadcast_gradient(d_p, p_values),
                _match_broadcast_gradient(d_a, a_values),
                d_q,
                d_l,
                d_u,
            )

        _solve_custom.defvjp(_solve_custom_fwd, _solve_custom_bwd)
        solve_callable = _solve_custom

        @jax.custom_vjp
        def _solve_warm_start_custom(
            p_values: jax.Array,
            a_values: jax.Array,
            q: jax.Array,
            l: jax.Array,
            u: jax.Array,
            warm_start_x: jax.Array,
            warm_start_z: jax.Array,
            warm_start_y: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return solve_warm_start(
                p_values,
                a_values,
                q,
                l,
                u,
                warm_start_x,
                warm_start_z,
                warm_start_y,
            )

        def _solve_warm_start_fwd(
            p_values,
            a_values,
            q,
            l,
            u,
            warm_start_x,
            warm_start_z,
            warm_start_y,
        ):
            out = solve_warm_start(
                p_values,
                a_values,
                q,
                l,
                u,
                warm_start_x,
                warm_start_z,
                warm_start_y,
            )
            x_out, _, y_out, _, _, _ = out
            return out, (
                p_values,
                a_values,
                q,
                l,
                u,
                x_out,
                y_out,
                warm_start_x,
                warm_start_z,
                warm_start_y,
            )

        def _solve_warm_start_bwd(residual, cotangents):
            (
                p_values,
                a_values,
                q,
                l,
                u,
                x_out,
                y_out,
                warm_start_x,
                warm_start_z,
                warm_start_y,
            ) = residual
            x_bar, _, y_bar, _, _, obj_bar = cotangents
            px = p_matvec(p_values, x_out)
            dx_total = x_bar + obj_bar[:, None] * (px + q)
            state = _adjoint_derivative_compute_jit(
                p_values,
                a_values,
                l,
                u,
                x_out,
                y_out,
                dx_total,
                y_bar,
            )
            d_p, d_a = _adjoint_derivative_get_mat_jit(x_out, y_out, state)
            d_q, d_l, d_u = _adjoint_derivative_get_vec_jit(state)
            p_diag = d_p_upper_rows == d_p_upper_cols
            direct_p_scale = jnp.where(p_diag, 0.5, 1.0).astype(jdtype)
            d_p = d_p + obj_bar[:, None] * direct_p_scale[None, :] * (
                x_out[:, d_p_upper_rows] * x_out[:, d_p_upper_cols]
            )
            d_q = d_q + obj_bar[:, None] * x_out
            return (
                _match_broadcast_gradient(d_p, p_values),
                _match_broadcast_gradient(d_a, a_values),
                d_q,
                d_l,
                d_u,
                jnp.zeros_like(warm_start_x),
                jnp.zeros_like(warm_start_z),
                jnp.zeros_like(warm_start_y),
            )

        _solve_warm_start_custom.defvjp(
            _solve_warm_start_fwd,
            _solve_warm_start_bwd,
        )
        solve_warm_start_callable = _solve_warm_start_custom

    return CompiledOSQP(
        plan=plan,
        settings=settings,
        qdldl=qdldl,
        solve=solve_callable,
        factor=factor,
        solve_with_factor=solve_with_factor,
        variant=qdldl.variant,
        dtype=dtype,
        linear_solve_loop=linear_solve_loop,
        init_warm_start=init_warm_start,
        solve_warm_start=solve_warm_start_callable,
        solve_with_factor_warm_start=solve_with_factor_warm_start,
        adjoint_derivative_compute=adjoint_derivative_compute,
        adjoint_derivative_get_mat=adjoint_derivative_get_mat,
        adjoint_derivative_get_vec=adjoint_derivative_get_vec,
        solve_primal=solve_primal,
        solve_xy=solve_xy,
    )
