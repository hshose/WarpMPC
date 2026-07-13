"""Batched fixed-pattern QDLDL in JAX.

The CPU side performs the symbolic work once:

* AMD permutation through the qdldl Python binding
* symmetric permutation of the upper-triangular input
* QDLDL elimination tree and L sparsity pattern
* a padded numeric "program" for each factorization row

The JAX side only sees fixed-shape index arrays and values.  Its numerical
factorization is intentionally close to resources/qdldl/src/qdldl.c.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import qdldl
import scipy.sparse as sp
from jax import lax

from .types import CompiledQDLDL, QDLDLPlan

QDLDL_UNKNOWN = -1


def _require_x64_for_float64(dtype: np.dtype) -> None:
    if dtype == np.dtype(np.float64) and not jax.config.jax_enable_x64:
        raise ValueError(
            "QDLDL float64 compilation requires jax_enable_x64=True. "
            "Set it in the benchmark or example entry point before compiling."
        )



def _as_upper_csc(matrix: sp.spmatrix, upper: bool) -> sp.csc_matrix:
    matrix = sp.csc_matrix(matrix)
    if upper:
        out = matrix.copy()
    else:
        out = sp.triu(matrix, format="csc")
    out.sum_duplicates()
    out.sort_indices()
    return out


def _pinv(p: np.ndarray) -> np.ndarray:
    pinv = np.empty_like(p)
    pinv[p] = np.arange(p.size, dtype=p.dtype)
    return pinv


def _symperm_upper(
    matrix: sp.csc_matrix, pinv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of resources/qdldl-python/c/amd/src/perm.c:symperm."""

    n = matrix.shape[0]
    ap = matrix.indptr
    ai = matrix.indices
    ax = matrix.data
    counts = np.zeros(n, dtype=np.int64)

    for j in range(n):
        j2 = int(pinv[j])
        for ptr in range(ap[j], ap[j + 1]):
            i = int(ai[ptr])
            if i > j:
                continue
            i2 = int(pinv[i])
            counts[max(i2, j2)] += 1

    cp = np.empty(n + 1, dtype=np.int64)
    cp[0] = 0
    np.cumsum(counts, out=cp[1:])

    ci = np.empty(int(cp[-1]), dtype=np.int32)
    cx = np.empty(int(cp[-1]), dtype=ax.dtype)
    orig_to_perm = np.full(ap[-1], -1, dtype=np.int32)
    next_slot = cp[:-1].copy()

    for j in range(n):
        j2 = int(pinv[j])
        for ptr in range(ap[j], ap[j + 1]):
            i = int(ai[ptr])
            if i > j:
                continue
            i2 = int(pinv[i])
            col = max(i2, j2)
            row = min(i2, j2)
            dest = int(next_slot[col])
            next_slot[col] += 1
            ci[dest] = row
            cx[dest] = ax[ptr]
            orig_to_perm[ptr] = dest

    if np.any(orig_to_perm < 0):
        raise ValueError("input matrix contains lower-triangular entries")

    return cp.astype(np.int32), ci, cx, orig_to_perm


def _qdldl_etree(
    n: int, ap: np.ndarray, ai: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Port of QDLDL_etree for upper-triangular CSC patterns."""

    work = np.zeros(n, dtype=np.int32)
    lnz = np.zeros(n, dtype=np.int32)
    etree = np.full(n, QDLDL_UNKNOWN, dtype=np.int32)

    for i in range(n):
        if ap[i] == ap[i + 1]:
            raise ValueError(f"column {i} is empty; QDLDL requires a nonempty column")

    for j in range(n):
        work[j] = j
        for ptr in range(ap[j], ap[j + 1]):
            i = int(ai[ptr])
            if i > j:
                raise ValueError("matrix is not upper triangular after permutation")
            while work[i] != j:
                if etree[i] == QDLDL_UNKNOWN:
                    etree[i] = j
                lnz[i] += 1
                work[i] = j
                i = int(etree[i])

    return lnz, etree, int(np.sum(lnz, dtype=np.int64))


def _pad_2d(
    rows: list[list[int]], pad_value: int = 0, dtype: np.dtype = np.int32
) -> tuple[np.ndarray, np.ndarray]:
    width = max((len(row) for row in rows), default=0)
    out = np.full((len(rows), width), pad_value, dtype=dtype)
    mask = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = np.asarray(row, dtype=dtype)
            mask[i, : len(row)] = True
    return out, mask


def _build_numeric_program(
    n: int, ap: np.ndarray, ai: np.ndarray, lnz: np.ndarray, etree: np.ndarray
) -> dict[str, np.ndarray]:
    lp = np.empty(n + 1, dtype=np.int32)
    lp[0] = 0
    np.cumsum(lnz, out=lp[1:])
    nnz_l = int(lp[-1])
    li = np.empty(nnz_l, dtype=np.int32)
    next_slot = lp[:-1].copy()

    diag_pos = np.full(n, -1, dtype=np.int32)
    init_rows_by_k: list[list[int]] = []
    init_pos_by_k: list[list[int]] = []
    process_cols_by_k: list[list[int]] = []
    process_lidx_by_k: list[list[int]] = []
    process_offsets_by_k: list[list[int]] = []

    for k in range(n):
        init_rows: list[int] = []
        init_pos: list[int] = []

        for ptr in range(ap[k], ap[k + 1]):
            row = int(ai[ptr])
            if row == k:
                diag_pos[k] = ptr
            else:
                init_rows.append(row)
                init_pos.append(ptr)

        if diag_pos[k] < 0:
            raise ValueError(f"column {k} has no diagonal entry")

        init_rows_by_k.append(init_rows)
        init_pos_by_k.append(init_pos)

        y_markers = np.zeros(n, dtype=bool)
        y_idx: list[int] = []

        if k > 0:
            for ptr in range(ap[k], ap[k + 1]):
                bidx = int(ai[ptr])
                if bidx == k:
                    continue

                next_idx = bidx
                if not y_markers[next_idx]:
                    y_markers[next_idx] = True
                    elim_buffer = [next_idx]
                    next_idx = int(etree[bidx])

                    while next_idx != QDLDL_UNKNOWN and next_idx < k:
                        if y_markers[next_idx]:
                            break
                        y_markers[next_idx] = True
                        elim_buffer.append(next_idx)
                        next_idx = int(etree[next_idx])

                    while elim_buffer:
                        y_idx.append(elim_buffer.pop())

        process_cols = list(reversed(y_idx))
        process_lidx: list[int] = []
        process_offsets: list[int] = []

        for cidx in process_cols:
            dest = int(next_slot[cidx])
            process_lidx.append(dest)
            process_offsets.append(dest - int(lp[cidx]))
            li[dest] = k
            next_slot[cidx] += 1

        process_cols_by_k.append(process_cols)
        process_lidx_by_k.append(process_lidx)
        process_offsets_by_k.append(process_offsets)

    expected_next = lp[1:]
    if not np.array_equal(next_slot, expected_next):
        raise AssertionError("internal symbolic program did not fill L correctly")

    init_rows, init_mask = _pad_2d(init_rows_by_k)
    init_pos, _ = _pad_2d(init_pos_by_k)
    process_cols, process_mask = _pad_2d(process_cols_by_k)
    process_lidx, _ = _pad_2d(process_lidx_by_k)
    process_offsets, _ = _pad_2d(process_offsets_by_k)

    col_rows_by_col: list[list[int]] = []
    col_lidx_by_col: list[list[int]] = []
    for col in range(n):
        start = int(lp[col])
        stop = int(lp[col + 1])
        col_rows_by_col.append(li[start:stop].astype(int).tolist())
        col_lidx_by_col.append(list(range(start, stop)))
    col_rows, col_mask = _pad_2d(col_rows_by_col)
    col_lidx, _ = _pad_2d(col_lidx_by_col)

    return {
        "lp": lp,
        "li": li,
        "diag_pos": diag_pos,
        "init_rows": init_rows,
        "init_pos": init_pos,
        "init_mask": init_mask,
        "process_cols": process_cols,
        "process_lidx": process_lidx,
        "process_offsets": process_offsets,
        "process_mask": process_mask,
        "col_rows": col_rows,
        "col_lidx": col_lidx,
        "col_mask": col_mask,
    }


def build_qdldl_plan(matrix: sp.spmatrix, upper: bool = False) -> QDLDLPlan:
    """Build a fixed-pattern JAX QDLDL plan from a symmetric matrix.

    Parameters
    ----------
    matrix:
        Symmetric sparse matrix.  Pass only its upper triangle with
        ``upper=True`` if it is already in QDLDL's expected format.
    upper:
        Whether ``matrix`` is already upper-triangular CSC data.
    """

    a_upper = _as_upper_csc(matrix, upper=upper)
    n = a_upper.shape[0]
    if a_upper.shape[0] != a_upper.shape[1]:
        raise ValueError("QDLDL matrix must be square")

    solver = qdldl.Solver(a_upper, upper=True)
    cpu_l, _, p = solver.factors()
    p = np.asarray(p, dtype=np.int32)
    pinv = _pinv(p)

    aperm_p, aperm_i, aperm_x, orig_to_perm = _symperm_upper(a_upper, pinv)
    perm_to_orig = np.empty_like(orig_to_perm)
    perm_to_orig[orig_to_perm] = np.arange(orig_to_perm.size, dtype=np.int32)

    lnz, etree, nnz_l = _qdldl_etree(n, aperm_p, aperm_i)
    program = _build_numeric_program(n, aperm_p, aperm_i, lnz, etree)

    if nnz_l != int(cpu_l.indptr[-1]):
        raise AssertionError("symbolic L nonzero count differs from qdldl")
    if not np.array_equal(program["lp"], np.asarray(cpu_l.indptr, dtype=np.int32)):
        raise AssertionError("symbolic L column pointers differ from qdldl")
    if not np.array_equal(program["li"], np.asarray(cpu_l.indices, dtype=np.int32)):
        raise AssertionError("symbolic L row indices differ from qdldl")

    return QDLDLPlan(
        n=n,
        nnz_a=int(a_upper.indptr[-1]),
        nnz_l=nnz_l,
        a_indptr=np.asarray(a_upper.indptr, dtype=np.int32),
        a_indices=np.asarray(a_upper.indices, dtype=np.int32),
        a_data=np.asarray(a_upper.data),
        aperm_indptr=aperm_p,
        aperm_indices=aperm_i,
        perm_to_orig=perm_to_orig,
        orig_to_perm=orig_to_perm,
        p=p,
        pinv=pinv,
        etree=etree,
        lnz=lnz,
        **program,
    )


def compile_qdldl(plan: QDLDLPlan, dtype: np.dtype | str = np.float64) -> CompiledQDLDL:
    """Compile batched factorization and solve kernels for ``plan``."""

    return compile_qdldl_variant(plan, dtype=dtype)


def _segment_cost(counts: np.ndarray, start: int, stop: int) -> int:
    if stop <= start:
        return 0
    return int(stop - start) * int(np.max(counts[start:stop], initial=0))


def _best_segment_split(counts: np.ndarray, start: int, stop: int) -> tuple[int | None, int]:
    """Return the best split point and saved padded work for one segment."""

    length = stop - start
    if length <= 1:
        return None, 0
    local = counts[start:stop]
    parent_cost = length * int(np.max(local, initial=0))
    left_max = np.maximum.accumulate(local[:-1])
    right_max = np.maximum.accumulate(local[:0:-1])[::-1]
    split_offsets = np.arange(1, length, dtype=np.int64)
    left_cost = split_offsets * left_max
    right_cost = (length - split_offsets) * right_max
    savings = parent_cost - left_cost - right_cost
    best_index = int(np.argmax(savings))
    best_saving = int(savings[best_index])
    if best_saving <= 0:
        return None, 0
    return start + best_index + 1, best_saving


def _make_fixed_segments(mask: np.ndarray, segment_budget: int) -> list[tuple[int, np.ndarray]]:
    """Split columns into ``segment_budget`` nearly equal fixed-width segments."""

    n = int(mask.shape[0])
    if n == 0:
        return []
    if segment_budget <= 0:
        raise ValueError("segment_budget must be positive")
    target = min(int(segment_budget), n)
    edges = np.linspace(0, n, target + 1, dtype=np.int64)
    segments: list[tuple[int, np.ndarray]] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        segments.append((int(start), np.arange(start, stop, dtype=np.int32)))
    return segments


def _make_adaptive_segments(mask: np.ndarray, segment_budget: int) -> list[tuple[int, np.ndarray]]:
    """Greedily split contiguous columns to reduce padded masked work.

    Each candidate segment ``[i, j)`` costs ``(j - i) * max(counts[i:j])``.
    The splitter repeatedly applies the split that saves the most padded work
    until the budget is exhausted or no split gives a positive saving.
    """

    n = int(mask.shape[0])
    if n == 0:
        return []
    if segment_budget <= 0:
        raise ValueError("segment_budget must be positive")
    counts = np.asarray(np.sum(mask, axis=1), dtype=np.int64)
    intervals: list[tuple[int, int]] = [(0, n)]
    target = min(int(segment_budget), n)
    while len(intervals) < target:
        best_i = -1
        best_split: int | None = None
        best_saving = 0
        for i, (start, stop) in enumerate(intervals):
            split, saving = _best_segment_split(counts, start, stop)
            if saving > best_saving:
                best_i = i
                best_split = split
                best_saving = saving
        if best_split is None or best_saving <= 0:
            break
        start, stop = intervals[best_i]
        intervals[best_i : best_i + 1] = [(start, best_split), (best_split, stop)]
    return [
        (start, np.arange(start, stop, dtype=np.int32))
        for start, stop in intervals
    ]


def _make_optimal_segments(mask: np.ndarray, segment_budget: int) -> list[tuple[int, np.ndarray]]:
    """Globally optimal contiguous segmentation for a padded-work budget.

    This solves the exact dynamic program

    ``dp[k, j] = min_i dp[k - 1, i] + (j - i) * max(counts[i:j])``

    for ``k = min(segment_budget, n)`` segments.  The implementation avoids the
    naive ``O(k*n^2)`` loop by maintaining monotone groups of starts with the
    same current segment maximum for each fixed end column.  For each group it
    stores ``min_i dp_prev[i] - i * group_max``, so querying all starts for a
    given end only scans the small set of backward record maxima.
    """

    n = int(mask.shape[0])
    if n == 0:
        return []
    if segment_budget <= 0:
        raise ValueError("segment_budget must be positive")
    target = min(int(segment_budget), n)
    counts = np.asarray(np.sum(mask, axis=1), dtype=np.int64)
    indices = np.arange(n, dtype=np.float64)
    inf = np.inf
    prev = np.full(n + 1, inf, dtype=np.float64)
    prev[0] = 0.0
    backptr = np.full((target + 1, n + 1), -1, dtype=np.int32)

    for k in range(1, target + 1):
        curr = np.full(n + 1, inf, dtype=np.float64)
        groups: list[tuple[int, int, int, float, int]] = []
        for t, count in enumerate(counts):
            max_value = int(count)
            merge_start = t
            while groups and groups[-1][2] <= max_value:
                merge_start = groups.pop()[0]

            values = prev[merge_start : t + 1] - indices[merge_start : t + 1] * max_value
            local_arg = int(np.argmin(values))
            best_value = float(values[local_arg])
            best_start = merge_start + local_arg
            groups.append((merge_start, t + 1, max_value, best_value, best_start))

            end = t + 1
            if end < k:
                continue

            best_cost = inf
            best_split = -1
            for _, _, group_max, group_best, group_split in groups:
                candidate = group_best + end * group_max
                if candidate < best_cost:
                    best_cost = candidate
                    best_split = group_split
            curr[end] = best_cost
            backptr[k, end] = best_split
        prev = curr

    intervals: list[tuple[int, int]] = []
    end = n
    for k in range(target, 0, -1):
        start = int(backptr[k, end])
        if start < 0:
            raise AssertionError("optimal segmentation backtracking failed")
        intervals.append((start, end))
        end = start
    if end != 0:
        raise AssertionError("optimal segmentation did not cover all columns")
    intervals.reverse()
    return [
        (start, np.arange(start, stop, dtype=np.int32))
        for start, stop in intervals
    ]


def _make_segments(
    mask: np.ndarray,
    *,
    segment_budget: int,
    segment_strategy: str = "optimal",
) -> list[tuple[int, np.ndarray]]:
    if segment_strategy == "fixed":
        return _make_fixed_segments(mask, segment_budget)
    if segment_strategy == "greedy":
        return _make_adaptive_segments(mask, segment_budget)
    if segment_strategy == "optimal":
        return _make_optimal_segments(mask, segment_budget)
    raise ValueError(f"unknown segment_strategy {segment_strategy!r}")


def _make_solve_levels(
    col_rows: np.ndarray,
    col_mask: np.ndarray,
) -> list[np.ndarray]:
    """Level sets for parallel triangular solves.

    For the forward solve, an edge ``col -> row`` means ``row`` depends on
    ``col``.  Columns with the same level have no dependencies between each
    other and can scatter their updates together.  The transpose solve reuses
    the same levels in reverse order.
    """

    n = int(col_rows.shape[0])
    levels = np.zeros(n, dtype=np.int32)
    for col in range(n):
        active_rows = col_rows[col, np.asarray(col_mask[col], dtype=bool)]
        for row in active_rows:
            row_i = int(row)
            levels[row_i] = max(levels[row_i], levels[col] + 1)
    return [
        np.flatnonzero(levels == level).astype(np.int32)
        for level in range(int(np.max(levels, initial=0)) + 1)
    ]


def _normalize_backend_name(name: str, *, role: str) -> str:
    if name not in {"jax", "warp"}:
        raise ValueError(f"unknown QDLDL {role} backend {name!r}")
    return name


def _mixed_backend_variant(
    *,
    factor_backend: str,
    solve_backend: str,
    transpose_work: bool,
    segmented: bool,
    segment_budget: int,
    segment_strategy: str,
    level_scheduled_solve: bool,
    level_scheduled_solve_threshold: int,
) -> str:
    variant_parts = [f"factor-{factor_backend}", f"solve-{solve_backend}"]
    if transpose_work:
        variant_parts.append("transpose")
    if segmented:
        variant_parts.append("segmented")
        variant_parts.append(f"budget{segment_budget}")
        variant_parts.append(segment_strategy)
    if level_scheduled_solve:
        variant_parts.append("levelsolve")
        if level_scheduled_solve_threshold != 1:
            variant_parts.append(f"threshold{level_scheduled_solve_threshold}")
    return "+".join(variant_parts)


def compile_qdldl_variant(
    plan: QDLDLPlan,
    dtype: np.dtype | str = np.float64,
    *,
    backend: str = "jax",
    factor_backend: str | None = None,
    solve_backend: str | None = None,
    transpose_work: bool = False,
    segmented: bool = False,
    segment_budget: int = 64,
    segment_strategy: str = "optimal",
    level_scheduled_solve: bool = False,
    level_scheduled_solve_threshold: int = 1,
) -> CompiledQDLDL:
    """Compile a fixed-pattern QDLDL variant.

    ``transpose_work`` uses symbolic-major dense arrays: ``(n, batch)`` and
    ``(nnz_L, batch)``.  ``segmented`` splits scans into contiguous column
    blocks so each block has narrower padding.  ``level_scheduled_solve``
    replaces sequential triangular-solve column scans with precomputed
    dependency levels.  Warp levelsolve fuses consecutive levels up to
    ``level_scheduled_solve_threshold`` columns into serial runs.
    """

    dtype = np.dtype(dtype)
    _require_x64_for_float64(dtype)
    factor_backend = backend if factor_backend is None else factor_backend
    solve_backend = backend if solve_backend is None else solve_backend
    factor_backend = _normalize_backend_name(factor_backend, role="factor")
    solve_backend = _normalize_backend_name(solve_backend, role="solve")

    if factor_backend != solve_backend:
        if "warp" in {factor_backend, solve_backend} and not transpose_work:
            raise ValueError("mixed Warp/JAX QDLDL backends require transpose_work=True")

        factor_level_scheduled_solve = (
            level_scheduled_solve
            if factor_backend == solve_backend or factor_backend != "warp"
            else False
        )
        factor_compiled = compile_qdldl_variant(
            plan,
            dtype=dtype,
            backend=factor_backend,
            transpose_work=transpose_work,
            segmented=segmented,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
            level_scheduled_solve=factor_level_scheduled_solve,
            level_scheduled_solve_threshold=level_scheduled_solve_threshold,
        )
        solve_compiled = compile_qdldl_variant(
            plan,
            dtype=dtype,
            backend=solve_backend,
            transpose_work=transpose_work,
            segmented=segmented,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
            level_scheduled_solve=level_scheduled_solve,
            level_scheduled_solve_threshold=level_scheduled_solve_threshold,
        )

        @jax.jit
        def factor_and_solve(
            a_values: jax.Array, rhs: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            lx, d, dinv = factor_compiled.factor(a_values)
            x = solve_compiled.solve(lx, dinv, rhs)
            return x, lx, d

        return CompiledQDLDL(
            plan=plan,
            dtype=np.dtype(dtype),
            factor=factor_compiled.factor,
            solve=solve_compiled.solve,
            factor_and_solve=factor_and_solve,
            variant=_mixed_backend_variant(
                factor_backend=factor_backend,
                solve_backend=solve_backend,
                transpose_work=transpose_work,
                segmented=segmented,
                segment_budget=segment_budget,
                segment_strategy=segment_strategy,
                level_scheduled_solve=level_scheduled_solve,
                level_scheduled_solve_threshold=level_scheduled_solve_threshold,
            ),
            values_layout=factor_compiled.values_layout,
            rhs_layout=solve_compiled.rhs_layout,
            backend=f"{factor_backend}/{solve_backend}",
            factor_backend=factor_backend,
            solve_backend=solve_backend,
        )

    if factor_backend == "warp":
        from .warp_backend import compile_qdldl_warp

        return compile_qdldl_warp(
            plan,
            dtype=dtype,
            transpose_work=transpose_work,
            segmented=segmented,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
            level_scheduled_solve=level_scheduled_solve,
            level_scheduled_solve_threshold=level_scheduled_solve_threshold,
        )
    if factor_backend != "jax":
        raise ValueError(f"unknown QDLDL backend {factor_backend!r}")

    jdtype = jnp.dtype(dtype)

    perm_to_orig = jnp.asarray(plan.perm_to_orig, dtype=jnp.int32)
    p = jnp.asarray(plan.p, dtype=jnp.int32)
    pinv = jnp.asarray(plan.pinv, dtype=jnp.int32)
    diag_pos = jnp.asarray(plan.diag_pos, dtype=jnp.int32)
    init_rows = jnp.asarray(plan.init_rows, dtype=jnp.int32)
    init_pos = jnp.asarray(plan.init_pos, dtype=jnp.int32)
    init_mask = jnp.asarray(plan.init_mask)
    process_cols = jnp.asarray(plan.process_cols, dtype=jnp.int32)
    process_lidx = jnp.asarray(plan.process_lidx, dtype=jnp.int32)
    process_offsets = jnp.asarray(plan.process_offsets, dtype=jnp.int32)
    process_mask = jnp.asarray(plan.process_mask)
    col_rows = jnp.asarray(plan.col_rows, dtype=jnp.int32)
    col_lidx = jnp.asarray(plan.col_lidx, dtype=jnp.int32)
    col_mask = jnp.asarray(plan.col_mask)
    col_pos = jnp.arange(plan.max_col_nnz, dtype=jnp.int32)
    row_pos = jnp.arange(plan.max_row_nnz, dtype=jnp.int32)
    n = plan.n
    nnz_l = plan.nnz_l
    if segmented:
        process_segments = _make_segments(
            plan.process_mask,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
        )
        col_segments = _make_segments(
            plan.col_mask,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
        )
    else:
        process_segments = ()
        col_segments = ()
    solve_levels = _make_solve_levels(plan.col_rows, plan.col_mask)
    variant_parts = []
    if transpose_work:
        variant_parts.append("transpose")
    if segmented:
        variant_parts.append("segmented")
    if segmented:
        variant_parts.append(f"budget{segment_budget}")
    if segmented:
        variant_parts.append(segment_strategy)
    if level_scheduled_solve:
        variant_parts.append("levelsolve")
    variant = "+".join(variant_parts) if variant_parts else "baseline"
    values_layout = "original_symbolic" if transpose_work else "original_batch"
    rhs_layout = "symbolic" if transpose_work else "batch"

    @jax.jit
    def factor_batch(a_values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        a_values = jnp.asarray(a_values, dtype=jdtype)
        a_perm = a_values[:, perm_to_orig]
        batch_size = a_values.shape[0]
        y0 = jnp.zeros((batch_size, n), dtype=jdtype)
        lx0 = jnp.zeros((batch_size, nnz_l), dtype=jdtype)
        d0 = jnp.zeros((batch_size, n), dtype=jdtype)
        dinv0 = jnp.zeros((batch_size, n), dtype=jdtype)

        def factor_row(carry, k, row_positions):
            y, lx, d, dinv = carry

            d_k = a_perm[:, diag_pos[k]]
            init_vals = a_perm[:, init_pos[k]]
            init_scale = init_mask[k].astype(jdtype)
            y = y.at[:, init_rows[k]].add(init_vals * init_scale[None, :])

            def process_entry(inner_carry, r):
                y_inner, lx_inner, d_inner = inner_carry
                active = process_mask[k, r]
                cidx = process_cols[k, r]
                dest = process_lidx[k, r]
                offset = process_offsets[k, r]
                val = y_inner[:, cidx]

                use_col = (col_pos < offset) & col_mask[cidx] & active
                update = (
                    -lx_inner[:, col_lidx[cidx]]
                    * val[:, None]
                    * use_col.astype(jdtype)[None, :]
                )
                y_inner = y_inner.at[:, col_rows[cidx]].add(update)

                lval = val * dinv[:, cidx]
                old_lval = lx_inner[:, dest]
                lx_inner = lx_inner.at[:, dest].set(
                    jnp.where(active, lval, old_lval)
                )
                d_inner = d_inner - jnp.where(active, val * lval, 0.0)
                return (y_inner, lx_inner, d_inner), None

            (y, lx, d_k), _ = lax.scan(process_entry, (y, lx, d_k), row_positions)
            d = d.at[:, k].set(d_k)
            dinv = dinv.at[:, k].set(1.0 / d_k)
            y = y.at[:, process_cols[k]].set(0.0)
            return (y, lx, d, dinv), None

        if segmented:
            carry = (y0, lx0, d0, dinv0)
            for _, k_np in process_segments:
                width = int(np.max(np.sum(plan.process_mask[k_np], axis=1), initial=0))
                local_row_pos = jnp.arange(width, dtype=jnp.int32)

                def scan_row(carry_inner, k):
                    return factor_row(carry_inner, k, local_row_pos)

                carry, _ = lax.scan(scan_row, carry, jnp.asarray(k_np, dtype=jnp.int32))
            _, lx, d, dinv = carry
        else:
            def scan_row(carry, k):
                return factor_row(carry, k, row_pos)

            (_, lx, d, dinv), _ = lax.scan(
                scan_row, (y0, lx0, d0, dinv0), jnp.arange(n, dtype=jnp.int32)
            )
        return lx, d, dinv

    @jax.jit
    def solve_batch_layout(lx: jax.Array, dinv: jax.Array, rhs: jax.Array) -> jax.Array:
        lx = jnp.asarray(lx, dtype=jdtype)
        dinv = jnp.asarray(dinv, dtype=jdtype)
        rhs = jnp.asarray(rhs, dtype=jdtype)
        x0 = rhs[:, p]

        def lsolve_col(x, col, col_positions):
            val = x[:, col]
            update = (
                -lx[:, col_lidx[col, col_positions]]
                * val[:, None]
                * col_mask[col, col_positions].astype(jdtype)[None, :]
            )
            x = x.at[:, col_rows[col, col_positions]].add(update)
            return x, None

        if level_scheduled_solve:
            x = x0
            for cols_np in solve_levels:
                cols = jnp.asarray(cols_np, dtype=jnp.int32)

                def lsolve_level(x_inner):
                    rows = col_rows[cols]
                    lidx = col_lidx[cols]
                    mask = col_mask[cols].astype(jdtype)
                    val = x_inner[:, cols]
                    update = -lx[:, lidx] * val[:, :, None] * mask[None, :, :]
                    return x_inner.at[:, rows.reshape(-1)].add(
                        update.reshape((x_inner.shape[0], -1))
                    )

                x = lsolve_level(x)
        elif segmented:
            x = x0
            for _, col_np in col_segments:
                width = int(np.max(np.sum(plan.col_mask[col_np], axis=1), initial=0))
                local_col_pos = jnp.arange(width, dtype=jnp.int32)

                def scan_col(x_inner, col):
                    return lsolve_col(x_inner, col, local_col_pos)

                x, _ = lax.scan(scan_col, x, jnp.asarray(col_np, dtype=jnp.int32))
        else:
            def scan_col(x, col):
                return lsolve_col(x, col, col_pos)

            x, _ = lax.scan(scan_col, x0, jnp.arange(n, dtype=jnp.int32))
        x = x * dinv

        def ltsolve_col(x, offset, col_positions):
            col = n - 1 - offset
            correction = jnp.sum(
                lx[:, col_lidx[col, col_positions]]
                * x[:, col_rows[col, col_positions]]
                * col_mask[col, col_positions].astype(jdtype)[None, :],
                axis=1,
            )
            x = x.at[:, col].set(x[:, col] - correction)
            return x, None

        if level_scheduled_solve:
            for cols_np in reversed(solve_levels):
                cols = jnp.asarray(cols_np, dtype=jnp.int32)

                def ltsolve_level(x_inner):
                    rows = col_rows[cols]
                    lidx = col_lidx[cols]
                    mask = col_mask[cols].astype(jdtype)
                    correction = jnp.sum(
                        lx[:, lidx]
                        * x_inner[:, rows]
                        * mask[None, :, :],
                        axis=2,
                    )
                    return x_inner.at[:, cols].add(-correction)

                x = ltsolve_level(x)
        elif segmented:
            for _, col_np in reversed(col_segments):
                width = int(np.max(np.sum(plan.col_mask[col_np], axis=1), initial=0))
                local_col_pos = jnp.arange(width, dtype=jnp.int32)
                offsets_np = (n - 1 - col_np[::-1]).astype(np.int32)

                def scan_lt(x_inner, offset):
                    return ltsolve_col(x_inner, offset, local_col_pos)

                x, _ = lax.scan(scan_lt, x, jnp.asarray(offsets_np, dtype=jnp.int32))
        else:
            def scan_lt(x, offset):
                return ltsolve_col(x, offset, col_pos)

            x, _ = lax.scan(scan_lt, x, jnp.arange(n, dtype=jnp.int32))
        return x[:, pinv]

    @jax.jit
    def factor_symbolic(a_values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        a_values = jnp.asarray(a_values, dtype=jdtype)
        a_perm = a_values[perm_to_orig, :]
        batch_size = a_values.shape[1]
        y0 = jnp.zeros((n, batch_size), dtype=jdtype)
        lx0 = jnp.zeros((nnz_l, batch_size), dtype=jdtype)
        d0 = jnp.zeros((n, batch_size), dtype=jdtype)
        dinv0 = jnp.zeros((n, batch_size), dtype=jdtype)

        def factor_row(carry, k, row_positions):
            y, lx, d, dinv = carry

            d_k = a_perm[diag_pos[k], :]
            init_vals = a_perm[init_pos[k], :]
            init_scale = init_mask[k].astype(jdtype)
            y = y.at[init_rows[k], :].add(init_vals * init_scale[:, None])

            def process_entry(inner_carry, r):
                y_inner, lx_inner, d_inner = inner_carry
                active = process_mask[k, r]
                cidx = process_cols[k, r]
                dest = process_lidx[k, r]
                offset = process_offsets[k, r]
                val = y_inner[cidx, :]

                use_col = (col_pos < offset) & col_mask[cidx] & active
                update = (
                    -lx_inner[col_lidx[cidx], :]
                    * val[None, :]
                    * use_col.astype(jdtype)[:, None]
                )
                y_inner = y_inner.at[col_rows[cidx], :].add(update)

                lval = val * dinv[cidx, :]
                old_lval = lx_inner[dest, :]
                lx_inner = lx_inner.at[dest, :].set(
                    jnp.where(active, lval, old_lval)
                )
                d_inner = d_inner - jnp.where(active, val * lval, 0.0)
                return (y_inner, lx_inner, d_inner), None

            (y, lx, d_k), _ = lax.scan(process_entry, (y, lx, d_k), row_positions)
            d = d.at[k, :].set(d_k)
            dinv = dinv.at[k, :].set(1.0 / d_k)
            y = y.at[process_cols[k], :].set(0.0)
            return (y, lx, d, dinv), None

        if segmented:
            carry = (y0, lx0, d0, dinv0)
            for _, k_np in process_segments:
                width = int(np.max(np.sum(plan.process_mask[k_np], axis=1), initial=0))
                local_row_pos = jnp.arange(width, dtype=jnp.int32)

                def scan_row(carry_inner, k):
                    return factor_row(carry_inner, k, local_row_pos)

                carry, _ = lax.scan(scan_row, carry, jnp.asarray(k_np, dtype=jnp.int32))
            _, lx, d, dinv = carry
        else:
            def scan_row(carry, k):
                return factor_row(carry, k, row_pos)

            (_, lx, d, dinv), _ = lax.scan(
                scan_row, (y0, lx0, d0, dinv0), jnp.arange(n, dtype=jnp.int32)
            )
        return lx, d, dinv

    @jax.jit
    def solve_symbolic(lx: jax.Array, dinv: jax.Array, rhs: jax.Array) -> jax.Array:
        lx = jnp.asarray(lx, dtype=jdtype)
        dinv = jnp.asarray(dinv, dtype=jdtype)
        rhs = jnp.asarray(rhs, dtype=jdtype)
        x0 = rhs[p, :]

        def lsolve_col(x, col, col_positions):
            val = x[col, :]
            update = (
                -lx[col_lidx[col, col_positions], :]
                * val[None, :]
                * col_mask[col, col_positions].astype(jdtype)[:, None]
            )
            x = x.at[col_rows[col, col_positions], :].add(update)
            return x, None

        if level_scheduled_solve:
            x = x0
            for cols_np in solve_levels:
                cols = jnp.asarray(cols_np, dtype=jnp.int32)

                def lsolve_level(x_inner):
                    rows = col_rows[cols]
                    lidx = col_lidx[cols]
                    mask = col_mask[cols].astype(jdtype)
                    val = x_inner[cols, :]
                    update = -lx[lidx, :] * val[:, None, :] * mask[:, :, None]
                    return x_inner.at[rows.reshape(-1), :].add(
                        update.reshape((-1, x_inner.shape[1]))
                    )

                x = lsolve_level(x)
        elif segmented:
            x = x0
            for _, col_np in col_segments:
                width = int(np.max(np.sum(plan.col_mask[col_np], axis=1), initial=0))
                local_col_pos = jnp.arange(width, dtype=jnp.int32)

                def scan_col(x_inner, col):
                    return lsolve_col(x_inner, col, local_col_pos)

                x, _ = lax.scan(scan_col, x, jnp.asarray(col_np, dtype=jnp.int32))
        else:
            def scan_col(x, col):
                return lsolve_col(x, col, col_pos)

            x, _ = lax.scan(scan_col, x0, jnp.arange(n, dtype=jnp.int32))
        x = x * dinv

        def ltsolve_col(x, offset, col_positions):
            col = n - 1 - offset
            correction = jnp.sum(
                lx[col_lidx[col, col_positions], :]
                * x[col_rows[col, col_positions], :]
                * col_mask[col, col_positions].astype(jdtype)[:, None],
                axis=0,
            )
            x = x.at[col, :].set(x[col, :] - correction)
            return x, None

        if level_scheduled_solve:
            for cols_np in reversed(solve_levels):
                cols = jnp.asarray(cols_np, dtype=jnp.int32)

                def ltsolve_level(x_inner):
                    rows = col_rows[cols]
                    lidx = col_lidx[cols]
                    mask = col_mask[cols].astype(jdtype)
                    correction = jnp.sum(
                        lx[lidx, :]
                        * x_inner[rows, :]
                        * mask[:, :, None],
                        axis=1,
                    )
                    return x_inner.at[cols, :].add(-correction)

                x = ltsolve_level(x)
        elif segmented:
            for _, col_np in reversed(col_segments):
                width = int(np.max(np.sum(plan.col_mask[col_np], axis=1), initial=0))
                local_col_pos = jnp.arange(width, dtype=jnp.int32)
                offsets_np = (n - 1 - col_np[::-1]).astype(np.int32)

                def scan_lt(x_inner, offset):
                    return ltsolve_col(x_inner, offset, local_col_pos)

                x, _ = lax.scan(scan_lt, x, jnp.asarray(offsets_np, dtype=jnp.int32))
        else:
            def scan_lt(x, offset):
                return ltsolve_col(x, offset, col_pos)

            x, _ = lax.scan(scan_lt, x, jnp.arange(n, dtype=jnp.int32))
        return x[pinv, :]

    factor = factor_symbolic if transpose_work else factor_batch
    solve = solve_symbolic if transpose_work else solve_batch_layout

    @jax.jit
    def factor_and_solve(a_values: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        lx, d, dinv = factor(a_values)
        x = solve(lx, dinv, rhs)
        return x, lx, d

    return CompiledQDLDL(
        plan=plan,
        dtype=dtype,
        factor=factor,
        solve=solve,
        factor_and_solve=factor_and_solve,
        variant=variant,
        values_layout=values_layout,
        rhs_layout=rhs_layout,
        backend="jax",
        factor_backend="jax",
        solve_backend="jax",
    )


def verify_against_qdldl(
    plan: QDLDLPlan,
    compiled: CompiledQDLDL | None = None,
    a_values: np.ndarray | None = None,
    rhs: np.ndarray | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> dict[str, float]:
    """Check JAX factors and solve against the qdldl Python binding."""

    if compiled is None:
        compiled = compile_qdldl(plan, dtype=np.float64)

    if a_values is None:
        a_values = plan.a_data
    a_values = np.asarray(a_values, dtype=compiled.dtype)
    if a_values.ndim == 1:
        a_values = a_values[None, :]

    if rhs is None:
        rng = np.random.default_rng(0)
        rhs = rng.standard_normal((a_values.shape[0], plan.n)).astype(compiled.dtype)
    rhs = np.asarray(rhs, dtype=compiled.dtype)
    if rhs.ndim == 1:
        rhs = rhs[None, :]

    x_jax, lx_jax, d_jax = compiled.factor_and_solve(
        jnp.asarray(a_values), jnp.asarray(rhs)
    )
    x_jax, lx_jax, d_jax = jax.device_get((x_jax, lx_jax, d_jax))

    base = sp.csc_matrix(
        (np.asarray(a_values[0]), plan.a_indices, plan.a_indptr),
        shape=(plan.n, plan.n),
    )
    solver = qdldl.Solver(base, upper=True)

    max_lx = 0.0
    max_d = 0.0
    max_x = 0.0

    for b in range(a_values.shape[0]):
        mat = sp.csc_matrix(
            (np.asarray(a_values[b]), plan.a_indices, plan.a_indptr),
            shape=(plan.n, plan.n),
        )
        if b == 0:
            cpu_l, cpu_d, cpu_p = solver.factors()
        else:
            solver.update(mat, upper=True)
            cpu_l, cpu_d, cpu_p = solver.factors()

        if not np.array_equal(np.asarray(cpu_p, dtype=np.int32), plan.p):
            raise AssertionError("AMD permutation changed unexpectedly")
        if not np.array_equal(np.asarray(cpu_l.indptr, dtype=np.int32), plan.lp):
            raise AssertionError("L column pointers do not match qdldl")
        if not np.array_equal(np.asarray(cpu_l.indices, dtype=np.int32), plan.li):
            raise AssertionError("L row indices do not match qdldl")

        cpu_x = solver.solve(np.asarray(rhs[b], dtype=compiled.dtype))
        max_lx = max(max_lx, float(np.max(np.abs(cpu_l.data - lx_jax[b]))))
        max_d = max(max_d, float(np.max(np.abs(cpu_d - d_jax[b]))))
        max_x = max(max_x, float(np.max(np.abs(cpu_x - x_jax[b]))))

        np.testing.assert_allclose(lx_jax[b], cpu_l.data, rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_jax[b], cpu_d, rtol=rtol, atol=atol)
        np.testing.assert_allclose(x_jax[b], cpu_x, rtol=rtol, atol=atol)

    return {"max_abs_lx": max_lx, "max_abs_d": max_d, "max_abs_x": max_x}
