"""Warp backend for fixed-pattern QDLDL factorization and triangular solves."""

from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from .core import _make_segments, _make_solve_levels
from .types import CompiledQDLDL, QDLDLPlan


def _require_x64_for_float64(dtype: np.dtype) -> None:
    if dtype == np.dtype(np.float64) and not jax.config.jax_enable_x64:
        raise ValueError(
            "Warp QDLDL float64 compilation requires jax_enable_x64=True. "
            "Set it in the benchmark or example entry point before compiling."
        )


try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - exercised only without Warp.
    wp = None
    _jax_callable = None
    _WARP_IMPORT_ERROR = exc
else:
    try:
        _jax_callable = wp.jax_callable
    except AttributeError:
        from warp.jax_experimental import jax_callable as _jax_callable

    _WARP_IMPORT_ERROR = None


if wp is not None:

    @wp.kernel
    def _qdldl_factor_transpose_kernel(
        a_values: wp.array2d[Any],
        perm_to_orig: wp.array(dtype=wp.int32),
        diag_pos: wp.array(dtype=wp.int32),
        init_rows: wp.array2d(dtype=wp.int32),
        init_pos: wp.array2d(dtype=wp.int32),
        init_mask: wp.array2d(dtype=wp.int32),
        process_cols: wp.array2d(dtype=wp.int32),
        process_lidx: wp.array2d(dtype=wp.int32),
        process_offsets: wp.array2d(dtype=wp.int32),
        process_mask: wp.array2d(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        lx: wp.array2d[Any],
        d: wp.array2d[Any],
        dinv: wp.array2d[Any],
        y: wp.array2d[Any],
    ):
        batch = wp.tid()
        n = d.shape[0]
        max_init = init_rows.shape[1]
        max_process = process_cols.shape[1]
        max_col = col_rows.shape[1]

        for row in range(n):
            y[row, batch] = y.dtype(0.0)

        for k in range(n):
            d_k = a_values[perm_to_orig[diag_pos[k]], batch]

            for r in range(max_init):
                if init_mask[k, r] != 0:
                    row = init_rows[k, r]
                    pos = init_pos[k, r]
                    y[row, batch] += a_values[perm_to_orig[pos], batch]

            for r in range(max_process):
                if process_mask[k, r] != 0:
                    cidx = process_cols[k, r]
                    dest = process_lidx[k, r]
                    offset = process_offsets[k, r]
                    val = y[cidx, batch]

                    for col_pos in range(max_col):
                        if col_pos < offset and col_mask[cidx, col_pos] != 0:
                            row = col_rows[cidx, col_pos]
                            lidx = col_lidx[cidx, col_pos]
                            y[row, batch] += -lx[lidx, batch] * val

                    lval = val * dinv[cidx, batch]
                    lx[dest, batch] = lval
                    d_k -= val * lval

            d[k, batch] = d_k
            dinv[k, batch] = dinv.dtype(1.0) / d_k

            for r in range(max_process):
                if process_mask[k, r] != 0:
                    y[process_cols[k, r], batch] = y.dtype(0.0)


    @wp.kernel
    def _qdldl_factor_segmented_transpose_kernel(
        a_values: wp.array2d[Any],
        perm_to_orig: wp.array(dtype=wp.int32),
        diag_pos: wp.array(dtype=wp.int32),
        init_rows: wp.array2d(dtype=wp.int32),
        init_pos: wp.array2d(dtype=wp.int32),
        init_mask: wp.array2d(dtype=wp.int32),
        process_cols: wp.array2d(dtype=wp.int32),
        process_lidx: wp.array2d(dtype=wp.int32),
        process_offsets: wp.array2d(dtype=wp.int32),
        process_mask: wp.array2d(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        segment_starts: wp.array(dtype=wp.int32),
        segment_counts: wp.array(dtype=wp.int32),
        segment_widths: wp.array(dtype=wp.int32),
        lx: wp.array2d[Any],
        d: wp.array2d[Any],
        dinv: wp.array2d[Any],
        y: wp.array2d[Any],
    ):
        batch = wp.tid()
        n = d.shape[0]
        max_init = init_rows.shape[1]
        max_col = col_rows.shape[1]
        num_segments = segment_starts.shape[0]

        for row in range(n):
            y[row, batch] = y.dtype(0.0)

        for segment_index in range(num_segments):
            row_start = segment_starts[segment_index]
            row_count = segment_counts[segment_index]
            max_process = segment_widths[segment_index]

            for row_offset in range(row_count):
                k = row_start + row_offset
                d_k = a_values[perm_to_orig[diag_pos[k]], batch]

                for r in range(max_init):
                    if init_mask[k, r] != 0:
                        row = init_rows[k, r]
                        pos = init_pos[k, r]
                        y[row, batch] += a_values[perm_to_orig[pos], batch]

                for r in range(max_process):
                    if process_mask[k, r] != 0:
                        cidx = process_cols[k, r]
                        dest = process_lidx[k, r]
                        offset = process_offsets[k, r]
                        val = y[cidx, batch]

                        for col_pos in range(max_col):
                            if col_pos < offset and col_mask[cidx, col_pos] != 0:
                                row = col_rows[cidx, col_pos]
                                lidx = col_lidx[cidx, col_pos]
                                y[row, batch] += -lx[lidx, batch] * val

                        lval = val * dinv[cidx, batch]
                        lx[dest, batch] = lval
                        d_k -= val * lval

                d[k, batch] = d_k
                dinv[k, batch] = dinv.dtype(1.0) / d_k

                for r in range(max_process):
                    if process_mask[k, r] != 0:
                        y[process_cols[k, r], batch] = y.dtype(0.0)


    @wp.kernel
    def _qdldl_solve_serial_transpose_kernel(
        lx: wp.array2d[Any],
        dinv: wp.array2d[Any],
        rhs: wp.array2d[Any],
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        out: wp.array2d[Any],
        work: wp.array2d[Any],
    ):
        batch = wp.tid()
        n = rhs.shape[0]
        max_col = col_rows.shape[1]

        for i in range(n):
            work[i, batch] = rhs[p[i], batch]

        for col in range(n):
            val = work[col, batch]
            for pos in range(max_col):
                if col_mask[col, pos] != 0:
                    row = col_rows[col, pos]
                    lidx = col_lidx[col, pos]
                    work[row, batch] += -lx[lidx, batch] * val

        for i in range(n):
            work[i, batch] *= dinv[i, batch]

        for offset in range(n):
            col = n - 1 - offset
            correction = work.dtype(0.0)
            for pos in range(max_col):
                if col_mask[col, pos] != 0:
                    row = col_rows[col, pos]
                    lidx = col_lidx[col, pos]
                    correction += lx[lidx, batch] * work[row, batch]
            work[col, batch] -= correction

        for i in range(n):
            out[i, batch] = work[pinv[i], batch]


    @wp.kernel
    def _qdldl_solve_segmented_transpose_kernel(
        lx: wp.array2d[Any],
        dinv: wp.array2d[Any],
        rhs: wp.array2d[Any],
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        segment_starts: wp.array(dtype=wp.int32),
        segment_counts: wp.array(dtype=wp.int32),
        segment_widths: wp.array(dtype=wp.int32),
        out: wp.array2d[Any],
        work: wp.array2d[Any],
    ):
        batch = wp.tid()
        n = rhs.shape[0]
        num_segments = segment_starts.shape[0]

        for i in range(n):
            work[i, batch] = rhs[p[i], batch]

        for segment_index in range(num_segments):
            col_start = segment_starts[segment_index]
            col_count = segment_counts[segment_index]
            max_col = segment_widths[segment_index]

            for col_offset in range(col_count):
                col = col_start + col_offset
                val = work[col, batch]
                for pos in range(max_col):
                    if col_mask[col, pos] != 0:
                        row = col_rows[col, pos]
                        lidx = col_lidx[col, pos]
                        work[row, batch] += -lx[lidx, batch] * val

        for i in range(n):
            work[i, batch] *= dinv[i, batch]

        for reverse_segment_offset in range(num_segments):
            segment_index = num_segments - 1 - reverse_segment_offset
            col_start = segment_starts[segment_index]
            col_count = segment_counts[segment_index]
            max_col = segment_widths[segment_index]

            for col_offset in range(col_count):
                col = col_start + col_count - 1 - col_offset
                correction = work.dtype(0.0)
                for pos in range(max_col):
                    if col_mask[col, pos] != 0:
                        row = col_rows[col, pos]
                        lidx = col_lidx[col, pos]
                        correction += lx[lidx, batch] * work[row, batch]
                work[col, batch] -= correction

        for i in range(n):
            out[i, batch] = work[pinv[i], batch]


    @wp.kernel
    def _qdldl_permute_rhs_transpose_kernel(
        rhs: wp.array2d[Any],
        p: wp.array(dtype=wp.int32),
        work: wp.array2d[Any],
    ):
        row, batch = wp.tid()
        work[row, batch] = rhs[p[row], batch]


    @wp.kernel
    def _qdldl_lsolve_level_transpose_kernel(
        lx: wp.array2d[Any],
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols: wp.array2d(dtype=wp.int32),
        level_mask: wp.array2d(dtype=wp.int32),
        level: int,
        work: wp.array2d[Any],
    ):
        level_pos, batch = wp.tid()
        if level_mask[level, level_pos] != 0:
            col = level_cols[level, level_pos]
            val = work[col, batch]
            max_col = col_rows.shape[1]
            for pos in range(max_col):
                if col_mask[col, pos] != 0:
                    row = col_rows[col, pos]
                    lidx = col_lidx[col, pos]
                    wp.atomic_add(work, row, batch, -lx[lidx, batch] * val)


    @wp.kernel
    def _qdldl_diag_scale_transpose_kernel(
        dinv: wp.array2d[Any],
        work: wp.array2d[Any],
    ):
        row, batch = wp.tid()
        work[row, batch] *= dinv[row, batch]


    @wp.kernel
    def _qdldl_ltsolve_level_transpose_kernel(
        lx: wp.array2d[Any],
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols: wp.array2d(dtype=wp.int32),
        level_mask: wp.array2d(dtype=wp.int32),
        level: int,
        work: wp.array2d[Any],
    ):
        level_pos, batch = wp.tid()
        if level_mask[level, level_pos] != 0:
            col = level_cols[level, level_pos]
            max_col = col_rows.shape[1]
            correction = work.dtype(0.0)
            for pos in range(max_col):
                if col_mask[col, pos] != 0:
                    row = col_rows[col, pos]
                    lidx = col_lidx[col, pos]
                    correction += lx[lidx, batch] * work[row, batch]
            work[col, batch] -= correction


    @wp.kernel
    def _qdldl_lsolve_level_ragged_transpose_kernel(
        lx: wp.array2d[Any],
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols_flat: wp.array(dtype=wp.int32),
        level_start: int,
        work: wp.array2d[Any],
    ):
        level_pos, batch = wp.tid()
        col = level_cols_flat[level_start + level_pos]
        val = work[col, batch]
        max_col = col_rows.shape[1]
        for pos in range(max_col):
            if col_mask[col, pos] != 0:
                row = col_rows[col, pos]
                lidx = col_lidx[col, pos]
                wp.atomic_add(work, row, batch, -lx[lidx, batch] * val)


    @wp.kernel
    def _qdldl_ltsolve_level_ragged_transpose_kernel(
        lx: wp.array2d[Any],
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols_flat: wp.array(dtype=wp.int32),
        level_start: int,
        work: wp.array2d[Any],
    ):
        level_pos, batch = wp.tid()
        col = level_cols_flat[level_start + level_pos]
        max_col = col_rows.shape[1]
        correction = work.dtype(0.0)
        for pos in range(max_col):
            if col_mask[col, pos] != 0:
                row = col_rows[col, pos]
                lidx = col_lidx[col, pos]
                correction += lx[lidx, batch] * work[row, batch]
        work[col, batch] -= correction


    @wp.kernel
    def _qdldl_lsolve_serial_levels_transpose_kernel(
        lx: wp.array2d[Any],
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols_flat: wp.array(dtype=wp.int32),
        level_offsets: wp.array(dtype=wp.int32),
        start_level: int,
        stop_level: int,
        work: wp.array2d[Any],
    ):
        batch = wp.tid()
        max_col = col_rows.shape[1]
        for level in range(start_level, stop_level):
            start = level_offsets[level]
            stop = level_offsets[level + 1]
            for flat_pos in range(start, stop):
                col = level_cols_flat[flat_pos]
                val = work[col, batch]
                for pos in range(max_col):
                    if col_mask[col, pos] != 0:
                        row = col_rows[col, pos]
                        lidx = col_lidx[col, pos]
                        work[row, batch] += -lx[lidx, batch] * val


    @wp.kernel
    def _qdldl_ltsolve_serial_levels_transpose_kernel(
        lx: wp.array2d[Any],
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols_flat: wp.array(dtype=wp.int32),
        level_offsets: wp.array(dtype=wp.int32),
        start_level: int,
        stop_level: int,
        work: wp.array2d[Any],
    ):
        batch = wp.tid()
        max_col = col_rows.shape[1]
        for reverse_offset in range(stop_level - start_level):
            level = stop_level - 1 - reverse_offset
            start = level_offsets[level]
            stop = level_offsets[level + 1]
            for flat_offset in range(stop - start):
                flat_pos = stop - 1 - flat_offset
                col = level_cols_flat[flat_pos]
                correction = work.dtype(0.0)
                for pos in range(max_col):
                    if col_mask[col, pos] != 0:
                        row = col_rows[col, pos]
                        lidx = col_lidx[col, pos]
                        correction += lx[lidx, batch] * work[row, batch]
                work[col, batch] -= correction


    @wp.kernel
    def _qdldl_unpermute_solution_transpose_kernel(
        work: wp.array2d[Any],
        pinv: wp.array(dtype=wp.int32),
        out: wp.array2d[Any],
    ):
        row, batch = wp.tid()
        out[row, batch] = work[pinv[row], batch]


    def _register_qdldl_kernel_overloads() -> None:
        int1d = wp.array(dtype=wp.int32)
        int2d = wp.array2d(dtype=wp.int32)
        for real in (wp.float32, wp.float64):
            real2d = wp.array2d(dtype=real)
            wp.overload(
                _qdldl_factor_transpose_kernel,
                [
                    real2d,
                    int1d,
                    int1d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    real2d,
                    real2d,
                    real2d,
                    real2d,
                ],
            )
            wp.overload(
                _qdldl_factor_segmented_transpose_kernel,
                [
                    real2d,
                    int1d,
                    int1d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int2d,
                    int1d,
                    int1d,
                    int1d,
                    real2d,
                    real2d,
                    real2d,
                    real2d,
                ],
            )
            wp.overload(
                _qdldl_solve_serial_transpose_kernel,
                [
                    real2d,
                    real2d,
                    real2d,
                    int1d,
                    int1d,
                    int2d,
                    int2d,
                    int2d,
                    real2d,
                    real2d,
                ],
            )
            wp.overload(
                _qdldl_solve_segmented_transpose_kernel,
                [
                    real2d,
                    real2d,
                    real2d,
                    int1d,
                    int1d,
                    int2d,
                    int2d,
                    int2d,
                    int1d,
                    int1d,
                    int1d,
                    real2d,
                    real2d,
                ],
            )
            wp.overload(
                _qdldl_permute_rhs_transpose_kernel,
                [real2d, int1d, real2d],
            )
            wp.overload(
                _qdldl_lsolve_level_transpose_kernel,
                [real2d, int2d, int2d, int2d, int2d, int2d, int, real2d],
            )
            wp.overload(
                _qdldl_diag_scale_transpose_kernel,
                [real2d, real2d],
            )
            wp.overload(
                _qdldl_ltsolve_level_transpose_kernel,
                [real2d, int2d, int2d, int2d, int2d, int2d, int, real2d],
            )
            wp.overload(
                _qdldl_lsolve_level_ragged_transpose_kernel,
                [real2d, int2d, int2d, int2d, int1d, int, real2d],
            )
            wp.overload(
                _qdldl_ltsolve_level_ragged_transpose_kernel,
                [real2d, int2d, int2d, int2d, int1d, int, real2d],
            )
            wp.overload(
                _qdldl_lsolve_serial_levels_transpose_kernel,
                [real2d, int2d, int2d, int2d, int1d, int1d, int, int, real2d],
            )
            wp.overload(
                _qdldl_ltsolve_serial_levels_transpose_kernel,
                [real2d, int2d, int2d, int2d, int1d, int1d, int, int, real2d],
            )
            wp.overload(
                _qdldl_unpermute_solution_transpose_kernel,
                [real2d, int1d, real2d],
            )


    _register_qdldl_kernel_overloads()


    def _qdldl_factor_transpose_callable(
        a_values: wp.array2d(dtype=wp.float32),
        perm_to_orig: wp.array(dtype=wp.int32),
        diag_pos: wp.array(dtype=wp.int32),
        init_rows: wp.array2d(dtype=wp.int32),
        init_pos: wp.array2d(dtype=wp.int32),
        init_mask: wp.array2d(dtype=wp.int32),
        process_cols: wp.array2d(dtype=wp.int32),
        process_lidx: wp.array2d(dtype=wp.int32),
        process_offsets: wp.array2d(dtype=wp.int32),
        process_mask: wp.array2d(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        lx: wp.array2d(dtype=wp.float32),
        d: wp.array2d(dtype=wp.float32),
        dinv: wp.array2d(dtype=wp.float32),
        y_work: wp.array2d(dtype=wp.float32),
    ):
        wp.launch(
            _qdldl_factor_transpose_kernel,
            dim=a_values.shape[1],
            inputs=[
                a_values,
                perm_to_orig,
                diag_pos,
                init_rows,
                init_pos,
                init_mask,
                process_cols,
                process_lidx,
                process_offsets,
                process_mask,
                col_rows,
                col_lidx,
                col_mask,
            ],
            outputs=[lx, d, dinv, y_work],
        )


    def _qdldl_factor_segmented_transpose_callable(
        a_values: wp.array2d(dtype=wp.float32),
        perm_to_orig: wp.array(dtype=wp.int32),
        diag_pos: wp.array(dtype=wp.int32),
        init_rows: wp.array2d(dtype=wp.int32),
        init_pos: wp.array2d(dtype=wp.int32),
        init_mask: wp.array2d(dtype=wp.int32),
        process_cols: wp.array2d(dtype=wp.int32),
        process_lidx: wp.array2d(dtype=wp.int32),
        process_offsets: wp.array2d(dtype=wp.int32),
        process_mask: wp.array2d(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        segment_starts: wp.array(dtype=wp.int32),
        segment_counts: wp.array(dtype=wp.int32),
        segment_widths: wp.array(dtype=wp.int32),
        lx: wp.array2d(dtype=wp.float32),
        d: wp.array2d(dtype=wp.float32),
        dinv: wp.array2d(dtype=wp.float32),
        y_work: wp.array2d(dtype=wp.float32),
    ):
        batch_size = a_values.shape[1]
        wp.launch(
            _qdldl_factor_segmented_transpose_kernel,
            dim=batch_size,
            inputs=[
                a_values,
                perm_to_orig,
                diag_pos,
                init_rows,
                init_pos,
                init_mask,
                process_cols,
                process_lidx,
                process_offsets,
                process_mask,
                col_rows,
                col_lidx,
                col_mask,
                segment_starts,
                segment_counts,
                segment_widths,
            ],
            outputs=[lx, d, dinv, y_work],
        )


    def _qdldl_solve_serial_transpose_callable(
        lx: wp.array2d(dtype=wp.float32),
        dinv: wp.array2d(dtype=wp.float32),
        rhs: wp.array2d(dtype=wp.float32),
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        out: wp.array2d(dtype=wp.float32),
        work: wp.array2d(dtype=wp.float32),
    ):
        wp.launch(
            _qdldl_solve_serial_transpose_kernel,
            dim=rhs.shape[1],
            inputs=[lx, dinv, rhs, p, pinv, col_rows, col_lidx, col_mask],
            outputs=[out, work],
        )


    def _qdldl_solve_segmented_transpose_callable(
        lx: wp.array2d(dtype=wp.float32),
        dinv: wp.array2d(dtype=wp.float32),
        rhs: wp.array2d(dtype=wp.float32),
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        segment_starts: wp.array(dtype=wp.int32),
        segment_counts: wp.array(dtype=wp.int32),
        segment_widths: wp.array(dtype=wp.int32),
        out: wp.array2d(dtype=wp.float32),
        work: wp.array2d(dtype=wp.float32),
    ):
        wp.launch(
            _qdldl_solve_segmented_transpose_kernel,
            dim=rhs.shape[1],
            inputs=[
                lx,
                dinv,
                rhs,
                p,
                pinv,
                col_rows,
                col_lidx,
                col_mask,
                segment_starts,
                segment_counts,
                segment_widths,
            ],
            outputs=[out, work],
        )


    def _qdldl_solve_level_transpose_callable(
        lx: wp.array2d(dtype=wp.float32),
        dinv: wp.array2d(dtype=wp.float32),
        rhs: wp.array2d(dtype=wp.float32),
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols: wp.array2d(dtype=wp.int32),
        level_mask: wp.array2d(dtype=wp.int32),
        out: wp.array2d(dtype=wp.float32),
        work: wp.array2d(dtype=wp.float32),
    ):
        n = rhs.shape[0]
        batch_size = rhs.shape[1]
        num_levels = level_cols.shape[0]
        max_level_width = level_cols.shape[1]

        wp.launch(
            _qdldl_permute_rhs_transpose_kernel,
            dim=(n, batch_size),
            inputs=[rhs, p],
            outputs=[work],
        )
        for level in range(num_levels):
            wp.launch(
                _qdldl_lsolve_level_transpose_kernel,
                dim=(max_level_width, batch_size),
                inputs=[
                    lx,
                    col_rows,
                    col_lidx,
                    col_mask,
                    level_cols,
                    level_mask,
                    level,
                    work,
                ],
            )
        wp.launch(
            _qdldl_diag_scale_transpose_kernel,
            dim=(n, batch_size),
            inputs=[dinv, work],
        )
        for offset in range(num_levels):
            level = num_levels - 1 - offset
            wp.launch(
                _qdldl_ltsolve_level_transpose_kernel,
                dim=(max_level_width, batch_size),
                inputs=[
                    lx,
                    col_rows,
                    col_lidx,
                    col_mask,
                    level_cols,
                    level_mask,
                    level,
                    work,
                ],
            )
        wp.launch(
            _qdldl_unpermute_solution_transpose_kernel,
            dim=(n, batch_size),
            inputs=[work, pinv],
            outputs=[out],
        )


    def _qdldl_factor_transpose_callable_f64(
        a_values: wp.array2d(dtype=wp.float64),
        perm_to_orig: wp.array(dtype=wp.int32),
        diag_pos: wp.array(dtype=wp.int32),
        init_rows: wp.array2d(dtype=wp.int32),
        init_pos: wp.array2d(dtype=wp.int32),
        init_mask: wp.array2d(dtype=wp.int32),
        process_cols: wp.array2d(dtype=wp.int32),
        process_lidx: wp.array2d(dtype=wp.int32),
        process_offsets: wp.array2d(dtype=wp.int32),
        process_mask: wp.array2d(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        lx: wp.array2d(dtype=wp.float64),
        d: wp.array2d(dtype=wp.float64),
        dinv: wp.array2d(dtype=wp.float64),
        y_work: wp.array2d(dtype=wp.float64),
    ):
        _qdldl_factor_transpose_callable(
            a_values,
            perm_to_orig,
            diag_pos,
            init_rows,
            init_pos,
            init_mask,
            process_cols,
            process_lidx,
            process_offsets,
            process_mask,
            col_rows,
            col_lidx,
            col_mask,
            lx,
            d,
            dinv,
            y_work,
        )


    def _qdldl_factor_segmented_transpose_callable_f64(
        a_values: wp.array2d(dtype=wp.float64),
        perm_to_orig: wp.array(dtype=wp.int32),
        diag_pos: wp.array(dtype=wp.int32),
        init_rows: wp.array2d(dtype=wp.int32),
        init_pos: wp.array2d(dtype=wp.int32),
        init_mask: wp.array2d(dtype=wp.int32),
        process_cols: wp.array2d(dtype=wp.int32),
        process_lidx: wp.array2d(dtype=wp.int32),
        process_offsets: wp.array2d(dtype=wp.int32),
        process_mask: wp.array2d(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        segment_starts: wp.array(dtype=wp.int32),
        segment_counts: wp.array(dtype=wp.int32),
        segment_widths: wp.array(dtype=wp.int32),
        lx: wp.array2d(dtype=wp.float64),
        d: wp.array2d(dtype=wp.float64),
        dinv: wp.array2d(dtype=wp.float64),
        y_work: wp.array2d(dtype=wp.float64),
    ):
        _qdldl_factor_segmented_transpose_callable(
            a_values,
            perm_to_orig,
            diag_pos,
            init_rows,
            init_pos,
            init_mask,
            process_cols,
            process_lidx,
            process_offsets,
            process_mask,
            col_rows,
            col_lidx,
            col_mask,
            segment_starts,
            segment_counts,
            segment_widths,
            lx,
            d,
            dinv,
            y_work,
        )


    def _qdldl_solve_serial_transpose_callable_f64(
        lx: wp.array2d(dtype=wp.float64),
        dinv: wp.array2d(dtype=wp.float64),
        rhs: wp.array2d(dtype=wp.float64),
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        out: wp.array2d(dtype=wp.float64),
        work: wp.array2d(dtype=wp.float64),
    ):
        _qdldl_solve_serial_transpose_callable(
            lx,
            dinv,
            rhs,
            p,
            pinv,
            col_rows,
            col_lidx,
            col_mask,
            out,
            work,
        )


    def _qdldl_solve_segmented_transpose_callable_f64(
        lx: wp.array2d(dtype=wp.float64),
        dinv: wp.array2d(dtype=wp.float64),
        rhs: wp.array2d(dtype=wp.float64),
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        segment_starts: wp.array(dtype=wp.int32),
        segment_counts: wp.array(dtype=wp.int32),
        segment_widths: wp.array(dtype=wp.int32),
        out: wp.array2d(dtype=wp.float64),
        work: wp.array2d(dtype=wp.float64),
    ):
        _qdldl_solve_segmented_transpose_callable(
            lx,
            dinv,
            rhs,
            p,
            pinv,
            col_rows,
            col_lidx,
            col_mask,
            segment_starts,
            segment_counts,
            segment_widths,
            out,
            work,
        )


    def _qdldl_solve_level_transpose_callable_f64(
        lx: wp.array2d(dtype=wp.float64),
        dinv: wp.array2d(dtype=wp.float64),
        rhs: wp.array2d(dtype=wp.float64),
        p: wp.array(dtype=wp.int32),
        pinv: wp.array(dtype=wp.int32),
        col_rows: wp.array2d(dtype=wp.int32),
        col_lidx: wp.array2d(dtype=wp.int32),
        col_mask: wp.array2d(dtype=wp.int32),
        level_cols: wp.array2d(dtype=wp.int32),
        level_mask: wp.array2d(dtype=wp.int32),
        out: wp.array2d(dtype=wp.float64),
        work: wp.array2d(dtype=wp.float64),
    ):
        _qdldl_solve_level_transpose_callable(
            lx,
            dinv,
            rhs,
            p,
            pinv,
            col_rows,
            col_lidx,
            col_mask,
            level_cols,
            level_mask,
            out,
            work,
        )


    def _make_qdldl_solve_ragged_serial_transpose_callable(
        schedule: tuple[tuple[str, int, int, int, int], ...],
        real_dtype: Any,
    ):
        real2d = wp.array2d(dtype=real_dtype)

        def _qdldl_solve_ragged_serial_transpose_callable(
            lx: real2d,
            dinv: real2d,
            rhs: real2d,
            p: wp.array(dtype=wp.int32),
            pinv: wp.array(dtype=wp.int32),
            col_rows: wp.array2d(dtype=wp.int32),
            col_lidx: wp.array2d(dtype=wp.int32),
            col_mask: wp.array2d(dtype=wp.int32),
            level_cols_flat: wp.array(dtype=wp.int32),
            level_offsets: wp.array(dtype=wp.int32),
            out: real2d,
            work: real2d,
        ):
            n = rhs.shape[0]
            batch_size = rhs.shape[1]

            wp.launch(
                _qdldl_permute_rhs_transpose_kernel,
                dim=(n, batch_size),
                inputs=[rhs, p],
                outputs=[work],
            )
            for kind, start_level, stop_level, flat_start, width in schedule:
                if kind == "parallel":
                    wp.launch(
                        _qdldl_lsolve_level_ragged_transpose_kernel,
                        dim=(width, batch_size),
                        inputs=[
                            lx,
                            col_rows,
                            col_lidx,
                            col_mask,
                            level_cols_flat,
                            flat_start,
                            work,
                        ],
                    )
                else:
                    wp.launch(
                        _qdldl_lsolve_serial_levels_transpose_kernel,
                        dim=batch_size,
                        inputs=[
                            lx,
                            col_rows,
                            col_lidx,
                            col_mask,
                            level_cols_flat,
                            level_offsets,
                            start_level,
                            stop_level,
                            work,
                        ],
                    )
            wp.launch(
                _qdldl_diag_scale_transpose_kernel,
                dim=(n, batch_size),
                inputs=[dinv, work],
            )
            for kind, start_level, stop_level, flat_start, width in reversed(schedule):
                if kind == "parallel":
                    wp.launch(
                        _qdldl_ltsolve_level_ragged_transpose_kernel,
                        dim=(width, batch_size),
                        inputs=[
                            lx,
                            col_rows,
                            col_lidx,
                            col_mask,
                            level_cols_flat,
                            flat_start,
                            work,
                        ],
                    )
                else:
                    wp.launch(
                        _qdldl_ltsolve_serial_levels_transpose_kernel,
                        dim=batch_size,
                        inputs=[
                            lx,
                            col_rows,
                            col_lidx,
                            col_mask,
                            level_cols_flat,
                            level_offsets,
                            start_level,
                            stop_level,
                            work,
                        ],
                    )
            wp.launch(
                _qdldl_unpermute_solution_transpose_kernel,
                dim=(n, batch_size),
                inputs=[work, pinv],
                outputs=[out],
            )

        for arg_name in ("lx", "dinv", "rhs", "out", "work"):
            _qdldl_solve_ragged_serial_transpose_callable.__annotations__[arg_name] = real2d

        return _qdldl_solve_ragged_serial_transpose_callable


def _require_warp() -> None:
    if wp is None:
        raise ImportError(
            "compile_qdldl_variant(..., backend='warp') requires the optional "
            "NVIDIA Warp package"
        ) from _WARP_IMPORT_ERROR


def _pad_levels(levels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    width = max((len(level) for level in levels), default=1)
    out = np.zeros((len(levels), width), dtype=np.int32)
    mask = np.zeros((len(levels), width), dtype=np.int32)
    for i, level in enumerate(levels):
        if level.size:
            out[i, : level.size] = np.asarray(level, dtype=np.int32)
            mask[i, : level.size] = 1
    return out, mask


def _pack_levels(levels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    flat: list[int] = []
    for level in levels:
        flat.extend(np.asarray(level, dtype=np.int32).tolist())
        offsets.append(len(flat))
    return np.asarray(flat, dtype=np.int32), np.asarray(offsets, dtype=np.int32)


def _make_level_schedule(
    levels: list[np.ndarray],
    offsets: np.ndarray,
    *,
    serial_width_threshold: int,
) -> tuple[tuple[str, int, int, int, int], ...]:
    schedule: list[tuple[str, int, int, int, int]] = []
    level = 0
    while level < len(levels):
        width = int(len(levels[level]))
        if width <= serial_width_threshold:
            start_level = level
            level += 1
            while level < len(levels) and int(len(levels[level])) <= serial_width_threshold:
                level += 1
            schedule.append(("serial", start_level, level, int(offsets[start_level]), 0))
        else:
            schedule.append(("parallel", level, level + 1, int(offsets[level]), width))
            level += 1
    return tuple(schedule)


def _make_segment_arrays(
    mask: np.ndarray,
    *,
    segment_budget: int,
    segment_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments = _make_segments(
        mask,
        segment_budget=segment_budget,
        segment_strategy=segment_strategy,
    )
    starts = np.asarray([start for start, _ in segments], dtype=np.int32)
    counts = np.asarray([columns.size for _, columns in segments], dtype=np.int32)
    widths = np.asarray(
        [
            int(np.max(np.sum(mask[columns], axis=1), initial=0))
            for _, columns in segments
        ],
        dtype=np.int32,
    )
    return starts, counts, widths


def _variant_name(
    *,
    transpose_work: bool,
    segmented: bool,
    segment_budget: int,
    segment_strategy: str,
    level_scheduled_solve: bool,
    level_scheduled_solve_threshold: int,
) -> str:
    variant_parts = ["warp"]
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


def compile_qdldl_warp(
    plan: QDLDLPlan,
    dtype: np.dtype | str = np.float32,
    *,
    transpose_work: bool = False,
    segmented: bool = False,
    segment_budget: int = 64,
    segment_strategy: str = "optimal",
    level_scheduled_solve: bool = False,
    level_scheduled_solve_threshold: int = 1,
) -> CompiledQDLDL:
    """Compile a Warp-backed QDLDL variant.

    Warp kernels currently operate on the symbolic-major layout used by the
    optimized JAX path.  ``transpose_work`` is accepted for API symmetry, but
    the returned layout is always symbolic-major for values and right-hand
    sides.
    """

    _require_warp()
    dtype = np.dtype(dtype)
    _require_x64_for_float64(dtype)
    if dtype == np.dtype(np.float32):
        warp_real_dtype = wp.float32
        factor_callback = (
            _qdldl_factor_segmented_transpose_callable
            if segmented
            else _qdldl_factor_transpose_callable
        )
        solve_serial_callback = (
            _qdldl_solve_segmented_transpose_callable
            if segmented
            else _qdldl_solve_serial_transpose_callable
        )
        solve_level_callback = None
    elif dtype == np.dtype(np.float64):
        warp_real_dtype = wp.float64
        factor_callback = (
            _qdldl_factor_segmented_transpose_callable_f64
            if segmented
            else _qdldl_factor_transpose_callable_f64
        )
        solve_serial_callback = (
            _qdldl_solve_segmented_transpose_callable_f64
            if segmented
            else _qdldl_solve_serial_transpose_callable_f64
        )
        solve_level_callback = None
    else:
        raise ValueError("the Warp QDLDL backend currently supports float32 and float64")
    if not transpose_work:
        raise ValueError("the Warp QDLDL backend currently requires transpose_work=True")
    if level_scheduled_solve_threshold < 1:
        raise ValueError("level_scheduled_solve_threshold must be at least 1")

    jdtype = jnp.dtype(dtype)
    perm_to_orig = jnp.asarray(plan.perm_to_orig, dtype=jnp.int32)
    p = jnp.asarray(plan.p, dtype=jnp.int32)
    pinv = jnp.asarray(plan.pinv, dtype=jnp.int32)
    diag_pos = jnp.asarray(plan.diag_pos, dtype=jnp.int32)
    init_rows = jnp.asarray(plan.init_rows, dtype=jnp.int32)
    init_pos = jnp.asarray(plan.init_pos, dtype=jnp.int32)
    init_mask = jnp.asarray(np.asarray(plan.init_mask, dtype=np.int32), dtype=jnp.int32)
    process_cols = jnp.asarray(plan.process_cols, dtype=jnp.int32)
    process_lidx = jnp.asarray(plan.process_lidx, dtype=jnp.int32)
    process_offsets = jnp.asarray(plan.process_offsets, dtype=jnp.int32)
    process_mask = jnp.asarray(np.asarray(plan.process_mask, dtype=np.int32), dtype=jnp.int32)
    col_rows = jnp.asarray(plan.col_rows, dtype=jnp.int32)
    col_lidx = jnp.asarray(plan.col_lidx, dtype=jnp.int32)
    col_mask = jnp.asarray(np.asarray(plan.col_mask, dtype=np.int32), dtype=jnp.int32)
    solve_levels = _make_solve_levels(plan.col_rows, plan.col_mask)
    level_cols_flat_np, level_offsets_np = _pack_levels(solve_levels)
    level_schedule = _make_level_schedule(
        solve_levels,
        level_offsets_np,
        serial_width_threshold=level_scheduled_solve_threshold,
    )
    level_cols_flat = jnp.asarray(level_cols_flat_np, dtype=jnp.int32)
    level_offsets = jnp.asarray(level_offsets_np, dtype=jnp.int32)
    if level_scheduled_solve:
        solve_level_callback = _make_qdldl_solve_ragged_serial_transpose_callable(
            level_schedule,
            warp_real_dtype,
        )
    if segmented:
        (
            process_segment_starts_np,
            process_segment_counts_np,
            process_segment_widths_np,
        ) = _make_segment_arrays(
            plan.process_mask,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
        )
        (
            col_segment_starts_np,
            col_segment_counts_np,
            col_segment_widths_np,
        ) = _make_segment_arrays(
            plan.col_mask,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
        )
        process_segment_starts = jnp.asarray(process_segment_starts_np, dtype=jnp.int32)
        process_segment_counts = jnp.asarray(process_segment_counts_np, dtype=jnp.int32)
        process_segment_widths = jnp.asarray(process_segment_widths_np, dtype=jnp.int32)
        col_segment_starts = jnp.asarray(col_segment_starts_np, dtype=jnp.int32)
        col_segment_counts = jnp.asarray(col_segment_counts_np, dtype=jnp.int32)
        col_segment_widths = jnp.asarray(col_segment_widths_np, dtype=jnp.int32)
    else:
        process_segment_starts = process_segment_counts = process_segment_widths = None
        col_segment_starts = col_segment_counts = col_segment_widths = None

    factor_callable = _jax_callable(
        factor_callback,
        num_outputs=4,
        vmap_method="broadcast_all",
    )
    solve_callable = _jax_callable(
        solve_level_callback if level_scheduled_solve else solve_serial_callback,
        num_outputs=2,
        vmap_method="broadcast_all",
    )

    n = plan.n
    nnz_l = plan.nnz_l

    @jax.jit
    def factor_symbolic(a_values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        a_values = jnp.asarray(a_values, dtype=jdtype)
        batch_size = a_values.shape[1]
        output_dims = {
            "lx": (nnz_l, batch_size),
            "d": (n, batch_size),
            "dinv": (n, batch_size),
            "y_work": (n, batch_size),
        }
        if segmented:
            lx, d, dinv, _ = factor_callable(
                a_values,
                perm_to_orig,
                diag_pos,
                init_rows,
                init_pos,
                init_mask,
                process_cols,
                process_lidx,
                process_offsets,
                process_mask,
                col_rows,
                col_lidx,
                col_mask,
                process_segment_starts,
                process_segment_counts,
                process_segment_widths,
                output_dims=output_dims,
            )
        else:
            lx, d, dinv, _ = factor_callable(
                a_values,
                perm_to_orig,
                diag_pos,
                init_rows,
                init_pos,
                init_mask,
                process_cols,
                process_lidx,
                process_offsets,
                process_mask,
                col_rows,
                col_lidx,
                col_mask,
                output_dims=output_dims,
            )
        return lx, d, dinv

    @jax.jit
    def solve_symbolic(lx: jax.Array, dinv: jax.Array, rhs: jax.Array) -> jax.Array:
        lx = jnp.asarray(lx, dtype=jdtype)
        dinv = jnp.asarray(dinv, dtype=jdtype)
        rhs = jnp.asarray(rhs, dtype=jdtype)
        batch_size = rhs.shape[1]
        output_dims = {
            "out": (n, batch_size),
            "work": (n, batch_size),
        }
        if level_scheduled_solve:
            out, _ = solve_callable(
                lx,
                dinv,
                rhs,
                p,
                pinv,
                col_rows,
                col_lidx,
                col_mask,
                level_cols_flat,
                level_offsets,
                output_dims=output_dims,
            )
        elif segmented:
            out, _ = solve_callable(
                lx,
                dinv,
                rhs,
                p,
                pinv,
                col_rows,
                col_lidx,
                col_mask,
                col_segment_starts,
                col_segment_counts,
                col_segment_widths,
                output_dims=output_dims,
            )
        else:
            out, _ = solve_callable(
                lx,
                dinv,
                rhs,
                p,
                pinv,
                col_rows,
                col_lidx,
                col_mask,
                output_dims=output_dims,
            )
        return out

    @jax.jit
    def factor_and_solve(a_values: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        lx, d, dinv = factor_symbolic(a_values)
        x = solve_symbolic(lx, dinv, rhs)
        return x, lx, d

    return CompiledQDLDL(
        plan=plan,
        dtype=dtype,
        factor=factor_symbolic,
        solve=solve_symbolic,
        factor_and_solve=factor_and_solve,
        variant=_variant_name(
            transpose_work=transpose_work,
            segmented=segmented,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
            level_scheduled_solve=level_scheduled_solve,
            level_scheduled_solve_threshold=level_scheduled_solve_threshold,
        ),
        values_layout="original_symbolic",
        rhs_layout="symbolic",
        backend="warp",
        factor_backend="warp",
        solve_backend="warp",
    )
