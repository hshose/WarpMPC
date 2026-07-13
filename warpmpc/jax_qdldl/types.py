"""Public dataclasses used by the fixed-pattern JAX QDLDL implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class QDLDLPlan:
    """All pattern-dependent data needed by the JAX numeric factorization."""

    n: int
    nnz_a: int
    nnz_l: int
    a_indptr: np.ndarray
    a_indices: np.ndarray
    a_data: np.ndarray
    aperm_indptr: np.ndarray
    aperm_indices: np.ndarray
    perm_to_orig: np.ndarray
    orig_to_perm: np.ndarray
    p: np.ndarray
    pinv: np.ndarray
    etree: np.ndarray
    lnz: np.ndarray
    lp: np.ndarray
    li: np.ndarray
    diag_pos: np.ndarray
    init_rows: np.ndarray
    init_pos: np.ndarray
    init_mask: np.ndarray
    process_cols: np.ndarray
    process_lidx: np.ndarray
    process_offsets: np.ndarray
    process_mask: np.ndarray
    col_rows: np.ndarray
    col_lidx: np.ndarray
    col_mask: np.ndarray

    @property
    def max_init(self) -> int:
        return int(self.init_rows.shape[1])

    @property
    def max_row_nnz(self) -> int:
        return int(self.process_cols.shape[1])

    @property
    def max_col_nnz(self) -> int:
        return int(self.col_rows.shape[1])


@dataclass(frozen=True)
class CompiledQDLDL:
    """JIT-compiled fixed-pattern factor and solve callables."""

    plan: QDLDLPlan
    dtype: np.dtype
    factor: Callable
    solve: Callable
    factor_and_solve: Callable
    variant: str = "baseline"
    values_layout: str = "original_batch"
    rhs_layout: str = "batch"
    backend: str = "jax"
    factor_backend: str = "jax"
    solve_backend: str = "jax"


__all__ = ["CompiledQDLDL", "QDLDLPlan"]
