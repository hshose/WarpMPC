"""Public dataclasses used by the fixed-pattern JAX OSQP implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp

from warpmpc.jax_qdldl import CompiledQDLDL, QDLDLPlan


@dataclass(frozen=True)
class OSQPDerivativePlan:
    """Pattern-dependent data for OSQP's adjoint derivative linear system."""

    n: int
    m: int
    n_ineq_l: int
    n_ineq_u: int
    n_eq: int
    adjoint_dim: int
    adjoint_upper: sp.csc_matrix
    qdldl_plan: QDLDLPlan
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    eq_indices: np.ndarray
    p_upper_rows: np.ndarray
    p_upper_cols: np.ndarray
    p_full_rows: np.ndarray
    p_full_cols: np.ndarray
    p_full_value_map: np.ndarray
    g_rows: np.ndarray
    g_cols: np.ndarray
    g_a_value_map: np.ndarray
    g_sign: np.ndarray
    aeq_rows: np.ndarray
    aeq_cols: np.ndarray
    aeq_a_value_map: np.ndarray
    diag_first_pos: np.ndarray
    diag_second_pos: np.ndarray
    p_block_pos: np.ndarray
    g_block_pos: np.ndarray
    aeq_block_pos: np.ndarray
    glambda_block_pos: np.ndarray
    slack_diag_pos: np.ndarray
    aeq_t_block_pos: np.ndarray
    eps: float = 1e-6

    @property
    def n_ineq(self) -> int:
        return self.n_ineq_l + self.n_ineq_u

    @property
    def half_dim(self) -> int:
        return self.n + self.n_ineq + self.n_eq


@dataclass(frozen=True)
class OSQPSettings:
    """Subset of OSQP settings used by the fixed-pattern JAX implementation."""

    rho: float = 0.1
    sigma: float = 1e-6
    alpha: float = 1.6
    max_iter: int = 4000
    eps_abs: float = 1e-3
    eps_rel: float = 1e-3
    scaling: int = 0
    adaptive_rho: bool = False
    rho_is_vec: bool = False
    check_termination: int = 0
    warm_starting: bool = False
    polishing: bool = False


class OSQPWarmStart(NamedTuple):
    """ADMM iterate state used to warm-start a fixed-pattern OSQP solve."""

    x: object
    z: object
    y: object


@dataclass(frozen=True)
class FixedOSQPPlan:
    """Pattern-dependent data for a direct-solver OSQP workspace."""

    n: int
    m: int
    p_upper: sp.csc_matrix
    a_matrix: sp.csc_matrix
    kkt_upper: sp.csc_matrix
    qdldl_plan: QDLDLPlan
    p_to_kkt: np.ndarray
    p_diag_sigma: np.ndarray
    a_to_kkt: np.ndarray
    rho_to_kkt: np.ndarray
    a_rows: np.ndarray
    a_cols: np.ndarray
    p_full_rows: np.ndarray
    p_full_cols: np.ndarray
    p_full_value_map: np.ndarray
    rho_vec: np.ndarray
    rho_inv_vec: np.ndarray
    constr_type: np.ndarray
    scaling_c: float
    scaling_cinv: float
    scaling_d: np.ndarray
    scaling_dinv: np.ndarray
    scaling_e: np.ndarray
    scaling_einv: np.ndarray
    p_scale: np.ndarray
    a_scale: np.ndarray
    q_scale: np.ndarray
    y_scale: np.ndarray
    y_unscale: np.ndarray
    derivative_plan: OSQPDerivativePlan | None
    settings: OSQPSettings

    @property
    def nnz_p(self) -> int:
        return int(self.p_upper.nnz)

    @property
    def nnz_a(self) -> int:
        return int(self.a_matrix.nnz)

    @property
    def nnz_kkt(self) -> int:
        return int(self.kkt_upper.nnz)


@dataclass(frozen=True)
class CompiledOSQP:
    """JIT-compiled fixed-pattern OSQP solve callables."""

    plan: FixedOSQPPlan
    settings: OSQPSettings
    qdldl: CompiledQDLDL
    solve: Callable
    factor: Callable
    solve_with_factor: Callable
    variant: str
    dtype: np.dtype
    linear_solve_loop: Callable | None = None
    init_warm_start: Callable | None = None
    solve_warm_start: Callable | None = None
    solve_with_factor_warm_start: Callable | None = None
    adjoint_derivative_compute: Callable | None = None
    adjoint_derivative_get_mat: Callable | None = None
    adjoint_derivative_get_vec: Callable | None = None
    solve_primal: Callable | None = None
    solve_xy: Callable | None = None


__all__ = [
    "CompiledOSQP",
    "FixedOSQPPlan",
    "OSQPDerivativePlan",
    "OSQPSettings",
    "OSQPWarmStart",
]
