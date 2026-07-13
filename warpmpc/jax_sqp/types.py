"""Public dataclasses for fixed-pattern sparse SQP in JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp

from warpmpc.jax_osqp import CompiledOSQP, FixedOSQPPlan, OSQPWarmStart


class SQPLinearization(NamedTuple):
    """Batched QP data produced by one SQP linearization."""

    p_values: object
    a_values: object
    q: object
    l: object
    u: object
    cost: object
    g: object
    constraint_l: object
    constraint_u: object


class SQPSolveResult(NamedTuple):
    """Result of solving the QP subproblem at one SQP iterate."""

    direction: object
    z: object
    y: object
    prim_res: object
    dual_res: object
    obj_val: object
    linearization: SQPLinearization


class SQPStepResult(NamedTuple):
    """Result of applying one damped SQP step."""

    z_next: object
    solve: SQPSolveResult


class SQPLineSearchStepResult(NamedTuple):
    """Result of applying one SQP step with a filter line search."""

    z_next: object
    solve: SQPSolveResult
    line_search: object
    is_finite: object


@dataclass(frozen=True)
class StageAssembly:
    """Global scatter maps for one stage's sparse derivative values."""

    z_offset: int
    next_z_offset: int
    param_offset: int
    constraint_offset: int
    q_cols: np.ndarray
    p_pos: np.ndarray
    a_pos: np.ndarray


@dataclass(frozen=True)
class SparseMPCPlan:
    """Fixed sparsity and index data for an MPC SQP problem."""

    stage_var_dims: tuple[int, ...]
    param_dims: tuple[int, ...]
    constraint_dims: tuple[int, ...]
    z_offsets: np.ndarray
    param_offsets: np.ndarray
    constraint_offsets: np.ndarray
    p_pattern: sp.csc_matrix
    a_pattern: sp.csc_matrix
    representative_l: np.ndarray
    representative_u: np.ndarray
    osqp_plan: FixedOSQPPlan
    assemblies: tuple[StageAssembly, ...]

    @property
    def n_variables(self) -> int:
        return int(self.z_offsets[-1])

    @property
    def n_parameters(self) -> int:
        return int(self.param_offsets[-1])

    @property
    def n_constraints(self) -> int:
        return int(self.constraint_offsets[-1])


@dataclass(frozen=True)
class CompiledSparseMPCSQP:
    """JIT-compiled fixed-pattern SQP update callables."""

    plan: SparseMPCPlan
    qp_solver: str
    osqp: object
    dtype: np.dtype
    evaluate_nlp: Callable
    linearize_nlp: Callable
    build_qp: Callable
    lower_evaluate_nlp: Callable
    lower_linearize_nlp: Callable
    lower_solve_qp_data: Callable
    solve_qp_data: Callable
    compute_direction: Callable
    fixed_step: Callable
    step: Callable
    step_split_compile: Callable
    init_state: Callable


__all__ = [
    "CompiledSparseMPCSQP",
    "SQPLinearization",
    "SQPLineSearchStepResult",
    "SQPSolveResult",
    "SQPStepResult",
    "SparseMPCPlan",
    "OSQPWarmStart",
    "StageAssembly",
]
