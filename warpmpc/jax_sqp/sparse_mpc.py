"""Fixed-pattern sparse MPC SQP assembly and OSQP-backed QP solves."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from collections.abc import Sequence
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental.sparse import BCOO

from warpmpc.jax_osqp import OSQPSettings, OSQPWarmStart, build_osqp_plan, compile_osqp

from .casadi_stage import CasadiStageFunction
from .line_search import (
    FilterLineSearchSettings,
    constraint_violation,
    filter_line_search_from_evaluations,
    make_step_lengths,
)
from .types import (
    CompiledSparseMPCSQP,
    SQPLinearization,
    SQPLineSearchStepResult,
    SQPSolveResult,
    SQPStepResult,
    SparseMPCPlan,
    StageAssembly,
)

_OSQP_INFTY = 1e30
_OSQP_MIN_SCALING = 1e-4


def _require_x64_for_float64(dtype: np.dtype) -> None:
    if dtype == np.dtype(np.float64) and not jax.config.jax_enable_x64:
        raise ValueError(
            "compile_sparse_mpc_sqp(..., dtype=float64) requires "
            "jax_enable_x64=True. Set it in the benchmark or example entry "
            "point before compiling."
        )


@dataclass(frozen=True)
class MPAXSettings:
    """Settings for the MPAX sparse QP backend used by sparse MPC SQP."""

    eps_abs: float = 1e-3
    eps_rel: float = 1e-3
    iteration_limit: int = 25
    termination_evaluation_frequency: int = 100
    l_inf_ruiz_iterations: int = 10
    pock_chambolle_alpha: float = 1.0
    regularization: float = 0.0
    unroll: bool = False


@dataclass(frozen=True)
class _CompiledMPAXQP:
    settings: MPAXSettings
    n_transformed_constraints: int
    init_warm_start: object
    solve_warm_start: object


@dataclass(frozen=True)
class SparseMPCProblem:
    """A stage-ordered nonlinear MPC problem with fixed local sparsity."""

    stages: tuple[CasadiStageFunction, ...]

    @classmethod
    def from_stage_functions(
        cls,
        *,
        horizon: int,
        first: CasadiStageFunction,
        intermediate: CasadiStageFunction | Sequence[CasadiStageFunction],
        terminal: CasadiStageFunction,
    ) -> "SparseMPCProblem":
        """Build an MPC problem from reusable first/middle/terminal functions.

        ``horizon`` is the number of intervals. The resulting problem has
        ``horizon + 1`` stage functions. ``intermediate`` may be one reusable
        function or a sequence of length ``horizon - 1``.
        """

        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if isinstance(intermediate, CasadiStageFunction):
            middle = [intermediate] * max(0, horizon - 1)
        else:
            middle = list(intermediate)
            if len(middle) != max(0, horizon - 1):
                raise ValueError("intermediate sequence must have length horizon - 1")
        stages = (first, *middle, terminal)
        return cls(stages=stages)


@dataclass(frozen=True)
class _StageEvaluationGroup:
    """A set of stages that reuse one CasADi/JAX stage function."""

    stage: CasadiStageFunction
    assemblies: tuple[StageAssembly, ...]
    z_offsets: np.ndarray
    next_z_offsets: np.ndarray
    param_offsets: np.ndarray
    constraint_rows: np.ndarray
    q_cols: np.ndarray
    p_pos: np.ndarray
    a_pos: np.ndarray

    @property
    def size(self) -> int:
        return len(self.assemblies)


def _make_stage_evaluation_groups(
    stage_functions: Sequence[CasadiStageFunction],
    assemblies: Sequence[StageAssembly],
    *,
    group_repeated_stages: bool,
) -> tuple[_StageEvaluationGroup, ...]:
    """Group repeated stage-function objects for stage-axis vmapping."""

    grouped: dict[int | tuple[int, int], list[tuple[CasadiStageFunction, StageAssembly]]] = {}
    order: list[int | tuple[int, int]] = []
    for stage_index, (stage, assembly) in enumerate(zip(stage_functions, assemblies, strict=True)):
        key: int | tuple[int, int]
        key = id(stage) if group_repeated_stages else (id(stage), stage_index)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((stage, assembly))

    out: list[_StageEvaluationGroup] = []
    for key in order:
        entries = grouped[key]
        stage = entries[0][0]
        group_assemblies = tuple(assembly for _, assembly in entries)
        constraint_offsets = np.asarray(
            [assembly.constraint_offset for assembly in group_assemblies],
            dtype=np.int32,
        )
        constraint_rows = (
            constraint_offsets[:, None]
            + np.arange(stage.constraint_dim, dtype=np.int32)[None, :]
        )
        out.append(
            _StageEvaluationGroup(
                stage=stage,
                assemblies=group_assemblies,
                z_offsets=np.asarray(
                    [assembly.z_offset for assembly in group_assemblies],
                    dtype=np.int32,
                ),
                next_z_offsets=np.asarray(
                    [assembly.next_z_offset for assembly in group_assemblies],
                    dtype=np.int32,
                ),
                param_offsets=np.asarray(
                    [assembly.param_offset for assembly in group_assemblies],
                    dtype=np.int32,
                ),
                constraint_rows=constraint_rows,
                q_cols=np.stack(
                    [assembly.q_cols for assembly in group_assemblies]
                ).astype(np.int32),
                p_pos=np.stack([assembly.p_pos for assembly in group_assemblies]).astype(np.int32),
                a_pos=np.stack([assembly.a_pos for assembly in group_assemblies]).astype(np.int32),
            )
        )
    return tuple(out)


def _offsets(widths: Sequence[int]) -> np.ndarray:
    out = np.zeros(len(widths) + 1, dtype=np.int32)
    out[1:] = np.cumsum(np.asarray(widths, dtype=np.int32))
    return out


def _csc_rows_cols(matrix: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(matrix.indices, dtype=np.int32)
    cols = np.repeat(
        np.arange(matrix.shape[1], dtype=np.int32),
        np.diff(matrix.indptr),
    )
    return rows, cols.astype(np.int32, copy=False)


def _csc_pattern(
    shape: tuple[int, int],
    keys: set[tuple[int, int]],
) -> tuple[sp.csc_matrix, dict[tuple[int, int], int]]:
    ordered = sorted(keys, key=lambda key: (key[1], key[0]))
    rows = np.asarray([key[0] for key in ordered], dtype=np.int32)
    cols = np.asarray([key[1] for key in ordered], dtype=np.int32)
    data = np.zeros(len(ordered), dtype=np.float64)
    matrix = sp.csc_matrix((data, (rows, cols)), shape=shape)
    matrix.sum_duplicates()
    matrix.sort_indices()
    key_to_pos: dict[tuple[int, int], int] = {}
    for col in range(matrix.shape[1]):
        for ptr in range(matrix.indptr[col], matrix.indptr[col + 1]):
            key_to_pos[(int(matrix.indices[ptr]), col)] = int(ptr)
    return matrix, key_to_pos


def _load_mpax():
    try:
        from mpax import create_qp, raPDHG
    except ImportError as exc:
        raise ImportError(
            "qp_solver='mpax' requires MPAX. Install MPAX separately; "
            "WarpMPC does not install benchmark-only solvers by default."
        ) from exc
    return create_qp, raPDHG


def _mpax_bound_rows(
    representative_l: np.ndarray,
    representative_u: np.ndarray,
    *,
    eq_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(representative_l, dtype=np.float64)
    upper = np.asarray(representative_u, dtype=np.float64)
    inf_threshold = _OSQP_INFTY * _OSQP_MIN_SCALING
    finite_lower = lower > -inf_threshold
    finite_upper = upper < inf_threshold
    equality = finite_lower & finite_upper & (np.abs(upper - lower) <= eq_tol)
    lower_only = finite_lower & ~equality
    upper_only = finite_upper & ~equality
    return (
        np.flatnonzero(equality).astype(np.int32),
        np.flatnonzero(lower_only).astype(np.int32),
        np.flatnonzero(upper_only).astype(np.int32),
    )


def _selected_csc_entries(
    matrix: sp.csc_matrix,
    selected_rows: np.ndarray,
    *,
    row_offset: int = 0,
    sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = _csc_rows_cols(matrix)
    row_map = np.full(matrix.shape[0], -1, dtype=np.int32)
    row_map[np.asarray(selected_rows, dtype=np.int32)] = np.arange(
        selected_rows.size,
        dtype=np.int32,
    )
    mapped_rows = row_map[rows]
    mask = mapped_rows >= 0
    value_map = np.flatnonzero(mask).astype(np.int32)
    return (
        (mapped_rows[mask] + int(row_offset)).astype(np.int32, copy=False),
        cols[mask].astype(np.int32, copy=False),
        value_map,
        np.full(value_map.shape, float(sign), dtype=np.float64),
    )


def _compile_mpax_qp(
    plan: SparseMPCPlan,
    *,
    dtype: np.dtype,
    settings: MPAXSettings,
) -> _CompiledMPAXQP:
    create_qp, raPDHG = _load_mpax()
    jdtype = jnp.dtype(dtype)
    n = plan.n_variables
    m = plan.n_constraints

    equality_rows, lower_rows, upper_rows = _mpax_bound_rows(
        plan.representative_l,
        plan.representative_u,
    )
    n_eq = int(equality_rows.size)
    n_lower = int(lower_rows.size)
    n_upper = int(upper_rows.size)
    n_ineq = n_lower + n_upper
    n_transformed_constraints = n_eq + n_ineq

    p_rows, p_cols = _csc_rows_cols(plan.p_pattern)
    p_value_map = np.arange(plan.p_pattern.nnz, dtype=np.int32)
    p_offdiag = p_rows != p_cols
    p_full_rows = np.concatenate([p_rows, p_cols[p_offdiag]]).astype(np.int32)
    p_full_cols = np.concatenate([p_cols, p_rows[p_offdiag]]).astype(np.int32)
    p_full_value_map = np.concatenate([p_value_map, p_value_map[p_offdiag]]).astype(np.int32)
    p_full_diag = p_full_rows == p_full_cols
    p_full_indices = jnp.asarray(np.stack([p_full_rows, p_full_cols], axis=1), dtype=jnp.int32)
    p_full_value_map_jax = jnp.asarray(p_full_value_map, dtype=jnp.int32)
    p_full_diag_jax = jnp.asarray(p_full_diag, dtype=bool)

    eq_a_rows, eq_a_cols, eq_a_map, eq_a_sign = _selected_csc_entries(
        plan.a_pattern,
        equality_rows,
        sign=1.0,
    )
    lower_a_rows, lower_a_cols, lower_a_map, lower_a_sign = _selected_csc_entries(
        plan.a_pattern,
        lower_rows,
        row_offset=0,
        sign=1.0,
    )
    upper_a_rows, upper_a_cols, upper_a_map, upper_a_sign = _selected_csc_entries(
        plan.a_pattern,
        upper_rows,
        row_offset=n_lower,
        sign=-1.0,
    )
    g_a_rows = np.concatenate([lower_a_rows, upper_a_rows]).astype(np.int32)
    g_a_cols = np.concatenate([lower_a_cols, upper_a_cols]).astype(np.int32)
    g_a_map = np.concatenate([lower_a_map, upper_a_map]).astype(np.int32)
    g_a_sign = np.concatenate([lower_a_sign, upper_a_sign]).astype(np.float64)

    eq_a_indices = jnp.asarray(np.stack([eq_a_rows, eq_a_cols], axis=1), dtype=jnp.int32)
    g_a_indices = jnp.asarray(np.stack([g_a_rows, g_a_cols], axis=1), dtype=jnp.int32)
    eq_a_map_jax = jnp.asarray(eq_a_map, dtype=jnp.int32)
    g_a_map_jax = jnp.asarray(g_a_map, dtype=jnp.int32)
    eq_a_sign_jax = jnp.asarray(eq_a_sign, dtype=jdtype)
    g_a_sign_jax = jnp.asarray(g_a_sign, dtype=jdtype)

    equality_rows_jax = jnp.asarray(equality_rows, dtype=jnp.int32)
    lower_rows_jax = jnp.asarray(lower_rows, dtype=jnp.int32)
    upper_rows_jax = jnp.asarray(upper_rows, dtype=jnp.int32)

    a_rows, a_cols = _csc_rows_cols(plan.a_pattern)
    a_indices = jnp.asarray(np.stack([a_rows, a_cols], axis=1), dtype=jnp.int32)

    variable_lower = jnp.full((n,), -jnp.inf, dtype=jdtype)
    variable_upper = jnp.full((n,), jnp.inf, dtype=jdtype)
    q_placeholder = BCOO(
        (jnp.zeros((p_full_value_map.size,), dtype=jdtype), p_full_indices),
        shape=(n, n),
    )
    a_eq_placeholder = BCOO(
        (jnp.zeros((eq_a_map.size,), dtype=jdtype), eq_a_indices),
        shape=(n_eq, n),
    )
    g_placeholder = BCOO(
        (jnp.zeros((g_a_map.size,), dtype=jdtype), g_a_indices),
        shape=(n_ineq, n),
    )
    base_problem = create_qp(
        q_placeholder,
        jnp.zeros((n,), dtype=jdtype),
        a_eq_placeholder,
        jnp.zeros((n_eq,), dtype=jdtype),
        g_placeholder,
        jnp.zeros((n_ineq,), dtype=jdtype),
        variable_lower,
        variable_upper,
        use_sparse_matrix=True,
    )
    solver = raPDHG(
        eps_abs=settings.eps_abs,
        eps_rel=settings.eps_rel,
        iteration_limit=settings.iteration_limit,
        termination_evaluation_frequency=settings.termination_evaluation_frequency,
        l_inf_ruiz_iterations=settings.l_inf_ruiz_iterations,
        pock_chambolle_alpha=settings.pock_chambolle_alpha,
        verbose=False,
        debug=False,
        jit=True,
        unroll=settings.unroll,
        warm_start=False,
        feasibility_polishing=False,
    )

    def init_warm_start(batch_size: int) -> OSQPWarmStart:
        return OSQPWarmStart(
            x=jnp.zeros((batch_size, n), dtype=jdtype),
            z=jnp.zeros((batch_size, m), dtype=jdtype),
            y=jnp.zeros((batch_size, m), dtype=jdtype),
        )

    def _solve_one(p_values, a_values, q, lower_qp, upper_qp):
        p_full_data = p_values[p_full_value_map_jax]
        if settings.regularization:
            p_full_data = p_full_data + jnp.where(
                p_full_diag_jax,
                jnp.asarray(settings.regularization, dtype=jdtype),
                jnp.asarray(0.0, dtype=jdtype),
            )
        objective_matrix = BCOO((p_full_data, p_full_indices), shape=(n, n))
        a_eq = BCOO(
            (a_values[eq_a_map_jax] * eq_a_sign_jax, eq_a_indices),
            shape=(n_eq, n),
        )
        g_ineq = BCOO(
            (a_values[g_a_map_jax] * g_a_sign_jax, g_a_indices),
            shape=(n_ineq, n),
        )
        constraint_matrix = BCOO(
            (
                jnp.concatenate([a_eq.data, g_ineq.data], axis=0),
                jnp.concatenate(
                    [
                        a_eq.indices,
                        g_ineq.indices + jnp.asarray([n_eq, 0], dtype=jnp.int32),
                    ],
                    axis=0,
                ),
            ),
            shape=(n_transformed_constraints, n),
        )
        right_hand_side = jnp.concatenate(
            [
                lower_qp[equality_rows_jax],
                lower_qp[lower_rows_jax],
                -upper_qp[upper_rows_jax],
            ],
            axis=0,
        )
        problem = dataclasses.replace(
            base_problem,
            objective_matrix=objective_matrix,
            objective_vector=q,
            constraint_matrix=constraint_matrix,
            constraint_matrix_t=constraint_matrix.T,
            right_hand_side=right_hand_side,
        )
        result = solver.optimize(problem)
        primal_solution = jnp.asarray(result.primal_solution, dtype=jdtype)
        dual_solution = jnp.asarray(result.dual_solution, dtype=jdtype)
        original_a = BCOO((a_values, a_indices), shape=(m, n))
        activity = original_a @ primal_solution
        z = jnp.minimum(upper_qp, jnp.maximum(lower_qp, activity))
        y = jnp.zeros((m,), dtype=jdtype)
        eq_dual = dual_solution[:n_eq]
        lower_dual = dual_solution[n_eq : n_eq + n_lower]
        upper_dual = dual_solution[n_eq + n_lower :]
        y = y.at[equality_rows_jax].set(-eq_dual)
        y = y.at[lower_rows_jax].add(-lower_dual)
        y = y.at[upper_rows_jax].add(upper_dual)
        return (
            primal_solution,
            z,
            y,
            jnp.asarray(result.primal_residual_norm, dtype=jdtype),
            jnp.asarray(result.dual_residual_norm, dtype=jdtype),
            jnp.asarray(result.primal_objective, dtype=jdtype),
        )

    solve_batch = jax.jit(jax.vmap(_solve_one, in_axes=(0, 0, 0, 0, 0)))

    def solve_warm_start(
        p_values,
        a_values,
        q,
        lower_qp,
        upper_qp,
        state_x,
        state_z,
        state_y,
    ):
        del state_x, state_z, state_y
        return solve_batch(p_values, a_values, q, lower_qp, upper_qp)

    return _CompiledMPAXQP(
        settings=settings,
        n_transformed_constraints=n_transformed_constraints,
        init_warm_start=init_warm_start,
        solve_warm_start=solve_warm_start,
    )


def _stage_bound_samples(stage: CasadiStageFunction) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate stage bounds at a few probes to infer fixed bound types."""

    lowers = []
    uppers = []
    for fill in (0.0, 1.0, -1.0):
        z = np.full(stage.z_dim, fill, dtype=np.float64)
        p = np.full(stage.param_dim, fill, dtype=np.float64)
        if stage.has_next:
            zn = np.full(stage.next_z_dim, fill, dtype=np.float64)
            _, _, lower, upper = stage.value_function(z, zn, p)
        else:
            _, _, lower, upper = stage.value_function(z, p)
        lowers.append(np.asarray(lower, dtype=np.float64).reshape(-1))
        uppers.append(np.asarray(upper, dtype=np.float64).reshape(-1))
    return np.stack(lowers, axis=0), np.stack(uppers, axis=0)


def _classify_problem_bounds(problem: SparseMPCProblem) -> tuple[np.ndarray, np.ndarray]:
    """Build representative bounds preserving equality/lower/upper structure.

    OSQP's adjoint derivative setup needs the fixed bound type of each
    constraint row.  The actual SQP bounds are assembled at run time, so the
    numeric values here are only representatives used for symbolic derivative
    setup and vector-rho classification.
    """

    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    inf_threshold = _OSQP_INFTY * _OSQP_MIN_SCALING
    for stage in problem.stages:
        lower_samples, upper_samples = _stage_bound_samples(stage)
        equal = np.all(np.isclose(lower_samples, upper_samples, rtol=1e-9, atol=1e-9), axis=0)
        has_lower = np.any(lower_samples > -inf_threshold, axis=0)
        has_upper = np.any(upper_samples < inf_threshold, axis=0)

        lower = np.where(has_lower, 0.0, -_OSQP_INFTY).astype(np.float64)
        upper = np.where(has_upper, 1.0, _OSQP_INFTY).astype(np.float64)
        lower[equal] = 0.0
        upper[equal] = 0.0
        lower_parts.append(lower)
        upper_parts.append(upper)
    return np.concatenate(lower_parts), np.concatenate(upper_parts)


def _local_to_global(stage: CasadiStageFunction, z_offset: int, next_z_offset: int) -> np.ndarray:
    if not stage.has_next:
        return z_offset + np.arange(stage.z_dim, dtype=np.int32)
    return np.concatenate(
        [
            z_offset + np.arange(stage.z_dim, dtype=np.int32),
            next_z_offset + np.arange(stage.next_z_dim, dtype=np.int32),
        ]
    )


def build_sparse_mpc_plan(
    problem: SparseMPCProblem,
    *,
    osqp_settings: OSQPSettings | None = None,
    derivatives: bool = False,
    representative_z: np.ndarray | None = None,
    representative_params: np.ndarray | None = None,
) -> SparseMPCPlan:
    """Precompute all fixed index data for sparse SQP QP assembly."""

    stages = problem.stages
    if len(stages) < 2:
        raise ValueError("MPC SQP needs at least one transition and one terminal stage")
    for i, stage in enumerate(stages[:-1]):
        if not stage.has_next:
            raise ValueError(f"stage {i} must have a next-stage input")
    if stages[-1].has_next:
        raise ValueError("terminal stage must not have a next-stage input")

    stage_var_dims = [stages[0].z_dim]
    for stage in stages[:-1]:
        stage_var_dims.append(stage.next_z_dim)
    for i, (expected, stage) in enumerate(zip(stage_var_dims, stages, strict=True)):
        if stage.z_dim != expected:
            raise ValueError(f"stage {i} z_dim={stage.z_dim} does not match expected {expected}")
    param_dims = [stage.param_dim for stage in stages]
    constraint_dims = [stage.constraint_dim for stage in stages]
    z_offsets = _offsets(stage_var_dims)
    param_offsets = _offsets(param_dims)
    constraint_offsets = _offsets(constraint_dims)

    p_keys = {(i, i) for i in range(int(z_offsets[-1]))}
    a_keys: set[tuple[int, int]] = set()
    local_data = []
    for i, stage in enumerate(stages):
        z_offset = int(z_offsets[i])
        next_z_offset = int(z_offsets[i + 1]) if stage.has_next else 0
        c_offset = int(constraint_offsets[i])
        local_global = _local_to_global(stage, z_offset, next_z_offset)
        q_cols = local_global[stage.grad_cols]

        p_stage_keys: list[tuple[int, int]] = []
        for local_r, local_c in zip(stage.hess_rows, stage.hess_cols, strict=True):
            row = int(local_global[int(local_r)])
            col = int(local_global[int(local_c)])
            if row > col:
                row, col = col, row
            p_keys.add((row, col))
            p_stage_keys.append((row, col))

        a_stage_keys: list[tuple[int, int]] = []
        for local_r, local_c in zip(stage.jac_rows, stage.jac_cols, strict=True):
            row = c_offset + int(local_r)
            col = int(local_global[int(local_c)])
            a_keys.add((row, col))
            a_stage_keys.append((row, col))
        local_data.append((q_cols, p_stage_keys, a_stage_keys))

    p_pattern, p_key_to_pos = _csc_pattern((int(z_offsets[-1]), int(z_offsets[-1])), p_keys)
    a_pattern, a_key_to_pos = _csc_pattern(
        (int(constraint_offsets[-1]), int(z_offsets[-1])),
        a_keys,
    )
    assemblies = []
    for i, stage in enumerate(stages):
        q_cols, p_stage_keys, a_stage_keys = local_data[i]
        assemblies.append(
            StageAssembly(
                z_offset=int(z_offsets[i]),
                next_z_offset=int(z_offsets[i + 1]) if stage.has_next else 0,
                param_offset=int(param_offsets[i]),
                constraint_offset=int(constraint_offsets[i]),
                q_cols=np.asarray(q_cols, dtype=np.int32),
                p_pos=np.asarray([p_key_to_pos[key] for key in p_stage_keys], dtype=np.int32),
                a_pos=np.asarray([a_key_to_pos[key] for key in a_stage_keys], dtype=np.int32),
            )
        )

    settings = osqp_settings or OSQPSettings(
        rho=0.1,
        sigma=1e-6,
        alpha=1.6,
        max_iter=25,
        scaling=0,
        adaptive_rho=False,
        rho_is_vec=False,
        check_termination=0,
        warm_starting=False,
        polishing=False,
    )
    lower0, upper0 = _classify_problem_bounds(problem)

    p_for_osqp = p_pattern
    a_for_osqp = a_pattern
    scaling_q = None
    if (representative_z is None) != (representative_params is None):
        raise ValueError("representative_z and representative_params must be provided together")
    if representative_z is not None and representative_params is not None:
        z_ref = np.asarray(representative_z, dtype=np.float64)
        p_ref = np.asarray(representative_params, dtype=np.float64)
        if z_ref.ndim == 2:
            z_ref = z_ref[0]
        if p_ref.ndim == 2:
            p_ref = p_ref[0]
        if z_ref.shape != (int(z_offsets[-1]),):
            raise ValueError(f"representative_z must have shape {(int(z_offsets[-1]),)}, got {z_ref.shape}")
        if p_ref.shape != (int(param_offsets[-1]),):
            raise ValueError(
                f"representative_params must have shape {(int(param_offsets[-1]),)}, got {p_ref.shape}"
            )

        p_values = np.zeros(p_pattern.nnz, dtype=np.float64)
        a_values = np.zeros(a_pattern.nnz, dtype=np.float64)
        q_values = np.zeros(int(z_offsets[-1]), dtype=np.float64)
        for stage, assembly in zip(stages, assemblies, strict=True):
            z_stage = z_ref[assembly.z_offset : assembly.z_offset + stage.z_dim]
            p_stage = p_ref[assembly.param_offset : assembly.param_offset + stage.param_dim]
            if stage.has_next:
                zn_stage = z_ref[
                    assembly.next_z_offset : assembly.next_z_offset + stage.next_z_dim
                ]
                raw = stage.sparse_function(z_stage, zn_stage, p_stage)
            else:
                raw = stage.sparse_function(z_stage, p_stage)
            grad = np.asarray(raw[4], dtype=np.float64).reshape(-1)
            hess = np.asarray(raw[5], dtype=np.float64).reshape(-1)
            jac = np.asarray(raw[6], dtype=np.float64).reshape(-1)
            q_values[assembly.q_cols] += grad
            p_values[assembly.p_pos] += hess
            a_values[assembly.a_pos] += jac

        p_for_osqp = p_pattern.copy()
        a_for_osqp = a_pattern.copy()
        p_for_osqp.data = p_values
        a_for_osqp.data = a_values
        scaling_q = q_values

    osqp_plan = build_osqp_plan(
        p_for_osqp,
        a_for_osqp,
        lower0,
        upper0,
        settings,
        derivatives=derivatives,
        scaling_q=scaling_q,
    )
    return SparseMPCPlan(
        stage_var_dims=tuple(int(v) for v in stage_var_dims),
        param_dims=tuple(int(v) for v in param_dims),
        constraint_dims=tuple(int(v) for v in constraint_dims),
        z_offsets=z_offsets,
        param_offsets=param_offsets,
        constraint_offsets=constraint_offsets,
        p_pattern=p_pattern,
        a_pattern=a_pattern,
        representative_l=lower0,
        representative_u=upper0,
        osqp_plan=osqp_plan,
        assemblies=tuple(assemblies),
    )


def _as_batch(array, width: int, dtype: np.dtype):
    out = jnp.asarray(array, dtype=jnp.dtype(dtype))
    if out.ndim == 1:
        out = out[None, :]
    if out.shape[1] != width:
        raise ValueError(f"expected width {width}, got {out.shape[1]}")
    return out


def compile_sparse_mpc_sqp(
    problem: SparseMPCProblem,
    plan: SparseMPCPlan | None = None,
    *,
    dtype: np.dtype | str = np.float64,
    qp_solver: str = "jax_osqp",
    osqp_settings: OSQPSettings | None = None,
    mpax_settings: MPAXSettings | None = None,
    transpose_work: bool = True,
    segmented: bool = True,
    segment_budget: int = 64,
    segment_strategy: str = "optimal",
    level_scheduled_solve: bool = False,
    level_scheduled_solve_threshold: int = 1,
    qdldl_backend: str = "jax",
    qdldl_factor_backend: str | None = None,
    qdldl_solve_backend: str | None = None,
    derivatives: bool = False,
    derivative_refinement_iters: int = 100,
    line_search_settings: FilterLineSearchSettings | None = None,
    group_repeated_stages: bool = True,
) -> CompiledSparseMPCSQP:
    """Compile fixed-pattern SQP linearization and QP solve functions.

    ``qdldl_backend`` selects both QDLDL factorization and solve backends for
    normal use.  The factor/solve-specific keyword arguments are advanced
    benchmarking hooks for mixed backend variants.
    """

    dtype = np.dtype(dtype)
    _require_x64_for_float64(dtype)
    jdtype = jnp.dtype(dtype)
    if qp_solver not in {"jax_osqp", "mpax"}:
        raise ValueError(f"unsupported qp_solver: {qp_solver!r}")
    if derivatives and qp_solver != "jax_osqp":
        raise ValueError("derivatives=True is only supported with qp_solver='jax_osqp'")
    if plan is None:
        plan = build_sparse_mpc_plan(
            problem,
            osqp_settings=osqp_settings,
            derivatives=derivatives,
        )
    elif derivatives and plan.osqp_plan.derivative_plan is None:
        raise ValueError(
            "compile_sparse_mpc_sqp(..., derivatives=True) requires a plan built "
            "with build_sparse_mpc_plan(..., derivatives=True), or no explicit plan."
        )
    if qp_solver == "jax_osqp":
        qp_backend = compile_osqp(
            plan.osqp_plan,
            dtype=dtype,
            transpose_work=transpose_work,
            segmented=segmented,
            segment_budget=segment_budget,
            segment_strategy=segment_strategy,
            level_scheduled_solve=level_scheduled_solve,
            level_scheduled_solve_threshold=level_scheduled_solve_threshold,
            qdldl_backend=qdldl_backend,
            qdldl_factor_backend=qdldl_factor_backend,
            qdldl_solve_backend=qdldl_solve_backend,
            derivatives=derivatives,
            derivative_refinement_iters=derivative_refinement_iters,
        )
        assert qp_backend.solve_warm_start is not None
    else:
        warnings.warn(
            "MPAX support in warpmpc.jax_sqp is experimental and not finalized yet.",
            RuntimeWarning,
            stacklevel=2,
        )
        qp_backend = _compile_mpax_qp(
            plan,
            dtype=dtype,
            settings=mpax_settings or MPAXSettings(),
        )

    stage_functions = problem.stages
    assemblies = plan.assemblies
    stage_groups = _make_stage_evaluation_groups(
        stage_functions,
        assemblies,
        group_repeated_stages=group_repeated_stages,
    )
    n = plan.n_variables
    m = plan.n_constraints
    nnz_p = plan.p_pattern.nnz
    nnz_a = plan.a_pattern.nnz
    line_search_settings = line_search_settings or FilterLineSearchSettings()
    line_search_step_lengths_np = np.asarray(
        make_step_lengths(line_search_settings), dtype=dtype
    )
    line_search_step_lengths = jnp.asarray(line_search_step_lengths_np, dtype=jdtype)

    def _eval_stage_group(group: _StageEvaluationGroup, z, params, *, values_only: bool):
        stage = group.stage
        z_cur = jnp.stack(
            [z[:, int(offset) : int(offset) + stage.z_dim] for offset in group.z_offsets],
            axis=0,
        )
        p_stage = jnp.stack(
            [
                params[:, int(offset) : int(offset) + stage.param_dim]
                for offset in group.param_offsets
            ],
            axis=0,
        )
        stage_function = stage.jax_value_function if values_only else stage.jax_function
        if stage.has_next:
            z_next = jnp.stack(
                [
                    z[:, int(offset) : int(offset) + stage.next_z_dim]
                    for offset in group.next_z_offsets
                ],
                axis=0,
            )

            def eval_batch(z_cur_batch, z_next_batch, p_batch):
                return jax.vmap(stage_function)(z_cur_batch, z_next_batch, p_batch)

            raw = jax.vmap(eval_batch)(z_cur, z_next, p_stage)
        else:

            def eval_batch(z_cur_batch, p_batch):
                return jax.vmap(stage_function)(z_cur_batch, p_batch)

            raw = jax.vmap(eval_batch)(z_cur, p_stage)
        return tuple(
            jnp.asarray(value, dtype=jdtype).reshape((group.size, z.shape[0], -1))
            for value in raw
        )

    def _batched_scatter_indices(batch: int, positions):
        return (
            jnp.arange(batch, dtype=jnp.int32)[:, None, None],
            jnp.asarray(positions, dtype=jnp.int32)[None, :, :],
        )

    def _evaluate_values(z, params):
        batch = z.shape[0]
        g_all = jnp.zeros((batch, m), dtype=jdtype)
        lower_all = jnp.zeros((batch, m), dtype=jdtype)
        upper_all = jnp.zeros((batch, m), dtype=jdtype)
        cost_total = jnp.zeros((batch,), dtype=jdtype)
        for group in stage_groups:
            cost, g, lower, upper = _eval_stage_group(group, z, params, values_only=True)
            cost_total = cost_total + jnp.sum(cost[:, :, 0], axis=0)
            batch_idx, rows = _batched_scatter_indices(batch, group.constraint_rows)
            g_all = g_all.at[batch_idx, rows].set(jnp.swapaxes(g, 0, 1))
            lower_all = lower_all.at[batch_idx, rows].set(jnp.swapaxes(lower, 0, 1))
            upper_all = upper_all.at[batch_idx, rows].set(jnp.swapaxes(upper, 0, 1))
        return cost_total, g_all, lower_all, upper_all

    @jax.jit
    def _linearize_jit(z, params) -> SQPLinearization:
        z = jnp.asarray(z, dtype=jdtype)
        params = jnp.asarray(params, dtype=jdtype)
        batch = z.shape[0]
        p_values = jnp.zeros((batch, nnz_p), dtype=jdtype)
        a_values = jnp.zeros((batch, nnz_a), dtype=jdtype)
        q = jnp.zeros((batch, n), dtype=jdtype)
        lower_qp = jnp.zeros((batch, m), dtype=jdtype)
        upper_qp = jnp.zeros((batch, m), dtype=jdtype)
        g_all = jnp.zeros((batch, m), dtype=jdtype)
        lower_all = jnp.zeros((batch, m), dtype=jdtype)
        upper_all = jnp.zeros((batch, m), dtype=jdtype)
        cost_total = jnp.zeros((batch,), dtype=jdtype)

        for group in stage_groups:
            cost, g, lower, upper, grad, hess, jac = _eval_stage_group(
                group,
                z,
                params,
                values_only=False,
            )
            cost_total = cost_total + jnp.sum(cost[:, :, 0], axis=0)
            batch_idx, q_cols = _batched_scatter_indices(batch, group.q_cols)
            q = q.at[batch_idx, q_cols].add(jnp.swapaxes(grad, 0, 1))
            _, p_pos = _batched_scatter_indices(batch, group.p_pos)
            p_values = p_values.at[batch_idx, p_pos].add(jnp.swapaxes(hess, 0, 1))
            _, a_pos = _batched_scatter_indices(batch, group.a_pos)
            a_values = a_values.at[batch_idx, a_pos].add(jnp.swapaxes(jac, 0, 1))
            _, rows = _batched_scatter_indices(batch, group.constraint_rows)
            lower_qp = lower_qp.at[batch_idx, rows].set(jnp.swapaxes(lower - g, 0, 1))
            upper_qp = upper_qp.at[batch_idx, rows].set(jnp.swapaxes(upper - g, 0, 1))
            g_all = g_all.at[batch_idx, rows].set(jnp.swapaxes(g, 0, 1))
            lower_all = lower_all.at[batch_idx, rows].set(jnp.swapaxes(lower, 0, 1))
            upper_all = upper_all.at[batch_idx, rows].set(jnp.swapaxes(upper, 0, 1))

        return SQPLinearization(
            p_values=p_values,
            a_values=a_values,
            q=q,
            l=lower_qp,
            u=upper_qp,
            cost=cost_total,
            g=g_all,
            constraint_l=lower_all,
            constraint_u=upper_all,
        )

    @jax.jit
    def _evaluate_jit(z, params):
        z = jnp.asarray(z, dtype=jdtype)
        params = jnp.asarray(params, dtype=jdtype)
        return _evaluate_values(z, params)

    @jax.jit
    def _solve_linearized_qp_with_state_jit(
        p_values,
        a_values,
        q,
        lower_qp,
        upper_qp,
        cost,
        g,
        constraint_l,
        constraint_u,
        state_x,
        state_z,
        state_y,
    ) -> SQPSolveResult:
        linearization = SQPLinearization(
            p_values=p_values,
            a_values=a_values,
            q=q,
            l=lower_qp,
            u=upper_qp,
            cost=cost,
            g=g,
            constraint_l=constraint_l,
            constraint_u=constraint_u,
        )
        direction, qp_z, qp_y, prim_res, dual_res, obj_val = qp_backend.solve_warm_start(
            p_values,
            a_values,
            q,
            lower_qp,
            upper_qp,
            state_x,
            state_z,
            state_y,
        )
        return SQPSolveResult(
            direction=direction,
            z=qp_z,
            y=qp_y,
            prim_res=prim_res,
            dual_res=dual_res,
            obj_val=obj_val,
            linearization=linearization,
        )

    @jax.jit
    def _solve_qp_with_state_jit(
        z,
        params,
        state_x,
        state_z,
        state_y,
    ) -> SQPSolveResult:
        linearization = _linearize_jit(z, params)
        return _solve_linearized_qp_with_state_jit(
            linearization.p_values,
            linearization.a_values,
            linearization.q,
            linearization.l,
            linearization.u,
            linearization.cost,
            linearization.g,
            linearization.constraint_l,
            linearization.constraint_u,
            state_x,
            state_z,
            state_y,
        )

    @jax.jit
    def _fixed_step_with_state_jit(
        z,
        params,
        beta,
        state_x,
        state_z,
        state_y,
    ) -> SQPStepResult:
        solve = _solve_qp_with_state_jit(
            z,
            params,
            state_x,
            state_z,
            state_y,
        )
        return SQPStepResult(z_next=z + beta * solve.direction, solve=solve)

    @jax.jit
    def _line_search_step_with_state_jit(
        z,
        params,
        state_x,
        state_z,
        state_y,
    ) -> SQPLineSearchStepResult:
        z = jnp.asarray(z, dtype=jdtype)
        params = jnp.asarray(params, dtype=jdtype)
        solve = _solve_qp_with_state_jit(
            z,
            params,
            state_x,
            state_z,
            state_y,
        )
        baseline = solve.linearization
        direction = solve.direction
        baseline_violation = constraint_violation(
            baseline.constraint_l,
            baseline.g,
            baseline.constraint_u,
            line_search_settings.line_search_constraint_scale,
        )
        armijo_descent_metric = jnp.sum(baseline.q * direction, axis=1)

        def evaluate_candidate(step_length):
            candidate_z = z + step_length * direction
            cost, g, lower, upper = _evaluate_values(candidate_z, params)
            violation = constraint_violation(
                lower,
                g,
                upper,
                line_search_settings.line_search_constraint_scale,
            )
            return cost, violation

        candidate_costs, candidate_violations = jax.vmap(evaluate_candidate)(
            line_search_step_lengths
        )
        line_search = filter_line_search_from_evaluations(
            settings=line_search_settings,
            step_lengths=line_search_step_lengths,
            baseline_cost=baseline.cost,
            baseline_constraint_violation=baseline_violation,
            armijo_descent_metric=armijo_descent_metric,
            candidate_costs=candidate_costs,
            candidate_constraint_violations=candidate_violations,
        )
        z_next = z + line_search.step_length[:, None] * direction
        return SQPLineSearchStepResult(
            z_next=z_next,
            solve=solve,
            line_search=line_search,
            is_finite=jnp.all(jnp.isfinite(z_next), axis=1),
        )

    def evaluate_nlp(z, params):
        return _evaluate_jit(
            _as_batch(z, n, dtype),
            _as_batch(params, plan.n_parameters, dtype),
        )

    def lower_evaluate_nlp(z, params):
        return _evaluate_jit.lower(
            _as_batch(z, n, dtype),
            _as_batch(params, plan.n_parameters, dtype),
        )

    def linearize_nlp(z, params) -> SQPLinearization:
        return _linearize_jit(
            _as_batch(z, n, dtype),
            _as_batch(params, plan.n_parameters, dtype),
        )

    def lower_linearize_nlp(z, params):
        return _linearize_jit.lower(
            _as_batch(z, n, dtype),
            _as_batch(params, plan.n_parameters, dtype),
        )

    build_qp = linearize_nlp

    def init_state(batch_size: int) -> OSQPWarmStart:
        return qp_backend.init_warm_start(batch_size)

    def _state_from_solve(solve: SQPSolveResult) -> OSQPWarmStart:
        return OSQPWarmStart(x=solve.direction, z=solve.z, y=solve.y)

    def _state_or_zeros(state: OSQPWarmStart | None, batch_size: int) -> OSQPWarmStart:
        return init_state(batch_size) if state is None else state

    def solve_qp_data(
        linearization: SQPLinearization,
        *,
        state: OSQPWarmStart | None = None,
    ) -> tuple[SQPSolveResult, OSQPWarmStart]:
        state = _state_or_zeros(state, int(linearization.q.shape[0]))
        solve = _solve_linearized_qp_with_state_jit(
            linearization.p_values,
            linearization.a_values,
            linearization.q,
            linearization.l,
            linearization.u,
            linearization.cost,
            linearization.g,
            linearization.constraint_l,
            linearization.constraint_u,
            state.x,
            state.z,
            state.y,
        )
        return solve, _state_from_solve(solve)

    def lower_solve_qp_data(
        linearization: SQPLinearization,
        *,
        state: OSQPWarmStart | None = None,
    ):
        state = _state_or_zeros(state, int(linearization.q.shape[0]))
        return _solve_linearized_qp_with_state_jit.lower(
            linearization.p_values,
            linearization.a_values,
            linearization.q,
            linearization.l,
            linearization.u,
            linearization.cost,
            linearization.g,
            linearization.constraint_l,
            linearization.constraint_u,
            state.x,
            state.z,
            state.y,
        )

    def compute_direction(
        z,
        params,
        *,
        state: OSQPWarmStart | None = None,
    ) -> tuple[SQPSolveResult, OSQPWarmStart]:
        z_batch = _as_batch(z, n, dtype)
        state = _state_or_zeros(state, int(z_batch.shape[0]))
        solve = _solve_qp_with_state_jit(
            z_batch,
            _as_batch(params, plan.n_parameters, dtype),
            _as_batch(state.x, n, dtype),
            _as_batch(state.z, m, dtype),
            _as_batch(state.y, m, dtype),
        )
        return solve, _state_from_solve(solve)

    def fixed_step(
        z,
        params,
        beta=1.0,
        *,
        state: OSQPWarmStart | None = None,
    ) -> tuple[SQPStepResult, OSQPWarmStart]:
        z_batch = _as_batch(z, n, dtype)
        state = _state_or_zeros(state, int(z_batch.shape[0]))
        beta_array = jnp.asarray(beta, dtype=jdtype)
        if beta_array.ndim == 0:
            beta_array = jnp.broadcast_to(beta_array, (z_batch.shape[0], 1))
        elif beta_array.ndim == 1:
            beta_array = beta_array[:, None]
        result = _fixed_step_with_state_jit(
            z_batch,
            _as_batch(params, plan.n_parameters, dtype),
            beta_array,
            _as_batch(state.x, n, dtype),
            _as_batch(state.z, m, dtype),
            _as_batch(state.y, m, dtype),
        )
        return result, _state_from_solve(result.solve)

    def step(
        z,
        params,
        *,
        state: OSQPWarmStart | None = None,
    ) -> tuple[SQPLineSearchStepResult, OSQPWarmStart]:
        z_batch = _as_batch(z, n, dtype)
        state = _state_or_zeros(state, int(z_batch.shape[0]))
        result = _line_search_step_with_state_jit(
            z_batch,
            _as_batch(params, plan.n_parameters, dtype),
            _as_batch(state.x, n, dtype),
            _as_batch(state.z, m, dtype),
            _as_batch(state.y, m, dtype),
        )
        return result, _state_from_solve(result.solve)

    def step_split_compile(
        z,
        params,
        *,
        state: OSQPWarmStart | None = None,
    ) -> tuple[SQPLineSearchStepResult, OSQPWarmStart]:
        """Apply one SQP line-search step using smaller compiled pieces.

        This avoids compiling the full linearize + QP solve + candidate
        evaluation pipeline as one XLA executable. It is useful for very large
        stage functions where the monolithic ``step`` can exceed GPU executable
        size limits.
        """

        z_batch = _as_batch(z, n, dtype)
        params_batch = _as_batch(params, plan.n_parameters, dtype)
        state = _state_or_zeros(state, int(z_batch.shape[0]))
        linearization = linearize_nlp(z_batch, params_batch)
        solve, next_state = solve_qp_data(linearization, state=state)
        direction = solve.direction
        baseline = solve.linearization
        baseline_violation = constraint_violation(
            baseline.constraint_l,
            baseline.g,
            baseline.constraint_u,
            line_search_settings.line_search_constraint_scale,
        )
        armijo_descent_metric = jnp.sum(baseline.q * direction, axis=1)

        candidate_costs = []
        candidate_violations = []
        for step_length in line_search_step_lengths_np:
            candidate_z = z_batch + jnp.asarray(step_length, dtype=jdtype) * direction
            cost, g, lower, upper = evaluate_nlp(candidate_z, params_batch)
            violation = constraint_violation(
                lower,
                g,
                upper,
                line_search_settings.line_search_constraint_scale,
            )
            candidate_costs.append(cost)
            candidate_violations.append(violation)

        candidate_costs_array = jnp.stack(candidate_costs, axis=0)
        candidate_violations_array = jnp.stack(candidate_violations, axis=0)
        line_search = filter_line_search_from_evaluations(
            settings=line_search_settings,
            step_lengths=line_search_step_lengths,
            baseline_cost=baseline.cost,
            baseline_constraint_violation=baseline_violation,
            armijo_descent_metric=armijo_descent_metric,
            candidate_costs=candidate_costs_array,
            candidate_constraint_violations=candidate_violations_array,
        )
        z_next = z_batch + line_search.step_length[:, None] * direction
        result = SQPLineSearchStepResult(
            z_next=z_next,
            solve=solve,
            line_search=line_search,
            is_finite=jnp.all(jnp.isfinite(z_next), axis=1),
        )
        return result, next_state

    return CompiledSparseMPCSQP(
        plan=plan,
        qp_solver=qp_solver,
        osqp=qp_backend,
        dtype=dtype,
        evaluate_nlp=evaluate_nlp,
        linearize_nlp=linearize_nlp,
        build_qp=build_qp,
        lower_evaluate_nlp=lower_evaluate_nlp,
        lower_linearize_nlp=lower_linearize_nlp,
        lower_solve_qp_data=lower_solve_qp_data,
        solve_qp_data=solve_qp_data,
        compute_direction=compute_direction,
        fixed_step=fixed_step,
        step=step,
        step_split_compile=step_split_compile,
        init_state=init_state,
    )


__all__ = [
    "MPAXSettings",
    "SparseMPCProblem",
    "build_sparse_mpc_plan",
    "compile_sparse_mpc_sqp",
]
