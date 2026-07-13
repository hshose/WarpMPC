"""TurboMPC adapters for this repo's CasADi sparse MPC benchmark problems."""

from __future__ import annotations

import copy
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from warpmpc.jax_sqp import SparseMPCProblem


ROOT = pathlib.Path(__file__).resolve().parents[2]
TURBOMPC_ROOT = ROOT / "resources" / "turbompc"
if TURBOMPC_ROOT.exists() and str(TURBOMPC_ROOT) not in sys.path:
    sys.path.insert(0, str(TURBOMPC_ROOT))

from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver
from turbompc.utils.load_params import load_solver_params


_INFTY = 1.0e30


@dataclass(frozen=True)
class DenseStageDescription:
    n_variables: int
    n_constraints: int
    nnz_p: int
    nnz_a: int
    horizon: int
    block_dim: int
    state_dim: int
    control_dim: int
    inequality_dim: int
    stage_var_dims: tuple[int, ...]


def _offsets(widths: list[int] | tuple[int, ...]) -> np.ndarray:
    out = np.zeros(len(widths) + 1, dtype=np.int32)
    out[1:] = np.cumsum(np.asarray(widths, dtype=np.int32))
    return out


def _stage_bound_equality_mask(stage) -> np.ndarray:
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
    lower_samples = np.stack(lowers, axis=0)
    upper_samples = np.stack(uppers, axis=0)
    return np.all(np.isclose(lower_samples, upper_samples, rtol=1e-9, atol=1e-9), axis=0)


def _dense_from_sparse_values(
    stage,
    raw: tuple[Any, ...],
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cost, g, lower, upper, grad_values, hess_values, jac_values = raw
    local_dim = stage.z_dim + (stage.next_z_dim if stage.has_next else 0)
    grad = jnp.zeros((local_dim,), dtype=dtype)
    hess = jnp.zeros((local_dim, local_dim), dtype=dtype)
    jac = jnp.zeros((stage.constraint_dim, local_dim), dtype=dtype)

    grad_rows = jnp.asarray(stage.grad_cols, dtype=jnp.int32)
    hess_rows = jnp.asarray(stage.hess_rows, dtype=jnp.int32)
    hess_cols = jnp.asarray(stage.hess_cols, dtype=jnp.int32)
    jac_rows = jnp.asarray(stage.jac_rows, dtype=jnp.int32)
    jac_cols = jnp.asarray(stage.jac_cols, dtype=jnp.int32)

    grad_values = jnp.asarray(grad_values, dtype=dtype).reshape(-1)
    hess_values = jnp.asarray(hess_values, dtype=dtype).reshape(-1)
    jac_values = jnp.asarray(jac_values, dtype=dtype).reshape(-1)
    grad = grad.at[grad_rows].add(grad_values)
    hess = hess.at[hess_rows, hess_cols].add(hess_values)
    offdiag_values = jnp.where(hess_rows == hess_cols, jnp.asarray(0.0, dtype=dtype), hess_values)
    hess = hess.at[hess_cols, hess_rows].add(offdiag_values)
    jac = jac.at[jac_rows, jac_cols].add(jac_values)
    return (
        jnp.asarray(cost, dtype=dtype).reshape(()),
        jnp.asarray(g, dtype=dtype).reshape(-1),
        jnp.asarray(lower, dtype=dtype).reshape(-1),
        jnp.asarray(upper, dtype=dtype).reshape(-1),
        grad,
        hess,
        jac,
    )


class DenseStageTurbompcProblem(OptimalControlProblem):
    """Present a ``SparseMPCProblem`` as dense time-sparse TurboMPC blocks.

    TurboMPC stores one dense block per time node and couples only neighboring
    nodes.  This adapter keeps the original stage functions and pads variable
    and constraint rows where the benchmark formulation has smaller later
    stages.
    """

    def __init__(
        self,
        sparse_problem: SparseMPCProblem,
        *,
        state_dim: int,
        control_dim: int | None = None,
        name: str = "DenseStageTurbompcProblem",
    ) -> None:
        self._sparse_problem = sparse_problem
        self._stages = tuple(sparse_problem.stages)
        self._horizon = len(self._stages) - 1
        if self._horizon < 1:
            raise ValueError("TurboMPC benchmark problems need at least one interval")

        stage_var_dims = [self._stages[0].z_dim]
        for stage in self._stages[:-1]:
            stage_var_dims.append(stage.next_z_dim)
        for index, (stage, expected) in enumerate(zip(self._stages, stage_var_dims, strict=True)):
            if stage.z_dim != expected:
                raise ValueError(
                    f"stage {index} z_dim={stage.z_dim} does not match expected {expected}"
                )
        if any(width < state_dim for width in stage_var_dims):
            raise ValueError(
                f"all stage widths must contain the {state_dim} state entries: {stage_var_dims}"
            )
        self._stage_var_dims = tuple(int(width) for width in stage_var_dims)
        self._param_dims = tuple(int(stage.param_dim) for stage in self._stages)
        self._z_offsets = _offsets(self._stage_var_dims)
        self._param_offsets = _offsets(self._param_dims)
        self._nx = int(state_dim)
        inferred_nu = max(width - self._nx for width in self._stage_var_dims)
        self._nu = int(inferred_nu if control_dim is None else control_dim)
        if any(width - self._nx > self._nu for width in self._stage_var_dims):
            raise ValueError("control_dim is smaller than one of the stage control widths")
        self._block_dim = self._nx + self._nu
        self._stage_control_dims = tuple(width - self._nx for width in self._stage_var_dims)
        self._name = name
        self._params = {"horizon": self._horizon}
        self._use_slack_variables = False
        self._rescale_optimization_variables = False
        self._constrain_initial_control = False

        self._block_index = tuple(
            np.concatenate(
                [
                    np.arange(self._nx, dtype=np.int32),
                    self._nx + np.arange(control_width, dtype=np.int32),
                ]
            )
            for control_width in self._stage_control_dims
        )

        self._a0_rows: np.ndarray | None = None
        dynamic_rows: list[np.ndarray] = []
        local_rows: list[np.ndarray] = []
        local_counts: list[int] = []
        for stage_index, stage in enumerate(self._stages):
            equality = _stage_bound_equality_mask(stage)
            row_has_next = np.zeros(stage.constraint_dim, dtype=bool)
            if stage.has_next and stage.jac_rows.size:
                next_mask = stage.jac_cols >= stage.z_dim
                if np.any(next_mask):
                    row_has_next[np.unique(stage.jac_rows[next_mask])] = True

            dyn = np.flatnonzero(equality & row_has_next).astype(np.int32)
            if dyn.size > self._nx:
                raise ValueError(
                    f"stage {stage_index} has {dyn.size} cross-stage equality rows, "
                    f"but TurboMPC state_dim is {self._nx}"
                )
            unsupported = np.flatnonzero((~equality) & row_has_next)
            if unsupported.size:
                raise ValueError(
                    f"stage {stage_index} has cross-stage inequality rows, which "
                    "TurboMPC cannot represent faithfully"
                )

            a0 = np.zeros((0,), dtype=np.int32)
            if stage_index == 0:
                candidates = np.flatnonzero(equality & ~row_has_next).astype(np.int32)
                if candidates.size < self._nx:
                    raise ValueError(
                        f"first stage has only {candidates.size} local equality rows; "
                        f"{self._nx} are needed for the initial equality block"
                    )
                a0 = candidates[: self._nx]
                self._a0_rows = a0

            exclude = np.zeros(stage.constraint_dim, dtype=bool)
            exclude[dyn] = True
            exclude[a0] = True
            loc = np.flatnonzero(~exclude).astype(np.int32)
            pad_controls = self._nu - self._stage_control_dims[stage_index]
            dynamic_rows.append(dyn)
            local_rows.append(loc)
            local_counts.append(int(loc.size + pad_controls))

        if self._a0_rows is None:
            raise ValueError("missing first-stage initial equality rows")
        self._dynamic_rows = tuple(dynamic_rows)
        self._local_rows = tuple(local_rows)
        self._num_ineq = max(local_counts) if local_counts else 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def num_state_variables(self) -> int:
        return self._nx

    @property
    def num_control_variables(self) -> int:
        return self._nu

    @property
    def num_variables(self) -> int:
        return (self._horizon + 1) * self._block_dim

    @property
    def num_inequality_constraints(self) -> int:
        return self._num_ineq

    @property
    def use_slack_variables(self) -> bool:
        return self._use_slack_variables

    @property
    def rescale_optimization_variables(self) -> bool:
        return self._rescale_optimization_variables

    @property
    def constrain_initial_control(self) -> bool:
        return self._constrain_initial_control

    def describe_dense_blocks(self) -> DenseStageDescription:
        n_vars = self.num_variables
        n_constraints = self._nx + self._horizon * self._nx + (self._horizon + 1) * self._num_ineq
        return DenseStageDescription(
            n_variables=n_vars,
            n_constraints=n_constraints,
            nnz_p=(self._horizon + 1) * self._block_dim * self._block_dim
            + self._horizon * self._block_dim * self._block_dim,
            nnz_a=self._nx * self._block_dim
            + self._horizon * 2 * self._nx * self._block_dim
            + (self._horizon + 1) * self._num_ineq * self._block_dim,
            horizon=self._horizon,
            block_dim=self._block_dim,
            state_dim=self._nx,
            control_dim=self._nu,
            inequality_dim=self._num_ineq,
            stage_var_dims=self._stage_var_dims,
        )

    def scale_states_controls(
        self, states: jnp.ndarray, controls: jnp.ndarray, params: dict[str, Any]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del params
        return states, controls

    def unscale_states_controls(
        self, states: jnp.ndarray, controls: jnp.ndarray, params: dict[str, Any]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del params
        return states, controls

    def split_packed_z(self, z: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        z = jnp.asarray(z)
        states = []
        controls = []
        for stage_index, width in enumerate(self._stage_var_dims):
            start = int(self._z_offsets[stage_index])
            stage_z = z[start : start + width]
            states.append(stage_z[: self._nx])
            control = jnp.zeros((self._nu,), dtype=z.dtype)
            control_width = self._stage_control_dims[stage_index]
            if control_width:
                control = control.at[:control_width].set(stage_z[self._nx : self._nx + control_width])
            controls.append(control)
        return jnp.stack(states, axis=0), jnp.stack(controls, axis=0)

    def pack_states_controls(self, states: Any, controls: Any) -> jnp.ndarray:
        states = jnp.asarray(states)
        controls = jnp.asarray(controls)
        pieces = []
        for stage_index, control_width in enumerate(self._stage_control_dims):
            if control_width:
                pieces.append(jnp.concatenate([states[stage_index], controls[stage_index, :control_width]]))
            else:
                pieces.append(states[stage_index])
        return jnp.concatenate(pieces, axis=0)

    def initial_guess(self, params: dict[str, Any] | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        if params is None or "initial_guess_z" not in params:
            dtype = jnp.float32
            return (
                jnp.zeros((self._horizon + 1, self._nx), dtype=dtype),
                jnp.zeros((self._horizon + 1, self._nu), dtype=dtype),
            )
        return self.split_packed_z(params["initial_guess_z"])

    def _stage_z(self, states: jnp.ndarray, controls: jnp.ndarray, stage_index: int) -> jnp.ndarray:
        control_width = self._stage_control_dims[stage_index]
        if control_width:
            return jnp.concatenate([states[stage_index], controls[stage_index, :control_width]])
        return states[stage_index]

    def _stage_params(self, params: dict[str, Any], stage_index: int) -> jnp.ndarray:
        packed = params["packed_params"]
        start = int(self._param_offsets[stage_index])
        width = self._param_dims[stage_index]
        return packed[start : start + width]

    def _eval_sparse(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
        stage_index: int,
    ):
        stage = self._stages[stage_index]
        z = self._stage_z(states, controls, stage_index)
        p = self._stage_params(params, stage_index)
        if stage.has_next:
            zn = self._stage_z(states, controls, stage_index + 1)
            return stage.jax_function(z, zn, p)
        return stage.jax_function(z, p)

    def _eval_values(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
        stage_index: int,
    ):
        stage = self._stages[stage_index]
        z = self._stage_z(states, controls, stage_index)
        p = self._stage_params(params, stage_index)
        if stage.has_next:
            zn = self._stage_z(states, controls, stage_index + 1)
            return stage.jax_value_function(z, zn, p)
        return stage.jax_value_function(z, p)

    def _block_matrix_from_actual(
        self, values: jnp.ndarray, stage_index: int
    ) -> jnp.ndarray:
        values = jnp.asarray(values)
        out = jnp.zeros((values.shape[0], self._block_dim), dtype=values.dtype)
        cols = jnp.asarray(self._block_index[stage_index], dtype=jnp.int32)
        return out.at[:, cols].add(values)

    def cost(self, states: jnp.ndarray, controls: jnp.ndarray, params: dict[str, Any]) -> jnp.ndarray:
        dtype = states.dtype
        total = jnp.asarray(0.0, dtype=dtype)
        for stage_index in range(self._horizon + 1):
            cost, *_ = self._eval_values(states, controls, params, stage_index)
            total = total + jnp.asarray(cost, dtype=dtype).reshape(())
        return total

    def get_cost_linearized_blocks(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dtype = states.dtype
        n = self._block_dim
        D = jnp.zeros((self._horizon + 1, n, n), dtype=dtype)
        E = jnp.zeros((self._horizon, n, n), dtype=dtype)
        q = jnp.zeros((self._horizon + 1, n), dtype=dtype)

        for stage_index, stage in enumerate(self._stages):
            raw = self._eval_sparse(states, controls, params, stage_index)
            _, _, _, _, grad, hess, _ = _dense_from_sparse_values(stage, raw, dtype)
            z_cur = self._stage_z(states, controls, stage_index)
            local_z = z_cur
            if stage.has_next:
                local_z = jnp.concatenate([local_z, self._stage_z(states, controls, stage_index + 1)])
            q_local = grad - hess @ local_z

            z_dim = stage.z_dim
            cur_cols = jnp.asarray(self._block_index[stage_index], dtype=jnp.int32)
            D = D.at[stage_index, cur_cols[:, None], cur_cols[None, :]].add(hess[:z_dim, :z_dim])
            q = q.at[stage_index, cur_cols].add(q_local[:z_dim])
            if stage.has_next:
                next_cols = jnp.asarray(self._block_index[stage_index + 1], dtype=jnp.int32)
                D = D.at[stage_index + 1, next_cols[:, None], next_cols[None, :]].add(hess[z_dim:, z_dim:])
                E = E.at[stage_index, next_cols[:, None], cur_cols[None, :]].add(hess[z_dim:, :z_dim])
                q = q.at[stage_index + 1, next_cols].add(q_local[z_dim:])
        return D, E, q

    def get_initial_equality_linearized_matrices(
        self, params: dict[str, Any], dtype: jnp.dtype
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        states, controls = self.initial_guess(params)
        states = states.astype(dtype)
        controls = controls.astype(dtype)
        stage = self._stages[0]
        raw = self._eval_sparse(states, controls, params, 0)
        _, g, lower, _, _, _, jac = _dense_from_sparse_values(stage, raw, dtype)
        rows = jnp.asarray(self._a0_rows, dtype=jnp.int32)
        A0 = self._block_matrix_from_actual(jac[rows, : stage.z_dim], 0)
        x0 = jnp.concatenate([states[0], controls[0]], axis=0)
        c0 = lower[rows] - g[rows] + A0 @ x0
        return A0, c0

    def get_dynamics_linearized_matrices(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dtype = states.dtype
        n = self._block_dim
        As_next = jnp.zeros((self._horizon, self._nx, self._nx), dtype=dtype)
        Bs_next = jnp.zeros((self._horizon, self._nx, self._nu), dtype=dtype)
        As = jnp.zeros((self._horizon, self._nx, self._nx), dtype=dtype)
        Bs = jnp.zeros((self._horizon, self._nx, self._nu), dtype=dtype)
        Cs = jnp.zeros((self._horizon + 1, self._nx), dtype=dtype)
        Cs = Cs.at[0].set(states[0])

        x_blocks = jnp.concatenate([states, controls], axis=-1)
        for stage_index in range(self._horizon):
            stage = self._stages[stage_index]
            rows_np = self._dynamic_rows[stage_index]
            if rows_np.size == 0:
                continue
            raw = self._eval_sparse(states, controls, params, stage_index)
            _, g, lower, _, _, _, jac = _dense_from_sparse_values(stage, raw, dtype)
            rows = jnp.asarray(rows_np, dtype=jnp.int32)
            row_count = int(rows_np.size)
            z_dim = stage.z_dim
            A_cur = self._block_matrix_from_actual(jac[rows, :z_dim], stage_index)
            A_next = self._block_matrix_from_actual(jac[rows, z_dim:], stage_index + 1)
            c_rows = lower[rows] - g[rows] + A_cur @ x_blocks[stage_index] + A_next @ x_blocks[stage_index + 1]
            As = As.at[stage_index, :row_count, :].set(A_cur[:, : self._nx])
            Bs = Bs.at[stage_index, :row_count, :].set(A_cur[:, self._nx :])
            As_next = As_next.at[stage_index, :row_count, :].set(A_next[:, : self._nx])
            Bs_next = Bs_next.at[stage_index, :row_count, :].set(A_next[:, self._nx :])
            Cs = Cs.at[stage_index + 1, :row_count].set(c_rows)
        return As_next, Bs_next, As, Bs, Cs

    def get_inequalities_linearized_matrices(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dtype = states.dtype
        n = self._block_dim
        G = jnp.zeros((self._horizon + 1, self._num_ineq, n), dtype=dtype)
        lower_all = -jnp.asarray(_INFTY, dtype=dtype) * jnp.ones((self._horizon + 1, self._num_ineq), dtype=dtype)
        upper_all = jnp.asarray(_INFTY, dtype=dtype) * jnp.ones((self._horizon + 1, self._num_ineq), dtype=dtype)
        x_blocks = jnp.concatenate([states, controls], axis=-1)

        for stage_index, stage in enumerate(self._stages):
            rows_np = self._local_rows[stage_index]
            cursor = 0
            if rows_np.size:
                raw = self._eval_sparse(states, controls, params, stage_index)
                _, g, lower, upper, _, _, jac = _dense_from_sparse_values(stage, raw, dtype)
                rows = jnp.asarray(rows_np, dtype=jnp.int32)
                row_count = int(rows_np.size)
                A = self._block_matrix_from_actual(jac[rows, : stage.z_dim], stage_index)
                offset = g[rows] - A @ x_blocks[stage_index]
                G = G.at[stage_index, :row_count, :].set(A)
                lower_all = lower_all.at[stage_index, :row_count].set(lower[rows] - offset)
                upper_all = upper_all.at[stage_index, :row_count].set(upper[rows] - offset)
                cursor = row_count

            control_width = self._stage_control_dims[stage_index]
            for control_index in range(control_width, self._nu):
                col = self._nx + control_index
                G = G.at[stage_index, cursor, col].set(jnp.asarray(1.0, dtype=dtype))
                lower_all = lower_all.at[stage_index, cursor].set(jnp.asarray(0.0, dtype=dtype))
                upper_all = upper_all.at[stage_index, cursor].set(jnp.asarray(0.0, dtype=dtype))
                cursor += 1
        return G, lower_all, upper_all

    def inequality_constraints(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dtype = states.dtype
        values = jnp.zeros((self._horizon + 1, self._num_ineq), dtype=dtype)
        lower_all = -jnp.asarray(_INFTY, dtype=dtype) * jnp.ones_like(values)
        upper_all = jnp.asarray(_INFTY, dtype=dtype) * jnp.ones_like(values)
        for stage_index, stage in enumerate(self._stages):
            rows_np = self._local_rows[stage_index]
            cursor = 0
            if rows_np.size:
                cost, g, lower, upper = self._eval_values(states, controls, params, stage_index)
                del cost
                rows = jnp.asarray(rows_np, dtype=jnp.int32)
                row_count = int(rows_np.size)
                values = values.at[stage_index, :row_count].set(jnp.asarray(g, dtype=dtype).reshape(-1)[rows])
                lower_all = lower_all.at[stage_index, :row_count].set(jnp.asarray(lower, dtype=dtype).reshape(-1)[rows])
                upper_all = upper_all.at[stage_index, :row_count].set(jnp.asarray(upper, dtype=dtype).reshape(-1)[rows])
                cursor = row_count

            control_width = self._stage_control_dims[stage_index]
            for control_index in range(control_width, self._nu):
                values = values.at[stage_index, cursor].set(controls[stage_index, control_index])
                lower_all = lower_all.at[stage_index, cursor].set(jnp.asarray(0.0, dtype=dtype))
                upper_all = upper_all.at[stage_index, cursor].set(jnp.asarray(0.0, dtype=dtype))
                cursor += 1
        return values, lower_all, upper_all

    def equality_constraints(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
    ) -> jnp.ndarray:
        dtype = states.dtype
        parts = []
        for stage_index, stage in enumerate(self._stages):
            if stage_index == 0:
                rows_np = self._a0_rows
            elif stage_index <= self._horizon:
                rows_np = self._dynamic_rows[stage_index - 1] if stage_index - 1 < self._horizon else np.zeros((0,), dtype=np.int32)
            else:
                rows_np = np.zeros((0,), dtype=np.int32)
            if rows_np is None or rows_np.size == 0:
                continue
            eval_stage = 0 if stage_index == 0 else stage_index - 1
            eval_obj = self._stages[eval_stage]
            _, g, lower, _ = self._eval_values(states, controls, params, eval_stage)
            rows = jnp.asarray(rows_np, dtype=jnp.int32)
            residual = jnp.asarray(g, dtype=dtype).reshape(-1)[rows] - jnp.asarray(lower, dtype=dtype).reshape(-1)[rows]
            if stage_index > 0 and residual.shape[0] < self._nx:
                residual = jnp.pad(residual, (0, self._nx - residual.shape[0]))
            parts.append(residual)
        if not parts:
            return jnp.zeros((0,), dtype=dtype)
        return jnp.concatenate(parts, axis=0)


def make_problem_params(packed_params: Any, initial_guess_z: Any | None = None) -> dict[str, Any]:
    out = {"packed_params": jnp.asarray(packed_params)}
    if initial_guess_z is not None:
        out["initial_guess_z"] = jnp.asarray(initial_guess_z)
    return out


def make_solver_params(
    *,
    sqp_iterations: int,
    admm_max_iter: int,
    rho: float,
    sigma: float,
    alpha: float,
    eps_abs: float,
    eps_rel: float,
    line_search_step_min: float,
    line_search_step_decay: float = 0.5,
    fixed_sqp_iterations: bool = True,
) -> dict[str, Any]:
    params = copy.deepcopy(load_solver_params("turbompc.yaml"))
    params["num_sqp_iteration_max"] = int(sqp_iterations)
    params["tol_convergence"] = -1.0 if fixed_sqp_iterations else max(float(eps_abs), float(eps_rel))
    params["convergence_criterion"] = "step"
    params["linesearch"] = True
    step = 1.0
    alphas: list[float] = []
    while step + max(1e-12, 1e-12 * line_search_step_min) >= line_search_step_min:
        alphas.append(float(step))
        step *= line_search_step_decay
    params["linesearch_alphas"] = list(reversed(alphas))
    params["admm"]["rho"] = float(rho)
    params["admm"]["sigma"] = float(sigma)
    params["admm"]["max_iter"] = int(admm_max_iter)
    params["admm"]["eps_abs"] = float(eps_abs)
    params["admm"]["eps_rel"] = float(eps_rel)
    params["admm"]["relaxation_parameter"] = float(alpha)
    params["admm"]["check_termination_every"] = int(admm_max_iter) + 1
    params["admm"]["adapt_rho_every"] = int(admm_max_iter) + 1
    return params


def make_solver(
    problem: DenseStageTurbompcProblem,
    solver_params: dict[str, Any],
    *,
    forward_backend: str = "admm_fused_cudss",
    backward_backend: str = "direct_cudss_ffi",
) -> TurboMPCSolver:
    return TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=forward_backend,
        backward_backend=backward_backend,
        use_full_hessian=False,
    )


def initial_admm_state_batch(
    solver: TurboMPCSolver,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    packed_params: jnp.ndarray,
    *,
    rho: float,
):
    def init_one(x, u, p):
        params = make_problem_params(p)
        qp_data = solver._build_qp_data(x, u, params)
        return solver._admm_solver_fwd.initial_state(
            qp_data,
            rho_bar=rho,
            states0=x,
            controls0=u,
        )

    return jax.vmap(init_one)(states, controls, packed_params)
