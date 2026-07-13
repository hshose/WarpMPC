"""TurboMPC-autodiff adapters for value-only CasADi MPC stages.

This is a parallel implementation to ``turbompc_adapter.py``.  It intentionally
does not build or convert CasADi gradient, Hessian, or Jacobian functions.
Instead, the CasADi stages are converted only as value functions
``cost, g, lower, upper`` and JAX autodiff creates the QP linearization blocks.
"""

from __future__ import annotations

import copy
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from warpmpc.numpysadi import convert as numpysadi_convert


ROOT = pathlib.Path(__file__).resolve().parents[2]
TURBOMPC_ROOT = ROOT / "resources" / "turbompc"
if TURBOMPC_ROOT.exists() and str(TURBOMPC_ROOT) not in sys.path:
    sys.path.insert(0, str(TURBOMPC_ROOT))

from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver
from turbompc.utils.load_params import load_solver_params


_INFTY = 1.0e30


def _flat_dim(function: ca.Function, index: int, *, is_input: bool) -> int:
    if is_input:
        return int(function.size1_in(index) * function.size2_in(index))
    return int(function.size1_out(index) * function.size2_out(index))


def _dense_column(expr) -> ca.SX:
    return ca.densify(ca.reshape(expr, int(expr.numel()), 1))


def _as_expanded_sx_function(function: ca.Function) -> ca.Function:
    try:
        return function.expand()
    except Exception as exc:  # pragma: no cover - CasADi exception type varies.
        raise ValueError("stage function could not be expanded to SX") from exc


@dataclass(frozen=True)
class ValueCasadiStageFunction:
    """CasADi stage converted only as values, with no derivative outputs."""

    name: str
    function: ca.Function
    has_next: bool
    z_dim: int
    next_z_dim: int
    param_dim: int
    constraint_dim: int
    row_has_next: np.ndarray
    value_function: ca.Function
    jax_value_function: Callable

    @classmethod
    def from_function(
        cls,
        function: ca.Function,
        *,
        has_next: bool | None = None,
        name: str | None = None,
        compile_jax: bool = True,
    ) -> "ValueCasadiStageFunction":
        """Create a value-only JAX evaluator from a CasADi stage function."""

        function_sx = _as_expanded_sx_function(function)
        if function_sx.n_out() != 4:
            raise ValueError("stage function must return exactly cost, g, l, u")
        if has_next is None:
            has_next = function_sx.n_in() == 3
        expected_inputs = 3 if has_next else 2
        if function_sx.n_in() != expected_inputs:
            raise ValueError(
                f"expected {expected_inputs} inputs for has_next={has_next}, "
                f"got {function_sx.n_in()}"
            )

        z_dim = _flat_dim(function_sx, 0, is_input=True)
        next_z_dim = _flat_dim(function_sx, 1, is_input=True) if has_next else 0
        param_index = 2 if has_next else 1
        param_dim = _flat_dim(function_sx, param_index, is_input=True)
        if _flat_dim(function_sx, 0, is_input=False) != 1:
            raise ValueError("stage cost output must be scalar")
        constraint_dim = _flat_dim(function_sx, 1, is_input=False)
        if _flat_dim(function_sx, 2, is_input=False) != constraint_dim:
            raise ValueError("stage lower-bound output must match g dimension")
        if _flat_dim(function_sx, 3, is_input=False) != constraint_dim:
            raise ValueError("stage upper-bound output must match g dimension")

        z = ca.SX.sym("z", z_dim)
        p = ca.SX.sym("p", param_dim)
        if has_next:
            zn = ca.SX.sym("zn", next_z_dim)
            cost, g, lower, upper = function_sx(z, zn, p)
            inputs = [z, zn, p]
            row_has_next = np.asarray(ca.which_depends(g, zn, 1, True), dtype=bool)
        else:
            cost, g, lower, upper = function_sx(z, p)
            inputs = [z, p]
            row_has_next = np.zeros(constraint_dim, dtype=bool)

        cost = ca.reshape(cost, 1, 1)
        g = ca.reshape(g, constraint_dim, 1)
        lower = ca.reshape(lower, constraint_dim, 1)
        upper = ca.reshape(upper, constraint_dim, 1)

        stage_name = name or function_sx.name()
        value_function = ca.Function(
            f"{stage_name}_values_only",
            inputs,
            [
                _dense_column(cost),
                _dense_column(g),
                _dense_column(lower),
                _dense_column(upper),
            ],
        )
        print(
            "ValueCasadiStageFunction instructions:",
            f"stage={stage_name}",
            f"value_function={value_function.n_instructions()}",
            flush=True,
        )
        jax_value_function = numpysadi_convert(value_function, jit=compile_jax)
        return cls(
            name=stage_name,
            function=function_sx,
            has_next=bool(has_next),
            z_dim=z_dim,
            next_z_dim=next_z_dim,
            param_dim=param_dim,
            constraint_dim=constraint_dim,
            row_has_next=row_has_next,
            value_function=value_function,
            jax_value_function=jax_value_function,
        )


@dataclass(frozen=True)
class ValueSparseMPCProblem:
    """A stage-ordered value-only nonlinear MPC problem."""

    stages: tuple[ValueCasadiStageFunction, ...]

    @classmethod
    def from_stage_functions(
        cls,
        *,
        horizon: int,
        first: ValueCasadiStageFunction,
        intermediate: ValueCasadiStageFunction | Sequence[ValueCasadiStageFunction],
        terminal: ValueCasadiStageFunction,
    ) -> "ValueSparseMPCProblem":
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if isinstance(intermediate, ValueCasadiStageFunction):
            middle = [intermediate] * max(0, horizon - 1)
        else:
            middle = list(intermediate)
            if len(middle) != max(0, horizon - 1):
                raise ValueError("intermediate sequence must have length horizon - 1")
        return cls(stages=(first, *middle, terminal))


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


def _stage_bound_equality_mask(stage: ValueCasadiStageFunction) -> np.ndarray:
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


class AutodiffDenseStageTurbompcProblem(OptimalControlProblem):
    """Present value-only stages as dense TurboMPC blocks using JAX autodiff."""

    def __init__(
        self,
        value_problem: ValueSparseMPCProblem,
        *,
        state_dim: int,
        control_dim: int | None = None,
        name: str = "AutodiffDenseStageTurbompcProblem",
    ) -> None:
        self._value_problem = value_problem
        self._stages = tuple(value_problem.stages)
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
            row_has_next = stage.row_has_next

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

    def _local_z(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        stage_index: int,
    ) -> jnp.ndarray:
        stage = self._stages[stage_index]
        local_z = self._stage_z(states, controls, stage_index)
        if stage.has_next:
            local_z = jnp.concatenate([local_z, self._stage_z(states, controls, stage_index + 1)])
        return local_z

    def _stage_values_from_local(
        self,
        stage: ValueCasadiStageFunction,
        local_z: jnp.ndarray,
        p: jnp.ndarray,
    ):
        if stage.has_next:
            return stage.jax_value_function(local_z[: stage.z_dim], local_z[stage.z_dim :], p)
        return stage.jax_value_function(local_z, p)

    def _eval_values(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
        stage_index: int,
    ):
        stage = self._stages[stage_index]
        local_z = self._local_z(states, controls, stage_index)
        p = self._stage_params(params, stage_index)
        return self._stage_values_from_local(stage, local_z, p)

    def _cost_grad_hess(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
        stage_index: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        stage = self._stages[stage_index]
        local_z = self._local_z(states, controls, stage_index)
        p = self._stage_params(params, stage_index)

        def cost_fn(local):
            cost, *_ = self._stage_values_from_local(stage, local, p)
            return jnp.asarray(cost, dtype=local.dtype).reshape(())

        return jax.grad(cost_fn)(local_z), jax.hessian(cost_fn)(local_z)

    def _constraints_jacobian(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: dict[str, Any],
        stage_index: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        stage = self._stages[stage_index]
        local_z = self._local_z(states, controls, stage_index)
        p = self._stage_params(params, stage_index)

        def g_fn(local):
            _, g, _, _ = self._stage_values_from_local(stage, local, p)
            return jnp.asarray(g, dtype=local.dtype).reshape(-1)

        _, g, lower, upper = self._stage_values_from_local(stage, local_z, p)
        jac = jax.jacfwd(g_fn)(local_z)
        dtype = local_z.dtype
        return (
            jnp.asarray(g, dtype=dtype).reshape(-1),
            jnp.asarray(lower, dtype=dtype).reshape(-1),
            jnp.asarray(upper, dtype=dtype).reshape(-1),
            jac,
        )

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
            grad, hess = self._cost_grad_hess(states, controls, params, stage_index)
            local_z = self._local_z(states, controls, stage_index)
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
        g, lower, _, jac = self._constraints_jacobian(states, controls, params, 0)
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
            g, lower, _, jac = self._constraints_jacobian(states, controls, params, stage_index)
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
                g, lower, upper, jac = self._constraints_jacobian(states, controls, params, stage_index)
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
        for stage_index, _stage in enumerate(self._stages):
            if stage_index == 0:
                rows_np = self._a0_rows
            elif stage_index <= self._horizon:
                rows_np = self._dynamic_rows[stage_index - 1] if stage_index - 1 < self._horizon else np.zeros((0,), dtype=np.int32)
            else:
                rows_np = np.zeros((0,), dtype=np.int32)
            if rows_np is None or rows_np.size == 0:
                continue
            eval_stage = 0 if stage_index == 0 else stage_index - 1
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
    problem: AutodiffDenseStageTurbompcProblem,
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
