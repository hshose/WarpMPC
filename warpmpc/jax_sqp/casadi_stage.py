"""CasADi-backed stage functions with sparse derivative outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import casadi as ca
import numpy as np
from warpmpc.numpysadi import convert as numpysadi_convert


def _flat_dim(function: ca.Function, index: int, *, is_input: bool) -> int:
    if is_input:
        return int(function.size1_in(index) * function.size2_in(index))
    return int(function.size1_out(index) * function.size2_out(index))


def _entries(expr, rows: np.ndarray, cols: np.ndarray):
    if rows.size == 0:
        return ca.SX.zeros(0, 1)
    return ca.vertcat(*[expr[int(r), int(c)] for r, c in zip(rows, cols, strict=True)])


def _dense_column(expr) -> ca.SX:
    """Return a dense column SX expression for C-code-based conversion."""

    return ca.densify(ca.reshape(expr, int(expr.numel()), 1))


def _as_expanded_sx_function(function: ca.Function) -> ca.Function:
    """Return an SXFunction so sparsity is extracted from scalar SX graphs."""

    try:
        return function.expand()
    except Exception as exc:  # pragma: no cover - CasADi exception type varies.
        raise ValueError("stage function could not be expanded to SX") from exc


@dataclass(frozen=True)
class CasadiStageFunction:
    """One reusable stage function for a fixed-pattern MPC SQP problem.

    The underlying CasADi function must return ``cost, g, l, u``. Non-terminal
    stage functions have inputs ``z_stage, z_next, p_stage``. Terminal functions
    have inputs ``z_stage, p_stage``.
    """

    name: str
    function: ca.Function
    has_next: bool
    z_dim: int
    next_z_dim: int
    param_dim: int
    constraint_dim: int
    grad_cols: np.ndarray
    hess_rows: np.ndarray
    hess_cols: np.ndarray
    jac_rows: np.ndarray
    jac_cols: np.ndarray
    value_function: ca.Function
    jax_value_function: Callable
    sparse_function: ca.Function
    jax_function: Callable

    @classmethod
    def from_function(
        cls,
        function: ca.Function,
        *,
        has_next: bool | None = None,
        name: str | None = None,
        compile_jax: bool = True,
    ) -> "CasadiStageFunction":
        """Create sparse-value evaluators from a CasADi stage function."""

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
            local_variables = ca.vertcat(z, zn)
        else:
            cost, g, lower, upper = function_sx(z, p)
            inputs = [z, p]
            local_variables = z

        cost = ca.sparsify(ca.reshape(cost, 1, 1))
        g = ca.sparsify(ca.reshape(g, constraint_dim, 1))
        lower = ca.sparsify(ca.reshape(lower, constraint_dim, 1))
        upper = ca.sparsify(ca.reshape(upper, constraint_dim, 1))
        grad = ca.sparsify(ca.gradient(cost, local_variables))
        hess = ca.sparsify(ca.hessian(cost, local_variables)[0])
        jac = ca.sparsify(ca.jacobian(g, local_variables))

        grad_rows, grad_cols = grad.sparsity().get_triplet()
        grad_rows = np.asarray(grad_rows, dtype=np.int32)
        grad_cols = np.asarray(grad_cols, dtype=np.int32)
        if grad_cols.size and np.any(grad_cols != 0):
            raise ValueError("cost gradient is expected to be a column vector")

        hess_rows_all, hess_cols_all = hess.sparsity().get_triplet()
        hess_rows_all = np.asarray(hess_rows_all, dtype=np.int32)
        hess_cols_all = np.asarray(hess_cols_all, dtype=np.int32)
        hess_upper = hess_rows_all <= hess_cols_all
        hess_rows = hess_rows_all[hess_upper]
        hess_cols = hess_cols_all[hess_upper]

        jac_rows, jac_cols = jac.sparsity().get_triplet()
        jac_rows = np.asarray(jac_rows, dtype=np.int32)
        jac_cols = np.asarray(jac_cols, dtype=np.int32)

        stage_name = name or function_sx.name()
        value_function = ca.Function(
            f"{stage_name}_values",
            inputs,
            [
                _dense_column(cost),
                _dense_column(g),
                _dense_column(lower),
                _dense_column(upper),
            ],
        )
        sparse_function = ca.Function(
            f"{stage_name}_sparse_values",
            inputs,
            [
                _dense_column(cost),
                _dense_column(g),
                _dense_column(lower),
                _dense_column(upper),
                _dense_column(_entries(grad, grad_rows, np.zeros_like(grad_rows))),
                _dense_column(_entries(hess, hess_rows, hess_cols)),
                _dense_column(_entries(jac, jac_rows, jac_cols)),
            ],
        )
        print(
            "CasadiStageFunction instructions:",
            f"stage={stage_name}",
            f"value_function={value_function.n_instructions()}",
            f"sparse_function={sparse_function.n_instructions()}",
            flush=True,
        )
        jax_value_function = numpysadi_convert(value_function, jit=compile_jax)
        jax_function = numpysadi_convert(sparse_function, jit=compile_jax)
        return cls(
            name=stage_name,
            function=function_sx,
            has_next=bool(has_next),
            z_dim=z_dim,
            next_z_dim=next_z_dim,
            param_dim=param_dim,
            constraint_dim=constraint_dim,
            grad_cols=grad_rows,
            hess_rows=hess_rows,
            hess_cols=hess_cols,
            jac_rows=jac_rows,
            jac_cols=jac_cols,
            value_function=value_function,
            jax_value_function=jax_value_function,
            sparse_function=sparse_function,
            jax_function=jax_function,
        )


__all__ = ["CasadiStageFunction"]
