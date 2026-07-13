"""Mixed full/simplified Crazyflie MPC SQP example."""

from __future__ import annotations

import casadi as ca
import numpy as np

from warpmpc.jax_sqp import CasadiStageFunction, SparseMPCProblem

from .crazyflie_sqp import (
    CRAZYFLIE_NU,
    CRAZYFLIE_NX,
    CRAZYFLIE_NZ,
    CRAZYFLIE_Q,
    CRAZYFLIE_THRUST_MAX,
    _crazyflie_euler_step,
    _diag_quadratic,
    _motor_mixing,
    _split_stage,
    _stage_cost,
    crazyflie_dt_schedule,
)


MIXED_FULL_MODEL_NODES = 25
MIXED_SIMPLE_MODEL_NODES = 25
MIXED_HORIZON = MIXED_FULL_MODEL_NODES + MIXED_SIMPLE_MODEL_NODES - 1
MIXED_SIMPLE_NX = 6
MIXED_SIMPLE_NU = 3
MIXED_SIMPLE_NZ = MIXED_SIMPLE_NX + MIXED_SIMPLE_NU
MIXED_SIMPLE_ACCEL_LIMIT = 5.0

MIXED_SIMPLE_Q = np.diag(
    [
        CRAZYFLIE_Q[0, 0],
        CRAZYFLIE_Q[1, 1],
        CRAZYFLIE_Q[2, 2],
        CRAZYFLIE_Q[6, 6],
        CRAZYFLIE_Q[7, 7],
        CRAZYFLIE_Q[8, 8],
    ]
)
MIXED_SIMPLE_R = 0.02 * np.eye(MIXED_SIMPLE_NU)


def mixed_crazyflie_dt_schedule(
    *,
    full_model_nodes: int = MIXED_FULL_MODEL_NODES,
    simple_model_nodes: int = MIXED_SIMPLE_MODEL_NODES,
) -> np.ndarray:
    """Variable shooting step sizes for the mixed-horizon intervals."""

    return crazyflie_dt_schedule(full_model_nodes + simple_model_nodes - 1)


def mixed_crazyflie_stage_dims(
    *,
    full_model_nodes: int = MIXED_FULL_MODEL_NODES,
    simple_model_nodes: int = MIXED_SIMPLE_MODEL_NODES,
) -> tuple[int, ...]:
    """Return the stage variable widths for full and simplified nodes."""

    return (CRAZYFLIE_NZ,) * full_model_nodes + (MIXED_SIMPLE_NZ,) * simple_model_nodes


def mixed_crazyflie_z_offsets(
    *,
    full_model_nodes: int = MIXED_FULL_MODEL_NODES,
    simple_model_nodes: int = MIXED_SIMPLE_MODEL_NODES,
) -> np.ndarray:
    """Return offsets for a packed mixed-model trajectory."""

    dims = np.asarray(
        mixed_crazyflie_stage_dims(
            full_model_nodes=full_model_nodes,
            simple_model_nodes=simple_model_nodes,
        ),
        dtype=np.int32,
    )
    out = np.zeros(dims.size + 1, dtype=np.int32)
    out[1:] = np.cumsum(dims)
    return out


def _split_simple_stage(z):
    return z[:MIXED_SIMPLE_NX], z[MIXED_SIMPLE_NX:]


def _project_full_to_simple(x):
    return ca.vertcat(x[0], x[1], x[2], x[6], x[7], x[8])


def _simple_step(x, u, dt):
    position = x[:3]
    velocity = x[3:]
    return ca.vertcat(position + dt * velocity, velocity + dt * u)


def _simple_stage_cost(x, u, x_ref, dt):
    return dt * (
        _diag_quadratic(np.diag(MIXED_SIMPLE_Q), x - x_ref)
        + _diag_quadratic(np.diag(MIXED_SIMPLE_R), u)
    )


def _simple_terminal_cost(x, x_ref):
    return _diag_quadratic(2.0 * np.diag(MIXED_SIMPLE_Q), x - x_ref)


def _validate_node_counts(full_model_nodes: int, simple_model_nodes: int) -> None:
    if full_model_nodes < 2:
        raise ValueError("full_model_nodes must be at least 2")
    if simple_model_nodes < 2:
        raise ValueError("simple_model_nodes must be at least 2")


def make_mixed_crazyflie_sqp_problem(
    *,
    full_model_nodes: int = MIXED_FULL_MODEL_NODES,
    simple_model_nodes: int = MIXED_SIMPLE_MODEL_NODES,
) -> SparseMPCProblem:
    """Create an MPC problem with full dynamics first and simple dynamics later."""

    _validate_node_counts(full_model_nodes, simple_model_nodes)
    horizon = full_model_nodes + simple_model_nodes - 1

    def make_first() -> ca.Function:
        z = ca.SX.sym("z0", CRAZYFLIE_NZ)
        zn = ca.SX.sym("z1", CRAZYFLIE_NZ)
        p = ca.SX.sym("p0", 2 * CRAZYFLIE_NX + 1)
        x, u = _split_stage(z)
        x_next, _ = _split_stage(zn)
        x0 = p[:CRAZYFLIE_NX]
        x_ref = p[CRAZYFLIE_NX : 2 * CRAZYFLIE_NX]
        dt = p[2 * CRAZYFLIE_NX]
        cost = _stage_cost(x, u, x_ref, dt)
        g = ca.vertcat(
            x - x0,
            x_next - _crazyflie_euler_step(x, u, dt),
            _motor_mixing(u),
        )
        lower = ca.vertcat(
            np.zeros(CRAZYFLIE_NX),
            np.zeros(CRAZYFLIE_NX),
            np.zeros(CRAZYFLIE_NU),
        )
        upper = ca.vertcat(
            np.zeros(CRAZYFLIE_NX),
            np.zeros(CRAZYFLIE_NX),
            CRAZYFLIE_THRUST_MAX * np.ones(CRAZYFLIE_NU),
        )
        return ca.Function("mixed_crazyflie_first", [z, zn, p], [cost, g, lower, upper])

    def make_full_middle() -> ca.Function:
        z = ca.SX.sym("z", CRAZYFLIE_NZ)
        zn = ca.SX.sym("zn", CRAZYFLIE_NZ)
        p = ca.SX.sym("p", CRAZYFLIE_NX + 1)
        x, u = _split_stage(z)
        x_next, _ = _split_stage(zn)
        x_ref = p[:CRAZYFLIE_NX]
        dt = p[CRAZYFLIE_NX]
        cost = _stage_cost(x, u, x_ref, dt)
        g = ca.vertcat(x_next - _crazyflie_euler_step(x, u, dt), _motor_mixing(u))
        lower = ca.vertcat(np.zeros(CRAZYFLIE_NX), np.zeros(CRAZYFLIE_NU))
        upper = ca.vertcat(
            np.zeros(CRAZYFLIE_NX),
            CRAZYFLIE_THRUST_MAX * np.ones(CRAZYFLIE_NU),
        )
        return ca.Function(
            "mixed_crazyflie_full_middle",
            [z, zn, p],
            [cost, g, lower, upper],
        )

    def make_bridge() -> ca.Function:
        z = ca.SX.sym("z_bridge", CRAZYFLIE_NZ)
        zn = ca.SX.sym("zn_simple", MIXED_SIMPLE_NZ)
        p = ca.SX.sym("p_bridge", CRAZYFLIE_NX + 1)
        x, u = _split_stage(z)
        x_next_simple, _ = _split_simple_stage(zn)
        x_ref = p[:CRAZYFLIE_NX]
        dt = p[CRAZYFLIE_NX]
        cost = _stage_cost(x, u, x_ref, dt)
        projected_next = _project_full_to_simple(_crazyflie_euler_step(x, u, dt))
        g = ca.vertcat(x_next_simple - projected_next, _motor_mixing(u))
        lower = ca.vertcat(np.zeros(MIXED_SIMPLE_NX), np.zeros(CRAZYFLIE_NU))
        upper = ca.vertcat(
            np.zeros(MIXED_SIMPLE_NX),
            CRAZYFLIE_THRUST_MAX * np.ones(CRAZYFLIE_NU),
        )
        return ca.Function("mixed_crazyflie_bridge", [z, zn, p], [cost, g, lower, upper])

    def make_simple_middle() -> ca.Function:
        z = ca.SX.sym("z_simple", MIXED_SIMPLE_NZ)
        zn = ca.SX.sym("zn_simple", MIXED_SIMPLE_NZ)
        p = ca.SX.sym("p_simple", MIXED_SIMPLE_NX + 1)
        x, u = _split_simple_stage(z)
        x_next, _ = _split_simple_stage(zn)
        x_ref = p[:MIXED_SIMPLE_NX]
        dt = p[MIXED_SIMPLE_NX]
        cost = _simple_stage_cost(x, u, x_ref, dt)
        g = ca.vertcat(x_next - _simple_step(x, u, dt), u)
        lower = ca.vertcat(
            np.zeros(MIXED_SIMPLE_NX),
            -MIXED_SIMPLE_ACCEL_LIMIT * np.ones(MIXED_SIMPLE_NU),
        )
        upper = ca.vertcat(
            np.zeros(MIXED_SIMPLE_NX),
            MIXED_SIMPLE_ACCEL_LIMIT * np.ones(MIXED_SIMPLE_NU),
        )
        return ca.Function(
            "mixed_crazyflie_simple_middle",
            [z, zn, p],
            [cost, g, lower, upper],
        )

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("z_terminal", MIXED_SIMPLE_NZ)
        p = ca.SX.sym("p_terminal", MIXED_SIMPLE_NX)
        x, u = _split_simple_stage(z)
        cost = _simple_terminal_cost(x, p)
        g = u
        lower = np.zeros(MIXED_SIMPLE_NU)
        upper = np.zeros(MIXED_SIMPLE_NU)
        return ca.Function("mixed_crazyflie_terminal", [z, p], [cost, g, lower, upper])

    first = CasadiStageFunction.from_function(make_first(), has_next=True)
    full_middle = CasadiStageFunction.from_function(make_full_middle(), has_next=True)
    bridge = CasadiStageFunction.from_function(make_bridge(), has_next=True)
    simple_middle = CasadiStageFunction.from_function(make_simple_middle(), has_next=True)
    terminal = CasadiStageFunction.from_function(make_terminal(), has_next=False)
    intermediate = (
        [full_middle] * (full_model_nodes - 2)
        + [bridge]
        + [simple_middle] * (simple_model_nodes - 1)
    )
    return SparseMPCProblem.from_stage_functions(
        horizon=horizon,
        first=first,
        intermediate=intermediate,
        terminal=terminal,
    )


def _project_full_batch_to_simple(x: np.ndarray) -> np.ndarray:
    return x[:, [0, 1, 2, 6, 7, 8]]


def mixed_crazyflie_initial_guess_and_params(
    x0: np.ndarray,
    *,
    full_model_nodes: int = MIXED_FULL_MODEL_NODES,
    simple_model_nodes: int = MIXED_SIMPLE_MODEL_NODES,
    dtype: np.dtype | str = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a packed mixed-model trajectory and stage-parameter vector."""

    _validate_node_counts(full_model_nodes, simple_model_nodes)
    dtype = np.dtype(dtype)
    x0 = np.asarray(x0, dtype=dtype)
    if x0.ndim == 1:
        x0 = x0[None, :]
    batch = x0.shape[0]
    horizon = full_model_nodes + simple_model_nodes - 1
    dt_schedule = crazyflie_dt_schedule(horizon).astype(dtype)
    full_ref = np.zeros((batch, CRAZYFLIE_NX), dtype=dtype)
    simple_ref = np.zeros((batch, MIXED_SIMPLE_NX), dtype=dtype)
    offsets = mixed_crazyflie_z_offsets(
        full_model_nodes=full_model_nodes,
        simple_model_nodes=simple_model_nodes,
    )
    z = np.zeros((batch, int(offsets[-1])), dtype=dtype)

    for node in range(full_model_nodes):
        frac = node / max(1, horizon)
        x_guess = (1.0 - frac) * x0 + frac * full_ref
        start = int(offsets[node])
        z[:, start : start + CRAZYFLIE_NX] = x_guess

    for simple_index in range(simple_model_nodes):
        node = full_model_nodes + simple_index
        frac = node / max(1, horizon)
        x_guess = (1.0 - frac) * x0 + frac * full_ref
        simple_guess = _project_full_batch_to_simple(x_guess)
        start = int(offsets[node])
        z[:, start : start + MIXED_SIMPLE_NX] = simple_guess

    param_dim = (
        (2 * CRAZYFLIE_NX + 1)
        + (full_model_nodes - 2) * (CRAZYFLIE_NX + 1)
        + (CRAZYFLIE_NX + 1)
        + (simple_model_nodes - 1) * (MIXED_SIMPLE_NX + 1)
        + MIXED_SIMPLE_NX
    )
    params = np.zeros((batch, param_dim), dtype=dtype)
    offset = 0
    params[:, offset : offset + CRAZYFLIE_NX] = x0
    offset += CRAZYFLIE_NX
    params[:, offset : offset + CRAZYFLIE_NX] = full_ref
    offset += CRAZYFLIE_NX
    params[:, offset] = dt_schedule[0]
    offset += 1

    for stage in range(1, full_model_nodes - 1):
        params[:, offset : offset + CRAZYFLIE_NX] = full_ref
        offset += CRAZYFLIE_NX
        params[:, offset] = dt_schedule[stage]
        offset += 1

    bridge_stage = full_model_nodes - 1
    params[:, offset : offset + CRAZYFLIE_NX] = full_ref
    offset += CRAZYFLIE_NX
    params[:, offset] = dt_schedule[bridge_stage]
    offset += 1

    for stage in range(full_model_nodes, horizon):
        params[:, offset : offset + MIXED_SIMPLE_NX] = simple_ref
        offset += MIXED_SIMPLE_NX
        params[:, offset] = dt_schedule[stage]
        offset += 1

    params[:, offset : offset + MIXED_SIMPLE_NX] = simple_ref
    return z, params


def update_mixed_crazyflie_params_initial_state(params: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Return mixed-stage parameters with the first full-state parameter updated."""

    params = np.array(params, copy=True)
    x0 = np.asarray(x0, dtype=params.dtype)
    if x0.ndim == 1:
        x0 = x0[None, :]
    params[:, :CRAZYFLIE_NX] = x0
    return params


__all__ = [
    "MIXED_FULL_MODEL_NODES",
    "MIXED_HORIZON",
    "MIXED_SIMPLE_MODEL_NODES",
    "MIXED_SIMPLE_NU",
    "MIXED_SIMPLE_NX",
    "MIXED_SIMPLE_NZ",
    "make_mixed_crazyflie_sqp_problem",
    "mixed_crazyflie_dt_schedule",
    "mixed_crazyflie_initial_guess_and_params",
    "mixed_crazyflie_stage_dims",
    "mixed_crazyflie_z_offsets",
    "update_mixed_crazyflie_params_initial_state",
]
