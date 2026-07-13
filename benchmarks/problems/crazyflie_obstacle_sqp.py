"""Crazyflie MPC SQP problem with fixed vertical cylinder obstacles."""

from __future__ import annotations

import casadi as ca
import jax.numpy as jnp
import numpy as np

from benchmarks.problems.crazyflie_sqp import (
    CRAZYFLIE_NX,
    CRAZYFLIE_NU,
    CRAZYFLIE_NZ,
    CRAZYFLIE_Q,
    CRAZYFLIE_THRUST_MAX,
    _crazyflie_euler_step,
    _diag_quadratic,
    _motor_mixing,
    _split_stage,
    _stage_cost,
    crazyflie_initial_guess_and_params,
)
from warpmpc.jax_sqp import CasadiStageFunction, SparseMPCProblem


CRAZYFLIE_OBSTACLE_UPPER = 1.0e30
CRAZYFLIE_OBSTACLE_SAFETY_RADIUS = 0.10
CRAZYFLIE_CYLINDER_OBSTACLES = np.array(
    [
        [0.58, 0.42, 0.17],
        [-0.62, 0.56, 0.22],
        [0.55, -0.62, 0.15],
        [-0.78, -0.35, 0.19],
    ],
    dtype=np.float64,
)


def crazyflie_obstacle_inflated_radii() -> np.ndarray:
    return CRAZYFLIE_CYLINDER_OBSTACLES[:, 2] + CRAZYFLIE_OBSTACLE_SAFETY_RADIUS


def _obstacle_distance_squared_expr(x) -> ca.SX:
    values = []
    for cx, cy, _radius in CRAZYFLIE_CYLINDER_OBSTACLES:
        dx = x[0] - float(cx)
        dy = x[1] - float(cy)
        values.append(dx * dx + dy * dy)
    return ca.vertcat(*values)


def _obstacle_lower() -> np.ndarray:
    return crazyflie_obstacle_inflated_radii() ** 2


def _obstacle_upper() -> np.ndarray:
    return CRAZYFLIE_OBSTACLE_UPPER * np.ones(CRAZYFLIE_CYLINDER_OBSTACLES.shape[0])


def make_crazyflie_obstacle_sqp_problem(
    n_steps: int,
    *,
    obstacle_constraint_scale: float = 1.0,
) -> SparseMPCProblem:
    """Create a Crazyflie sparse MPC SQP problem with cylinder avoidance rows."""

    obstacle_constraint_scale = float(obstacle_constraint_scale)
    if obstacle_constraint_scale <= 0.0:
        raise ValueError("obstacle constraint scale must be positive")
    obstacle_lower = obstacle_constraint_scale * _obstacle_lower()
    obstacle_upper = _obstacle_upper()

    def scaled_obstacle_distance_squared_expr(x) -> ca.SX:
        return obstacle_constraint_scale * _obstacle_distance_squared_expr(x)

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
            scaled_obstacle_distance_squared_expr(x),
        )
        lower = ca.vertcat(
            np.zeros(CRAZYFLIE_NX),
            np.zeros(CRAZYFLIE_NX),
            np.zeros(CRAZYFLIE_NU),
            obstacle_lower,
        )
        upper = ca.vertcat(
            np.zeros(CRAZYFLIE_NX),
            np.zeros(CRAZYFLIE_NX),
            CRAZYFLIE_THRUST_MAX * np.ones(CRAZYFLIE_NU),
            obstacle_upper,
        )
        return ca.Function("crazyflie_obstacle_first", [z, zn, p], [cost, g, lower, upper])

    def make_middle() -> ca.Function:
        z = ca.SX.sym("z", CRAZYFLIE_NZ)
        zn = ca.SX.sym("zn", CRAZYFLIE_NZ)
        p = ca.SX.sym("p", CRAZYFLIE_NX + 1)
        x, u = _split_stage(z)
        x_next, _ = _split_stage(zn)
        x_ref = p[:CRAZYFLIE_NX]
        dt = p[CRAZYFLIE_NX]
        cost = _stage_cost(x, u, x_ref, dt)
        g = ca.vertcat(
            x_next - _crazyflie_euler_step(x, u, dt),
            _motor_mixing(u),
            scaled_obstacle_distance_squared_expr(x),
        )
        lower = ca.vertcat(np.zeros(CRAZYFLIE_NX), np.zeros(CRAZYFLIE_NU), obstacle_lower)
        upper = ca.vertcat(
            np.zeros(CRAZYFLIE_NX),
            CRAZYFLIE_THRUST_MAX * np.ones(CRAZYFLIE_NU),
            obstacle_upper,
        )
        return ca.Function("crazyflie_obstacle_middle", [z, zn, p], [cost, g, lower, upper])

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("zN", CRAZYFLIE_NZ)
        p = ca.SX.sym("pN", CRAZYFLIE_NX)
        x, u = _split_stage(z)
        cost = _diag_quadratic(np.diag(CRAZYFLIE_Q), x - p)
        g = ca.vertcat(u, scaled_obstacle_distance_squared_expr(x))
        lower = ca.vertcat(np.zeros(CRAZYFLIE_NU), obstacle_lower)
        upper = ca.vertcat(np.zeros(CRAZYFLIE_NU), obstacle_upper)
        return ca.Function("crazyflie_obstacle_terminal", [z, p], [cost, g, lower, upper])

    first = CasadiStageFunction.from_function(make_first(), has_next=True)
    middle = CasadiStageFunction.from_function(make_middle(), has_next=True)
    terminal = CasadiStageFunction.from_function(make_terminal(), has_next=False)
    return SparseMPCProblem.from_stage_functions(
        horizon=n_steps,
        first=first,
        intermediate=middle,
        terminal=terminal,
    )


def crazyflie_obstacle_initial_guess_and_params(
    x0: np.ndarray,
    *,
    n_steps: int,
    dtype: np.dtype | str = np.float64,
    trajectory_initialization: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    z, params = crazyflie_initial_guess_and_params(x0, n_steps=n_steps, dtype=dtype)
    if trajectory_initialization == "linear":
        return z, params
    if trajectory_initialization != "initial_state":
        raise ValueError(f"unsupported trajectory initialization: {trajectory_initialization!r}")

    x0_arr = np.asarray(x0, dtype=np.dtype(dtype))
    if x0_arr.ndim == 1:
        x0_arr = x0_arr[None, :]
    stages = z.reshape((x0_arr.shape[0], n_steps + 1, CRAZYFLIE_NZ))
    stages[:, :, :CRAZYFLIE_NX] = x0_arr[:, None, :]
    return z, params


def sample_crazyflie_obstacle_initial_states(
    batch_size: int,
    seed: int,
    dtype: np.dtype | str,
    *,
    xy_limit: float = 1.15,
    z_min: float = -0.35,
    z_max: float = 0.75,
    extra_clearance: float = 0.04,
) -> np.ndarray:
    """Sample standstill poses with random yaw outside all inflated cylinders."""

    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    x0 = np.zeros((batch_size, CRAZYFLIE_NX), dtype=dtype)
    centers = CRAZYFLIE_CYLINDER_OBSTACLES[:, :2]
    safe_radii = crazyflie_obstacle_inflated_radii() + float(extra_clearance)
    filled = 0
    while filled < batch_size:
        need = batch_size - filled
        count = max(64, 3 * need)
        xy = rng.uniform(-xy_limit, xy_limit, size=(count, 2))
        dist = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)
        ok = np.all(dist >= safe_radii[None, :], axis=1)
        accepted = xy[ok][:need]
        n_accept = accepted.shape[0]
        if n_accept == 0:
            continue
        sl = slice(filled, filled + n_accept)
        x0[sl, 0:2] = accepted.astype(dtype)
        x0[sl, 2] = rng.uniform(z_min, z_max, size=n_accept).astype(dtype)
        x0[sl, 5] = rng.uniform(-np.pi, np.pi, size=n_accept).astype(dtype)
        filled += n_accept
    return x0


def sample_crazyflie_obstacle_border_initial_states(
    batch_size: int,
    seed: int,
    dtype: np.dtype | str,
    *,
    xy_limit: float = 1.15,
    z_min: float = -0.35,
    z_max: float = 0.75,
    edge_jitter: float = 0.04,
    extra_clearance: float = 0.04,
) -> np.ndarray:
    """Sample standstill poses near the xy boundary while avoiding inflated cylinders."""

    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    x0 = np.zeros((batch_size, CRAZYFLIE_NX), dtype=dtype)
    centers = CRAZYFLIE_CYLINDER_OBSTACLES[:, :2]
    safe_radii = crazyflie_obstacle_inflated_radii() + float(extra_clearance)
    filled = 0
    attempts = 0
    samples_per_side = max(1, (batch_size + 3) // 4)
    while filled < batch_size:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("could not sample collision-free border initial states")
        candidate = attempts - 1
        side = candidate % 4
        edge_slot = (candidate // 4) % samples_per_side
        along = -xy_limit + 2.0 * xy_limit * ((edge_slot + 0.5) / samples_per_side)
        along += rng.uniform(-edge_jitter, edge_jitter)
        inward = rng.uniform(0.0, edge_jitter)
        if side == 0:
            xy = np.array([-xy_limit + inward, along], dtype=np.float64)
        elif side == 1:
            xy = np.array([xy_limit - inward, along], dtype=np.float64)
        elif side == 2:
            xy = np.array([along, -xy_limit + inward], dtype=np.float64)
        else:
            xy = np.array([along, xy_limit - inward], dtype=np.float64)
        xy = np.clip(xy, -xy_limit, xy_limit)
        dist = np.linalg.norm(xy[None, :] - centers, axis=1)
        if not np.all(dist >= safe_radii):
            continue
        x0[filled, 0:2] = xy.astype(dtype)
        x0[filled, 2] = np.asarray(rng.uniform(z_min, z_max), dtype=dtype)
        x0[filled, 5] = np.asarray(rng.uniform(-np.pi, np.pi), dtype=dtype)
        filled += 1
    return x0


def crazyflie_obstacle_squared_margins(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states)
    centers = CRAZYFLIE_CYLINDER_OBSTACLES[:, :2]
    inflated = crazyflie_obstacle_inflated_radii()
    xy = states[..., :2]
    dist_sq = np.sum((xy[..., None, :] - centers) ** 2, axis=-1)
    return dist_sq - inflated**2


def crazyflie_obstacle_penetration(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states)
    centers = CRAZYFLIE_CYLINDER_OBSTACLES[:, :2]
    inflated = crazyflie_obstacle_inflated_radii()
    xy = states[..., :2]
    dist = np.linalg.norm(xy[..., None, :] - centers, axis=-1)
    return np.maximum(inflated - dist, 0.0)


def crazyflie_obstacle_squared_violation_jax(x):
    centers = jnp.asarray(CRAZYFLIE_CYLINDER_OBSTACLES[:, :2], dtype=x.dtype)
    inflated = jnp.asarray(crazyflie_obstacle_inflated_radii(), dtype=x.dtype)
    xy = x[..., :2]
    dist_sq = jnp.sum((xy[..., None, :] - centers) ** 2, axis=-1)
    return jnp.maximum(inflated**2 - dist_sq, 0.0)


__all__ = [
    "CRAZYFLIE_CYLINDER_OBSTACLES",
    "CRAZYFLIE_OBSTACLE_SAFETY_RADIUS",
    "crazyflie_obstacle_inflated_radii",
    "crazyflie_obstacle_initial_guess_and_params",
    "crazyflie_obstacle_penetration",
    "sample_crazyflie_obstacle_border_initial_states",
    "crazyflie_obstacle_squared_margins",
    "crazyflie_obstacle_squared_violation_jax",
    "make_crazyflie_obstacle_sqp_problem",
    "sample_crazyflie_obstacle_initial_states",
]
