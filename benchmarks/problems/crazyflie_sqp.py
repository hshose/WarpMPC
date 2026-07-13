"""Crazyflie MPC SQP problem using Euler angles and variable shooting times."""

from __future__ import annotations

import casadi as ca
import jax.numpy as jnp
import numpy as np

from warpmpc.jax_sqp import CasadiStageFunction, SparseMPCProblem


CRAZYFLIE_G = 9.8066
CRAZYFLIE_M = 40.2e-3
CRAZYFLIE_IXX = 1.8e-5
CRAZYFLIE_IYY = 1.8e-5
CRAZYFLIE_IZZ = 3.6e-5
CRAZYFLIE_ARM = 65e-3 / 2.0
CRAZYFLIE_THRUST_MAX = 0.18
CRAZYFLIE_THRUST_TO_TORQUE = 0.0051648627905205285
CRAZYFLIE_ACTION_SCALING = np.array([1e-2, 1e-2, 1e-3, 1.0])
CRAZYFLIE_NX = 12
CRAZYFLIE_NU = 4
CRAZYFLIE_NZ = CRAZYFLIE_NX + CRAZYFLIE_NU
CRAZYFLIE_N_STEPS = 25

CRAZYFLIE_Q = np.diag(
    [
        200.0,
        200.0,
        100.0,
        1.0,
        1.0,
        1000.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        10.0,
    ]
) / 200.0
CRAZYFLIE_R = np.diag([200.0, 200.0, 10.0, 100.0]) / 200.0


def crazyflie_dt_schedule(n_steps: int = CRAZYFLIE_N_STEPS) -> np.ndarray:
    """Variable shooting step sizes ``0.01 * 1.05**i``."""

    return 0.01 * 1.05 ** np.arange(n_steps, dtype=np.float64)


def _diag_quadratic(diagonal: np.ndarray, residual) -> ca.SX:
    out = ca.SX(0)
    for i, value in enumerate(diagonal):
        if value != 0.0:
            out = out + 0.5 * float(value) * residual[i] ** 2
    return out


def _euler_body_z_axis(phi, theta, psi):
    cphi = ca.cos(phi)
    sphi = ca.sin(phi)
    ctheta = ca.cos(theta)
    stheta = ca.sin(theta)
    cpsi = ca.cos(psi)
    spsi = ca.sin(psi)
    return (
        cpsi * stheta * cphi + spsi * sphi,
        spsi * stheta * cphi - cpsi * sphi,
        ctheta * cphi,
    )


def _crazyflie_continuous_dynamics(x, u):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vx = x[6]
    vy = x[7]
    vz = x[8]
    wx = x[9]
    wy = x[10]
    wz = x[11]

    mx = u[0] * CRAZYFLIE_ACTION_SCALING[0]
    my = u[1] * CRAZYFLIE_ACTION_SCALING[1]
    mz = u[2] * CRAZYFLIE_ACTION_SCALING[2]
    thrust = u[3] * CRAZYFLIE_ACTION_SCALING[3] + CRAZYFLIE_M * CRAZYFLIE_G

    r13, r23, r33 = _euler_body_z_axis(phi, theta, psi)
    dvx = r13 * thrust / CRAZYFLIE_M
    dvy = r23 * thrust / CRAZYFLIE_M
    dvz = -CRAZYFLIE_G + r33 * thrust / CRAZYFLIE_M

    tan_theta = ca.tan(theta)
    inv_ctheta = 1.0 / ca.cos(theta)
    cphi = ca.cos(phi)
    sphi = ca.sin(phi)
    dphi = wx + sphi * tan_theta * wy + cphi * tan_theta * wz
    dtheta = cphi * wy - sphi * wz
    dpsi = sphi * inv_ctheta * wy + cphi * inv_ctheta * wz

    b_drag = 5e-6
    dwx = (mx - (CRAZYFLIE_IZZ - CRAZYFLIE_IYY) * wy * wz - b_drag * wx) / CRAZYFLIE_IXX
    dwy = (my - (CRAZYFLIE_IXX - CRAZYFLIE_IZZ) * wx * wz - b_drag * wy) / CRAZYFLIE_IYY
    dwz = (mz - (CRAZYFLIE_IYY - CRAZYFLIE_IXX) * wx * wy - b_drag * wz) / CRAZYFLIE_IZZ

    return ca.vertcat(vx, vy, vz, dphi, dtheta, dpsi, dvx, dvy, dvz, dwx, dwy, dwz)


def _crazyflie_euler_step(x, u, dt):
    return x + dt * _crazyflie_continuous_dynamics(x, u)


def _motor_mixing(u):
    inv_arm = 1.0 / CRAZYFLIE_ARM
    inv_k = 1.0 / CRAZYFLIE_THRUST_TO_TORQUE
    mixer = 0.25 * np.array(
        [
            [-inv_arm, -inv_arm, -inv_k, 1.0],
            [-inv_arm, +inv_arm, +inv_k, 1.0],
            [+inv_arm, +inv_arm, -inv_k, 1.0],
            [+inv_arm, -inv_arm, +inv_k, 1.0],
        ]
    )
    wrench = ca.vertcat(
        CRAZYFLIE_ACTION_SCALING[0] * u[0],
        CRAZYFLIE_ACTION_SCALING[1] * u[1],
        CRAZYFLIE_ACTION_SCALING[2] * u[2],
        CRAZYFLIE_ACTION_SCALING[3] * u[3] + CRAZYFLIE_M * CRAZYFLIE_G,
    )
    out = []
    for row in range(4):
        value = ca.SX(0)
        for col in range(4):
            value = value + float(mixer[row, col]) * wrench[col]
        out.append(value)
    return ca.vertcat(*out)


def _split_stage(z):
    return z[:CRAZYFLIE_NX], z[CRAZYFLIE_NX:]


def _stage_cost(x, u, x_ref, dt):
    return dt * (
        _diag_quadratic(np.diag(CRAZYFLIE_Q), x - x_ref)
        + _diag_quadratic(np.diag(CRAZYFLIE_R), u)
    )


def make_crazyflie_sqp_problem(n_steps: int = CRAZYFLIE_N_STEPS) -> SparseMPCProblem:
    """Create a Crazyflie sparse MPC SQP problem with Euler-angle dynamics."""

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
        lower = ca.vertcat(np.zeros(CRAZYFLIE_NX), np.zeros(CRAZYFLIE_NX), np.zeros(CRAZYFLIE_NU))
        upper = ca.vertcat(
            np.zeros(CRAZYFLIE_NX),
            np.zeros(CRAZYFLIE_NX),
            CRAZYFLIE_THRUST_MAX * np.ones(CRAZYFLIE_NU),
        )
        return ca.Function("crazyflie_first", [z, zn, p], [cost, g, lower, upper])

    def make_middle() -> ca.Function:
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
        upper = ca.vertcat(np.zeros(CRAZYFLIE_NX), CRAZYFLIE_THRUST_MAX * np.ones(CRAZYFLIE_NU))
        return ca.Function("crazyflie_middle", [z, zn, p], [cost, g, lower, upper])

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("zN", CRAZYFLIE_NZ)
        p = ca.SX.sym("pN", CRAZYFLIE_NX)
        x, u = _split_stage(z)
        cost = _diag_quadratic(np.diag(CRAZYFLIE_Q), x - p)
        g = u
        lower = np.zeros(CRAZYFLIE_NU)
        upper = np.zeros(CRAZYFLIE_NU)
        return ca.Function("crazyflie_terminal", [z, p], [cost, g, lower, upper])

    first = CasadiStageFunction.from_function(make_first(), has_next=True)
    middle = CasadiStageFunction.from_function(make_middle(), has_next=True)
    terminal = CasadiStageFunction.from_function(make_terminal(), has_next=False)
    return SparseMPCProblem.from_stage_functions(
        horizon=n_steps,
        first=first,
        intermediate=middle,
        terminal=terminal,
    )


def crazyflie_initial_guess_and_params(
    x0: np.ndarray,
    *,
    n_steps: int = CRAZYFLIE_N_STEPS,
    dtype: np.dtype | str = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a warm-start trajectory and packed stage parameters from initial states."""

    dtype = np.dtype(dtype)
    x0 = np.asarray(x0, dtype=dtype)
    if x0.ndim == 1:
        x0 = x0[None, :]
    batch = x0.shape[0]
    dt_schedule = crazyflie_dt_schedule(n_steps).astype(dtype)
    x_ref = np.zeros((batch, CRAZYFLIE_NX), dtype=dtype)
    z = np.zeros((batch, (n_steps + 1) * CRAZYFLIE_NZ), dtype=dtype)
    for stage in range(n_steps + 1):
        frac = stage / max(1, n_steps)
        x_guess = (1.0 - frac) * x0 + frac * x_ref
        offset = stage * CRAZYFLIE_NZ
        z[:, offset : offset + CRAZYFLIE_NX] = x_guess

    param_dim = (2 * CRAZYFLIE_NX + 1) + (n_steps - 1) * (CRAZYFLIE_NX + 1) + CRAZYFLIE_NX
    params = np.zeros((batch, param_dim), dtype=dtype)
    offset = 0
    params[:, offset : offset + CRAZYFLIE_NX] = x0
    offset += CRAZYFLIE_NX
    params[:, offset : offset + CRAZYFLIE_NX] = x_ref
    offset += CRAZYFLIE_NX
    params[:, offset] = dt_schedule[0]
    offset += 1
    for i in range(1, n_steps):
        params[:, offset : offset + CRAZYFLIE_NX] = x_ref
        offset += CRAZYFLIE_NX
        params[:, offset] = dt_schedule[i]
        offset += 1
    params[:, offset : offset + CRAZYFLIE_NX] = x_ref
    return z, params


def update_crazyflie_params_initial_state(params: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Return packed parameters with the first-stage initial state replaced."""

    params = np.array(params, copy=True)
    x0 = np.asarray(x0, dtype=params.dtype)
    if x0.ndim == 1:
        x0 = x0[None, :]
    params[:, :CRAZYFLIE_NX] = x0
    return params


def crazyflie_jax_dynamics(x, u):
    """Nominal continuous-time Crazyflie dynamics in JAX."""

    dtype = x.dtype
    action_scaling = jnp.asarray(CRAZYFLIE_ACTION_SCALING, dtype=dtype)
    mass = jnp.asarray(CRAZYFLIE_M, dtype=dtype)
    gravity = jnp.asarray(CRAZYFLIE_G, dtype=dtype)
    ixx = jnp.asarray(CRAZYFLIE_IXX, dtype=dtype)
    iyy = jnp.asarray(CRAZYFLIE_IYY, dtype=dtype)
    izz = jnp.asarray(CRAZYFLIE_IZZ, dtype=dtype)
    drag = jnp.asarray(5e-6, dtype=dtype)
    phi = x[:, 3]
    theta = x[:, 4]
    psi = x[:, 5]
    wx = x[:, 9]
    wy = x[:, 10]
    wz = x[:, 11]
    thrust = u[:, 3] * action_scaling[3] + mass * gravity
    cphi = jnp.cos(phi)
    sphi = jnp.sin(phi)
    ctheta = jnp.cos(theta)
    stheta = jnp.sin(theta)
    cpsi = jnp.cos(psi)
    spsi = jnp.sin(psi)
    r13 = cpsi * stheta * cphi + spsi * sphi
    r23 = spsi * stheta * cphi - cpsi * sphi
    r33 = ctheta * cphi

    tan_theta = jnp.tan(theta)
    inv_ctheta = 1.0 / ctheta
    dphi = wx + sphi * tan_theta * wy + cphi * tan_theta * wz
    dtheta = cphi * wy - sphi * wz
    dpsi = sphi * inv_ctheta * wy + cphi * inv_ctheta * wz
    mx = u[:, 0] * action_scaling[0]
    my = u[:, 1] * action_scaling[1]
    mz = u[:, 2] * action_scaling[2]

    dx = jnp.zeros_like(x)
    dx = dx.at[:, 0:3].set(x[:, 6:9])
    dx = dx.at[:, 3].set(dphi)
    dx = dx.at[:, 4].set(dtheta)
    dx = dx.at[:, 5].set(dpsi)
    dx = dx.at[:, 6].set(r13 * thrust / mass)
    dx = dx.at[:, 7].set(r23 * thrust / mass)
    dx = dx.at[:, 8].set(-gravity + r33 * thrust / mass)
    dx = dx.at[:, 9].set((mx - (izz - iyy) * wy * wz - drag * wx) / ixx)
    dx = dx.at[:, 10].set((my - (ixx - izz) * wx * wz - drag * wy) / iyy)
    dx = dx.at[:, 11].set((mz - (iyy - ixx) * wx * wy - drag * wz) / izz)
    return dx


def crazyflie_jax_euler_step(x, u, dt: float = 0.01):
    """One nominal Euler integration step in JAX."""

    return x + dt * crazyflie_jax_dynamics(x, u)


__all__ = [
    "CRAZYFLIE_N_STEPS",
    "CRAZYFLIE_NU",
    "CRAZYFLIE_NX",
    "CRAZYFLIE_NZ",
    "crazyflie_dt_schedule",
    "crazyflie_initial_guess_and_params",
    "crazyflie_jax_dynamics",
    "crazyflie_jax_euler_step",
    "make_crazyflie_sqp_problem",
    "update_crazyflie_params_initial_state",
]
