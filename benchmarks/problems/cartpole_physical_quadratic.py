"""Pure-quadratic identified-parameter cartpole sparse SQP MPC problem."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from warpmpc.jax_sqp import CasadiStageFunction, SparseMPCProblem


CARTPOLE_G = 9.81
CARTPOLE_M_CART = 0.4472965
CARTPOLE_M_ROD = 0.01840051
CARTPOLE_M_TIP = 0.09
CARTPOLE_L_ROD = 0.6696171
CARTPOLE_J_MOT_EQ = 0.0
CARTPOLE_AB = -1.959361
CARTPOLE_AC = 0.5374264
CARTPOLE_B_EQ = 1.959361
CARTPOLE_B_P = 0.00172243
CARTPOLE_M_ADD = 0.0

CARTPOLE_NX = 4
CARTPOLE_NU = 1
CARTPOLE_NZ = CARTPOLE_NX + CARTPOLE_NU
CARTPOLE_N_STEPS = 100
CARTPOLE_DT_START = 100e-3
CARTPOLE_DT_GROWTH = 1.0
CARTPOLE_U_MIN = -9.0
CARTPOLE_U_MAX = 9.0
CARTPOLE_RAIL_LIMIT = 0.35

CARTPOLE_M = CARTPOLE_M_CART
CARTPOLE_m = CARTPOLE_M_ROD + CARTPOLE_M_TIP
CARTPOLE_l = (CARTPOLE_M_ROD * (CARTPOLE_L_ROD / 2.0) + CARTPOLE_M_TIP * CARTPOLE_L_ROD) / CARTPOLE_m
CARTPOLE_J_PIVOT = CARTPOLE_M_ROD * CARTPOLE_L_ROD**2 / 3.0 + CARTPOLE_M_TIP * CARTPOLE_L_ROD**2
CARTPOLE_J = CARTPOLE_J_PIVOT - CARTPOLE_m * CARTPOLE_l**2
CARTPOLE_AB_MINUS_BEQ = CARTPOLE_AB - CARTPOLE_B_EQ


CARTPOLE_COST_PARAMETER_NAMES = (
    "stage_q_x",
    "stage_q_v",
    "stage_q_theta",
    "stage_q_omega",
    "stage_r_u",
    "terminal_q_x",
    "terminal_q_v",
    "terminal_q_theta",
    "terminal_q_omega",
    "tube_rho",
    "tube_w",
    "tighten_u",
    "tighten_x",
)
CARTPOLE_N_COST_PARAMS = len(CARTPOLE_COST_PARAMETER_NAMES)
CARTPOLE_N_QUADRATIC_COST_PARAMS = 9
CARTPOLE_TUBE_RHO_INDEX = 9
CARTPOLE_TUBE_W_INDEX = 10
CARTPOLE_TIGHTEN_U_INDEX = 11
CARTPOLE_TIGHTEN_X_INDEX = 12

CARTPOLE_NOMINAL_COST_PARAMS = np.asarray(
    [
        1.0,
        2.0,
        1.5,
        1.0,
        5.0e-2,
        1.0,
        2.0,
        1.5,
        1.0,
        2.0,
        5.0e-3,
        1.0,
        0.1,
    ],
    dtype=np.float64,
)

CARTPOLE_COST_PARAM_LOWER = np.full(CARTPOLE_N_COST_PARAMS, 1.0e-4, dtype=np.float64)
CARTPOLE_COST_PARAM_UPPER = np.full(CARTPOLE_N_COST_PARAMS, 1.0e4, dtype=np.float64)
CARTPOLE_COST_PARAM_LOWER[CARTPOLE_TUBE_RHO_INDEX] = 0.2
CARTPOLE_COST_PARAM_UPPER[CARTPOLE_TUBE_RHO_INDEX] = 20.0
CARTPOLE_COST_PARAM_LOWER[CARTPOLE_TUBE_W_INDEX] = 1.0e-5
CARTPOLE_COST_PARAM_UPPER[CARTPOLE_TUBE_W_INDEX] = 0.1
CARTPOLE_COST_PARAM_LOWER[CARTPOLE_TIGHTEN_U_INDEX] = 1.0e-5
CARTPOLE_COST_PARAM_UPPER[CARTPOLE_TIGHTEN_U_INDEX] = 10.0
CARTPOLE_COST_PARAM_LOWER[CARTPOLE_TIGHTEN_X_INDEX] = 1.0e-5
CARTPOLE_COST_PARAM_UPPER[CARTPOLE_TIGHTEN_X_INDEX] = 1.0


@dataclass(frozen=True)
class CartpoleCostParameterSpec:
    names: tuple[str, ...]
    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def cartpole_cost_parameter_spec() -> CartpoleCostParameterSpec:
    return CartpoleCostParameterSpec(
        names=CARTPOLE_COST_PARAMETER_NAMES,
        nominal=CARTPOLE_NOMINAL_COST_PARAMS.copy(),
        lower=CARTPOLE_COST_PARAM_LOWER.copy(),
        upper=CARTPOLE_COST_PARAM_UPPER.copy(),
    )


def cartpole_dt_schedule(
    n_steps: int = CARTPOLE_N_STEPS,
    *,
    dt_start: float = CARTPOLE_DT_START,
    dt_growth: float = CARTPOLE_DT_GROWTH,
) -> np.ndarray:
    return dt_start * dt_growth ** np.arange(n_steps, dtype=np.float64)


def _split_stage(z):
    state = z[:CARTPOLE_NX]
    action = z[CARTPOLE_NX : CARTPOLE_NX + CARTPOLE_NU]
    return state, action


def _cartpole_continuous_dynamics_ca(state, action):
    _, v, theta, omega = state[0], state[1], state[2], state[3]
    u = action[0]
    h1 = CARTPOLE_M + CARTPOLE_m
    h2 = CARTPOLE_m * CARTPOLE_l
    h4 = CARTPOLE_m * CARTPOLE_l**2 + CARTPOLE_J
    h7 = CARTPOLE_m * CARTPOLE_l * CARTPOLE_G
    force = CARTPOLE_AB_MINUS_BEQ * v + CARTPOLE_AC * u
    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    denominator = h2**2 * cos_theta**2 - h1 * h4
    vdot = (
        h2 * h4 * omega**2 * sin_theta
        - h2 * h7 * cos_theta * sin_theta
        + h4 * force
        - h2 * cos_theta * CARTPOLE_B_P * omega
    ) / (-denominator)
    omegadot = (
        h2**2 * omega**2 * cos_theta * sin_theta
        - h1 * h7 * sin_theta
        + h2 * cos_theta * force
        + h1 * CARTPOLE_B_P * omega
    ) / denominator
    return ca.vertcat(v, vdot, omega, omegadot)


def _cartpole_euler_step_ca(state, action, dt):
    return state + dt * _cartpole_continuous_dynamics_ca(state, action)


def _write_stage_guess(z: np.ndarray, stage: int, state: np.ndarray, action: np.ndarray | None) -> None:
    offset = stage * CARTPOLE_NZ
    z[:, offset : offset + CARTPOLE_NX] = state
    if action is not None:
        z[:, offset + CARTPOLE_NX : offset + CARTPOLE_NX + CARTPOLE_NU] = action


def _stage_cost_ca(state, action, dt, cost_params):
    x = state[0]
    v = state[1]
    theta = state[2]
    omega = state[3]
    u = action[0]
    return dt * 0.5 * (
        cost_params[0] * x**2
        + cost_params[1] * v**2
        + cost_params[2] * theta**2
        + cost_params[3] * omega**2
        + cost_params[4] * u**2
    )


def _terminal_cost_ca(state, dt, cost_params):
    x = state[0]
    v = state[1]
    theta = state[2]
    omega = state[3]
    return dt * 0.5 * (
        cost_params[5] * x**2
        + cost_params[6] * v**2
        + cost_params[7] * theta**2
        + cost_params[8] * omega**2
    )


def _tightened_action_bounds_ca(cost_params, tube_s):
    tightening = cost_params[CARTPOLE_TIGHTEN_U_INDEX] * tube_s
    lower = CARTPOLE_U_MIN + tightening
    upper = CARTPOLE_U_MAX - tightening
    return lower, upper


def _tightened_rail_bounds_ca(cost_params, tube_s):
    tightening = cost_params[CARTPOLE_TIGHTEN_X_INDEX] * tube_s
    lower = -CARTPOLE_RAIL_LIMIT + tightening
    upper = CARTPOLE_RAIL_LIMIT - tightening
    return lower, upper


def make_cartpole_sqp_problem(
    n_steps: int = CARTPOLE_N_STEPS,
    *,
    rail_constraint: bool = False,
) -> SparseMPCProblem:
    if n_steps < 1:
        raise ValueError("cartpole MPC needs at least one shooting interval")

    def make_first() -> ca.Function:
        z = ca.SX.sym("cartpole_phys_quad_z0", CARTPOLE_NZ)
        zn = ca.SX.sym("cartpole_phys_quad_z1", CARTPOLE_NZ)
        p = ca.SX.sym("cartpole_phys_quad_p0", CARTPOLE_NX + 1 + CARTPOLE_N_COST_PARAMS + 1)
        state, action = _split_stage(z)
        next_state, _ = _split_stage(zn)
        x0 = p[:CARTPOLE_NX]
        dt = p[CARTPOLE_NX]
        cost_params = p[CARTPOLE_NX + 1 : CARTPOLE_NX + 1 + CARTPOLE_N_COST_PARAMS]
        tube_s = p[CARTPOLE_NX + 1 + CARTPOLE_N_COST_PARAMS]
        cost = _stage_cost_ca(state, action, dt, cost_params)
        action_lower, action_upper = _tightened_action_bounds_ca(cost_params, tube_s)
        g = ca.vertcat(
            state - x0,
            next_state - _cartpole_euler_step_ca(state, action, dt),
            action[0],
        )
        lower = ca.vertcat(np.zeros(CARTPOLE_NX), np.zeros(CARTPOLE_NX), action_lower)
        upper = ca.vertcat(np.zeros(CARTPOLE_NX), np.zeros(CARTPOLE_NX), action_upper)
        if rail_constraint:
            rail_lower, rail_upper = _tightened_rail_bounds_ca(cost_params, tube_s)
            g = ca.vertcat(g, state[0])
            lower = ca.vertcat(lower, rail_lower)
            upper = ca.vertcat(upper, rail_upper)
        return ca.Function("cartpole_phys_quad_first", [z, zn, p], [cost, g, lower, upper])

    def make_middle() -> ca.Function:
        z = ca.SX.sym("cartpole_phys_quad_z", CARTPOLE_NZ)
        zn = ca.SX.sym("cartpole_phys_quad_zn", CARTPOLE_NZ)
        p = ca.SX.sym("cartpole_phys_quad_p", 1 + CARTPOLE_N_COST_PARAMS + 1)
        state, action = _split_stage(z)
        next_state, _ = _split_stage(zn)
        dt = p[0]
        cost_params = p[1 : 1 + CARTPOLE_N_COST_PARAMS]
        tube_s = p[1 + CARTPOLE_N_COST_PARAMS]
        cost = _stage_cost_ca(state, action, dt, cost_params)
        action_lower, action_upper = _tightened_action_bounds_ca(cost_params, tube_s)
        g = ca.vertcat(next_state - _cartpole_euler_step_ca(state, action, dt), action[0])
        lower = ca.vertcat(np.zeros(CARTPOLE_NX), action_lower)
        upper = ca.vertcat(np.zeros(CARTPOLE_NX), action_upper)
        if rail_constraint:
            rail_lower, rail_upper = _tightened_rail_bounds_ca(cost_params, tube_s)
            g = ca.vertcat(g, state[0])
            lower = ca.vertcat(lower, rail_lower)
            upper = ca.vertcat(upper, rail_upper)
        return ca.Function("cartpole_phys_quad_middle", [z, zn, p], [cost, g, lower, upper])

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("cartpole_phys_quad_zN", CARTPOLE_NZ)
        p = ca.SX.sym("cartpole_phys_quad_pN", 1 + CARTPOLE_N_COST_PARAMS + 1)
        state, action = _split_stage(z)
        dt = p[0]
        cost_params = p[1 : 1 + CARTPOLE_N_COST_PARAMS]
        tube_s = p[1 + CARTPOLE_N_COST_PARAMS]
        cost = _terminal_cost_ca(state, dt, cost_params)
        g = ca.vertcat(action[0])
        lower = ca.vertcat(0.0)
        upper = ca.vertcat(0.0)
        if rail_constraint:
            rail_lower, rail_upper = _tightened_rail_bounds_ca(cost_params, tube_s)
            g = ca.vertcat(g, state[0])
            lower = ca.vertcat(lower, rail_lower)
            upper = ca.vertcat(upper, rail_upper)
        return ca.Function("cartpole_phys_quad_terminal", [z, p], [cost, g, lower, upper])

    first = CasadiStageFunction.from_function(make_first(), has_next=True)
    middle = CasadiStageFunction.from_function(make_middle(), has_next=True)
    terminal = CasadiStageFunction.from_function(make_terminal(), has_next=False)
    return SparseMPCProblem.from_stage_functions(
        horizon=n_steps,
        first=first,
        intermediate=middle,
        terminal=terminal,
    )


def cartpole_tube_schedule(
    cost_params: np.ndarray,
    dt_schedule: np.ndarray,
) -> np.ndarray:
    cost_params = np.asarray(cost_params)
    if cost_params.ndim == 1:
        cost_params = cost_params[None, :]
    dt_schedule = np.asarray(dt_schedule, dtype=cost_params.dtype)
    rho = cost_params[:, CARTPOLE_TUBE_RHO_INDEX]
    w = cost_params[:, CARTPOLE_TUBE_W_INDEX]
    tube = np.zeros((cost_params.shape[0], dt_schedule.size + 1), dtype=cost_params.dtype)
    s = np.zeros((cost_params.shape[0],), dtype=cost_params.dtype)
    for stage, dt in enumerate(dt_schedule):
        decay = np.exp(-rho * dt)
        s = decay * s + (1.0 - decay) * w / rho
        tube[:, stage + 1] = s
    return tube


def cartpole_initial_guess_and_params(
    x0: np.ndarray,
    cost_params: np.ndarray | None = None,
    *,
    n_steps: int = CARTPOLE_N_STEPS,
    dt_start: float = CARTPOLE_DT_START,
    dt_growth: float = CARTPOLE_DT_GROWTH,
    dtype: np.dtype | str = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.dtype(dtype)
    x0 = np.asarray(x0, dtype=dtype)
    if x0.ndim == 1:
        x0 = x0[None, :]
    batch = x0.shape[0]
    if cost_params is None:
        cost_params = np.broadcast_to(CARTPOLE_NOMINAL_COST_PARAMS, (batch, CARTPOLE_N_COST_PARAMS))
    cost_params = np.asarray(cost_params, dtype=dtype)
    if cost_params.ndim == 1:
        cost_params = np.broadcast_to(cost_params[None, :], (batch, CARTPOLE_N_COST_PARAMS))
    if cost_params.shape != (batch, CARTPOLE_N_COST_PARAMS):
        raise ValueError(
            f"expected cost_params shape {(batch, CARTPOLE_N_COST_PARAMS)}, got {cost_params.shape}"
        )

    dt_schedule = cartpole_dt_schedule(n_steps, dt_start=dt_start, dt_growth=dt_growth).astype(dtype)
    tube_schedule = cartpole_tube_schedule(cost_params, dt_schedule).astype(dtype, copy=False)
    z = np.zeros((batch, (n_steps + 1) * CARTPOLE_NZ), dtype=dtype)
    target = np.zeros_like(x0)
    for stage in range(n_steps + 1):
        alpha = np.asarray(stage / n_steps, dtype=dtype)
        state_guess = (1.0 - alpha) * x0 + alpha * target
        _write_stage_guess(z, stage, state_guess, None)

    param_dim = (
        (CARTPOLE_NX + 1 + CARTPOLE_N_COST_PARAMS + 1)
        + (n_steps - 1) * (1 + CARTPOLE_N_COST_PARAMS + 1)
        + (1 + CARTPOLE_N_COST_PARAMS + 1)
    )
    params = np.zeros((batch, param_dim), dtype=dtype)
    offset = 0
    params[:, offset : offset + CARTPOLE_NX] = x0
    offset += CARTPOLE_NX
    params[:, offset] = dt_schedule[0]
    offset += 1
    params[:, offset : offset + CARTPOLE_N_COST_PARAMS] = cost_params
    offset += CARTPOLE_N_COST_PARAMS
    params[:, offset] = tube_schedule[:, 0]
    offset += 1
    for i in range(1, n_steps):
        params[:, offset] = dt_schedule[i]
        offset += 1
        params[:, offset : offset + CARTPOLE_N_COST_PARAMS] = cost_params
        offset += CARTPOLE_N_COST_PARAMS
        params[:, offset] = tube_schedule[:, i]
        offset += 1
    params[:, offset] = dt_schedule[-1]
    offset += 1
    params[:, offset : offset + CARTPOLE_N_COST_PARAMS] = cost_params
    offset += CARTPOLE_N_COST_PARAMS
    params[:, offset] = tube_schedule[:, -1]
    return z, params


def update_cartpole_params_initial_state(params, x):
    return params.at[:, :CARTPOLE_NX].set(x)


def cartpole_jax_dynamics(x, u):
    dtype = x.dtype
    h1 = jnp.asarray(CARTPOLE_M + CARTPOLE_m, dtype=dtype)
    h2 = jnp.asarray(CARTPOLE_m * CARTPOLE_l, dtype=dtype)
    h4 = jnp.asarray(CARTPOLE_m * CARTPOLE_l**2 + CARTPOLE_J, dtype=dtype)
    h7 = jnp.asarray(CARTPOLE_m * CARTPOLE_l * CARTPOLE_G, dtype=dtype)
    ab_minus_beq = jnp.asarray(CARTPOLE_AB_MINUS_BEQ, dtype=dtype)
    ac = jnp.asarray(CARTPOLE_AC, dtype=dtype)
    b_p = jnp.asarray(CARTPOLE_B_P, dtype=dtype)
    v = x[:, 1]
    theta = x[:, 2]
    omega = x[:, 3]
    force = ab_minus_beq * v + ac * u[:, 0]
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    denominator = h2**2 * cos_theta**2 - h1 * h4
    vdot = (
        h2 * h4 * omega**2 * sin_theta
        - h2 * h7 * cos_theta * sin_theta
        + h4 * force
        - h2 * cos_theta * b_p * omega
    ) / (-denominator)
    omegadot = (
        h2**2 * omega**2 * cos_theta * sin_theta
        - h1 * h7 * sin_theta
        + h2 * cos_theta * force
        + h1 * b_p * omega
    ) / denominator
    return jnp.stack((v, vdot, omega, omegadot), axis=1)


def cartpole_jax_euler_step(x, u, dt):
    return x + dt * cartpole_jax_dynamics(x, u)


def make_cartpole_initialization_kernel(
    *,
    n_steps: int = CARTPOLE_N_STEPS,
    dt_start: float = CARTPOLE_DT_START,
    dt_growth: float = CARTPOLE_DT_GROWTH,
    dtype: np.dtype | str = np.float64,
):
    dtype = np.dtype(dtype)
    jdtype = jnp.dtype(dtype)
    dt_schedule = jnp.asarray(
        cartpole_dt_schedule(n_steps, dt_start=dt_start, dt_growth=dt_growth),
        dtype=jdtype,
    )
    param_dim = (
        (CARTPOLE_NX + 1 + CARTPOLE_N_COST_PARAMS + 1)
        + (n_steps - 1) * (1 + CARTPOLE_N_COST_PARAMS + 1)
        + (1 + CARTPOLE_N_COST_PARAMS + 1)
    )

    def write_stage(z, stage: int, state, action):
        offset = stage * CARTPOLE_NZ
        z = z.at[:, offset : offset + CARTPOLE_NX].set(state)
        return z.at[:, offset + CARTPOLE_NX : offset + CARTPOLE_NX + CARTPOLE_NU].set(action)

    @jax.jit
    def initialize(x0, cost_params):
        x0 = jnp.asarray(x0, dtype=jdtype)
        cost_params = jnp.asarray(cost_params, dtype=jdtype)
        batch = x0.shape[0]
        z = jnp.zeros((batch, (n_steps + 1) * CARTPOLE_NZ), dtype=jdtype)
        params = jnp.zeros((batch, param_dim), dtype=jdtype)
        target = jnp.zeros_like(x0)
        zero_action = jnp.zeros((batch, CARTPOLE_NU), dtype=jdtype)
        rho = cost_params[:, CARTPOLE_TUBE_RHO_INDEX]
        w = cost_params[:, CARTPOLE_TUBE_W_INDEX]
        tube_schedule = jnp.zeros((batch, n_steps + 1), dtype=jdtype)
        tube_s = jnp.zeros((batch,), dtype=jdtype)
        for stage in range(n_steps):
            decay = jnp.exp(-rho * dt_schedule[stage])
            tube_s = decay * tube_s + (jnp.asarray(1.0, dtype=jdtype) - decay) * w / rho
            tube_schedule = tube_schedule.at[:, stage + 1].set(tube_s)
        for stage in range(n_steps + 1):
            alpha = jnp.asarray(stage / n_steps, dtype=jdtype)
            state_guess = (jnp.asarray(1.0, dtype=jdtype) - alpha) * x0 + alpha * target
            z = write_stage(z, stage, state_guess, zero_action)

        offset = 0
        params = params.at[:, offset : offset + CARTPOLE_NX].set(x0)
        offset += CARTPOLE_NX
        params = params.at[:, offset].set(dt_schedule[0])
        offset += 1
        params = params.at[:, offset : offset + CARTPOLE_N_COST_PARAMS].set(cost_params)
        offset += CARTPOLE_N_COST_PARAMS
        params = params.at[:, offset].set(tube_schedule[:, 0])
        offset += 1
        for stage in range(1, n_steps):
            params = params.at[:, offset].set(dt_schedule[stage])
            offset += 1
            params = params.at[:, offset : offset + CARTPOLE_N_COST_PARAMS].set(cost_params)
            offset += CARTPOLE_N_COST_PARAMS
            params = params.at[:, offset].set(tube_schedule[:, stage])
            offset += 1
        params = params.at[:, offset].set(dt_schedule[-1])
        offset += 1
        params = params.at[:, offset : offset + CARTPOLE_N_COST_PARAMS].set(cost_params)
        offset += CARTPOLE_N_COST_PARAMS
        params = params.at[:, offset].set(tube_schedule[:, -1])
        return z, params

    return initialize


def log_to_cost_params(log_params):
    return np.exp(np.asarray(log_params, dtype=np.float64))


def cost_params_to_log(cost_params):
    return np.log(np.asarray(cost_params, dtype=np.float64))


def format_cost_parameters(cost_params: np.ndarray, *, max_lines: int | None = None) -> str:
    values = np.asarray(cost_params, dtype=np.float64).reshape(-1)
    lines = [
        f"{name}: {value:.4g}"
        for name, value in zip(CARTPOLE_COST_PARAMETER_NAMES, values, strict=True)
    ]
    if max_lines is not None and len(lines) > max_lines:
        remaining = len(lines) - max_lines
        lines = lines[:max_lines] + [f"... {remaining} more"]
    return "\n".join(lines)


def cartpole_cost_function_description() -> str:
    return (
        "Pure-quadratic identified-parameter MPC. For stages k=0..N-1,\n"
        "L_k = dt_k * 0.5*(w0*x_k^2 + w1*v_k^2 + w2*theta_k^2\n"
        "      + w3*omega_k^2 + w4*u_k^2).\n"
        "Terminal cost Phi_N = dt_N * 0.5*(w5*x_N^2 + w6*v_N^2\n"
        "      + w7*theta_N^2 + w8*omega_N^2), with dt_N=dt_{N-1}.\n"
        "Dynamics use the identified nominal cartpole model,\n"
        "plus a fixed 90 g point mass at the pole tip:\n"
        "M_cart=0.4472965, m_rod=0.01840051,\n"
        "m_tip=0.09, L_rod=0.6696171, AB=-1.959361,\n"
        "AC=0.5374264, B_eq=1.959361,\n"
        "B_p=0.00172243, discretized by Euler steps.\n"
        "The SQP can optionally add hard rail bounds |x|<=0.35. There are no angle wrap/sin(theta) terms in the MPC cost;\n"
        "theta is controlled directly toward zero.\n"
        "The final four tuning parameters are deterministic tube-tightening hyperparameters:\n"
        "s_0=0, s_{k+1}=exp(-rho*dt_k)*s_k + (1-exp(-rho*dt_k))*w/rho.\n"
        "Action bounds are tightened symmetrically to [u_min+c_u*s_k, u_max-c_u*s_k],\n"
        "and rail bounds are tightened symmetrically to [-x_max+c_x*s_k, x_max-c_x*s_k].\n"
        "All tuning parameters are optimized in log space."
    )


def cartpole_warm_start_description() -> str:
    return (
        "Initial SQP primal warm start: linearly interpolate state nodes from the sampled "
        "initial state to [0, 0, 0, 0] over the horizon, with all control nodes initialized "
        "to zero. During closed-loop rollout, each subsequent SQP call reuses the previous "
        "solution directly; no trajectory shifting is applied."
    )


__all__ = [
    "CARTPOLE_COST_PARAMETER_NAMES",
    "CARTPOLE_COST_PARAM_LOWER",
    "CARTPOLE_COST_PARAM_UPPER",
    "CARTPOLE_DT_GROWTH",
    "CARTPOLE_DT_START",
    "CARTPOLE_M_TIP",
    "CARTPOLE_NOMINAL_COST_PARAMS",
    "CARTPOLE_NX",
    "CARTPOLE_NU",
    "CARTPOLE_NZ",
    "CARTPOLE_N_COST_PARAMS",
    "CARTPOLE_N_STEPS",
    "CARTPOLE_RAIL_LIMIT",
    "CARTPOLE_U_MAX",
    "CARTPOLE_U_MIN",
    "cartpole_cost_function_description",
    "cartpole_cost_parameter_spec",
    "cartpole_dt_schedule",
    "cartpole_initial_guess_and_params",
    "cartpole_jax_dynamics",
    "cartpole_jax_euler_step",
    "cartpole_tube_schedule",
    "cartpole_warm_start_description",
    "cost_params_to_log",
    "format_cost_parameters",
    "log_to_cost_params",
    "make_cartpole_initialization_kernel",
    "make_cartpole_sqp_problem",
    "update_cartpole_params_initial_state",
]
