#!/usr/bin/env python3
"""Closed-loop MPX relaxed-barrier benchmark for the tuned cartpole MPC.

This formulation uses the physical cartpole state as the MPX state and the
physical voltage/action as the MPX control.  The initial state and nominal
Euler dynamics are therefore handled by MPX's own multiple-shooting dynamics
constraints.  The non-dynamics action and optional rail bounds remain relaxed
barrier terms, matching the style used in MPX examples.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MPX_ROOT = ROOT / "resources" / "mpx"
if MPX_ROOT.exists() and str(MPX_ROOT) not in sys.path:
    sys.path.insert(0, str(MPX_ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp

from benchmarks.nonlinear_mpc.evaluate_cartpole_quadratic_mpc import (
    TUNED_COST_PARAMS,
    _plot_sampled_rollout,
    _summarize_rollout,
)
from benchmarks.nonlinear_mpc.mpx_penalty_adapter import (
    MPXBarrierSettings,
    relaxed_barrier,
    relaxed_barrier_curvature,
    settings_array,
)
from benchmarks.problems.cartpole_physical_quadratic import (
    CARTPOLE_M_TIP,
    CARTPOLE_N_COST_PARAMS,
    CARTPOLE_NU,
    CARTPOLE_NX,
    CARTPOLE_NZ,
    CARTPOLE_RAIL_LIMIT,
    CARTPOLE_TIGHTEN_U_INDEX,
    CARTPOLE_TIGHTEN_X_INDEX,
    CARTPOLE_U_MAX,
    CARTPOLE_U_MIN,
    cartpole_dt_schedule,
    cartpole_initial_guess_and_params,
    cartpole_jax_euler_step,
    cartpole_tube_schedule,
)
from examples.cartpole_tuning.tune_cartpole_physical_quadratic_mpc import (
    RAIL_MAX_PENALTY,
    RAIL_MEAN_SQUARED_PENALTY,
    RAIL_RATE_PENALTY,
    SUCCESS_ANGLE_BOUND,
    SYSID_ENV_PARAMETER_NAMES,
    SYSID_ENV_PARAMETER_NOMINAL,
    _sample_initial_states,
)
from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers as mpx_optimizers


ZERO_PROCESS_NOISE_SCALE = "0,0,0,0"
ZERO_INPUT_NOISE_SCALE = "0"


class CartpoleMPXBatchOutput(NamedTuple):
    states: object
    commanded_actions: object
    applied_actions: object
    rollout_returns: object
    rollout_success: object
    rollout_rail_violation: object
    experiment_returns: object
    experiment_success_rates: object
    experiment_rail_violation_rates: object
    step_lengths: object
    constraint_violations: object
    line_search_accepted: object
    prim_res: object
    dual_res: object
    sqp_finite: object
    action_constraint_violations: object
    rail_constraint_violations: object
    dynamics_defects: object
    initial_defects: object


@dataclass(frozen=True)
class CartpolePhysicalMPXProblem:
    horizon: int
    rail_constraint: bool
    settings: MPXBarrierSettings
    solve: object
    violation: object
    action_violation: object
    rail_violation: object
    dynamics_defect: object
    initial_defect: object

    @property
    def n_variables(self) -> int:
        return (self.horizon + 1) * CARTPOLE_NX + self.horizon * CARTPOLE_NU

    @property
    def n_hard_constraints(self) -> int:
        return (self.horizon + 1) * CARTPOLE_NX

    @property
    def n_barrier_constraints(self) -> int:
        return self.horizon * CARTPOLE_NU + (self.horizon + 1 if self.rail_constraint else 0)


def _nominal_env_parameters(batch_size: int, dtype: np.dtype) -> np.ndarray:
    nominal = SYSID_ENV_PARAMETER_NOMINAL.astype(dtype, copy=False)
    return np.broadcast_to(nominal[None, :], (batch_size, nominal.size)).copy()


def _split_reference(reference, t: int, horizon: int):
    idx = jnp.minimum(t, horizon)
    return reference[idx, 0], reference[idx, 1], reference[idx, 2:]


def _action_bounds(cost_params, tube_s):
    tightening = cost_params[CARTPOLE_TIGHTEN_U_INDEX] * tube_s
    return CARTPOLE_U_MIN + tightening, CARTPOLE_U_MAX - tightening


def _rail_bounds(cost_params, tube_s):
    tightening = cost_params[CARTPOLE_TIGHTEN_X_INDEX] * tube_s
    return -CARTPOLE_RAIL_LIMIT + tightening, CARTPOLE_RAIL_LIMIT - tightening


def compile_physical_cartpole_mpx_problem(
    horizon: int,
    *,
    rail_constraint: bool,
    settings: MPXBarrierSettings,
    dtype: np.dtype | str,
) -> CartpolePhysicalMPXProblem:
    jdtype = jnp.dtype(dtype)

    def action_barrier(u, cost_params, tube_s, settings_vec):
        lower, upper = _action_bounds(cost_params, tube_s)
        alpha = jnp.asarray(settings_vec[1], dtype=u.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=u.dtype)
        return relaxed_barrier(u[0] - lower, alpha, sigma) + relaxed_barrier(upper - u[0], alpha, sigma)

    def action_barrier_curvature(u, cost_params, tube_s, settings_vec):
        lower, upper = _action_bounds(cost_params, tube_s)
        alpha = jnp.asarray(settings_vec[1], dtype=u.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=u.dtype)
        return relaxed_barrier_curvature(u[0] - lower, alpha, sigma) + relaxed_barrier_curvature(
            upper - u[0], alpha, sigma
        )

    def rail_barrier(x, cost_params, tube_s, settings_vec):
        if not rail_constraint:
            return jnp.asarray(0.0, dtype=x.dtype)
        lower, upper = _rail_bounds(cost_params, tube_s)
        alpha = jnp.asarray(settings_vec[1], dtype=x.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=x.dtype)
        return relaxed_barrier(x[0] - lower, alpha, sigma) + relaxed_barrier(upper - x[0], alpha, sigma)

    def rail_barrier_curvature(x, cost_params, tube_s, settings_vec):
        if not rail_constraint:
            return jnp.asarray(0.0, dtype=x.dtype)
        lower, upper = _rail_bounds(cost_params, tube_s)
        alpha = jnp.asarray(settings_vec[1], dtype=x.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=x.dtype)
        return relaxed_barrier_curvature(x[0] - lower, alpha, sigma) + relaxed_barrier_curvature(
            upper - x[0], alpha, sigma
        )

    def cost(settings_vec, reference, x, u, t):
        dt, tube_s, cost_params = _split_reference(reference, t, horizon)
        stage_state_cost = 0.5 * jnp.sum(cost_params[:CARTPOLE_NX] * x * x)
        control_cost = 0.5 * cost_params[4] * u[0] * u[0]
        terminal_state_cost = 0.5 * jnp.sum(cost_params[5:9] * x * x)
        stage_cost = (
            dt * (stage_state_cost + control_cost)
            + action_barrier(u, cost_params, tube_s, settings_vec)
            + rail_barrier(x, cost_params, tube_s, settings_vec)
        )
        terminal_cost = dt * terminal_state_cost + rail_barrier(x, cost_params, tube_s, settings_vec)
        return jnp.where(t < horizon, stage_cost, terminal_cost)

    def dynamics(x, u, t, parameter):
        dt = parameter[jnp.minimum(t, horizon - 1)]
        return cartpole_jax_euler_step(x[None, :], u[None, :], dt)[0]

    def hessian_approx(settings_vec, reference, x, u, t):
        dt, tube_s, cost_params = _split_reference(reference, t, horizon)
        q_stage_diag = dt * cost_params[:CARTPOLE_NX]
        q_terminal_diag = dt * cost_params[5:9]
        q_diag = jnp.where(t < horizon, q_stage_diag, q_terminal_diag)
        q = jnp.diag(q_diag)
        q = q.at[0, 0].add(rail_barrier_curvature(x, cost_params, tube_s, settings_vec))
        r_stage = dt * cost_params[4] + action_barrier_curvature(u, cost_params, tube_s, settings_vec)
        r_value = jnp.where(t < horizon, r_stage, jnp.asarray(0.0, dtype=x.dtype))
        r = jnp.asarray([[r_value]], dtype=x.dtype)
        m = jnp.zeros((CARTPOLE_NX, CARTPOLE_NU), dtype=x.dtype)
        return q, r, m

    if settings.solver_mode != "primal_dual":
        raise ValueError("MPX cartpole benchmark uses solver mode primal_dual")

    work = partial(
        mpx_optimizers.mpc,
        cost,
        dynamics,
        hessian_approx,
        settings.limited_memory,
        num_alpha=settings.num_alpha,
    )

    def one_action_violation(u, reference):
        tube_s = reference[:horizon, 1]
        cost_params = reference[:horizon, 2:]
        tightening = cost_params[:, CARTPOLE_TIGHTEN_U_INDEX] * tube_s
        lower = CARTPOLE_U_MIN + tightening
        upper = CARTPOLE_U_MAX - tightening
        value = u[:, 0]
        return jnp.max(jnp.maximum(jnp.maximum(lower - value, value - upper), 0.0))

    def one_rail_violation(x, reference):
        if not rail_constraint:
            return jnp.asarray(0.0, dtype=x.dtype)
        tube_s = reference[:, 1]
        cost_params = reference[:, 2:]
        tightening = cost_params[:, CARTPOLE_TIGHTEN_X_INDEX] * tube_s
        lower = -CARTPOLE_RAIL_LIMIT + tightening
        upper = CARTPOLE_RAIL_LIMIT - tightening
        value = x[:, 0]
        return jnp.max(jnp.maximum(jnp.maximum(lower - value, value - upper), 0.0))

    def one_dynamics_defect(x, u, parameter):
        def stage_defect(t):
            return dynamics(x[t], u[t], t, parameter) - x[t + 1]

        defects = jax.vmap(stage_defect)(jnp.arange(horizon))
        return jnp.max(jnp.abs(defects))

    def one_initial_defect(x0, x):
        return jnp.max(jnp.abs(x0 - x[0]))

    def one_violation(x0, x, u, reference, parameter):
        return jnp.maximum(
            jnp.maximum(one_action_violation(u, reference), one_rail_violation(x, reference)),
            jnp.maximum(one_initial_defect(x0, x), one_dynamics_defect(x, u, parameter)),
        )

    return CartpolePhysicalMPXProblem(
        horizon=horizon,
        rail_constraint=rail_constraint,
        settings=settings,
        solve=jax.jit(jax.vmap(work)),
        violation=jax.jit(jax.vmap(one_violation)),
        action_violation=jax.jit(jax.vmap(one_action_violation)),
        rail_violation=jax.jit(jax.vmap(one_rail_violation)),
        dynamics_defect=jax.jit(jax.vmap(one_dynamics_defect)),
        initial_defect=jax.jit(jax.vmap(one_initial_defect)),
    )


def physical_initial_guess_and_reference(
    x0: np.ndarray,
    cost_params: np.ndarray,
    *,
    n_steps: int,
    dt_start: float,
    dt_growth: float,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z0, _params0 = cartpole_initial_guess_and_params(
        x0,
        cost_params,
        n_steps=n_steps,
        dt_start=dt_start,
        dt_growth=dt_growth,
        dtype=dtype,
    )
    stages = z0.reshape((x0.shape[0], n_steps + 1, CARTPOLE_NZ))
    x_nodes = stages[:, :, :CARTPOLE_NX].copy()
    u_nodes = stages[:, :n_steps, CARTPOLE_NX : CARTPOLE_NX + CARTPOLE_NU].copy()
    dt = cartpole_dt_schedule(n_steps, dt_start=dt_start, dt_growth=dt_growth).astype(dtype)
    tube = cartpole_tube_schedule(cost_params, dt).astype(dtype, copy=False)
    reference = np.zeros((x0.shape[0], n_steps + 1, 2 + CARTPOLE_N_COST_PARAMS), dtype=dtype)
    reference[:, :n_steps, 0] = dt[None, :]
    reference[:, n_steps, 0] = dt[-1]
    reference[:, :, 1] = tube
    reference[:, :, 2:] = cost_params[:, None, :]
    parameter = np.broadcast_to(dt[None, :], (x0.shape[0], n_steps)).copy()
    return x_nodes, u_nodes, reference, parameter


def _make_mpx_rollout(args: argparse.Namespace, problem: CartpolePhysicalMPXProblem, dtype: np.dtype):
    jdtype = jnp.dtype(dtype)
    control_dt = jnp.asarray(args.control_dt, dtype=jdtype)
    substeps = int(args.integrator_substeps)
    rollout_steps = int(args.rollout_steps)
    sqp_iterations = int(args.sqp_iterations)

    def env_cartpole_jax_dynamics(x, u, env_params):
        m_cart = env_params[:, 0]
        m_rod = env_params[:, 1]
        ab = env_params[:, 2]
        ac = env_params[:, 3]
        b_eq = env_params[:, 4]
        b_p = env_params[:, 5]
        l_rod = env_params[:, 6]
        m_tip = jnp.asarray(CARTPOLE_M_TIP, dtype=jdtype)
        m_pole = m_rod + m_tip
        center_of_mass = (m_rod * (0.5 * l_rod) + m_tip * l_rod) / m_pole
        pole_inertia_pivot = m_rod * l_rod**2 / jnp.asarray(3.0, dtype=jdtype) + m_tip * l_rod**2
        pole_inertia_com = pole_inertia_pivot - m_pole * center_of_mass**2

        v = x[:, 1]
        theta = x[:, 2]
        omega = x[:, 3]
        force = (ab - b_eq) * v + ac * u[:, 0]

        h1 = m_cart + m_pole
        h2 = m_pole * center_of_mass
        h4 = m_pole * center_of_mass**2 + pole_inertia_com
        h7 = m_pole * center_of_mass * jnp.asarray(9.81, dtype=jdtype)
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

    def env_cartpole_jax_euler_step(x, u, env_params, dt):
        return x + dt * env_cartpole_jax_dynamics(x, u, env_params)

    def deterministic_dynamics_step(x, u, env_params):
        u_applied = jnp.clip(
            u,
            jnp.asarray(CARTPOLE_U_MIN, dtype=jdtype),
            jnp.asarray(CARTPOLE_U_MAX, dtype=jdtype),
        )
        step_dt = control_dt / jnp.asarray(substeps, dtype=jdtype)

        def body(_, x_cur):
            return env_cartpole_jax_euler_step(x_cur, u_applied, env_params, step_dt)

        x_next = jax.lax.fori_loop(0, substeps, body, x)
        return x_next, u_applied

    def reward_fn(states, actions):
        theta = states[:, :, 2]
        angle = theta
        x_pos = states[:, :, 0]
        v = states[:, :, 1]
        omega = states[:, :, 3]
        tail_start = max(1, rollout_steps // 2)
        tail_angle = angle[tail_start:]
        tail_x = x_pos[tail_start:]
        tail_v = v[tail_start:]
        tail_omega = omega[tail_start:]
        upright_score = jnp.exp(-0.5 * (angle / 0.45) ** 2)
        upright_tail = jnp.exp(-0.5 * (tail_angle / 0.35) ** 2)
        center_tail = jnp.exp(-0.5 * (tail_x / 0.15) ** 2)
        swing_up_time = jnp.abs(angle) < 0.35
        swing_up = jnp.any(swing_up_time, axis=0)
        within_success_angle = jnp.abs(angle) <= jnp.asarray(SUCCESS_ANGLE_BOUND, dtype=jdtype)
        stay_up_from_time = jnp.flip(
            jnp.cumprod(jnp.flip(within_success_angle.astype(jdtype), axis=0), axis=0),
            axis=0,
        )
        success = jnp.any(stay_up_from_time[:-1] > jnp.asarray(0.5, dtype=jdtype), axis=0)
        near_upright_fraction = jnp.mean((jnp.abs(angle) < 0.55).astype(jdtype), axis=0)
        max_height_score = jnp.max(0.5 * (1.0 + jnp.cos(angle)), axis=0)
        balance_time = (jnp.abs(tail_angle) < 0.25) & (jnp.abs(tail_omega) < 1.0)
        balance_fraction = jnp.mean(balance_time.astype(jdtype), axis=0)
        best_upright_score = jnp.max(upright_score, axis=0)
        rail_violation = jnp.maximum(jnp.abs(x_pos) - jnp.asarray(CARTPOLE_RAIL_LIMIT, dtype=jdtype), 0.0)
        rail_violation_rate = jnp.mean((rail_violation > 0.0).astype(jdtype), axis=0)
        rail_penalty = (
            jnp.asarray(RAIL_RATE_PENALTY, dtype=jdtype) * rail_violation_rate
            + jnp.asarray(RAIL_MAX_PENALTY, dtype=jdtype) * jnp.max(rail_violation, axis=0)
            + jnp.asarray(RAIL_MEAN_SQUARED_PENALTY, dtype=jdtype) * jnp.mean(rail_violation**2, axis=0)
        )
        tail_velocity_penalty = 0.10 * jnp.mean(tail_v**2, axis=0) + 0.01 * jnp.mean(tail_omega**2, axis=0)
        action_penalty = 0.002 * jnp.mean(actions[:, :, 0] ** 2, axis=0)
        reward = (
            250.0 * swing_up.astype(jdtype)
            + 80.0 * max_height_score
            + 60.0 * best_upright_score
            + 50.0 * near_upright_fraction
            + 45.0 * balance_fraction
            + 25.0 * jnp.mean(upright_tail * center_tail, axis=0)
            - rail_penalty
            - tail_velocity_penalty
            - action_penalty
        )
        finite = jnp.all(jnp.isfinite(states), axis=(0, 2)) & jnp.all(jnp.isfinite(actions), axis=(0, 2))
        return jnp.where(finite, reward, -1.0e6), success & finite, rail_violation_rate

    @jax.jit
    def rollout(x_nodes, u_nodes, dual_nodes, reference, settings_vec, parameter, x, env_params, key, noise_scale):
        del noise_scale

        def step(carry, _):
            x_nodes_cur, u_nodes_cur, dual_nodes_cur, x_cur, key_cur = carry
            x_nodes_cur = x_nodes_cur.at[:, 0, :].set(x_cur)
            x0_physical = x_nodes_cur[:, 0]

            def mpx_body(iter_carry, _):
                x_iter, u_iter, dual_iter = iter_carry
                x_next, u_next, dual_next = problem.solve(
                    reference,
                    parameter,
                    settings_vec,
                    x0_physical,
                    x_iter,
                    u_iter,
                    dual_iter,
                )
                return (x_next, u_next, dual_next), None

            (x_nodes_next, u_nodes_next, dual_nodes_next), _ = jax.lax.scan(
                mpx_body,
                (x_nodes_cur, u_nodes_cur, dual_nodes_cur),
                xs=None,
                length=sqp_iterations,
            )
            commanded_u = u_nodes_next[:, 0, :]
            key_next = key_cur
            x_next, applied_u = deterministic_dynamics_step(x_cur, commanded_u, env_params)
            action_violation = problem.action_violation(u_nodes_next, reference)
            rail_violation = problem.rail_violation(x_nodes_next, reference)
            dynamics_defect = problem.dynamics_defect(x_nodes_next, u_nodes_next, parameter)
            initial_defect = problem.initial_defect(x0_physical, x_nodes_next)
            violation = problem.violation(x0_physical, x_nodes_next, u_nodes_next, reference, parameter)
            finite = (
                jnp.all(jnp.isfinite(x_nodes_next), axis=(1, 2))
                & jnp.all(jnp.isfinite(u_nodes_next), axis=(1, 2))
                & jnp.all(jnp.isfinite(x_cur), axis=1)
                & jnp.all(jnp.isfinite(commanded_u), axis=1)
            )
            output = (
                x_cur,
                commanded_u,
                applied_u,
                jnp.ones((x_cur.shape[0],), dtype=jdtype),
                violation,
                finite,
                jnp.full((x_cur.shape[0],), jnp.nan, dtype=jdtype),
                jnp.full((x_cur.shape[0],), jnp.nan, dtype=jdtype),
                finite,
                action_violation,
                rail_violation,
                dynamics_defect,
                initial_defect,
            )
            return (
                x_nodes_next,
                u_nodes_next,
                dual_nodes_next,
                x_next,
                key_next,
            ), output

        final_carry, outputs = jax.lax.scan(
            step,
            (x_nodes, u_nodes, dual_nodes, x, key),
            xs=None,
            length=rollout_steps,
        )
        _, _, _, x_final, _ = final_carry
        (
            states_before,
            commanded_actions,
            applied_actions,
            step_lengths,
            constraint_violations,
            line_search_accepted,
            prim_res,
            dual_res,
            sqp_finite,
            action_constraint_violations,
            rail_constraint_violations,
            dynamics_defects,
            initial_defects,
        ) = outputs
        states = jnp.concatenate([states_before, x_final[None, :, :]], axis=0)
        rollout_returns, rollout_success, rollout_rail_violation = reward_fn(states, applied_actions)
        experiment_returns = rollout_returns + 300.0 * rollout_success.astype(jdtype)
        return CartpoleMPXBatchOutput(
            states=states,
            commanded_actions=commanded_actions,
            applied_actions=applied_actions,
            rollout_returns=rollout_returns,
            rollout_success=rollout_success,
            rollout_rail_violation=rollout_rail_violation,
            experiment_returns=experiment_returns,
            experiment_success_rates=rollout_success.astype(jdtype),
            experiment_rail_violation_rates=rollout_rail_violation,
            step_lengths=step_lengths,
            constraint_violations=constraint_violations,
            line_search_accepted=line_search_accepted,
            prim_res=prim_res,
            dual_res=dual_res,
            sqp_finite=sqp_finite,
            action_constraint_violations=action_constraint_violations,
            rail_constraint_violations=rail_constraint_violations,
            dynamics_defects=dynamics_defects,
            initial_defects=initial_defects,
        )

    return rollout


def _max_output_field(output: CartpoleMPXBatchOutput, name: str) -> float:
    value = np.asarray(jax.device_get(getattr(output, name)), dtype=np.float64)
    return float(np.max(value)) if value.size else float("nan")


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    if args.mpx_solver_mode != "primal_dual":
        raise ValueError("MPX cartpole benchmark uses solver mode primal_dual")
    args.noise_scale = 0.0
    args.process_noise_scale = ZERO_PROCESS_NOISE_SCALE
    args.input_noise_scale = ZERO_INPUT_NOISE_SCALE

    args.experiments_per_episode = 1
    args.rollouts_per_experiment = args.batch_size

    x0 = _sample_initial_states(args, args.batch_size, args.seed, dtype)
    cost_params = np.broadcast_to(
        TUNED_COST_PARAMS.astype(dtype, copy=False)[None, :],
        (args.batch_size, CARTPOLE_N_COST_PARAMS),
    )

    print("building physical Cartpole MPX problem...", flush=True)
    problem_start = time.perf_counter()
    barrier_settings = MPXBarrierSettings(
        equality_weight=args.mpx_equality_weight,
        barrier_alpha=args.mpx_barrier_alpha,
        barrier_sigma=args.mpx_barrier_sigma,
        num_alpha=args.mpx_num_alpha,
        limited_memory=args.mpx_limited_memory,
        solver_mode=args.mpx_solver_mode,
    )
    problem = compile_physical_cartpole_mpx_problem(
        args.horizon_steps,
        rail_constraint=args.enable_rail_constraint,
        settings=barrier_settings,
        dtype=dtype,
    )
    problem_build_s = time.perf_counter() - problem_start
    compile_setup_s = 0.0

    init_start = time.perf_counter()
    x_nodes0_np, u_nodes0_np, reference0_np, parameter_np = physical_initial_guess_and_reference(
        x0,
        cost_params,
        n_steps=args.horizon_steps,
        dt_start=args.dt_start,
        dt_growth=args.dt_growth,
        dtype=dtype,
    )
    dual0_np = np.zeros_like(x_nodes0_np)
    initialization_s = time.perf_counter() - init_start

    x_nodes0 = jnp.asarray(x_nodes0_np)
    u_nodes0 = jnp.asarray(u_nodes0_np)
    dual0 = jnp.asarray(dual0_np)
    reference0 = jnp.asarray(reference0_np)
    parameter = jnp.asarray(parameter_np)
    settings_vec = jnp.asarray(settings_array(barrier_settings, args.batch_size, dtype))
    x_jax = jnp.asarray(x0)
    env_params_jax = jnp.asarray(_nominal_env_parameters(args.batch_size, dtype))
    key = jax.random.PRNGKey(args.seed + 1_000_003)
    noise_scale = jnp.asarray(args.noise_scale, dtype=jnp.dtype(dtype))

    rollout_fn = _make_mpx_rollout(args, problem, dtype)
    compile_start = time.perf_counter()
    compiled_rollout = rollout_fn.lower(
        x_nodes0,
        u_nodes0,
        dual0,
        reference0,
        settings_vec,
        parameter,
        x_jax,
        env_params_jax,
        key,
        noise_scale,
    ).compile()
    rollout_compile_s = time.perf_counter() - compile_start

    print(
        "Cartpole MPX MPC:",
        f"batch={args.batch_size}",
        f"dtype={dtype}",
        f"solver_mode={args.mpx_solver_mode}",
        f"horizon_steps={args.horizon_steps}",
        f"rollout_steps={args.rollout_steps}",
        f"sqp_iterations={args.sqp_iterations}",
        f"rail_constraint={args.enable_rail_constraint}",
        f"n={problem.n_variables}",
        f"hard_m={problem.n_hard_constraints}",
        f"barrier_m={problem.n_barrier_constraints}",
        flush=True,
    )
    print(
        "setup timings:",
        f"problem_build={problem_build_s:.3f}s",
        f"compile_setup={compile_setup_s:.3f}s",
        f"initialization={initialization_s:.3f}s",
        f"rollout_compile={rollout_compile_s:.3f}s",
        flush=True,
    )

    start = time.perf_counter()
    output = compiled_rollout(
        x_nodes0,
        u_nodes0,
        dual0,
        reference0,
        settings_vec,
        parameter,
        x_jax,
        env_params_jax,
        key,
        noise_scale,
    )
    terminal_dt = args.dt_start * args.dt_growth ** (args.horizon_steps - 1)
    rollout_summary = _summarize_rollout(output, dtype=dtype, terminal_dt=terminal_dt)
    elapsed_s = time.perf_counter() - start

    if not args.skip_state_plot:
        _plot_sampled_rollout(
            args.plot_path,
            output,
            batch_size=args.batch_size,
            plot_samples=args.plot_samples,
            control_dt=args.control_dt,
        )

    total_closed_loop_steps = args.batch_size * args.rollout_steps
    total_sqp_iterations = total_closed_loop_steps * args.sqp_iterations
    closed_loop_steps_per_s = total_closed_loop_steps / elapsed_s
    sqp_iterations_per_s = total_sqp_iterations / elapsed_s
    max_action_violation = _max_output_field(output, "action_constraint_violations")
    max_rail_constraint_violation = _max_output_field(output, "rail_constraint_violations")
    max_dynamics_defect = _max_output_field(output, "dynamics_defects")
    max_initial_defect = _max_output_field(output, "initial_defects")
    summary: dict[str, object] = {
        "batch_size": args.batch_size,
        "dtype": str(dtype),
        "horizon_steps": args.horizon_steps,
        "dt_start": args.dt_start,
        "dt_growth": args.dt_growth,
        "sim_time": args.sim_time,
        "control_dt": args.control_dt,
        "rollout_steps": args.rollout_steps,
        "integrator_substeps": args.integrator_substeps,
        "sqp_iterations": args.sqp_iterations,
        "qp_solver": "mpx",
        "mpx_solver_mode": args.mpx_solver_mode,
        "mpx_equality_weight": args.mpx_equality_weight,
        "mpx_barrier_alpha": args.mpx_barrier_alpha,
        "mpx_barrier_sigma": args.mpx_barrier_sigma,
        "mpx_num_alpha": args.mpx_num_alpha,
        "mpx_limited_memory": bool(args.mpx_limited_memory),
        "enable_rail_constraint": bool(args.enable_rail_constraint),
        "noise_scale": args.noise_scale,
        "process_noise_scale": args.process_noise_scale,
        "input_noise_scale": args.input_noise_scale,
        "simulation_parameter_mode": "nominal",
        "simulation_env_parameters": {
            name: float(value)
            for name, value in zip(SYSID_ENV_PARAMETER_NAMES, SYSID_ENV_PARAMETER_NOMINAL, strict=True)
        },
        "simulation_input_disturbance_bound": 0.0,
        "simulation_process_noise_scale_effective": 0.0,
        "simulation_input_noise_scale_effective": 0.0,
        "warm_starting": True,
        "n_variables": problem.n_variables,
        "n_constraints": problem.n_hard_constraints + problem.n_barrier_constraints,
        "n_hard_constraints": problem.n_hard_constraints,
        "n_barrier_constraints": problem.n_barrier_constraints,
        "nnz_p": "",
        "nnz_a": "",
        "problem_build_s": problem_build_s,
        "compile_setup_s": compile_setup_s,
        "solver_setup_s": problem_build_s + compile_setup_s,
        "initialization_s": initialization_s,
        "rollout_compile_s": rollout_compile_s,
        "elapsed_s": elapsed_s,
        "total_closed_loop_steps": total_closed_loop_steps,
        "total_sqp_iterations": total_sqp_iterations,
        "closed_loop_steps_per_s": closed_loop_steps_per_s,
        "sqp_iterations_per_s": sqp_iterations_per_s,
        "rti_steps_per_s": sqp_iterations_per_s,
        "max_action_violation": max_action_violation,
        "max_rail_constraint_violation": max_rail_constraint_violation,
        "max_dynamics_defect": max_dynamics_defect,
        "max_initial_defect": max_initial_defect,
        "plot_path": "" if args.skip_state_plot else str(args.plot_path),
    }
    summary.update(rollout_summary)
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({closed_loop_steps_per_s:.3g} closed-loop steps/s, "
        f"{sqp_iterations_per_s:.3g} MPX iterations/s), "
        f"return_mean={summary['return_mean']:.3g}, "
        f"final_state_cost_mean={summary['final_state_cost_mean']:.3g}, "
        f"success={summary['rollout_success_rate']:.2%}, "
        f"rail_violation={summary['rail_violation_rate']:.2%}, "
        f"max_rail_constraint_violation={max_rail_constraint_violation:.3e}, "
        f"max_violation={summary['max_violation']:.3e}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--dt-start", type=float, default=0.1)
    parser.add_argument("--dt-growth", type=float, default=1.0)
    parser.add_argument("--sim-time", type=float, default=10.0)
    parser.add_argument("--control-dt", type=float, default=0.1)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--integrator-substeps", type=int, default=1)
    parser.add_argument("--sqp-iterations", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--mpx-solver-mode", choices=("primal_dual",), default="primal_dual")
    parser.add_argument("--mpx-equality-weight", type=float, default=1.0e4)
    parser.add_argument("--mpx-barrier-alpha", type=float, default=0.1)
    parser.add_argument("--mpx-barrier-sigma", type=float, default=1.0)
    parser.add_argument("--mpx-num-alpha", type=int, default=11)
    parser.add_argument("--mpx-limited-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-rail-constraint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--process-noise-scale", default=ZERO_PROCESS_NOISE_SCALE)
    parser.add_argument("--input-noise-scale", default=ZERO_INPUT_NOISE_SCALE)
    parser.add_argument("--initial-position-range", type=float, default=0.30)
    parser.add_argument("--initial-angle-spread-deg", type=float, default=20.0)
    parser.add_argument("--initial-velocity-std", type=float, default=0.15)
    parser.add_argument("--initial-omega-std", type=float, default=0.5)
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/cartpole_quadratic_mpc_mpx_rollout.png"),
    )
    parser.add_argument("--plot-samples", type=int, default=2048)
    parser.add_argument("--skip-state-plot", action="store_true")
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.rollout_steps is None:
        args.rollout_steps = int(math.ceil(args.sim_time / args.control_dt))
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.integrator_substeps <= 0:
        raise ValueError("integrator substeps must be positive")
    if args.sqp_iterations <= 0:
        raise ValueError("sqp iterations must be positive")

    summary = run(args)
    if not args.skip_state_plot:
        print(f"Wrote {args.plot_path}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
