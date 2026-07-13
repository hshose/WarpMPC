#!/usr/bin/env python3
"""Closed-loop benchmark for the tuned quadratic cartpole SQP MPC."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp

from benchmarks.problems.cartpole_physical_quadratic import (
    CARTPOLE_COST_PARAMETER_NAMES,
    CARTPOLE_NOMINAL_COST_PARAMS,
    CARTPOLE_N_COST_PARAMS,
    CARTPOLE_NU,
    CARTPOLE_NX,
    CARTPOLE_RAIL_LIMIT,
    CARTPOLE_U_MAX,
    CARTPOLE_U_MIN,
    cartpole_jax_euler_step,
    format_cost_parameters,
    make_cartpole_initialization_kernel,
    update_cartpole_params_initial_state,
)
from examples.cartpole_tuning.tune_cartpole_physical_quadratic_mpc import (
    CartpoleBatchOutput,
    RAIL_MAX_PENALTY,
    RAIL_MEAN_SQUARED_PENALTY,
    RAIL_RATE_PENALTY,
    SUCCESS_ANGLE_BOUND,
    SYSID_ENV_PARAMETER_NAMES,
    SYSID_ENV_PARAMETER_NOMINAL,
    _make_sqp,
    _plot_experiment_state_distribution,
    _sample_initial_states,
)


ZERO_PROCESS_NOISE_SCALE = "0,0,0,0"
ZERO_INPUT_NOISE_SCALE = "0"


TUNED_COST_PARAMS_BY_NAME = {
    "stage_q_x": 4.87491613111937,
    "stage_q_v": 0.00010000000000000009,
    "stage_q_theta": 0.19748506311671338,
    "stage_q_omega": 0.053314181866045744,
    "stage_r_u": 0.010968784540611266,
    "terminal_q_x": 1825.9080085069231,
    "terminal_q_v": 10000.00000000001,
    "terminal_q_theta": 695.2242355039175,
    "terminal_q_omega": 9.132100349037795,
    "tube_rho": 3.7141252051386218,
    "tube_w": 1.9962829285664576e-05,
    "tighten_u": 0.2512645737717601,
    "tighten_x": 0.06702144014521097,
}

TUNED_COST_PARAMS = CARTPOLE_NOMINAL_COST_PARAMS.astype(np.float64, copy=True)
for index, name in enumerate(CARTPOLE_COST_PARAMETER_NAMES):
    if name in TUNED_COST_PARAMS_BY_NAME:
        TUNED_COST_PARAMS[index] = TUNED_COST_PARAMS_BY_NAME[name]


def _device_float(value) -> float:
    return float(np.asarray(jax.device_get(value)))


def _constraint_violation_stats(
    prefix: str,
    values: np.ndarray,
    *,
    threshold: float = 1.0e-6,
) -> tuple[dict[str, float], np.ndarray]:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.size == 0:
        return (
            {
                f"{prefix}_violation_rate_gt_1e-6": 0.0,
                f"{prefix}_constraint_satisfaction_gt_1e-6": 1.0,
                f"{prefix}_violation_mean": 0.0,
                f"{prefix}_violation_p95": 0.0,
                f"{prefix}_violation_p99": 0.0,
                f"{prefix}_violation_max": 0.0,
                f"{prefix}_rollout_violation_rate_gt_1e-6": 0.0,
                f"{prefix}_rollout_constraint_satisfaction_gt_1e-6": 1.0,
                f"{prefix}_rollout_violation_severity_mean": 0.0,
                f"{prefix}_rollout_violation_severity_p95": 0.0,
                f"{prefix}_rollout_violation_severity_p99": 0.0,
                f"{prefix}_rollout_violation_severity_max": 0.0,
            },
            np.empty((0,), dtype=np.float64),
        )
    values_arr = np.maximum(values_arr, 0.0)
    values_eval = np.where(np.isfinite(values_arr), values_arr, np.inf)
    if values_eval.ndim == 1:
        rollout_severity = values_eval
    else:
        rollout_severity = np.max(values_eval, axis=0)
    stats = {
        f"{prefix}_violation_rate_gt_1e-6": float(np.mean(values_eval > threshold)),
        f"{prefix}_constraint_satisfaction_gt_1e-6": float(np.mean(values_eval <= threshold)),
        f"{prefix}_violation_mean": float(np.mean(values_eval)),
        f"{prefix}_violation_p95": float(np.percentile(values_eval, 95.0)),
        f"{prefix}_violation_p99": float(np.percentile(values_eval, 99.0)),
        f"{prefix}_violation_max": float(np.max(values_eval)),
        f"{prefix}_rollout_violation_rate_gt_1e-6": float(np.mean(rollout_severity > threshold)),
        f"{prefix}_rollout_constraint_satisfaction_gt_1e-6": float(np.mean(rollout_severity <= threshold)),
        f"{prefix}_rollout_violation_severity_mean": float(np.mean(rollout_severity)),
        f"{prefix}_rollout_violation_severity_p95": float(np.percentile(rollout_severity, 95.0)),
        f"{prefix}_rollout_violation_severity_p99": float(np.percentile(rollout_severity, 99.0)),
        f"{prefix}_rollout_violation_severity_max": float(np.max(rollout_severity)),
    }
    return stats, rollout_severity


def _summarize_rollout(output, *, dtype: np.dtype, terminal_dt: float = 1.0) -> dict[str, float]:
    jdtype = jnp.dtype(dtype)
    summary_device = {
        "return_mean": jnp.mean(output.rollout_returns),
        "return_min": jnp.min(output.rollout_returns),
        "return_max": jnp.max(output.rollout_returns),
        "return_std": jnp.std(output.rollout_returns),
        "rollout_success_rate": jnp.mean(output.rollout_success.astype(jdtype)),
        "rail_violation_rate": jnp.mean(output.rollout_rail_violation),
        "mean_step": jnp.mean(output.step_lengths),
        "min_step": jnp.min(output.step_lengths),
        "max_violation": jnp.max(output.constraint_violations),
        "mean_qp_prim": jnp.mean(output.prim_res),
        "mean_qp_dual": jnp.mean(output.dual_res),
        "line_search_accept_rate": jnp.mean(output.line_search_accepted.astype(jdtype)),
        "sqp_finite_rate": jnp.mean(output.sqp_finite.astype(jdtype)),
    }
    jax.block_until_ready(tuple(summary_device.values()))
    summary = {key: _device_float(value) for key, value in summary_device.items()}

    returns = np.asarray(jax.device_get(output.rollout_returns), dtype=np.float64)
    states = np.asarray(jax.device_get(output.states), dtype=np.float64)
    commanded_actions = np.asarray(jax.device_get(output.commanded_actions), dtype=np.float64)
    final_state = states[-1]
    final_state_norm = np.linalg.norm(final_state, axis=1)
    terminal_weights = TUNED_COST_PARAMS[5:9].astype(np.float64, copy=False)
    final_state_cost = float(terminal_dt) * 0.5 * np.sum(final_state * terminal_weights[None, :] * final_state, axis=1)
    rail_violation = np.maximum(np.abs(states[:, :, 0]) - CARTPOLE_RAIL_LIMIT, 0.0)
    input_violation = np.maximum(
        np.maximum(CARTPOLE_U_MIN - commanded_actions[:, :, 0], commanded_actions[:, :, 0] - CARTPOLE_U_MAX),
        0.0,
    )
    rail_stats, rail_rollout_severity = _constraint_violation_stats("rail", rail_violation)
    input_stats, input_rollout_severity = _constraint_violation_stats("input", input_violation)
    if rail_rollout_severity.size and input_rollout_severity.size:
        combined_rollout_severity = np.maximum(rail_rollout_severity, input_rollout_severity)
        combined_rollout_satisfaction = float(np.mean(combined_rollout_severity <= 1.0e-6))
        combined_rollout_violation_rate = float(np.mean(combined_rollout_severity > 1.0e-6))
    else:
        combined_rollout_severity = np.empty((0,), dtype=np.float64)
        combined_rollout_satisfaction = 1.0
        combined_rollout_violation_rate = 0.0
    summary.update(
        {
            "return_p10": float(np.percentile(returns, 10.0)),
            "return_median": float(np.percentile(returns, 50.0)),
            "return_p90": float(np.percentile(returns, 90.0)),
            "final_state_norm_mean": float(np.mean(final_state_norm)),
            "final_state_norm_median": float(np.median(final_state_norm)),
            "final_state_norm_p95": float(np.percentile(final_state_norm, 95.0)),
            "final_state_norm_max": float(np.max(final_state_norm)),
            "final_state_cost_mean": float(np.mean(final_state_cost)),
            "final_state_cost_median": float(np.median(final_state_cost)),
            "final_state_cost_max": float(np.max(final_state_cost)),
            "rollout_rail_violation_mean": float(np.mean(rail_violation)),
            "rollout_rail_violation_max": float(np.max(rail_violation)),
            "combined_violation_rate_gt_1e-6": max(
                rail_stats["rail_violation_rate_gt_1e-6"],
                input_stats["input_violation_rate_gt_1e-6"],
            ),
            "combined_constraint_satisfaction_gt_1e-6": min(
                rail_stats["rail_constraint_satisfaction_gt_1e-6"],
                input_stats["input_constraint_satisfaction_gt_1e-6"],
            ),
            "combined_rollout_violation_rate_gt_1e-6": combined_rollout_violation_rate,
            "combined_rollout_constraint_satisfaction_gt_1e-6": combined_rollout_satisfaction,
            "combined_rollout_violation_severity_mean": float(np.mean(combined_rollout_severity))
            if combined_rollout_severity.size
            else 0.0,
            "combined_rollout_violation_severity_p95": float(np.percentile(combined_rollout_severity, 95.0))
            if combined_rollout_severity.size
            else 0.0,
            "combined_rollout_violation_severity_p99": float(np.percentile(combined_rollout_severity, 99.0))
            if combined_rollout_severity.size
            else 0.0,
            "combined_rollout_violation_severity_max": float(np.max(combined_rollout_severity))
            if combined_rollout_severity.size
            else 0.0,
        }
    )
    summary.update(rail_stats)
    summary.update(input_stats)
    return summary


def _plot_sampled_rollout(
    path: pathlib.Path,
    output,
    *,
    batch_size: int,
    plot_samples: int,
    control_dt: float,
) -> None:
    if plot_samples <= 0:
        return
    sample_count = min(batch_size, plot_samples)
    sample_index = np.linspace(0, batch_size - 1, sample_count, dtype=np.int64)
    states = np.asarray(jax.device_get(output.states[:, sample_index, :]))
    returns = np.asarray(jax.device_get(output.rollout_returns[sample_index]))
    time_grid = np.arange(states.shape[0], dtype=np.float64) * control_dt
    _plot_experiment_state_distribution(
        path,
        time_grid,
        states,
        returns,
        TUNED_COST_PARAMS,
        title=f"cartpole quadratic MPC batch={batch_size:,}",
        success_rate=float(np.mean(np.asarray(jax.device_get(output.rollout_success)))),
        rail_violation_rate=float(np.mean(np.asarray(jax.device_get(output.rollout_rail_violation)))),
    )


def _nominal_env_parameters(batch_size: int, dtype: np.dtype) -> np.ndarray:
    nominal = SYSID_ENV_PARAMETER_NOMINAL.astype(dtype, copy=False)
    return np.broadcast_to(nominal[None, :], (batch_size, nominal.size)).copy()


def _make_benchmark_rollout(args: argparse.Namespace, sqp, dtype: np.dtype):
    jdtype = jnp.dtype(dtype)
    control_dt = jnp.asarray(args.control_dt, dtype=jdtype)
    substeps = int(args.integrator_substeps)
    rollout_steps = int(args.rollout_steps)
    horizon_steps = int(args.horizon_steps)
    sqp_iterations = int(args.sqp_iterations)
    num_experiments = int(args.experiments_per_episode)
    rollouts_per_experiment = int(args.rollouts_per_experiment)

    def deterministic_dynamics_step(x, u):
        u_applied = jnp.clip(
            u,
            jnp.asarray(CARTPOLE_U_MIN, dtype=jdtype),
            jnp.asarray(CARTPOLE_U_MAX, dtype=jdtype),
        )
        step_dt = control_dt / jnp.asarray(substeps, dtype=jdtype)

        def body(_, x_cur):
            return cartpole_jax_euler_step(x_cur, u_applied, step_dt)

        return jax.lax.fori_loop(0, substeps, body, x), u_applied

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
    def rollout(z, params, x, env_params, key, noise_scale):
        del env_params, key, noise_scale

        def step(carry, _):
            z_cur, params_cur, x_cur, solver_state_cur = carry
            params_cur = update_cartpole_params_initial_state(params_cur, x_cur)

            def sqp_body(iter_carry, _):
                z_iter, solver_state_iter = iter_carry
                result_iter, solver_state_next = sqp.step(
                    z_iter,
                    params_cur,
                    state=solver_state_iter,
                )
                return (result_iter.z_next, solver_state_next), result_iter

            (z_next, solver_state_next), sqp_results = jax.lax.scan(
                sqp_body,
                (z_cur, solver_state_cur),
                xs=None,
                length=sqp_iterations,
            )
            result = jax.tree_util.tree_map(lambda leaf: leaf[-1], sqp_results)
            stages = result.z_next.reshape((x_cur.shape[0], horizon_steps + 1, CARTPOLE_NX + CARTPOLE_NU))
            commanded_u = stages[:, 0, CARTPOLE_NX : CARTPOLE_NX + CARTPOLE_NU]
            x_next, applied_u = deterministic_dynamics_step(x_cur, commanded_u)
            params_next = update_cartpole_params_initial_state(params_cur, x_next)
            finite = (
                result.is_finite
                & jnp.all(jnp.isfinite(x_cur), axis=1)
                & jnp.all(jnp.isfinite(commanded_u), axis=1)
            )
            output = (
                x_cur,
                commanded_u,
                applied_u,
                result.line_search.step_length,
                result.line_search.constraint_violation,
                result.line_search.accepted,
                result.solve.prim_res,
                result.solve.dual_res,
                finite,
            )
            return (z_next, params_next, x_next, solver_state_next), output

        solver_state = sqp.init_state(z.shape[0])
        final_carry, outputs = jax.lax.scan(
            step,
            (z, params, x, solver_state),
            xs=None,
            length=rollout_steps,
        )
        _, _, x_final, _ = final_carry
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
        ) = outputs
        states = jnp.concatenate([states_before, x_final[None, :, :]], axis=0)
        rollout_returns, rollout_success, rollout_rail_violation = reward_fn(states, applied_actions)
        exp_shape = (num_experiments, rollouts_per_experiment)
        experiment_returns = jnp.mean(rollout_returns.reshape(exp_shape), axis=1)
        experiment_success_rates = jnp.mean(rollout_success.reshape(exp_shape).astype(jdtype), axis=1)
        experiment_returns = experiment_returns + 300.0 * experiment_success_rates
        experiment_rail_violation_rates = jnp.mean(rollout_rail_violation.reshape(exp_shape), axis=1)
        return CartpoleBatchOutput(
            states=states,
            commanded_actions=commanded_actions,
            applied_actions=applied_actions,
            rollout_returns=rollout_returns,
            rollout_success=rollout_success,
            rollout_rail_violation=rollout_rail_violation,
            experiment_returns=experiment_returns,
            experiment_success_rates=experiment_success_rates,
            experiment_rail_violation_rates=experiment_rail_violation_rates,
            step_lengths=step_lengths,
            constraint_violations=constraint_violations,
            line_search_accepted=line_search_accepted,
            prim_res=prim_res,
            dual_res=dual_res,
            sqp_finite=sqp_finite,
        )

    return rollout


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    if args.qdldl_factor_backend is None:
        args.qdldl_factor_backend = args.qdldl_backend
    if args.qdldl_solve_backend is None:
        args.qdldl_solve_backend = args.qdldl_backend
    if args.mpax_iteration_limit is None:
        args.mpax_iteration_limit = args.osqp_max_iter
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

    print("building cartpole quadratic SQP solver...", flush=True)
    solver_setup_start = time.perf_counter()
    plan, sqp = _make_sqp(args, dtype)
    solver_setup_s = time.perf_counter() - solver_setup_start

    initialize_fn = make_cartpole_initialization_kernel(
        n_steps=args.horizon_steps,
        dt_start=args.dt_start,
        dt_growth=args.dt_growth,
        dtype=dtype,
    )
    rollout_fn = _make_benchmark_rollout(args, sqp, dtype)
    print(
        "Cartpole quadratic MPC:",
        f"batch={args.batch_size}",
        f"dtype={dtype}",
        f"qp_solver={args.qp_solver}",
        f"horizon_steps={args.horizon_steps}",
        f"rollout_steps={args.rollout_steps}",
        f"sqp_iterations={args.sqp_iterations}",
        f"osqp_max_iter={args.osqp_max_iter}",
        f"mpax_iteration_limit={args.mpax_iteration_limit}",
        f"qdldl_backend={args.qdldl_backend}",
        f"qdldl_factor_backend={args.qdldl_factor_backend}",
        f"qdldl_solve_backend={args.qdldl_solve_backend}",
        f"group_repeated_stages={args.group_repeated_stages}",
        f"level_scheduled_solve={args.level_scheduled_solve}",
        f"level_scheduled_solve_threshold={args.level_scheduled_solve_threshold}",
        f"rail_constraint={args.enable_rail_constraint}",
        f"n={plan.n_variables}",
        f"m={plan.n_constraints}",
        f"nnz_P={plan.p_pattern.nnz}",
        f"nnz_A={plan.a_pattern.nnz}",
        flush=True,
    )
    print("cost parameters:", flush=True)
    print(format_cost_parameters(TUNED_COST_PARAMS), flush=True)

    x_jax = jnp.asarray(x0)
    env_params_jax = jnp.asarray(_nominal_env_parameters(args.batch_size, dtype))
    cost_params_jax = jnp.asarray(cost_params)
    init_start = time.perf_counter()
    z0, params0 = initialize_fn(x_jax, cost_params_jax)
    jax.block_until_ready((z0, params0))
    initialization_s = time.perf_counter() - init_start

    key = jax.random.PRNGKey(args.seed + 1_000_003)
    noise_scale = jnp.asarray(args.noise_scale, dtype=jnp.dtype(dtype))
    compile_start = time.perf_counter()
    compiled_rollout = rollout_fn.lower(z0, params0, x_jax, env_params_jax, key, noise_scale).compile()
    rollout_compile_s = time.perf_counter() - compile_start
    print(
        "setup timings:",
        f"solver_setup={solver_setup_s:.3f}s",
        f"initialization={initialization_s:.3f}s",
        f"rollout_compile={rollout_compile_s:.3f}s",
        flush=True,
    )

    start = time.perf_counter()
    output = compiled_rollout(z0, params0, x_jax, env_params_jax, key, noise_scale)
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
        "qp_solver": args.qp_solver,
        "osqp_max_iter": args.osqp_max_iter,
        "mpax_iteration_limit": args.mpax_iteration_limit,
        "mpax_eps_abs": args.mpax_eps_abs,
        "mpax_eps_rel": args.mpax_eps_rel,
        "mpax_termination_evaluation_frequency": args.mpax_termination_evaluation_frequency,
        "mpax_l_inf_ruiz_iterations": args.mpax_l_inf_ruiz_iterations,
        "mpax_pock_chambolle_alpha": args.mpax_pock_chambolle_alpha,
        "mpax_regularization": args.mpax_regularization,
        "mpax_unroll": bool(args.mpax_unroll),
        "rho": args.rho,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "qdldl_backend": args.qdldl_backend,
        "qdldl_factor_backend": args.qdldl_factor_backend,
        "qdldl_solve_backend": args.qdldl_solve_backend,
        "transpose_work": bool(args.transpose_work),
        "segmented": bool(args.segmented),
        "segment_budget": args.segment_budget,
        "segment_strategy": args.segment_strategy,
        "line_search_step_min": args.line_search_step_min,
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
        "group_repeated_stages": bool(args.group_repeated_stages),
        "level_scheduled_solve": bool(args.level_scheduled_solve),
        "level_scheduled_solve_threshold": args.level_scheduled_solve_threshold,
        "n_variables": plan.n_variables,
        "n_constraints": plan.n_constraints,
        "nnz_p": plan.p_pattern.nnz,
        "nnz_a": plan.a_pattern.nnz,
        "solver_setup_s": solver_setup_s,
        "initialization_s": initialization_s,
        "rollout_compile_s": rollout_compile_s,
        "elapsed_s": elapsed_s,
        "total_closed_loop_steps": total_closed_loop_steps,
        "total_sqp_iterations": total_sqp_iterations,
        "closed_loop_steps_per_s": closed_loop_steps_per_s,
        "sqp_iterations_per_s": sqp_iterations_per_s,
        "rti_steps_per_s": sqp_iterations_per_s,
        "plot_path": "" if args.skip_state_plot else str(args.plot_path),
        "cost_params": {
            name: float(value)
            for name, value in zip(CARTPOLE_COST_PARAMETER_NAMES, TUNED_COST_PARAMS, strict=True)
        },
    }
    summary.update(rollout_summary)
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({closed_loop_steps_per_s:.3g} closed-loop steps/s, "
        f"{sqp_iterations_per_s:.3g} SQP iterations/s), "
        f"return_mean={summary['return_mean']:.3g}, "
        f"success={summary['rollout_success_rate']:.2%}, "
        f"rail_violation={summary['rail_violation_rate']:.2%}, "
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
    parser.add_argument("--qp-solver", choices=("jax_osqp", "mpax"), default="jax_osqp")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--osqp-max-iter", "--max-iter", dest="osqp_max_iter", type=int, default=25)
    parser.add_argument("--mpax-iteration-limit", type=int, default=None)
    parser.add_argument("--mpax-eps-abs", type=float, default=1e-3)
    parser.add_argument("--mpax-eps-rel", type=float, default=1e-3)
    parser.add_argument("--mpax-termination-evaluation-frequency", type=int, default=100)
    parser.add_argument("--mpax-l-inf-ruiz-iterations", type=int, default=10)
    parser.add_argument("--mpax-pock-chambolle-alpha", type=float, default=1.0)
    parser.add_argument("--mpax-regularization", type=float, default=0.0)
    parser.add_argument("--mpax-unroll", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="jax")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-budget", type=int, default=384)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action="store_true")
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
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
        "--no-group-repeated-stages",
        action="store_false",
        dest="group_repeated_stages",
    )
    parser.set_defaults(group_repeated_stages=True)
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/cartpole_quadratic_mpc_rollout.png"),
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
    if args.mpax_iteration_limit is None:
        args.mpax_iteration_limit = args.osqp_max_iter

    summary = run(args)
    if not args.skip_state_plot:
        print(f"Wrote {args.plot_path}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
