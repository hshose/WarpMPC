#!/usr/bin/env python3
"""Closed-loop tuned cartpole MPC benchmark using TurboMPC autodiff blocks."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import NamedTuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp

from benchmarks.nonlinear_mpc.evaluate_cartpole_quadratic_mpc import (
    TUNED_COST_PARAMS,
    _plot_sampled_rollout,
    _summarize_rollout,
)
from benchmarks.nonlinear_mpc.turbompc_autodiff_adapter import (
    AutodiffDenseStageTurbompcProblem,
    initial_admm_state_batch,
    make_problem_params,
    make_solver,
    make_solver_params,
)
from benchmarks.problems.turbompc_autodiff_value_problems import (
    make_cartpole_turbompc_autodiff_problem,
)
from benchmarks.problems.cartpole_physical_quadratic import (
    CARTPOLE_AB,
    CARTPOLE_AC,
    CARTPOLE_B_EQ,
    CARTPOLE_B_P,
    CARTPOLE_COST_PARAMETER_NAMES,
    CARTPOLE_M_TIP,
    CARTPOLE_NX,
    CARTPOLE_NU,
    CARTPOLE_RAIL_LIMIT,
    CARTPOLE_U_MAX,
    CARTPOLE_U_MIN,
    format_cost_parameters,
    make_cartpole_initialization_kernel,
    update_cartpole_params_initial_state,
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


ZERO_PROCESS_NOISE_SCALE = "0,0,0,0"
ZERO_INPUT_NOISE_SCALE = "0"


class CartpoleTurboMPCOutput(NamedTuple):
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


def _nominal_env_parameters(batch_size: int, dtype: np.dtype) -> np.ndarray:
    nominal = SYSID_ENV_PARAMETER_NOMINAL.astype(dtype, copy=False)
    return np.broadcast_to(nominal[None, :], (batch_size, nominal.size)).copy()


def _make_rollout(args: argparse.Namespace, solver, dtype: np.dtype):
    jdtype = jnp.dtype(dtype)
    control_dt = jnp.asarray(args.control_dt, dtype=jdtype)
    substeps = int(args.integrator_substeps)
    rollout_steps = int(args.rollout_steps)

    def solve_one(states, controls, packed_params, admm_state):
        return solver._solve_impl(
            states,
            controls,
            make_problem_params(packed_params),
            admm_state,
        )

    solve_batch = jax.vmap(solve_one)

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
        x_pos = states[:, :, 0]
        v = states[:, :, 1]
        omega = states[:, :, 3]
        tail_start = max(1, rollout_steps // 2)
        tail_angle = theta[tail_start:]
        tail_x = x_pos[tail_start:]
        tail_v = v[tail_start:]
        tail_omega = omega[tail_start:]
        upright_score = jnp.exp(-0.5 * (theta / 0.45) ** 2)
        upright_tail = jnp.exp(-0.5 * (tail_angle / 0.35) ** 2)
        center_tail = jnp.exp(-0.5 * (tail_x / 0.15) ** 2)
        swing_up = jnp.any(jnp.abs(theta) < 0.35, axis=0)
        within_success_angle = jnp.abs(theta) <= jnp.asarray(SUCCESS_ANGLE_BOUND, dtype=jdtype)
        stay_up_from_time = jnp.flip(
            jnp.cumprod(jnp.flip(within_success_angle.astype(jdtype), axis=0), axis=0),
            axis=0,
        )
        success = jnp.any(stay_up_from_time[:-1] > jnp.asarray(0.5, dtype=jdtype), axis=0)
        near_upright_fraction = jnp.mean((jnp.abs(theta) < 0.55).astype(jdtype), axis=0)
        max_height_score = jnp.max(0.5 * (1.0 + jnp.cos(theta)), axis=0)
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
    def rollout(states, controls, params, x, env_params, key, noise_scale, admm_state):
        del noise_scale

        def step(carry, _):
            states_cur, controls_cur, params_cur, x_cur, key_cur, admm_cur = carry
            params_cur = update_cartpole_params_initial_state(params_cur, x_cur)
            solution = solve_batch(states_cur, controls_cur, params_cur, admm_cur)
            commanded_u = solution.controls[:, 0, :CARTPOLE_NU]
            key_next = key_cur
            x_next, applied_u = deterministic_dynamics_step(x_cur, commanded_u, env_params)
            eq_v = solution.solver_stats.eq_constraints_violations[:, -1]
            ineq_v = solution.solver_stats.ineq_constraints_violations[:, -1]
            conv = solution.convergence_error
            step_len = solution.linesearch_alphas[:, -1]
            finite = (
                jnp.all(jnp.isfinite(solution.states), axis=(1, 2))
                & jnp.all(jnp.isfinite(commanded_u), axis=1)
                & jnp.all(jnp.isfinite(x_cur), axis=1)
            )
            output = (
                x_cur,
                commanded_u,
                applied_u,
                step_len,
                eq_v + ineq_v,
                step_len > jnp.asarray(args.line_search_step_min + 1e-12, dtype=jdtype),
                eq_v,
                conv,
                finite,
            )
            return (
                solution.states,
                solution.controls,
                params_cur.at[:, :CARTPOLE_NX].set(x_next),
                x_next,
                key_next,
                solution.admm_state,
            ), output

        final_carry, outputs = jax.lax.scan(
            step,
            (states, controls, params, x, key, admm_state),
            xs=None,
            length=rollout_steps,
        )
        _, _, _, x_final, _, _ = final_carry
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
        states_out = jnp.concatenate([states_before, x_final[None, :, :]], axis=0)
        rollout_returns, rollout_success, rollout_rail_violation = reward_fn(states_out, applied_actions)
        return CartpoleTurboMPCOutput(
            states=states_out,
            commanded_actions=commanded_actions,
            applied_actions=applied_actions,
            rollout_returns=rollout_returns,
            rollout_success=rollout_success,
            rollout_rail_violation=rollout_rail_violation,
            experiment_returns=rollout_returns[None, :],
            experiment_success_rates=rollout_success.astype(jdtype)[None, :],
            experiment_rail_violation_rates=rollout_rail_violation[None, :],
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
    args.noise_scale = 0.0
    args.process_noise_scale = ZERO_PROCESS_NOISE_SCALE
    args.input_noise_scale = ZERO_INPUT_NOISE_SCALE

    args.experiments_per_episode = 1
    args.rollouts_per_experiment = args.batch_size
    x0 = _sample_initial_states(args, args.batch_size, args.seed, dtype)
    cost_params = np.broadcast_to(
        TUNED_COST_PARAMS.astype(dtype, copy=False)[None, :],
        (args.batch_size, TUNED_COST_PARAMS.size),
    )

    print("building cartpole TurboMPC-autodiff dense-stage problem...", flush=True)
    problem_start = time.perf_counter()
    value_problem = make_cartpole_turbompc_autodiff_problem(
        args.horizon_steps,
        rail_constraint=args.enable_rail_constraint,
    )
    problem = AutodiffDenseStageTurbompcProblem(
        value_problem,
        state_dim=CARTPOLE_NX,
        control_dim=CARTPOLE_NU,
        name="CartpoleAutodiffDenseStageTurboMPC",
    )
    problem_build_s = time.perf_counter() - problem_start
    desc = problem.describe_dense_blocks()

    solver_params = make_solver_params(
        sqp_iterations=args.sqp_iterations,
        admm_max_iter=args.osqp_max_iter,
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        eps_abs=args.turbompc_eps_abs,
        eps_rel=args.turbompc_eps_rel,
        line_search_step_min=args.line_search_step_min,
        fixed_sqp_iterations=True,
    )
    solver_setup_start = time.perf_counter()
    solver = make_solver(
        problem,
        solver_params,
        forward_backend=args.turbompc_forward_backend,
        backward_backend=args.turbompc_backward_backend,
    )
    solver_setup_s = time.perf_counter() - solver_setup_start

    initialize_fn = make_cartpole_initialization_kernel(
        n_steps=args.horizon_steps,
        dt_start=args.dt_start,
        dt_growth=args.dt_growth,
        dtype=dtype,
    )
    init_start = time.perf_counter()
    z0, params0 = initialize_fn(jnp.asarray(x0), jnp.asarray(cost_params))
    states0, controls0 = jax.vmap(problem.split_packed_z)(z0)
    env_params_jax = jnp.asarray(_nominal_env_parameters(args.batch_size, dtype))
    admm_state0 = initial_admm_state_batch(
        solver,
        states0,
        controls0,
        params0,
        rho=args.rho,
    )
    jax.block_until_ready(admm_state0.x_blocks)
    initialization_s = time.perf_counter() - init_start

    rollout_fn = _make_rollout(args, solver, dtype)
    key = jax.random.PRNGKey(args.seed + 1_000_003)
    noise_scale = jnp.asarray(args.noise_scale, dtype=jnp.dtype(dtype))
    compile_start = time.perf_counter()
    compiled_rollout = rollout_fn.lower(
        states0,
        controls0,
        params0,
        jnp.asarray(x0),
        env_params_jax,
        key,
        noise_scale,
        admm_state0,
    ).compile()
    rollout_compile_s = time.perf_counter() - compile_start

    start = time.perf_counter()
    output = compiled_rollout(
        states0,
        controls0,
        params0,
        jnp.asarray(x0),
        env_params_jax,
        key,
        noise_scale,
        admm_state0,
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
        "qp_solver": "turbompc_autodiff",
        "linearization_backend": "turbompc_jax_autodiff",
        "turbompc_forward_backend": args.turbompc_forward_backend,
        "turbompc_backward_backend": args.turbompc_backward_backend,
        "osqp_max_iter": args.osqp_max_iter,
        "turbompc_eps_abs": args.turbompc_eps_abs,
        "turbompc_eps_rel": args.turbompc_eps_rel,
        "rho": args.rho,
        "sigma": args.sigma,
        "alpha": args.alpha,
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
        "n_variables": desc.n_variables,
        "n_constraints": desc.n_constraints,
        "nnz_p": desc.nnz_p,
        "nnz_a": desc.nnz_a,
        "dense_block_dim": desc.block_dim,
        "dense_inequality_dim": desc.inequality_dim,
        "problem_build_s": problem_build_s,
        "solver_setup_s": solver_setup_s,
        "initialization_s": initialization_s,
        "rollout_compile_s": rollout_compile_s,
        "elapsed_s": elapsed_s,
        "total_closed_loop_steps": total_closed_loop_steps,
        "total_sqp_iterations": total_sqp_iterations,
        "closed_loop_steps_per_s": total_closed_loop_steps / elapsed_s,
        "sqp_iterations_per_s": total_sqp_iterations / elapsed_s,
        "rti_steps_per_s": total_sqp_iterations / elapsed_s,
        "plot_path": "" if args.skip_state_plot else str(args.plot_path),
        "cost_params": {
            name: float(value)
            for name, value in zip(CARTPOLE_COST_PARAMETER_NAMES, TUNED_COST_PARAMS, strict=True)
        },
    }
    summary.update(rollout_summary)
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({summary['closed_loop_steps_per_s']:.3g} closed-loop steps/s, "
        f"{summary['sqp_iterations_per_s']:.3g} SQP iterations/s), "
        f"return_mean={summary['return_mean']:.3g}, "
        f"success={summary['rollout_success_rate']:.2%}",
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
    parser.add_argument("--osqp-max-iter", "--max-iter", dest="osqp_max_iter", type=int, default=25)
    parser.add_argument("--turbompc-eps-abs", type=float, default=1e-3)
    parser.add_argument("--turbompc-eps-rel", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
    parser.add_argument("--turbompc-forward-backend", default="admm_fused_cudss")
    parser.add_argument("--turbompc-backward-backend", default="direct_cudss_ffi")
    parser.add_argument("--enable-rail-constraint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--process-noise-scale", default=ZERO_PROCESS_NOISE_SCALE)
    parser.add_argument("--input-noise-scale", default=ZERO_INPUT_NOISE_SCALE)
    parser.add_argument("--initial-position-range", type=float, default=0.30)
    parser.add_argument("--initial-angle-spread-deg", type=float, default=20.0)
    parser.add_argument("--initial-velocity-std", type=float, default=0.15)
    parser.add_argument("--initial-omega-std", type=float, default=0.5)
    parser.add_argument("--plot-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/turbompc_autodiff_cartpole_rollout.png"))
    parser.add_argument("--plot-samples", type=int, default=2048)
    parser.add_argument("--skip-state-plot", action="store_true")
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.rollout_steps is None:
        args.rollout_steps = int(math.ceil(args.sim_time / args.control_dt))
    summary = run(args)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
