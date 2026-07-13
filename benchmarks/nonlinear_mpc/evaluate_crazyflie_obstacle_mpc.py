#!/usr/bin/env python3
"""Closed-loop Crazyflie obstacle-avoidance benchmark for JAX-OSQP and MPX."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from functools import partial

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

from benchmarks.nonlinear_mpc.evaluate_crazyflie_sqp import _plot_state_distribution
from benchmarks.nonlinear_mpc.evaluate_mpx_crazyflie_sqp import (
    _motor_bias,
    _motor_mixing_matrix,
    physical_initial_guess_and_reference,
)
from benchmarks.nonlinear_mpc.mpx_penalty_adapter import (
    MPXBarrierSettings,
    ilqr_mpc,
    relaxed_barrier,
    relaxed_barrier_curvature,
    settings_array,
)
from benchmarks.problems.crazyflie_obstacle_sqp import (
    CRAZYFLIE_CYLINDER_OBSTACLES,
    crazyflie_obstacle_inflated_radii,
    crazyflie_obstacle_initial_guess_and_params,
    crazyflie_obstacle_penetration,
    crazyflie_obstacle_squared_margins,
    crazyflie_obstacle_squared_violation_jax,
    make_crazyflie_obstacle_sqp_problem,
    sample_crazyflie_obstacle_border_initial_states,
    sample_crazyflie_obstacle_initial_states,
)
from benchmarks.problems.crazyflie_sqp import (
    CRAZYFLIE_ACTION_SCALING,
    CRAZYFLIE_ARM,
    CRAZYFLIE_G,
    CRAZYFLIE_M,
    CRAZYFLIE_NU,
    CRAZYFLIE_NX,
    CRAZYFLIE_NZ,
    CRAZYFLIE_Q,
    CRAZYFLIE_R,
    CRAZYFLIE_THRUST_MAX,
    CRAZYFLIE_THRUST_TO_TORQUE,
    crazyflie_dt_schedule,
    crazyflie_jax_euler_step,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    FilterLineSearchSettings,
    MPAXSettings,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)
from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers as mpx_optimizers


@dataclass(frozen=True)
class CrazyflieObstacleMPXProblem:
    horizon: int
    settings: MPXBarrierSettings
    solve: object
    motor_violation: object
    obstacle_violation: object
    dynamics_defect: object

    @property
    def n_variables(self) -> int:
        return (self.horizon + 1) * CRAZYFLIE_NX + self.horizon * CRAZYFLIE_NU

    @property
    def n_hard_constraints(self) -> int:
        return (self.horizon + 1) * CRAZYFLIE_NX

    @property
    def n_barrier_constraints(self) -> int:
        return self.horizon * CRAZYFLIE_NU + (self.horizon + 1) * CRAZYFLIE_CYLINDER_OBSTACLES.shape[0]


def _motor_matrix_np() -> tuple[np.ndarray, np.ndarray]:
    inv_arm = 1.0 / CRAZYFLIE_ARM
    inv_k = 1.0 / CRAZYFLIE_THRUST_TO_TORQUE
    mixer = 0.25 * np.array(
        [
            [-inv_arm, -inv_arm, -inv_k, 1.0],
            [-inv_arm, +inv_arm, +inv_k, 1.0],
            [+inv_arm, +inv_arm, -inv_k, 1.0],
            [+inv_arm, -inv_arm, +inv_k, 1.0],
        ],
        dtype=np.float64,
    )
    scale = np.asarray(CRAZYFLIE_ACTION_SCALING, dtype=np.float64)
    wrench_bias = np.asarray([0.0, 0.0, 0.0, CRAZYFLIE_M * CRAZYFLIE_G], dtype=np.float64)
    return mixer * scale[None, :], mixer @ wrench_bias


def _motor_constraint_violation_np(controls: np.ndarray) -> np.ndarray:
    if controls.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    matrix, offset = _motor_matrix_np()
    motors = np.einsum("ij,...j->...i", matrix, np.asarray(controls, dtype=np.float64)) + offset
    lower = np.maximum(-motors, 0.0)
    upper = np.maximum(motors - CRAZYFLIE_THRUST_MAX, 0.0)
    return np.max(np.maximum(lower, upper), axis=-1)


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


def _summarize_rollout(
    *,
    args: argparse.Namespace,
    states_arr: np.ndarray,
    controls_arr: np.ndarray,
    step_lengths_arr: np.ndarray,
    violations_arr: np.ndarray,
    motor_violations_arr: np.ndarray | None,
    obstacle_violations_arr: np.ndarray | None,
    dynamics_defects_arr: np.ndarray | None,
    finite_arr: np.ndarray | None,
    elapsed_s: float,
    extra: dict[str, object],
) -> dict[str, object]:
    if motor_violations_arr is None:
        motor_violations_arr = _motor_constraint_violation_np(controls_arr)
    obstacle_pen = crazyflie_obstacle_penetration(states_arr)
    obstacle_pen_max = np.max(obstacle_pen, axis=-1)
    obstacle_sq = np.maximum(-crazyflie_obstacle_squared_margins(states_arr), 0.0)
    obstacle_sq_max = np.max(obstacle_sq, axis=-1)
    obstacle_stats, obstacle_rollout_severity = _constraint_violation_stats("obstacle", obstacle_pen_max)
    motor_stats, motor_rollout_severity = _constraint_violation_stats("motor", motor_violations_arr)
    input_stats = {
        key.replace("motor_", "input_", 1): value
        for key, value in motor_stats.items()
        if key.startswith("motor_")
    }
    combined_rollout_severity = np.maximum(obstacle_rollout_severity, motor_rollout_severity)
    combined_rollout_satisfaction = float(np.mean(combined_rollout_severity <= 1.0e-6))
    combined_rollout_violation_rate = float(np.mean(combined_rollout_severity > 1.0e-6))
    final_state = states_arr[-1]
    final_position_norm = np.linalg.norm(final_state[:, :3], axis=1)
    final_state_cost = 0.5 * np.sum(final_state * (np.diag(CRAZYFLIE_Q)[None, :] * final_state), axis=1)
    finite_rate = float(np.mean(finite_arr)) if finite_arr is not None and finite_arr.size else 1.0
    rti_steps_per_s = args.batch_size * args.sim_steps / elapsed_s
    summary: dict[str, object] = {
        "batch_size": args.batch_size,
        "horizon_steps": args.horizon_steps,
        "sim_steps": args.sim_steps,
        "sim_time": args.sim_time,
        "control_dt": args.control_dt,
        "sqp_iterations": args.sqp_iterations,
        "dtype": str(np.dtype(args.dtype)),
        "solver": args.solver,
        "seed": args.seed,
        "trajectory_initialization": args.trajectory_initialization,
        "initial_state_sampling": args.initial_state_sampling,
        "obstacles": CRAZYFLIE_CYLINDER_OBSTACLES.tolist(),
        "obstacle_inflated_radii": crazyflie_obstacle_inflated_radii().tolist(),
        "finite_rate": finite_rate,
        "tracking_success_rate_10cm": float(np.mean(final_position_norm <= 0.10)),
        "tracking_success_rate_20cm": float(np.mean(final_position_norm <= 0.20)),
        "final_position_rms": float(np.sqrt(np.mean(final_state[:, :3] ** 2))),
        "final_position_norm_mean": float(np.mean(final_position_norm)),
        "final_position_norm_p95": float(np.percentile(final_position_norm, 95.0)),
        "final_state_rms": float(np.sqrt(np.mean(final_state**2))),
        "final_state_cost_mean": float(np.mean(final_state_cost)),
        "final_state_cost_median": float(np.median(final_state_cost)),
        "final_state_cost_max": float(np.max(final_state_cost)),
        "mean_step": float(np.mean(step_lengths_arr)) if step_lengths_arr.size else float("nan"),
        "max_violation": float(np.max(violations_arr)) if violations_arr.size else float("nan"),
        "combined_violation_rate_gt_1e-6": max(
            obstacle_stats["obstacle_violation_rate_gt_1e-6"],
            motor_stats["motor_violation_rate_gt_1e-6"],
        ),
        "combined_constraint_satisfaction_gt_1e-6": min(
            obstacle_stats["obstacle_constraint_satisfaction_gt_1e-6"],
            motor_stats["motor_constraint_satisfaction_gt_1e-6"],
        ),
        "combined_rollout_violation_rate_gt_1e-6": combined_rollout_violation_rate,
        "combined_rollout_constraint_satisfaction_gt_1e-6": combined_rollout_satisfaction,
        "combined_rollout_violation_severity_mean": float(np.mean(combined_rollout_severity)),
        "combined_rollout_violation_severity_p95": float(np.percentile(combined_rollout_severity, 95.0)),
        "combined_rollout_violation_severity_p99": float(np.percentile(combined_rollout_severity, 99.0)),
        "combined_rollout_violation_severity_max": float(np.max(combined_rollout_severity)),
        "obstacle_penetration_mean": float(np.mean(obstacle_pen_max)),
        "obstacle_penetration_p99": float(np.percentile(obstacle_pen_max, 99.0)),
        "obstacle_penetration_max": float(np.max(obstacle_pen_max)),
        "obstacle_squared_violation_mean": float(np.mean(obstacle_sq_max)),
        "obstacle_squared_violation_p99": float(np.percentile(obstacle_sq_max, 99.0)),
        "obstacle_squared_violation_max": float(np.max(obstacle_sq_max)),
        "elapsed_s": elapsed_s,
        "solve_s": elapsed_s,
        "rti_steps_per_s": rti_steps_per_s,
        "sqp_iterations_per_s": rti_steps_per_s * args.sqp_iterations,
    }
    summary.update(obstacle_stats)
    summary.update(motor_stats)
    summary.update(input_stats)
    if obstacle_violations_arr is not None and obstacle_violations_arr.size:
        summary["planned_obstacle_squared_violation_max"] = float(np.max(obstacle_violations_arr))
    if dynamics_defects_arr is not None and dynamics_defects_arr.size:
        summary["max_dynamics_defect"] = float(np.max(dynamics_defects_arr))
    summary.update(extra)
    return summary


def _plot_obstacle_position_cloud(path: pathlib.Path, states: np.ndarray, *, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = min(states.shape[1], 256)
    sample_idx = np.linspace(0, states.shape[1] - 1, sample_count, dtype=np.int64)
    stride = max(1, states.shape[0] // 100)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))

    ax = axes[0]
    for sample in sample_idx:
        trajectory = states[::stride, sample, :3]
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.22, linewidth=0.9)
    ax.scatter(states[0, sample_idx, 0], states[0, sample_idx, 1], s=10, color="tab:red", label="initial")
    ax.scatter(states[-1, sample_idx, 0], states[-1, sample_idx, 1], s=10, color="tab:green", label="final")
    inflated = crazyflie_obstacle_inflated_radii()
    for idx, (cx, cy, radius) in enumerate(CRAZYFLIE_CYLINDER_OBSTACLES):
        ax.add_patch(Circle((cx, cy), radius, color="#8b5a2b", alpha=0.35))
        ax.add_patch(Circle((cx, cy), inflated[idx], fill=False, color="#5c4033", linewidth=1.5))
    ax.scatter([0.0], [0.0], s=70, marker="*", color="black", label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.set_title("top-down trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax3 = fig.add_subplot(122, projection="3d")
    for sample in sample_idx:
        trajectory = states[::stride, sample, :3]
        ax3.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.18, linewidth=0.8)
    ax3.scatter(states[0, sample_idx, 0], states[0, sample_idx, 1], states[0, sample_idx, 2], s=8, color="tab:red")
    ax3.scatter(states[-1, sample_idx, 0], states[-1, sample_idx, 1], states[-1, sample_idx, 2], s=8, color="tab:green")
    ax3.set_xlabel("px")
    ax3.set_ylabel("py")
    ax3.set_zlabel("pz")
    ax3.set_title("3D trajectories")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_clearance_distribution(
    path: pathlib.Path,
    time_grid: np.ndarray,
    states: np.ndarray,
    *,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    penetration = np.max(crazyflie_obstacle_penetration(states), axis=-1)
    clearance = np.min(crazyflie_obstacle_squared_margins(states), axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)
    for ax, values, ylabel in (
        (axes[0], penetration, "max obstacle penetration [m]"),
        (axes[1], clearance, "min squared-distance margin [m^2]"),
    ):
        median = np.median(values, axis=1)
        low = np.percentile(values, 10.0, axis=1)
        high = np.percentile(values, 90.0, axis=1)
        max_value = np.max(values, axis=1)
        min_value = np.min(values, axis=1)
        ax.fill_between(time_grid, min_value, max_value, color="tab:purple", alpha=0.10, linewidth=0.0)
        ax.fill_between(time_grid, low, high, color="tab:purple", alpha=0.25, linewidth=0.0)
        ax.plot(time_grid, median, color="tab:purple", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compile_physical_crazyflie_obstacle_mpx_problem(
    horizon: int,
    *,
    settings: MPXBarrierSettings,
    dtype: np.dtype | str,
) -> CrazyflieObstacleMPXProblem:
    jdtype = jnp.dtype(dtype)
    q_diag = jnp.asarray(np.diag(CRAZYFLIE_Q), dtype=jdtype)
    r_diag = jnp.asarray(np.diag(CRAZYFLIE_R), dtype=jdtype)
    motor_matrix = _motor_mixing_matrix(jdtype)
    motor_offset = _motor_bias(jdtype)
    thrust_max = jnp.asarray(CRAZYFLIE_THRUST_MAX, dtype=jdtype)
    obstacle_centers = jnp.asarray(CRAZYFLIE_CYLINDER_OBSTACLES[:, :2], dtype=jdtype)
    obstacle_radii = jnp.asarray(crazyflie_obstacle_inflated_radii(), dtype=jdtype)

    def motors(u):
        return motor_matrix @ u + motor_offset

    def obstacle_margins(x):
        delta = x[:2][None, :] - obstacle_centers
        return jnp.sum(delta * delta, axis=1) - obstacle_radii * obstacle_radii

    def motor_barrier(u, settings_vec):
        motor = motors(u)
        alpha = jnp.asarray(settings_vec[1], dtype=u.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=u.dtype)
        return jnp.sum(relaxed_barrier(motor, alpha, sigma) + relaxed_barrier(thrust_max - motor, alpha, sigma))

    def obstacle_barrier(x, settings_vec):
        alpha = jnp.asarray(settings_vec[1], dtype=x.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=x.dtype)
        return jnp.sum(relaxed_barrier(obstacle_margins(x), alpha, sigma))

    def motor_barrier_hessian(u, settings_vec):
        motor = motors(u)
        alpha = jnp.asarray(settings_vec[1], dtype=u.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=u.dtype)
        weights = relaxed_barrier_curvature(motor, alpha, sigma)
        weights += relaxed_barrier_curvature(thrust_max - motor, alpha, sigma)
        return (motor_matrix * weights[:, None]).T @ motor_matrix

    def obstacle_barrier_hessian(x, settings_vec):
        alpha = jnp.asarray(settings_vec[1], dtype=x.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=x.dtype)
        delta = x[:2][None, :] - obstacle_centers
        gradients = 2.0 * delta
        weights = relaxed_barrier_curvature(obstacle_margins(x), alpha, sigma)
        hxy = (gradients * weights[:, None]).T @ gradients
        h = jnp.zeros((CRAZYFLIE_NX, CRAZYFLIE_NX), dtype=x.dtype)
        return h.at[:2, :2].set(hxy)

    def cost(settings_vec, reference, x, u, t):
        idx = jnp.minimum(t, horizon)
        x_ref = reference[idx, :CRAZYFLIE_NX]
        dt = reference[jnp.minimum(t, horizon - 1), CRAZYFLIE_NX]
        residual = x - x_ref
        state_cost = 0.5 * jnp.sum(q_diag * residual * residual)
        control_cost = 0.5 * jnp.sum(r_diag * u * u)
        obs_cost = obstacle_barrier(x, settings_vec)
        stage_cost = dt * (state_cost + control_cost) + motor_barrier(u, settings_vec) + obs_cost
        terminal_cost = state_cost + obs_cost
        return jnp.where(t < horizon, stage_cost, terminal_cost)

    def dynamics(x, u, t, parameter):
        dt = parameter[jnp.minimum(t, horizon - 1)]
        return crazyflie_jax_euler_step(x[None, :], u[None, :], dt)[0]

    def hessian_approx(settings_vec, reference, x, u, t):
        dt = reference[jnp.minimum(t, horizon - 1), CRAZYFLIE_NX]
        obs_h = obstacle_barrier_hessian(x, settings_vec)
        q_stage = jnp.diag(dt * q_diag) + obs_h
        r_stage = jnp.diag(dt * r_diag) + motor_barrier_hessian(u, settings_vec)
        q_terminal = jnp.diag(q_diag) + obs_h
        r_terminal = jnp.zeros((CRAZYFLIE_NU, CRAZYFLIE_NU), dtype=x.dtype)
        q = jnp.where(t < horizon, q_stage, q_terminal)
        r = jnp.where(t < horizon, r_stage, r_terminal)
        m = jnp.zeros((CRAZYFLIE_NX, CRAZYFLIE_NU), dtype=x.dtype)
        return q, r, m

    if settings.solver_mode == "primal_dual":
        work = partial(
            mpx_optimizers.mpc,
            cost,
            dynamics,
            hessian_approx,
            settings.limited_memory,
            num_alpha=settings.num_alpha,
        )
    elif settings.solver_mode == "fddp":

        def work(reference, parameter, settings_vec, x0, x, u, v):
            del v
            return mpx_optimizers.fddp_mpc(
                cost,
                dynamics,
                hessian_approx,
                settings.limited_memory,
                reference,
                parameter,
                settings_vec,
                x0,
                x,
                u,
                settings.num_alpha,
            )

    elif settings.solver_mode == "ilqr":

        def work(reference, parameter, settings_vec, x0, x, u, v):
            del v
            return ilqr_mpc(
                cost,
                dynamics,
                hessian_approx,
                settings.limited_memory,
                reference,
                parameter,
                settings_vec,
                x0,
                x,
                u,
                settings.num_alpha,
            )

    else:
        raise ValueError(f"unsupported MPX solver mode: {settings.solver_mode!r}")

    def one_motor_violation(u):
        motor = jax.vmap(motors)(u)
        lower_violation = jnp.maximum(-motor, 0.0)
        upper_violation = jnp.maximum(motor - thrust_max, 0.0)
        return jnp.max(jnp.maximum(lower_violation, upper_violation))

    def one_obstacle_violation(x):
        return jnp.max(crazyflie_obstacle_squared_violation_jax(x))

    def one_dynamics_defect(x, u, parameter):
        def stage_defect(t):
            return dynamics(x[t], u[t], t, parameter) - x[t + 1]

        defects = jax.vmap(stage_defect)(jnp.arange(horizon))
        return jnp.max(jnp.abs(defects))

    return CrazyflieObstacleMPXProblem(
        horizon=horizon,
        settings=settings,
        solve=jax.jit(jax.vmap(work)),
        motor_violation=jax.jit(jax.vmap(one_motor_violation)),
        obstacle_violation=jax.jit(jax.vmap(one_obstacle_violation)),
        dynamics_defect=jax.jit(jax.vmap(one_dynamics_defect)),
    )


def _run_jax_osqp(args: argparse.Namespace, x0: np.ndarray) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    settings = OSQPSettings(
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        max_iter=args.max_iter,
        scaling=args.osqp_scaling,
        adaptive_rho=False,
        rho_is_vec=args.osqp_rho_is_vec,
        check_termination=0,
        warm_starting=True,
        polishing=False,
    )
    line_search_settings = FilterLineSearchSettings(line_search_step_min=args.line_search_step_min)
    problem_start = time.perf_counter()
    problem = make_crazyflie_obstacle_sqp_problem(
        args.horizon_steps,
        obstacle_constraint_scale=args.obstacle_constraint_scale,
    )
    problem_build_s = time.perf_counter() - problem_start
    plan_start = time.perf_counter()
    plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
    plan_build_s = time.perf_counter() - plan_start
    compile_setup_start = time.perf_counter()
    sqp = compile_sparse_mpc_sqp(
        problem,
        plan,
        dtype=dtype,
        qp_solver="jax_osqp",
        osqp_settings=settings,
        mpax_settings=MPAXSettings(),
        transpose_work=args.transpose_work,
        segmented=args.segmented,
        segment_budget=args.segment_budget,
        segment_strategy=args.segment_strategy,
        level_scheduled_solve=args.level_scheduled_solve,
        level_scheduled_solve_threshold=args.level_scheduled_solve_threshold,
        qdldl_backend=args.qdldl_backend,
        qdldl_factor_backend=args.qdldl_factor_backend,
        qdldl_solve_backend=args.qdldl_solve_backend,
        line_search_settings=line_search_settings,
        group_repeated_stages=args.group_repeated_stages,
    )
    compile_setup_s = time.perf_counter() - compile_setup_start
    z0, params0 = crazyflie_obstacle_initial_guess_and_params(
        x0,
        n_steps=args.horizon_steps,
        dtype=dtype,
        trajectory_initialization=args.trajectory_initialization,
    )
    control_dt = jnp.asarray(args.control_dt, dtype=jnp.dtype(dtype))

    @jax.jit
    def closed_loop_step(z, params, x, solver_state):
        params = params.at[:, :CRAZYFLIE_NX].set(x)

        def sqp_body(carry, _):
            z_iter, solver_state_iter = carry
            result_iter, solver_state_next = sqp.step(z_iter, params, state=solver_state_iter)
            return (result_iter.z_next, solver_state_next), result_iter

        (z_next, solver_state_next), sqp_results = jax.lax.scan(
            sqp_body,
            (z, solver_state),
            xs=None,
            length=int(args.sqp_iterations),
        )
        result = jax.tree_util.tree_map(lambda leaf: leaf[-1], sqp_results)
        stages = z_next.reshape((z.shape[0], args.horizon_steps + 1, CRAZYFLIE_NZ))
        u0 = stages[:, 0, CRAZYFLIE_NX : CRAZYFLIE_NX + CRAZYFLIE_NU]
        x_next = crazyflie_jax_euler_step(x, u0, control_dt)
        params_next = params.at[:, :CRAZYFLIE_NX].set(x_next)
        return (
            z_next,
            params_next,
            x_next,
            u0,
            result.line_search.step_length,
            result.line_search.constraint_violation,
            result.solve.prim_res,
            result.solve.dual_res,
            solver_state_next,
        )

    z = jnp.asarray(z0)
    params = jnp.asarray(params0)
    x = jnp.asarray(x0)
    solver_state = sqp.init_state(args.batch_size)
    warmup_start = time.perf_counter()
    warmup = closed_loop_step(z, params, x, solver_state)
    jax.block_until_ready(warmup[2])
    warmup_s = time.perf_counter() - warmup_start

    states = [np.asarray(jax.device_get(x))]
    controls = []
    step_lengths = []
    violations = []
    prim_res = []
    dual_res = []
    start = time.perf_counter()
    for _ in range(args.sim_steps):
        z, params, x, u0, step_len, violation, primal, dual, solver_state = closed_loop_step(
            z,
            params,
            x,
            solver_state,
        )
        states.append(np.asarray(jax.device_get(x)))
        controls.append(np.asarray(jax.device_get(u0)))
        step_lengths.append(np.asarray(jax.device_get(step_len)))
        violations.append(np.asarray(jax.device_get(violation)))
        prim_res.append(np.asarray(jax.device_get(primal)))
        dual_res.append(np.asarray(jax.device_get(dual)))
    elapsed_s = time.perf_counter() - start

    states_arr = np.stack(states, axis=0)
    controls_arr = np.stack(controls, axis=0)
    step_lengths_arr = np.stack(step_lengths, axis=0)
    violations_arr = np.stack(violations, axis=0)
    prim_res_arr = np.stack(prim_res, axis=0)
    dual_res_arr = np.stack(dual_res, axis=0)
    return _finish_run(
        args,
        states_arr=states_arr,
        controls_arr=controls_arr,
        step_lengths_arr=step_lengths_arr,
        violations_arr=violations_arr,
        motor_violations_arr=None,
        obstacle_violations_arr=None,
        dynamics_defects_arr=None,
        finite_arr=None,
        elapsed_s=elapsed_s,
        extra={
            "qp_solver": "jax_osqp",
            "max_iter": args.max_iter,
            "osqp_scaling": args.osqp_scaling,
            "osqp_rho_is_vec": bool(args.osqp_rho_is_vec),
            "line_search_step_min": args.line_search_step_min,
            "obstacle_constraint_scale": args.obstacle_constraint_scale,
            "trajectory_initialization": args.trajectory_initialization,
            "qdldl_backend": args.qdldl_backend,
            "qdldl_factor_backend": args.qdldl_factor_backend,
            "qdldl_solve_backend": args.qdldl_solve_backend,
            "transpose_work": bool(args.transpose_work),
            "segmented": bool(args.segmented),
            "segment_budget": args.segment_budget,
            "segment_strategy": args.segment_strategy,
            "group_repeated_stages": bool(args.group_repeated_stages),
            "level_scheduled_solve": bool(args.level_scheduled_solve),
            "level_scheduled_solve_threshold": args.level_scheduled_solve_threshold,
            "n_variables": plan.n_variables,
            "n_constraints": plan.n_constraints,
            "nnz_p": plan.p_pattern.nnz,
            "nnz_a": plan.a_pattern.nnz,
            "problem_build_s": problem_build_s,
            "plan_build_s": plan_build_s,
            "compile_setup_s": compile_setup_s,
            "warmup_compile_and_run_s": warmup_s,
            "compile_s": warmup_s,
            "total_compile_s": problem_build_s + plan_build_s + compile_setup_s + warmup_s,
            "mean_qp_prim": float(np.mean(prim_res_arr)),
            "mean_qp_dual": float(np.mean(dual_res_arr)),
        },
    )


def _run_mpx(args: argparse.Namespace, x0: np.ndarray) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    barrier_settings = MPXBarrierSettings(
        equality_weight=args.mpx_equality_weight,
        barrier_alpha=args.mpx_barrier_alpha,
        barrier_sigma=args.mpx_barrier_sigma,
        num_alpha=args.mpx_num_alpha,
        limited_memory=args.mpx_limited_memory,
        solver_mode=args.mpx_solver_mode,
    )
    problem_start = time.perf_counter()
    problem = compile_physical_crazyflie_obstacle_mpx_problem(
        args.horizon_steps,
        settings=barrier_settings,
        dtype=dtype,
    )
    problem_build_s = time.perf_counter() - problem_start
    init_start = time.perf_counter()
    x_nodes0_np, u_nodes0_np, reference0_np, parameter_np = physical_initial_guess_and_reference(
        x0,
        n_steps=args.horizon_steps,
        dtype=dtype,
    )
    if args.trajectory_initialization == "initial_state":
        x_nodes0_np[:, :, :] = x0[:, None, :]
    elif args.trajectory_initialization != "linear":
        raise ValueError(f"unsupported trajectory initialization: {args.trajectory_initialization!r}")
    dual0_np = np.zeros_like(x_nodes0_np)
    initialization_s = time.perf_counter() - init_start
    x_nodes0 = jnp.asarray(x_nodes0_np)
    u_nodes0 = jnp.asarray(u_nodes0_np)
    dual0 = jnp.asarray(dual0_np)
    reference0 = jnp.asarray(reference0_np)
    parameter = jnp.asarray(parameter_np)
    settings_vec = jnp.asarray(settings_array(barrier_settings, args.batch_size, dtype))
    x = jnp.asarray(x0)
    control_dt = jnp.asarray(args.control_dt, dtype=jnp.dtype(dtype))

    @jax.jit
    def closed_loop_step(x_nodes, u_nodes, dual_nodes, reference, x_cur):
        x_nodes = x_nodes.at[:, 0, :].set(x_cur)
        x0_physical = x_nodes[:, 0]

        def mpx_body(iter_carry, _):
            x_iter, u_iter, dual_iter = iter_carry
            x_next_iter, u_next_iter, dual_next_iter = problem.solve(
                reference,
                parameter,
                settings_vec,
                x0_physical,
                x_iter,
                u_iter,
                dual_iter,
            )
            return (x_next_iter, u_next_iter, dual_next_iter), None

        (x_next_nodes, u_next_nodes, dual_next_nodes), _ = jax.lax.scan(
            mpx_body,
            (x_nodes, u_nodes, dual_nodes),
            xs=None,
            length=int(args.sqp_iterations),
        )
        u0 = u_next_nodes[:, 0, :]
        x_next = crazyflie_jax_euler_step(x_cur, u0, control_dt)
        motor_violation = problem.motor_violation(u_next_nodes)
        obstacle_violation = problem.obstacle_violation(x_next_nodes)
        dynamics_defect = problem.dynamics_defect(x_next_nodes, u_next_nodes, parameter)
        finite = (
            jnp.all(jnp.isfinite(x_next_nodes), axis=(1, 2))
            & jnp.all(jnp.isfinite(u_next_nodes), axis=(1, 2))
            & jnp.all(jnp.isfinite(x_next), axis=1)
            & jnp.all(jnp.isfinite(u0), axis=1)
        )
        violation = jnp.maximum(jnp.maximum(motor_violation, obstacle_violation), dynamics_defect)
        return (
            x_next_nodes,
            u_next_nodes,
            dual_next_nodes,
            reference,
            x_next,
            u0,
            jnp.ones((x_cur.shape[0],), dtype=x_cur.dtype),
            violation,
            motor_violation,
            obstacle_violation,
            dynamics_defect,
            finite,
        )

    warmup_start = time.perf_counter()
    warmup = closed_loop_step(x_nodes0, u_nodes0, dual0, reference0, x)
    jax.block_until_ready(warmup[4])
    warmup_s = time.perf_counter() - warmup_start
    x_nodes = x_nodes0
    u_nodes = u_nodes0
    dual_nodes = dual0
    reference = reference0
    states = [np.asarray(jax.device_get(x))]
    controls = []
    step_lengths = []
    violations = []
    motor_violations = []
    obstacle_violations = []
    dynamics_defects = []
    finite_flags = []
    start = time.perf_counter()
    for _ in range(args.sim_steps):
        (
            x_nodes,
            u_nodes,
            dual_nodes,
            reference,
            x,
            u0,
            step_len,
            violation,
            motor_violation,
            obstacle_violation,
            dynamics_defect,
            finite,
        ) = closed_loop_step(x_nodes, u_nodes, dual_nodes, reference, x)
        states.append(np.asarray(jax.device_get(x)))
        controls.append(np.asarray(jax.device_get(u0)))
        step_lengths.append(np.asarray(jax.device_get(step_len)))
        violations.append(np.asarray(jax.device_get(violation)))
        motor_violations.append(np.asarray(jax.device_get(motor_violation)))
        obstacle_violations.append(np.asarray(jax.device_get(obstacle_violation)))
        dynamics_defects.append(np.asarray(jax.device_get(dynamics_defect)))
        finite_flags.append(np.asarray(jax.device_get(finite)))
    elapsed_s = time.perf_counter() - start

    states_arr = np.stack(states, axis=0)
    controls_arr = np.stack(controls, axis=0)
    step_lengths_arr = np.stack(step_lengths, axis=0)
    violations_arr = np.stack(violations, axis=0)
    motor_violations_arr = np.stack(motor_violations, axis=0)
    obstacle_violations_arr = np.stack(obstacle_violations, axis=0)
    dynamics_defects_arr = np.stack(dynamics_defects, axis=0)
    finite_arr = np.stack(finite_flags, axis=0)
    return _finish_run(
        args,
        states_arr=states_arr,
        controls_arr=controls_arr,
        step_lengths_arr=step_lengths_arr,
        violations_arr=violations_arr,
        motor_violations_arr=motor_violations_arr,
        obstacle_violations_arr=obstacle_violations_arr,
        dynamics_defects_arr=dynamics_defects_arr,
        finite_arr=finite_arr,
        elapsed_s=elapsed_s,
        extra={
            "qp_solver": "mpx",
            "mpx_solver_mode": args.mpx_solver_mode,
            "mpx_equality_weight": args.mpx_equality_weight,
            "mpx_barrier_alpha": args.mpx_barrier_alpha,
            "mpx_barrier_sigma": args.mpx_barrier_sigma,
            "mpx_num_alpha": args.mpx_num_alpha,
            "mpx_limited_memory": bool(args.mpx_limited_memory),
            "trajectory_initialization": args.trajectory_initialization,
            "n_variables": problem.n_variables,
            "n_constraints": problem.n_hard_constraints + problem.n_barrier_constraints,
            "n_hard_constraints": problem.n_hard_constraints,
            "n_barrier_constraints": problem.n_barrier_constraints,
            "problem_build_s": problem_build_s,
            "initialization_s": initialization_s,
            "warmup_compile_and_run_s": warmup_s,
            "compile_s": warmup_s,
            "total_compile_s": problem_build_s + initialization_s + warmup_s,
            "mean_qp_prim": float("nan"),
            "mean_qp_dual": float("nan"),
        },
    )


def _finish_run(
    args: argparse.Namespace,
    *,
    states_arr: np.ndarray,
    controls_arr: np.ndarray,
    step_lengths_arr: np.ndarray,
    violations_arr: np.ndarray,
    motor_violations_arr: np.ndarray | None,
    obstacle_violations_arr: np.ndarray | None,
    dynamics_defects_arr: np.ndarray | None,
    finite_arr: np.ndarray | None,
    elapsed_s: float,
    extra: dict[str, object],
) -> dict[str, object]:
    time_grid = np.arange(args.sim_steps + 1, dtype=np.float64) * args.control_dt
    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_npz,
            time=time_grid,
            states=states_arr,
            controls=controls_arr,
            step_lengths=step_lengths_arr,
            constraint_violations=violations_arr,
            motor_constraint_violations=_motor_constraint_violation_np(controls_arr)
            if motor_violations_arr is None
            else motor_violations_arr,
            obstacle_penetration=np.max(crazyflie_obstacle_penetration(states_arr), axis=-1),
            obstacle_squared_violations=np.max(
                np.maximum(-crazyflie_obstacle_squared_margins(states_arr), 0.0),
                axis=-1,
            ),
            planned_obstacle_squared_violations=np.empty((0, args.batch_size))
            if obstacle_violations_arr is None
            else obstacle_violations_arr,
            dynamics_defects=np.empty((0, args.batch_size))
            if dynamics_defects_arr is None
            else dynamics_defects_arr,
            finite=np.ones((args.sim_steps, args.batch_size), dtype=bool)
            if finite_arr is None
            else finite_arr,
            initial_states=states_arr[0],
            obstacles=CRAZYFLIE_CYLINDER_OBSTACLES,
            obstacle_inflated_radii=crazyflie_obstacle_inflated_radii(),
        )
    if not args.skip_plots:
        _plot_state_distribution(args.plot_path, time_grid, states_arr)
        _plot_obstacle_position_cloud(
            args.position_plot_path,
            states_arr,
            title=f"Crazyflie obstacle MPC {args.solver} SQP={args.sqp_iterations}",
        )
        if args.clearance_plot_path is not None:
            _plot_clearance_distribution(
                args.clearance_plot_path,
                time_grid,
                states_arr,
                title=f"Crazyflie obstacle clearance {args.solver} SQP={args.sqp_iterations}",
            )
    summary = _summarize_rollout(
        args=args,
        states_arr=states_arr,
        controls_arr=controls_arr,
        step_lengths_arr=step_lengths_arr,
        violations_arr=violations_arr,
        motor_violations_arr=motor_violations_arr,
        obstacle_violations_arr=obstacle_violations_arr,
        dynamics_defects_arr=dynamics_defects_arr,
        finite_arr=finite_arr,
        elapsed_s=elapsed_s,
        extra=extra,
    )
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({summary['rti_steps_per_s']:.3g} RTI steps/s, "
        f"{summary['sqp_iterations_per_s']:.3g} SQP iterations/s), "
        f"final_position_rms={summary['final_position_rms']:.3e}, "
        f"track10={summary['tracking_success_rate_10cm']:.2%}, "
        f"obstacle_satisfaction={summary['obstacle_constraint_satisfaction_gt_1e-6']:.2%}, "
        f"obstacle_penetration_max={summary['obstacle_penetration_max']:.3e}, "
        f"motor_violation_max={summary['motor_violation_max']:.3e}, "
        f"max_violation={summary['max_violation']:.3e}",
        flush=True,
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    if args.qdldl_factor_backend is None:
        args.qdldl_factor_backend = args.qdldl_backend
    if args.qdldl_solve_backend is None:
        args.qdldl_solve_backend = args.qdldl_backend
    if args.initial_state_sampling == "border":
        x0 = sample_crazyflie_obstacle_border_initial_states(args.batch_size, args.seed, dtype)
    else:
        x0 = sample_crazyflie_obstacle_initial_states(args.batch_size, args.seed, dtype)
    print(
        "Crazyflie obstacle MPC:",
        f"solver={args.solver}",
        f"batch={args.batch_size}",
        f"dtype={dtype}",
        f"horizon_steps={args.horizon_steps}",
        f"sim_steps={args.sim_steps}",
        f"sqp_iterations={args.sqp_iterations}",
        f"initial_state_sampling={args.initial_state_sampling}",
        f"obstacles={CRAZYFLIE_CYLINDER_OBSTACLES.shape[0]}",
        flush=True,
    )
    if args.solver == "jax_osqp":
        return _run_jax_osqp(args, x0)
    if args.solver == "mpx":
        return _run_mpx(args, x0)
    raise ValueError(f"unsupported solver: {args.solver!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("jax_osqp", "mpx"), default="jax_osqp")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--sim-time", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--sqp-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
    parser.add_argument("--osqp-scaling", type=int, default=0)
    parser.add_argument("--osqp-rho-is-vec", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obstacle-constraint-scale", type=float, default=1.0)
    parser.add_argument("--trajectory-initialization", choices=("linear", "initial_state"), default="linear")
    parser.add_argument("--initial-state-sampling", choices=("uniform", "border"), default="uniform")
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="jax")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-budget", type=int, default=256)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action="store_true")
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument("--no-group-repeated-stages", action="store_false", dest="group_repeated_stages")
    parser.set_defaults(group_repeated_stages=True)
    parser.add_argument("--mpx-solver-mode", choices=("primal_dual", "fddp", "ilqr"), default="primal_dual")
    parser.add_argument("--mpx-equality-weight", type=float, default=1.0e4)
    parser.add_argument("--mpx-barrier-alpha", type=float, default=0.0007)
    parser.add_argument("--mpx-barrier-sigma", type=float, default=0.5)
    parser.add_argument("--mpx-num-alpha", type=int, default=11)
    parser.add_argument("--mpx-limited-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-npz",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_obstacle_mpc.npz"),
    )
    parser.add_argument("--skip-output-npz", action="store_true")
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_obstacle_mpc_states.png"),
    )
    parser.add_argument(
        "--position-plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_obstacle_mpc_positions.png"),
    )
    parser.add_argument(
        "--clearance-plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_obstacle_mpc_clearance.png"),
    )
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.skip_output_npz:
        args.output_npz = None
    if args.sim_steps is None:
        args.sim_steps = int(np.ceil(args.sim_time / args.control_dt))
    if args.sqp_iterations <= 0:
        raise ValueError("sqp iterations must be positive")
    summary = run(args)
    if args.output_npz is not None:
        print(f"Wrote {args.output_npz}")
    if not args.skip_plots:
        print(f"Wrote {args.plot_path}")
        print(f"Wrote {args.position_plot_path}")
        if args.clearance_plot_path is not None:
            print(f"Wrote {args.clearance_plot_path}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
