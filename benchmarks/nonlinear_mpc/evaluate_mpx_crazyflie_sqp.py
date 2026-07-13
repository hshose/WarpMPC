#!/usr/bin/env python3
"""Closed-loop MPX relaxed-barrier benchmark for Crazyflie MPC.

This formulation uses the physical Crazyflie state as the MPX state and the
physical action as the MPX control.  The initial state and Euler dynamics are
therefore handled by MPX's own multiple-shooting dynamics constraints, while
the non-dynamics motor-thrust bounds are kept in the scalar objective as the
relaxed barrier used by MPX examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import argparse
import json
import pathlib
import sys
import time

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
import numpy as np

from benchmarks.nonlinear_mpc.evaluate_crazyflie_sqp import (
    _plot_position_cloud,
    _plot_state_distribution,
    _sample_initial_states,
)
from benchmarks.nonlinear_mpc.mpx_penalty_adapter import (
    MPXBarrierSettings,
    ilqr_mpc,
    relaxed_barrier,
    relaxed_barrier_curvature,
    settings_array,
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
    crazyflie_initial_guess_and_params,
    crazyflie_jax_euler_step,
)
from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers as mpx_optimizers


@dataclass(frozen=True)
class CrazyfliePhysicalMPXProblem:
    horizon: int
    settings: MPXBarrierSettings
    solve: object
    violation: object
    dynamics_defect: object

    @property
    def n_variables(self) -> int:
        return (self.horizon + 1) * CRAZYFLIE_NX + self.horizon * CRAZYFLIE_NU

    @property
    def n_hard_constraints(self) -> int:
        return (self.horizon + 1) * CRAZYFLIE_NX

    @property
    def n_barrier_constraints(self) -> int:
        return self.horizon * CRAZYFLIE_NU


def _motor_mixing_matrix(dtype):
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
    return jnp.asarray(mixer * scale[None, :], dtype=dtype)


def _motor_bias(dtype):
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
    wrench_bias = np.asarray([0.0, 0.0, 0.0, CRAZYFLIE_M * CRAZYFLIE_G], dtype=np.float64)
    return jnp.asarray(mixer @ wrench_bias, dtype=dtype)


def compile_physical_crazyflie_mpx_problem(
    horizon: int,
    *,
    settings: MPXBarrierSettings,
    dtype: np.dtype | str,
) -> CrazyfliePhysicalMPXProblem:
    jdtype = jnp.dtype(dtype)
    q_diag = jnp.asarray(np.diag(CRAZYFLIE_Q), dtype=jdtype)
    r_diag = jnp.asarray(np.diag(CRAZYFLIE_R), dtype=jdtype)
    motor_matrix = _motor_mixing_matrix(jdtype)
    motor_offset = _motor_bias(jdtype)
    thrust_max = jnp.asarray(CRAZYFLIE_THRUST_MAX, dtype=jdtype)

    def motors(u):
        return motor_matrix @ u + motor_offset

    def motor_barrier(u, settings_vec):
        motor = motors(u)
        alpha = jnp.asarray(settings_vec[1], dtype=u.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=u.dtype)
        lower = relaxed_barrier(motor, alpha, sigma)
        upper = relaxed_barrier(thrust_max - motor, alpha, sigma)
        return jnp.sum(lower + upper)

    def motor_barrier_hessian(u, settings_vec):
        motor = motors(u)
        alpha = jnp.asarray(settings_vec[1], dtype=u.dtype)
        sigma = jnp.asarray(settings_vec[2], dtype=u.dtype)
        weights = relaxed_barrier_curvature(motor, alpha, sigma)
        weights += relaxed_barrier_curvature(thrust_max - motor, alpha, sigma)
        return (motor_matrix * weights[:, None]).T @ motor_matrix

    def cost(settings_vec, reference, x, u, t):
        idx = jnp.minimum(t, horizon)
        x_ref = reference[idx, :CRAZYFLIE_NX]
        dt = reference[jnp.minimum(t, horizon - 1), CRAZYFLIE_NX]
        residual = x - x_ref
        state_cost = 0.5 * jnp.sum(q_diag * residual * residual)
        control_cost = 0.5 * jnp.sum(r_diag * u * u)
        stage_cost = dt * (state_cost + control_cost) + motor_barrier(u, settings_vec)
        terminal_cost = state_cost
        return jnp.where(t < horizon, stage_cost, terminal_cost)

    def dynamics(x, u, t, parameter):
        dt = parameter[jnp.minimum(t, horizon - 1)]
        return crazyflie_jax_euler_step(x[None, :], u[None, :], dt)[0]

    def hessian_approx(settings_vec, reference, x, u, t):
        dt = reference[jnp.minimum(t, horizon - 1), CRAZYFLIE_NX]
        q_stage = jnp.diag(dt * q_diag)
        r_stage = jnp.diag(dt * r_diag) + motor_barrier_hessian(u, settings_vec)
        q_terminal = jnp.diag(q_diag)
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

    def one_violation(u):
        motor = jax.vmap(motors)(u)
        lower_violation = jnp.maximum(-motor, 0.0)
        upper_violation = jnp.maximum(motor - thrust_max, 0.0)
        return jnp.max(jnp.maximum(lower_violation, upper_violation))

    def one_dynamics_defect(x, u, parameter):
        def stage_defect(t):
            return dynamics(x[t], u[t], t, parameter) - x[t + 1]

        defects = jax.vmap(stage_defect)(jnp.arange(horizon))
        return jnp.max(jnp.abs(defects))

    return CrazyfliePhysicalMPXProblem(
        horizon=horizon,
        settings=settings,
        solve=jax.jit(jax.vmap(work)),
        violation=jax.jit(jax.vmap(one_violation)),
        dynamics_defect=jax.jit(jax.vmap(one_dynamics_defect)),
    )


def physical_initial_guess_and_reference(
    x0: np.ndarray,
    *,
    n_steps: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z0, _params0 = crazyflie_initial_guess_and_params(x0, n_steps=n_steps, dtype=dtype)
    stages = z0.reshape((x0.shape[0], n_steps + 1, CRAZYFLIE_NZ))
    x_nodes = stages[:, :, :CRAZYFLIE_NX].copy()
    u_nodes = stages[:, :n_steps, CRAZYFLIE_NX : CRAZYFLIE_NX + CRAZYFLIE_NU].copy()
    reference = np.zeros((x0.shape[0], n_steps + 1, CRAZYFLIE_NX + 1), dtype=dtype)
    dt = crazyflie_dt_schedule(n_steps).astype(dtype)
    reference[:, :n_steps, CRAZYFLIE_NX] = dt[None, :]
    parameter = np.broadcast_to(dt[None, :], (x0.shape[0], n_steps)).copy()
    return x_nodes, u_nodes, reference, parameter


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    if args.mpx_solver_mode not in {"primal_dual", "fddp", "ilqr"}:
        raise ValueError("MPX Crazyflie benchmark uses solver mode primal_dual, fddp, or ilqr")

    x0 = _sample_initial_states(args.batch_size, args.seed, dtype)

    print("building physical Crazyflie MPX problem...", flush=True)
    problem_start = time.perf_counter()
    barrier_settings = MPXBarrierSettings(
        equality_weight=args.mpx_equality_weight,
        barrier_alpha=args.mpx_barrier_alpha,
        barrier_sigma=args.mpx_barrier_sigma,
        num_alpha=args.mpx_num_alpha,
        limited_memory=args.mpx_limited_memory,
        solver_mode=args.mpx_solver_mode,
    )
    problem = compile_physical_crazyflie_mpx_problem(
        args.horizon_steps,
        settings=barrier_settings,
        dtype=dtype,
    )
    problem_build_s = time.perf_counter() - problem_start
    compile_setup_s = 0.0
    print(f"problem_build={problem_build_s:.3f}s", flush=True)

    init_start = time.perf_counter()
    x_nodes0_np, u_nodes0_np, reference0_np, parameter_np = physical_initial_guess_and_reference(
        x0,
        n_steps=args.horizon_steps,
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
    x = jnp.asarray(x0)
    control_dt = jnp.asarray(args.control_dt, dtype=jnp.dtype(dtype))

    print(
        "Crazyflie MPX MPC:",
        f"batch={args.batch_size}",
        f"dtype={dtype}",
        f"solver_mode={args.mpx_solver_mode}",
        f"horizon_steps={args.horizon_steps}",
        f"sim_steps={args.sim_steps}",
        f"sqp_iterations={args.sqp_iterations}",
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
        flush=True,
    )

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
        violation = problem.violation(u_next_nodes)
        dynamics_defect = problem.dynamics_defect(x_next_nodes, u_next_nodes, parameter)
        finite = (
            jnp.all(jnp.isfinite(x_next_nodes), axis=(1, 2))
            & jnp.all(jnp.isfinite(u_next_nodes), axis=(1, 2))
            & jnp.all(jnp.isfinite(x_next), axis=1)
            & jnp.all(jnp.isfinite(u0), axis=1)
        )
        return (
            x_next_nodes,
            u_next_nodes,
            dual_next_nodes,
            reference,
            x_next,
            u0,
            jnp.ones((x_cur.shape[0],), dtype=x_cur.dtype),
            jnp.maximum(violation, dynamics_defect),
            violation,
            dynamics_defect,
            finite,
        )

    warmup_start = time.perf_counter()
    warmup = closed_loop_step(x_nodes0, u_nodes0, dual0, reference0, x)
    jax.block_until_ready(warmup[4])
    warmup_s = time.perf_counter() - warmup_start
    print(f"warmup_compile_and_run={warmup_s:.3f}s", flush=True)

    x_nodes = x_nodes0
    u_nodes = u_nodes0
    dual_nodes = dual0
    reference = reference0
    states = [np.asarray(jax.device_get(x))]
    controls = []
    step_lengths = []
    violations = []
    motor_violations = []
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
            dynamics_defect,
            finite,
        ) = closed_loop_step(x_nodes, u_nodes, dual_nodes, reference, x)
        states.append(np.asarray(jax.device_get(x)))
        controls.append(np.asarray(jax.device_get(u0)))
        step_lengths.append(np.asarray(jax.device_get(step_len)))
        violations.append(np.asarray(jax.device_get(violation)))
        motor_violations.append(np.asarray(jax.device_get(motor_violation)))
        dynamics_defects.append(np.asarray(jax.device_get(dynamics_defect)))
        finite_flags.append(np.asarray(jax.device_get(finite)))
    elapsed_s = time.perf_counter() - start

    states_arr = np.stack(states, axis=0)
    controls_arr = (
        np.stack(controls, axis=0)
        if controls
        else np.empty((0, args.batch_size, CRAZYFLIE_NU))
    )
    step_lengths_arr = (
        np.stack(step_lengths, axis=0)
        if step_lengths
        else np.empty((0, args.batch_size))
    )
    violations_arr = np.stack(violations, axis=0) if violations else np.empty((0, args.batch_size))
    motor_violations_arr = (
        np.stack(motor_violations, axis=0) if motor_violations else np.empty((0, args.batch_size))
    )
    dynamics_defects_arr = (
        np.stack(dynamics_defects, axis=0) if dynamics_defects else np.empty((0, args.batch_size))
    )
    finite_arr = np.stack(finite_flags, axis=0) if finite_flags else np.empty((0, args.batch_size), dtype=bool)
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
            motor_constraint_violations=motor_violations_arr,
            dynamics_defects=dynamics_defects_arr,
            finite=finite_arr,
            initial_states=x0,
        )
    if not args.skip_plots:
        _plot_state_distribution(args.plot_path, time_grid, states_arr)
        if args.position_plot_path is not None:
            _plot_position_cloud(args.position_plot_path, states_arr)

    final_state = states_arr[-1]
    final_state_cost = 0.5 * np.sum(final_state * (np.diag(CRAZYFLIE_Q)[None, :] * final_state), axis=1)
    final_position_rms = float(np.sqrt(np.mean(final_state[:, :3] ** 2)))
    final_state_rms = float(np.sqrt(np.mean(final_state ** 2)))
    final_state_cost_mean = float(np.mean(final_state_cost))
    final_state_cost_median = float(np.median(final_state_cost))
    final_state_cost_max = float(np.max(final_state_cost))
    mean_step = float(np.mean(step_lengths_arr)) if step_lengths_arr.size else float("nan")
    max_violation = float(np.max(violations_arr)) if violations_arr.size else float("nan")
    max_motor_violation = float(np.max(motor_violations_arr)) if motor_violations_arr.size else float("nan")
    max_dynamics_defect = float(np.max(dynamics_defects_arr)) if dynamics_defects_arr.size else float("nan")
    finite_rate = float(np.mean(finite_arr)) if finite_arr.size else float("nan")
    rti_steps_per_s = args.batch_size * args.sim_steps / elapsed_s
    sqp_iterations_per_s = rti_steps_per_s * args.sqp_iterations
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({rti_steps_per_s:.3g} MPX-RTI steps/s, {sqp_iterations_per_s:.3g} SQP iterations/s), "
        f"final_position_rms={final_position_rms:.3e}, "
        f"final_state_rms={final_state_rms:.3e}, "
        f"final_state_cost_mean={final_state_cost_mean:.3e}, "
        f"mean_step={mean_step:.3g}, "
        f"max_violation={max_violation:.3e}, "
        f"max_motor_violation={max_motor_violation:.3e}, "
        f"max_dynamics_defect={max_dynamics_defect:.3e}, "
        f"finite_rate={finite_rate:.2%}",
        flush=True,
    )
    return {
        "batch_size": args.batch_size,
        "horizon_steps": args.horizon_steps,
        "sim_steps": args.sim_steps,
        "sqp_iterations": args.sqp_iterations,
        "control_dt": args.control_dt,
        "dtype": str(dtype),
        "qp_solver": "mpx",
        "mpx_solver_mode": args.mpx_solver_mode,
        "mpx_equality_weight": args.mpx_equality_weight,
        "mpx_barrier_alpha": args.mpx_barrier_alpha,
        "mpx_barrier_sigma": args.mpx_barrier_sigma,
        "mpx_num_alpha": args.mpx_num_alpha,
        "mpx_limited_memory": bool(args.mpx_limited_memory),
        "warm_starting": True,
        "n_variables": problem.n_variables,
        "n_constraints": problem.n_hard_constraints + problem.n_barrier_constraints,
        "n_hard_constraints": problem.n_hard_constraints,
        "n_barrier_constraints": problem.n_barrier_constraints,
        "nnz_p": "",
        "nnz_a": "",
        "problem_build_s": problem_build_s,
        "compile_setup_s": compile_setup_s,
        "initialization_s": initialization_s,
        "setup_s": problem_build_s + compile_setup_s + initialization_s,
        "warmup_compile_and_run_s": warmup_s,
        "compile_s": warmup_s,
        "elapsed_s": elapsed_s,
        "solve_s": elapsed_s,
        "total_compile_s": problem_build_s + compile_setup_s + initialization_s + warmup_s,
        "total_rti_steps": args.batch_size * args.sim_steps,
        "rti_steps_per_s": rti_steps_per_s,
        "sqp_iterations_per_s": sqp_iterations_per_s,
        "mean_step": mean_step,
        "final_position_rms": final_position_rms,
        "final_state_rms": final_state_rms,
        "final_state_cost_mean": final_state_cost_mean,
        "final_state_cost_median": final_state_cost_median,
        "final_state_cost_max": final_state_cost_max,
        "max_violation": max_violation,
        "max_motor_violation": max_motor_violation,
        "max_dynamics_defect": max_dynamics_defect,
        "finite_rate": finite_rate,
        "mean_qp_prim": float("nan"),
        "mean_qp_dual": float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--sim-time", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--sqp-iterations", type=int, default=1)
    parser.add_argument("--mpx-solver-mode", choices=("primal_dual", "fddp", "ilqr"), default="primal_dual")
    parser.add_argument("--mpx-equality-weight", type=float, default=1.0e4)
    parser.add_argument("--mpx-barrier-alpha", type=float, default=0.1)
    parser.add_argument("--mpx-barrier-sigma", type=float, default=1.0)
    parser.add_argument("--mpx-num-alpha", type=int, default=11)
    parser.add_argument("--mpx-limited-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-npz",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_mpx_closed_loop.npz"),
    )
    parser.add_argument(
        "--skip-output-npz",
        action="store_true",
        help="Do not write raw rollout arrays; plots and JSON summaries are still written.",
    )
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_mpx_closed_loop_states.png"),
    )
    parser.add_argument(
        "--position-plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_mpx_closed_loop_positions.png"),
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
        if args.position_plot_path is not None:
            print(f"Wrote {args.position_plot_path}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
