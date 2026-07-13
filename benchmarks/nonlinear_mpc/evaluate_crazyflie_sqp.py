#!/usr/bin/env python3
"""Closed-loop previous-solution warm-start evaluation for Crazyflie SQP MPC."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.problems.crazyflie_sqp import (
    CRAZYFLIE_Q,
    CRAZYFLIE_NU,
    CRAZYFLIE_NX,
    CRAZYFLIE_NZ,
    crazyflie_initial_guess_and_params,
    crazyflie_jax_euler_step,
    make_crazyflie_sqp_problem,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    FilterLineSearchSettings,
    MPAXSettings,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)


STATE_LABELS = (
    "px",
    "py",
    "pz",
    "phi",
    "theta",
    "psi",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
)


def _sample_initial_states(batch_size: int, seed: int, dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = np.zeros((batch_size, CRAZYFLIE_NX), dtype=dtype)
    x0[:, :3] = rng.uniform(-1.0, 1.0, size=(batch_size, 3)).astype(dtype)
    return x0


def _plot_state_distribution(path: pathlib.Path, time_grid: np.ndarray, states: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(12.0, 10.0), sharex=True)
    axes = axes.ravel()
    for idx, (ax, label) in enumerate(zip(axes, STATE_LABELS, strict=True)):
        values = states[:, :, idx]
        median = np.median(values, axis=1)
        low = np.percentile(values, 10.0, axis=1)
        high = np.percentile(values, 90.0, axis=1)
        min_value = np.min(values, axis=1)
        max_value = np.max(values, axis=1)
        ax.fill_between(
            time_grid,
            min_value,
            max_value,
            color="tab:blue",
            alpha=0.10,
            linewidth=0.0,
        )
        ax.fill_between(time_grid, low, high, color="tab:blue", alpha=0.25, linewidth=0.0)
        ax.plot(time_grid, median, color="tab:blue", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    for ax in axes[-3:]:
        ax.set_xlabel("time [s]")
    fig.suptitle("Crazyflie SQP MPC closed-loop state distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_position_cloud(path: pathlib.Path, states: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    stride = max(1, states.shape[0] // 20)
    for sample in range(states.shape[1]):
        trajectory = states[::stride, sample, :3]
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.20, linewidth=0.9)
    ax.scatter(
        states[0, :, 0],
        states[0, :, 1],
        states[0, :, 2],
        s=12,
        color="tab:red",
        label="initial",
    )
    ax.scatter(
        states[-1, :, 0],
        states[-1, :, 1],
        states[-1, :, 2],
        s=12,
        color="tab:green",
        label="final",
    )
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.set_zlabel("pz")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, np.ndarray | float | int]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    if args.qdldl_factor_backend is None:
        args.qdldl_factor_backend = args.qdldl_backend
    if args.qdldl_solve_backend is None:
        args.qdldl_solve_backend = args.qdldl_backend
    if args.mpax_iteration_limit is None:
        args.mpax_iteration_limit = args.max_iter

    x0 = _sample_initial_states(args.batch_size, args.seed, dtype)
    z0, params0 = crazyflie_initial_guess_and_params(
        x0,
        n_steps=args.horizon_steps,
        dtype=dtype,
    )
    settings = OSQPSettings(
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        max_iter=args.max_iter,
        scaling=0,
        adaptive_rho=False,
        rho_is_vec=False,
        check_termination=0,
        warm_starting=True,
        polishing=False,
    )
    line_search_settings = FilterLineSearchSettings(
        line_search_step_min=args.line_search_step_min,
    )
    print("building crazyflie stage functions...", flush=True)
    problem_start = time.perf_counter()
    problem = make_crazyflie_sqp_problem(args.horizon_steps)
    problem_build_s = time.perf_counter() - problem_start
    print(f"problem_build={problem_build_s:.3f}s", flush=True)

    print("building sparse MPC/QP plan...", flush=True)
    plan_start = time.perf_counter()
    plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
    plan_build_s = time.perf_counter() - plan_start
    print(f"plan_build={plan_build_s:.3f}s", flush=True)

    print("creating compiled SQP callables...", flush=True)
    compile_setup_start = time.perf_counter()
    sqp = compile_sparse_mpc_sqp(
        problem,
        plan,
        dtype=dtype,
        qp_solver=args.qp_solver,
        osqp_settings=settings,
        mpax_settings=MPAXSettings(
            eps_abs=args.mpax_eps_abs,
            eps_rel=args.mpax_eps_rel,
            iteration_limit=args.mpax_iteration_limit,
            termination_evaluation_frequency=args.mpax_termination_evaluation_frequency,
            l_inf_ruiz_iterations=args.mpax_l_inf_ruiz_iterations,
            pock_chambolle_alpha=args.mpax_pock_chambolle_alpha,
            regularization=args.mpax_regularization,
            unroll=args.mpax_unroll,
        ),
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
    print(f"compile_setup={compile_setup_s:.3f}s", flush=True)
    print(
        "Crazyflie SQP MPC:",
        f"batch={args.batch_size}",
        f"dtype={dtype}",
        f"qp_solver={args.qp_solver}",
        f"mpax_iteration_limit={args.mpax_iteration_limit}",
        f"qdldl_backend={args.qdldl_backend}",
        f"qdldl_factor_backend={args.qdldl_factor_backend}",
        f"qdldl_solve_backend={args.qdldl_solve_backend}",
        f"transpose_work={args.transpose_work}",
        f"segmented={args.segmented}",
        f"group_repeated_stages={args.group_repeated_stages}",
        f"horizon_steps={args.horizon_steps}",
        f"sim_steps={args.sim_steps}",
        f"level_scheduled_solve={args.level_scheduled_solve}",
        f"level_scheduled_solve_threshold={args.level_scheduled_solve_threshold}",
        f"n={plan.n_variables}",
        f"m={plan.n_constraints}",
        f"nnz_P={plan.p_pattern.nnz}",
        f"nnz_A={plan.a_pattern.nnz}",
        flush=True,
    )
    print(
        "setup timings:",
        f"problem_build={problem_build_s:.3f}s",
        f"plan_build={plan_build_s:.3f}s",
        f"compile_setup={compile_setup_s:.3f}s",
        flush=True,
    )

    control_dt = jnp.asarray(args.control_dt, dtype=jnp.dtype(dtype))

    @jax.jit
    def closed_loop_step(z, params, x, solver_state):
        params = params.at[:, :CRAZYFLIE_NX].set(x)
        step_result, solver_state_next = sqp.step(
            z,
            params,
            state=solver_state,
        )
        stages = step_result.z_next.reshape((z.shape[0], args.horizon_steps + 1, CRAZYFLIE_NZ))
        u0 = stages[:, 0, CRAZYFLIE_NX : CRAZYFLIE_NX + CRAZYFLIE_NU]
        x_next = crazyflie_jax_euler_step(x, u0, control_dt)
        params_next = params.at[:, :CRAZYFLIE_NX].set(x_next)
        return (
            step_result.z_next,
            params_next,
            x_next,
            u0,
            step_result.line_search.step_length,
            step_result.line_search.constraint_violation,
            step_result.line_search.reason,
            step_result.solve.prim_res,
            step_result.solve.dual_res,
            solver_state_next,
        )

    z = jnp.asarray(z0)
    params = jnp.asarray(params0)
    x = jnp.asarray(x0)
    solver_state = sqp.init_state(args.batch_size)
    warmup_start = time.perf_counter()
    warmup = closed_loop_step(
        z,
        params,
        x,
        solver_state,
    )
    jax.block_until_ready(warmup[2])
    warmup_s = time.perf_counter() - warmup_start
    print(f"warmup_compile_and_run={warmup_s:.3f}s", flush=True)

    states = [np.asarray(jax.device_get(x))]
    controls = []
    step_lengths = []
    violations = []
    reasons = []
    prim_res = []
    dual_res = []

    start = time.perf_counter()
    for _ in range(args.sim_steps):
        (
            z,
            params,
            x,
            u0,
            step_len,
            violation,
            reason,
            primal,
            dual,
            solver_state,
        ) = closed_loop_step(z, params, x, solver_state)
        states.append(np.asarray(jax.device_get(x)))
        controls.append(np.asarray(jax.device_get(u0)))
        step_lengths.append(np.asarray(jax.device_get(step_len)))
        violations.append(np.asarray(jax.device_get(violation)))
        reasons.append(np.asarray(jax.device_get(reason)))
        prim_res.append(np.asarray(jax.device_get(primal)))
        dual_res.append(np.asarray(jax.device_get(dual)))
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
    reasons_arr = (
        np.stack(reasons, axis=0)
        if reasons
        else np.empty((0, args.batch_size), dtype=np.int32)
    )
    prim_res_arr = np.stack(prim_res, axis=0) if prim_res else np.empty((0, args.batch_size))
    dual_res_arr = np.stack(dual_res, axis=0) if dual_res else np.empty((0, args.batch_size))
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
            line_search_reasons=reasons_arr,
            prim_res=prim_res_arr,
            dual_res=dual_res_arr,
            initial_states=x0,
        )
    if not args.skip_plots:
        _plot_state_distribution(args.plot_path, time_grid, states_arr)
        if args.position_plot_path is not None:
            _plot_position_cloud(args.position_plot_path, states_arr)

    final_state = states_arr[-1]
    final_state_cost = 0.5 * np.sum(final_state * (np.diag(CRAZYFLIE_Q)[None, :] * final_state), axis=1)
    final_position_norm = np.linalg.norm(final_state[:, :3], axis=1)
    final_state_norm = np.linalg.norm(final_state, axis=1)
    final_position_rms = float(np.sqrt(np.mean(final_state[:, :3] ** 2)))
    final_state_rms = float(np.sqrt(np.mean(final_state ** 2)))
    final_position_norm_mean = float(np.mean(final_position_norm))
    final_position_norm_median = float(np.median(final_position_norm))
    final_position_norm_p95 = float(np.percentile(final_position_norm, 95.0))
    final_position_norm_max = float(np.max(final_position_norm))
    final_state_norm_mean = float(np.mean(final_state_norm))
    final_state_norm_median = float(np.median(final_state_norm))
    final_state_norm_p95 = float(np.percentile(final_state_norm, 95.0))
    final_state_norm_max = float(np.max(final_state_norm))
    final_state_cost_mean = float(np.mean(final_state_cost))
    final_state_cost_median = float(np.median(final_state_cost))
    final_state_cost_max = float(np.max(final_state_cost))
    mean_step = float(np.mean(step_lengths_arr)) if step_lengths_arr.size else float("nan")
    max_violation = float(np.max(violations_arr)) if violations_arr.size else float("nan")
    mean_qp_prim = float(np.mean(prim_res_arr)) if prim_res_arr.size else float("nan")
    mean_qp_dual = float(np.mean(dual_res_arr)) if dual_res_arr.size else float("nan")
    rti_steps_per_s = args.batch_size * args.sim_steps / elapsed_s
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({rti_steps_per_s:.3g} MPC steps/s), "
        f"final_position_rms={final_position_rms:.3e}, "
        f"final_state_rms={final_state_rms:.3e}, "
        f"final_state_cost_mean={final_state_cost_mean:.3e}, "
        f"mean_step={mean_step:.3g}, "
        f"max_violation={max_violation:.3e}, "
        f"mean_qp_res=({mean_qp_prim:.2e}, {mean_qp_dual:.2e})"
    )
    return {
        "batch_size": args.batch_size,
        "horizon_steps": args.horizon_steps,
        "sim_steps": args.sim_steps,
        "control_dt": args.control_dt,
        "dtype": str(dtype),
        "qp_solver": args.qp_solver,
        "mpax_iteration_limit": args.mpax_iteration_limit,
        "mpax_eps_abs": args.mpax_eps_abs,
        "mpax_eps_rel": args.mpax_eps_rel,
        "mpax_termination_evaluation_frequency": args.mpax_termination_evaluation_frequency,
        "mpax_l_inf_ruiz_iterations": args.mpax_l_inf_ruiz_iterations,
        "mpax_pock_chambolle_alpha": args.mpax_pock_chambolle_alpha,
        "mpax_regularization": args.mpax_regularization,
        "mpax_unroll": bool(args.mpax_unroll),
        "qdldl_backend": args.qdldl_backend,
        "qdldl_factor_backend": args.qdldl_factor_backend,
        "qdldl_solve_backend": args.qdldl_solve_backend,
        "transpose_work": bool(args.transpose_work),
        "segmented": bool(args.segmented),
        "segment_budget": args.segment_budget,
        "segment_strategy": args.segment_strategy,
        "warm_starting": bool(settings.warm_starting),
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
        "setup_s": problem_build_s + plan_build_s + compile_setup_s,
        "warmup_compile_and_run_s": warmup_s,
        "compile_s": warmup_s,
        "elapsed_s": elapsed_s,
        "solve_s": elapsed_s,
        "total_compile_s": problem_build_s + plan_build_s + compile_setup_s + warmup_s,
        "total_rti_steps": args.batch_size * args.sim_steps,
        "rti_steps_per_s": rti_steps_per_s,
        "final_position_rms": final_position_rms,
        "final_state_rms": final_state_rms,
        "final_position_norm_mean": final_position_norm_mean,
        "final_position_norm_median": final_position_norm_median,
        "final_position_norm_p95": final_position_norm_p95,
        "final_position_norm_max": final_position_norm_max,
        "final_state_norm_mean": final_state_norm_mean,
        "final_state_norm_median": final_state_norm_median,
        "final_state_norm_p95": final_state_norm_p95,
        "final_state_norm_max": final_state_norm_max,
        "final_state_cost_mean": final_state_cost_mean,
        "final_state_cost_median": final_state_cost_median,
        "final_state_cost_max": final_state_cost_max,
        "mean_step": mean_step,
        "max_violation": max_violation,
        "mean_qp_prim": mean_qp_prim,
        "mean_qp_dual": mean_qp_dual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--sim-time", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--qp-solver", choices=("jax_osqp", "mpax"), default="jax_osqp")
    parser.add_argument("--max-iter", type=int, default=25)
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
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-budget", type=int, default=256)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action="store_true")
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="jax")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-group-repeated-stages",
        action="store_false",
        dest="group_repeated_stages",
    )
    parser.set_defaults(group_repeated_stages=True)
    parser.add_argument(
        "--output-npz",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_closed_loop.npz"),
    )
    parser.add_argument(
        "--skip-output-npz",
        action="store_true",
        help="Do not write raw rollout arrays; plots and JSON summaries are still written.",
    )
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_closed_loop_states.png"),
    )
    parser.add_argument(
        "--position-plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/crazyflie_sqp_closed_loop_positions.png"),
    )
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.skip_output_npz:
        args.output_npz = None
    if args.sim_steps is None:
        args.sim_steps = int(np.ceil(args.sim_time / args.control_dt))
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
