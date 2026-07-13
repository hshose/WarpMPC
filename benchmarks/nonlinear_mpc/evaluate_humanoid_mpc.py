#!/usr/bin/env python3
"""Closed-loop benchmark for the self-contained humanoid sparse SQP MPC."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.problems.humanoid_mpc import (
    BodyReferenceCommand,
    HUMANOID_NDQ,
    HUMANOID_NQ,
    HUMANOID_NTAU,
    HUMANOID_MPC_STANDING_REFERENCE_S,
    HUMANOID_MODEL_NAME,
    HUMANOID_PHI_VEL,
    humanoid_initial_guess_and_params,
    humanoid_jax_predicted_next_state,
    humanoid_make_references,
    humanoid_qhome,
    humanoid_walking_gait_schedule,
    load_humanoid_mpc_parameters,
    make_humanoid_sqp_problem,
    standing_gait_schedule,
    update_humanoid_params_mpc,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    FilterLineSearchSettings,
    MPAXSettings,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)


STATE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_initial_q(
    batch_size: int,
    seed: int,
    dtype: np.dtype,
    qhome: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = np.broadcast_to(qhome, (batch_size, HUMANOID_NQ)).copy()
    q[:, 0:2] += rng.uniform(-0.05, 0.05, size=(batch_size, 2))
    q[:, 5] += rng.uniform(-0.08, 0.08, size=(batch_size,))
    return q.astype(dtype)


def _plot_base_distribution(path: pathlib.Path, time_grid: np.ndarray, q_states: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), sharex=True)
    axes = axes.ravel()
    for idx, (ax, label) in enumerate(zip(axes, STATE_LABELS, strict=True)):
        values = q_states[:, :, idx]
        median = np.median(values, axis=1)
        low = np.percentile(values, 10.0, axis=1)
        high = np.percentile(values, 90.0, axis=1)
        ax.fill_between(time_grid, low, high, color="tab:blue", alpha=0.25, linewidth=0.0)
        ax.plot(time_grid, median, color="tab:blue", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    for ax in axes[-3:]:
        ax.set_xlabel("time [s]")
    fig.suptitle("Humanoid SQP MPC predicted-state rollout")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _default_related_path(path: pathlib.Path, suffix: str, extension: str) -> pathlib.Path:
    return path.with_name(f"{path.stem}_{suffix}{extension}")


def _write_step_throughput_csv(
    path: pathlib.Path,
    *,
    batch_size: int,
    control_dt: float,
    step_total_s: np.ndarray,
    param_update_s: np.ndarray,
    sqp_sync_s: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "closed_loop_step",
                "closed_loop_time_s",
                "total_step_s",
                "param_update_s",
                "sqp_step_and_sync_s",
                "rti_steps_per_s",
            ],
        )
        writer.writeheader()
        for step, total_s in enumerate(step_total_s):
            writer.writerow(
                {
                    "closed_loop_step": step,
                    "closed_loop_time_s": step * control_dt,
                    "total_step_s": float(total_s),
                    "param_update_s": float(param_update_s[step]),
                    "sqp_step_and_sync_s": float(sqp_sync_s[step]),
                    "rti_steps_per_s": float(batch_size / total_s),
                }
            )


def _plot_step_throughput(
    path: pathlib.Path,
    *,
    batch_size: int,
    control_dt: float,
    step_total_s: np.ndarray,
    overall_throughput: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(step_total_s.shape[0])
    throughput = batch_size / step_total_s
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(steps, throughput, marker="o", linewidth=1.7, markersize=4)
    ax.axhline(
        overall_throughput,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="overall",
    )
    ax.set_xlabel("closed-loop step")
    ax.set_ylabel("SQP-RTI steps / s")
    ax.set_title(f"Closed-loop throughput, dt={control_dt:g}s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    if args.qdldl_factor_backend is None:
        args.qdldl_factor_backend = args.qdldl_backend
    if args.qdldl_solve_backend is None:
        args.qdldl_solve_backend = args.qdldl_backend

    loaded_params = load_humanoid_mpc_parameters(args.parameters)
    if args.horizon_nodes is None:
        args.horizon_nodes = loaded_params.n_nodes
    if args.max_iter is None:
        args.max_iter = loaded_params.osqp_max_iter
    if args.rho is None:
        args.rho = loaded_params.osqp_rho
    if args.alpha is None:
        args.alpha = loaded_params.osqp_alpha
    if args.scaling is None:
        args.scaling = loaded_params.osqp_scaling
    if args.line_search_step_min is None:
        args.line_search_step_min = loaded_params.line_search_step_min
    if args.mpax_iteration_limit is None:
        args.mpax_iteration_limit = args.max_iter
    params_obj = replace(loaded_params, n_nodes=args.horizon_nodes)
    command = BodyReferenceCommand(
        velocity_x=args.reference_velocity_x,
        velocity_y=args.reference_velocity_y,
        yaw_rate=args.reference_yaw_rate,
        body_height=(
            float(params_obj.qhome[2])
            if args.reference_body_height is None
            else args.reference_body_height
        ),
    )

    q0 = _sample_initial_q(args.batch_size, args.seed, dtype, params_obj.qhome)
    dq0 = np.zeros((args.batch_size, HUMANOID_NDQ), dtype=dtype)
    z0, params0 = humanoid_initial_guess_and_params(
        q0,
        dq0,
        n_nodes=args.horizon_nodes,
        dtype=dtype,
        phi=0.0,
        phi_vel=args.phi_vel,
        parameters=params_obj,
    )
    if args.standing_reference_s > 0.0:
        plan_gait = standing_gait_schedule(
            args.horizon_nodes,
            params_obj.discretization_times,
        )
    else:
        plan_gait = humanoid_walking_gait_schedule(
            0.0,
            n_nodes=args.horizon_nodes,
            parameters=params_obj,
        )
    plan_qref, plan_dqref = humanoid_make_references(
        params_obj,
        q0,
        command,
        plan_gait.discretization_times,
    )
    plan_params0 = update_humanoid_params_mpc(
        params0,
        q0=q0,
        dq0=dq0,
        tau0=np.zeros((args.batch_size, HUMANOID_NTAU), dtype=dtype),
        reference_contacts=plan_gait.gait_pattern,
        reference_foot_height=plan_gait.foot_height,
        reference_q=plan_qref,
        reference_dq=plan_dqref,
        discretization_times=plan_gait.discretization_times,
        weights=params_obj,
    )
    settings = OSQPSettings(
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        max_iter=args.max_iter,
        scaling=args.scaling,
        adaptive_rho=False,
        rho_is_vec=True,
        check_termination=0,
        warm_starting=True,
        polishing=False,
    )
    print("building humanoid stage functions...", flush=True)
    problem_start = time.perf_counter()
    problem = make_humanoid_sqp_problem(
        args.horizon_nodes,
        model_name=args.model_name,
    )
    problem_build_s = time.perf_counter() - problem_start
    print(f"problem_build={problem_build_s:.3f}s", flush=True)

    print("building sparse MPC/QP plan...", flush=True)
    plan_start = time.perf_counter()
    plan = build_sparse_mpc_plan(
        problem,
        osqp_settings=settings,
        representative_z=z0,
        representative_params=plan_params0,
    )
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
        line_search_settings=FilterLineSearchSettings(
            line_search_g_max=params_obj.line_search_g_max,
            line_search_g_min=params_obj.line_search_g_min,
            line_search_gamma_c=params_obj.line_search_gamma_c,
            line_search_armijo_factor=params_obj.line_search_armijo_factor,
            line_search_step_decay=params_obj.line_search_step_decay,
            line_search_step_min=args.line_search_step_min,
            line_search_constraint_scale=params_obj.line_search_constraint_scale,
            line_search_cost_accept_uses_trial_cost=False,
        ),
        group_repeated_stages=args.group_repeated_stages,
    )
    compile_setup_s = time.perf_counter() - compile_setup_start
    print(f"compile_setup={compile_setup_s:.3f}s", flush=True)
    print(
        "Humanoid SQP MPC:",
        f"batch={args.batch_size}",
        f"dtype={dtype}",
        f"qp_solver={args.qp_solver}",
        f"mpax_iteration_limit={args.mpax_iteration_limit}",
        f"model_name={args.model_name}",
        f"horizon_nodes={args.horizon_nodes}",
        f"sim_steps={args.sim_steps}",
        f"qdldl_backend={args.qdldl_backend}",
        f"qdldl_factor_backend={args.qdldl_factor_backend}",
        f"qdldl_solve_backend={args.qdldl_solve_backend}",
        f"group_repeated_stages={args.group_repeated_stages}",
        f"transpose_work={args.transpose_work}",
        f"segmented={args.segmented}",
        f"segment_budget={args.segment_budget}",
        f"segment_strategy={args.segment_strategy}",
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

    def closed_loop_step(z, params, solver_state):
        result, solver_state_next = sqp.step(
            z,
            params,
            state=solver_state,
        )
        q_next, dq_next, tau0_next = humanoid_jax_predicted_next_state(result.z_next)
        return (
            result.z_next,
            q_next,
            dq_next,
            tau0_next,
            result.line_search.step_length,
            result.line_search.constraint_violation,
            result.solve.prim_res,
            result.solve.dual_res,
            solver_state_next,
        )

    z = jnp.asarray(z0)
    params_np = params0
    params = jnp.asarray(params_np)
    solver_state = sqp.init_state(args.batch_size)
    warmup_start = time.perf_counter()
    warmup = closed_loop_step(
        z,
        params,
        solver_state,
    )
    jax.block_until_ready(warmup[1])
    warmup_s = time.perf_counter() - warmup_start
    print(f"warmup_compile_and_run={warmup_s:.3f}s", flush=True)

    q_history = [q0]
    step_lengths = []
    violations = []
    prim_res = []
    dual_res = []
    step_total_s = []
    step_param_update_s = []
    step_sqp_sync_s = []
    q_measured = q0
    dq_measured = dq0
    tau0_measured = np.zeros((args.batch_size, HUMANOID_NTAU), dtype=dtype)

    start = time.perf_counter()
    for step in range(args.sim_steps):
        step_start = time.perf_counter()
        param_update_start = step_start
        mpc_time = step * args.control_dt
        if mpc_time < args.standing_reference_s:
            gait = standing_gait_schedule(
                args.horizon_nodes,
                params_obj.discretization_times,
            )
        else:
            gait = humanoid_walking_gait_schedule(
                mpc_time - args.standing_reference_s,
                n_nodes=args.horizon_nodes,
                parameters=params_obj,
            )
        qref, dqref = humanoid_make_references(
            params_obj,
            q_measured,
            command,
            gait.discretization_times,
        )
        params_np = update_humanoid_params_mpc(
            params_np,
            q0=q_measured,
            dq0=dq_measured,
            tau0=tau0_measured,
            reference_contacts=gait.gait_pattern,
            reference_foot_height=gait.foot_height,
            reference_q=qref,
            reference_dq=dqref,
            discretization_times=gait.discretization_times,
            weights=params_obj,
        )
        param_update_s = time.perf_counter() - param_update_start
        sqp_start = time.perf_counter()
        params = jnp.asarray(params_np)
        (
            z,
            q_next,
            dq_next,
            tau0_next,
            step_len,
            violation,
            primal,
            dual,
            solver_state,
        ) = closed_loop_step(
            z,
            params,
            solver_state,
        )
        (
            q_next,
            dq_next,
            tau0_next,
            step_len,
            violation,
            primal,
            dual,
            solver_state,
        ) = jax.block_until_ready(
            (
                q_next,
                dq_next,
                tau0_next,
                step_len,
                violation,
                primal,
                dual,
                solver_state,
            )
        )
        sqp_sync_s = time.perf_counter() - sqp_start
        q_measured = np.asarray(jax.device_get(q_next))
        dq_measured = np.asarray(jax.device_get(dq_next))
        tau0_measured = np.asarray(jax.device_get(tau0_next))
        q_history.append(q_measured)
        step_lengths.append(np.asarray(jax.device_get(step_len)))
        violations.append(np.asarray(jax.device_get(violation)))
        prim_res.append(np.asarray(jax.device_get(primal)))
        dual_res.append(np.asarray(jax.device_get(dual)))
        step_param_update_s.append(param_update_s)
        step_sqp_sync_s.append(sqp_sync_s)
        step_total_s.append(time.perf_counter() - step_start)
    elapsed_s = time.perf_counter() - start

    q_arr = np.stack(q_history, axis=0)
    step_lengths_arr = np.stack(step_lengths, axis=0)
    violations_arr = np.stack(violations, axis=0)
    prim_res_arr = np.stack(prim_res, axis=0)
    dual_res_arr = np.stack(dual_res, axis=0)
    step_total_s_arr = np.asarray(step_total_s, dtype=np.float64)
    step_param_update_s_arr = np.asarray(step_param_update_s, dtype=np.float64)
    step_sqp_sync_s_arr = np.asarray(step_sqp_sync_s, dtype=np.float64)
    time_grid = np.arange(args.sim_steps + 1, dtype=np.float64) * args.control_dt

    if not args.skip_plots:
        _plot_base_distribution(args.plot_path, time_grid, q_arr)
    throughput = args.batch_size * args.sim_steps / elapsed_s
    throughput_plot_path = args.throughput_plot_path or _default_related_path(
        args.plot_path,
        "throughput",
        ".png",
    )
    throughput_csv_path = args.throughput_csv_path or _default_related_path(
        args.plot_path,
        "throughput",
        ".csv",
    )
    _write_step_throughput_csv(
        throughput_csv_path,
        batch_size=args.batch_size,
        control_dt=args.control_dt,
        step_total_s=step_total_s_arr,
        param_update_s=step_param_update_s_arr,
        sqp_sync_s=step_sqp_sync_s_arr,
    )
    if not args.skip_plots:
        _plot_step_throughput(
            throughput_plot_path,
            batch_size=args.batch_size,
            control_dt=args.control_dt,
            step_total_s=step_total_s_arr,
            overall_throughput=throughput,
        )
    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_npz,
            time=time_grid,
            q=q_arr,
            step_lengths=step_lengths_arr,
            constraint_violations=violations_arr,
            prim_res=prim_res_arr,
            dual_res=dual_res_arr,
            step_total_s=step_total_s_arr,
            step_param_update_s=step_param_update_s_arr,
            step_sqp_sync_s=step_sqp_sync_s_arr,
            step_rti_steps_per_s=args.batch_size / step_total_s_arr,
        )

    step_throughput = args.batch_size / step_total_s_arr
    summary = {
        "batch_size": args.batch_size,
        "dtype": str(dtype),
        "model_name": args.model_name,
        "horizon_nodes": args.horizon_nodes,
        "sim_steps": args.sim_steps,
        "control_dt": args.control_dt,
        "standing_reference_s": args.standing_reference_s,
        "warm_starting": bool(settings.warm_starting),
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
        "group_repeated_stages": bool(args.group_repeated_stages),
        "level_scheduled_solve": bool(args.level_scheduled_solve),
        "level_scheduled_solve_threshold": args.level_scheduled_solve_threshold,
        "n_variables": plan.n_variables,
        "n_constraints": plan.n_constraints,
        "nnz_p": plan.p_pattern.nnz,
        "nnz_a": plan.a_pattern.nnz,
        "elapsed_s": elapsed_s,
        "problem_build_s": problem_build_s,
        "plan_build_s": plan_build_s,
        "compile_setup_s": compile_setup_s,
        "warmup_compile_and_run_s": warmup_s,
        "total_rti_steps": args.batch_size * args.sim_steps,
        "rti_steps_per_s": throughput,
        "min_step_rti_steps_per_s": float(np.min(step_throughput)),
        "max_step_rti_steps_per_s": float(np.max(step_throughput)),
        "mean_step_rti_steps_per_s": float(np.mean(step_throughput)),
        "step_throughput_plot_path": str(throughput_plot_path),
        "step_throughput_csv_path": str(throughput_csv_path),
        "mean_step": float(np.mean(step_lengths_arr)),
        "max_violation": float(np.max(violations_arr)),
        "mean_qp_prim": float(np.mean(prim_res_arr)),
        "mean_qp_dual": float(np.mean(dual_res_arr)),
    }
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({throughput:.3g} SQP-RTI steps/s), "
        f"mean_step={summary['mean_step']:.3g}, "
        f"max_violation={summary['max_violation']:.3e}, "
        f"mean_qp_res=({summary['mean_qp_prim']:.2e}, {summary['mean_qp_dual']:.2e})"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--horizon-nodes",
        type=int,
        default=None,
        help="Override the humanoid horizon node count from the parameter YAML.",
    )
    parser.add_argument("--sim-steps", type=int, default=20)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--standing-reference-s", type=float, default=0.0)
    parser.add_argument("--phi-vel", type=float, default=HUMANOID_PHI_VEL)
    parser.add_argument("--model-name", default=HUMANOID_MODEL_NAME)
    parser.add_argument(
        "--parameters",
        type=pathlib.Path,
        default=None,
        help="Optional humanoid MPC parameter YAML. Defaults to the tuned repo copy.",
    )
    parser.add_argument("--reference-velocity-x", type=float, default=0.0)
    parser.add_argument("--reference-velocity-y", type=float, default=0.0)
    parser.add_argument("--reference-yaw-rate", type=float, default=0.0)
    parser.add_argument("--reference-body-height", type=float, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--qp-solver", choices=("jax_osqp", "mpax"), default="jax_osqp")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--mpax-iteration-limit", type=int, default=None)
    parser.add_argument("--mpax-eps-abs", type=float, default=1e-3)
    parser.add_argument("--mpax-eps-rel", type=float, default=1e-3)
    parser.add_argument("--mpax-termination-evaluation-frequency", type=int, default=100)
    parser.add_argument("--mpax-l-inf-ruiz-iterations", type=int, default=10)
    parser.add_argument("--mpax-pock-chambolle-alpha", type=float, default=1.0)
    parser.add_argument("--mpax-regularization", type=float, default=0.0)
    parser.add_argument("--mpax-unroll", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--scaling", type=int, default=None)
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-budget", type=int, default=96)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action="store_true")
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="jax")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--line-search-step-min", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-group-repeated-stages",
        action="store_false",
        dest="group_repeated_stages",
    )
    parser.set_defaults(group_repeated_stages=True)
    parser.add_argument("--output-npz", type=pathlib.Path, default=None)
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/humanoid_mpc_rollout.png"),
    )
    parser.add_argument("--throughput-plot-path", type=pathlib.Path, default=None)
    parser.add_argument("--throughput-csv-path", type=pathlib.Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.parameters is None:
        args.parameters = pathlib.Path(__file__).resolve().parents[1] / "problems" / "humanoid_mpc_parameters.yaml"
    summary = run(args)
    if args.skip_plots:
        print("Skipped rollout plots")
    else:
        print(f"Wrote {args.plot_path}")
        print(f"Wrote {summary['step_throughput_plot_path']}")
    print(f"Wrote {summary['step_throughput_csv_path']}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
