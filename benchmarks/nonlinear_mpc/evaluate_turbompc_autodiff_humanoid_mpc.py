#!/usr/bin/env python3
"""Closed-loop humanoid sparse MPC benchmark using TurboMPC autodiff blocks."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp

from benchmarks.nonlinear_mpc.evaluate_humanoid_mpc import (
    _default_related_path,
    _plot_base_distribution,
    _plot_step_throughput,
    _sample_initial_q,
    _write_step_throughput_csv,
)
from benchmarks.nonlinear_mpc.turbompc_autodiff_adapter import (
    AutodiffDenseStageTurbompcProblem,
    initial_admm_state_batch,
    make_problem_params,
    make_solver,
    make_solver_params,
)
from benchmarks.problems.turbompc_autodiff_value_problems import (
    make_humanoid_turbompc_autodiff_problem,
)
from benchmarks.problems.humanoid_mpc import (
    BodyReferenceCommand,
    HUMANOID_MPC_STANDING_REFERENCE_S,
    HUMANOID_NDQ,
    HUMANOID_NQ,
    HUMANOID_NTAU,
    HUMANOID_PHI_VEL,
    HUMANOID_MODEL_NAME,
    humanoid_initial_guess_and_params,
    humanoid_make_references,
    humanoid_walking_gait_schedule,
    load_humanoid_mpc_parameters,
    standing_gait_schedule,
    update_humanoid_params_mpc,
)


HUMANOID_TURBOMPC_STATE_DIM = HUMANOID_NQ + HUMANOID_NDQ
HUMANOID_TURBOMPC_CONTROL_DIM = 30


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))

    loaded_params = load_humanoid_mpc_parameters(args.parameters)
    if args.horizon_nodes is None:
        args.horizon_nodes = loaded_params.n_nodes
    if args.max_iter is None:
        args.max_iter = loaded_params.osqp_max_iter
    if args.rho is None:
        args.rho = loaded_params.osqp_rho
    if args.alpha is None:
        args.alpha = loaded_params.osqp_alpha
    if args.line_search_step_min is None:
        args.line_search_step_min = loaded_params.line_search_step_min
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
    params0 = update_humanoid_params_mpc(
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

    print("building humanoid TurboMPC-autodiff dense-stage problem...", flush=True)
    problem_start = time.perf_counter()
    value_problem = make_humanoid_turbompc_autodiff_problem(
        args.horizon_nodes,
        model_name=args.model_name,
    )
    problem = AutodiffDenseStageTurbompcProblem(
        value_problem,
        state_dim=HUMANOID_TURBOMPC_STATE_DIM,
        control_dim=HUMANOID_TURBOMPC_CONTROL_DIM,
        name="HumanoidAutodiffDenseStageTurboMPC",
    )
    problem_build_s = time.perf_counter() - problem_start
    desc = problem.describe_dense_blocks()
    print(f"problem_build={problem_build_s:.3f}s", flush=True)

    solver_params = make_solver_params(
        sqp_iterations=1,
        admm_max_iter=args.max_iter,
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        eps_abs=args.turbompc_eps_abs,
        eps_rel=args.turbompc_eps_rel,
        line_search_step_min=args.line_search_step_min,
        fixed_sqp_iterations=True,
    )

    print("creating TurboMPC solver...", flush=True)
    setup_start = time.perf_counter()
    solver = make_solver(
        problem,
        solver_params,
        forward_backend=args.turbompc_forward_backend,
        backward_backend=args.turbompc_backward_backend,
    )
    setup_s = time.perf_counter() - setup_start
    print(f"setup={setup_s:.3f}s", flush=True)

    states, controls = jax.vmap(problem.split_packed_z)(jnp.asarray(z0))
    params = jnp.asarray(params0)
    admm_state = initial_admm_state_batch(
        solver,
        states,
        controls,
        params,
        rho=args.rho,
    )
    jax.block_until_ready(admm_state.x_blocks)

    def solve_one(states_one, controls_one, params_one, admm_state_one):
        return solver._solve_impl(
            states_one,
            controls_one,
            make_problem_params(params_one),
            admm_state_one,
        )

    solve_batch = jax.jit(jax.vmap(solve_one))

    warmup_start = time.perf_counter()
    warmup = solve_batch(states, controls, params, admm_state)
    jax.block_until_ready(warmup.states)
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
    params_np = params0

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
        solution = solve_batch(states, controls, params, admm_state)
        (
            states,
            controls,
            admm_state,
            step_len,
            eq_violation,
            ineq_violation,
            convergence_error,
        ) = jax.block_until_ready(
            (
                solution.states,
                solution.controls,
                solution.admm_state,
                solution.linesearch_alphas[:, -1],
                solution.solver_stats.eq_constraints_violations[:, -1],
                solution.solver_stats.ineq_constraints_violations[:, -1],
                solution.convergence_error,
            )
        )
        sqp_sync_s = time.perf_counter() - sqp_start
        q_measured = np.asarray(jax.device_get(states[:, 1, :HUMANOID_NQ]))
        dq_measured = np.asarray(jax.device_get(states[:, 1, HUMANOID_NQ:HUMANOID_NQ + HUMANOID_NDQ]))
        tau0_measured = np.asarray(jax.device_get(controls[:, 0, :HUMANOID_NTAU]))
        q_history.append(q_measured)
        step_lengths.append(np.asarray(jax.device_get(step_len)))
        violations.append(
            np.asarray(jax.device_get(eq_violation + ineq_violation))
        )
        prim_res.append(np.asarray(jax.device_get(eq_violation)))
        dual_res.append(np.asarray(jax.device_get(convergence_error)))
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
    if not args.skip_plots:
        _write_step_throughput_csv(
            throughput_csv_path,
            batch_size=args.batch_size,
            control_dt=args.control_dt,
            step_total_s=step_total_s_arr,
            param_update_s=step_param_update_s_arr,
            sqp_sync_s=step_sqp_sync_s_arr,
        )
        _plot_step_throughput(
            throughput_plot_path,
            batch_size=args.batch_size,
            control_dt=args.control_dt,
            step_total_s=step_total_s_arr,
            overall_throughput=throughput,
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
        "warm_starting": True,
        "qp_solver": "turbompc_autodiff",
        "linearization_backend": "turbompc_jax_autodiff",
        "turbompc_forward_backend": args.turbompc_forward_backend,
        "turbompc_backward_backend": args.turbompc_backward_backend,
        "max_iter": args.max_iter,
        "turbompc_sqp_iterations": 1,
        "turbompc_eps_abs": args.turbompc_eps_abs,
        "turbompc_eps_rel": args.turbompc_eps_rel,
        "rho": args.rho,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "n_variables": desc.n_variables,
        "n_constraints": desc.n_constraints,
        "nnz_p": desc.nnz_p,
        "nnz_a": desc.nnz_a,
        "dense_block_dim": desc.block_dim,
        "dense_inequality_dim": desc.inequality_dim,
        "elapsed_s": elapsed_s,
        "problem_build_s": problem_build_s,
        "setup_s": setup_s,
        "warmup_compile_and_run_s": warmup_s,
        "total_rti_steps": args.batch_size * args.sim_steps,
        "rti_steps_per_s": throughput,
        "min_step_rti_steps_per_s": float(np.min(step_throughput)),
        "max_step_rti_steps_per_s": float(np.max(step_throughput)),
        "mean_step_rti_steps_per_s": float(np.mean(step_throughput)),
        "step_throughput_plot_path": "" if args.skip_plots else str(throughput_plot_path),
        "step_throughput_csv_path": "" if args.skip_plots else str(throughput_csv_path),
        "mean_step": float(np.mean(step_lengths_arr)),
        "max_violation": float(np.max(violations_arr)),
        "mean_qp_prim": float(np.mean(prim_res_arr)),
        "mean_qp_dual": float(np.mean(dual_res_arr)),
    }
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({throughput:.3g} SQP-RTI steps/s), "
        f"mean_step={summary['mean_step']:.3g}, "
        f"max_violation={summary['max_violation']:.3e}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon-nodes", type=int, default=None)
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
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--turbompc-eps-abs", type=float, default=1e-3)
    parser.add_argument("--turbompc-eps-rel", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--line-search-step-min", type=float, default=None)
    parser.add_argument("--turbompc-forward-backend", default="admm_fused_cudss")
    parser.add_argument("--turbompc-backward-backend", default="direct_cudss_ffi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/turbompc_autodiff_humanoid_mpc_rollout.png"))
    parser.add_argument("--throughput-plot-path", type=pathlib.Path, default=None)
    parser.add_argument("--throughput-csv-path", type=pathlib.Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.parameters is None:
        args.parameters = pathlib.Path(__file__).resolve().parents[1] / "problems" / "humanoid_mpc_parameters.yaml"
    summary = run(args)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
