#!/usr/bin/env python3
"""Closed-loop MPX relaxed-barrier benchmark for the humanoid MPC."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import replace

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

from benchmarks.nonlinear_mpc.evaluate_humanoid_mpc import (
    _default_related_path,
    _plot_base_distribution,
    _plot_step_throughput,
    _sample_initial_q,
    _write_step_throughput_csv,
)
from benchmarks.nonlinear_mpc.mpx_penalty_adapter import (
    MPXBarrierSettings,
    compile_lifted_mpx_problem,
    pack_lifted_trajectory,
    settings_array,
    update_lifted_reference,
    zeros_parameter,
)
from benchmarks.problems.humanoid_mpc import (
    BodyReferenceCommand,
    HUMANOID_MODEL_NAME,
    HUMANOID_NDQ,
    HUMANOID_NQ,
    HUMANOID_NTAU,
    HUMANOID_PHI_VEL,
    humanoid_initial_guess_and_params,
    humanoid_jax_predicted_next_state,
    humanoid_make_references,
    humanoid_walking_gait_schedule,
    load_humanoid_mpc_parameters,
    make_humanoid_sqp_problem,
    standing_gait_schedule,
    update_humanoid_params_mpc,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    if args.mpx_solver_mode != "primal_dual":
        raise ValueError("MPX humanoid benchmark uses solver mode primal_dual")

    loaded_params = load_humanoid_mpc_parameters(args.parameters)
    if args.horizon_nodes is None:
        args.horizon_nodes = loaded_params.n_nodes
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

    print("building humanoid stage functions for MPX...", flush=True)
    problem_start = time.perf_counter()
    problem = make_humanoid_sqp_problem(
        args.horizon_nodes,
        model_name=args.model_name,
    )
    problem_build_s = time.perf_counter() - problem_start
    print(f"problem_build={problem_build_s:.3f}s", flush=True)

    print("compiling MPX primal_dual relaxed-barrier problem...", flush=True)
    compile_setup_start = time.perf_counter()
    barrier_settings = MPXBarrierSettings(
        equality_weight=args.mpx_equality_weight,
        barrier_alpha=args.mpx_barrier_alpha,
        barrier_sigma=args.mpx_barrier_sigma,
        num_alpha=args.mpx_num_alpha,
        limited_memory=args.mpx_limited_memory,
        solver_mode=args.mpx_solver_mode,
    )
    lifted = compile_lifted_mpx_problem(problem, settings=barrier_settings)
    compile_setup_s = time.perf_counter() - compile_setup_start
    print(f"compile_setup={compile_setup_s:.3f}s", flush=True)

    init_start = time.perf_counter()
    x_nodes0_np, u_nodes0_np, reference_np = pack_lifted_trajectory(z0, plan_params0, lifted)
    dual0_np = np.zeros_like(x_nodes0_np)
    initialization_s = time.perf_counter() - init_start

    x_nodes = jnp.asarray(x_nodes0_np)
    u_nodes = jnp.asarray(u_nodes0_np)
    dual_nodes = jnp.asarray(dual0_np)
    reference = jnp.asarray(reference_np)
    settings_vec = jnp.asarray(settings_array(barrier_settings, args.batch_size, dtype))
    parameter = jnp.asarray(zeros_parameter(args.batch_size, dtype))

    print(
        "Humanoid MPX MPC:",
        f"batch={args.batch_size}",
        f"dtype={dtype}",
        f"solver_mode={args.mpx_solver_mode}",
        f"model_name={args.model_name}",
        f"horizon_nodes={args.horizon_nodes}",
        f"sim_steps={args.sim_steps}",
        f"n={lifted.n_variables}",
        f"m={lifted.n_constraints}",
        flush=True,
    )
    print(
        "setup timings:",
        f"problem_build={problem_build_s:.3f}s",
        f"compile_setup={compile_setup_s:.3f}s",
        f"initialization={initialization_s:.3f}s",
        flush=True,
    )

    def lifted_to_flat_z_jax(x_lifted):
        pieces = [
            x_lifted[:, stage_index, :width]
            for stage_index, width in enumerate(lifted.stage_z_dims)
        ]
        return jnp.concatenate(pieces, axis=1)

    @jax.jit
    def closed_loop_step(x_nodes_cur, u_nodes_cur, dual_nodes_cur, reference_cur, q_cur, dq_cur, tau_cur):
        x_nodes_cur = x_nodes_cur.at[:, 0, :HUMANOID_NQ].set(q_cur)
        x_nodes_cur = x_nodes_cur.at[:, 0, HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ].set(dq_cur)
        x_nodes_cur = x_nodes_cur.at[
            :,
            0,
            HUMANOID_NQ + HUMANOID_NDQ : HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU,
        ].set(tau_cur)
        x_next_nodes, u_next_nodes, dual_next_nodes = lifted.solve(
            reference_cur,
            parameter,
            settings_vec,
            x_nodes_cur[:, 0],
            x_nodes_cur,
            u_nodes_cur,
            dual_nodes_cur,
        )
        z_flat = lifted_to_flat_z_jax(x_next_nodes)
        q_next, dq_next, tau0_next = humanoid_jax_predicted_next_state(z_flat)
        violation = lifted.violation(x_next_nodes, u_next_nodes, reference_cur)
        finite = (
            jnp.all(jnp.isfinite(x_next_nodes), axis=(1, 2))
            & jnp.all(jnp.isfinite(u_next_nodes), axis=(1, 2))
            & jnp.all(jnp.isfinite(q_next), axis=1)
            & jnp.all(jnp.isfinite(dq_next), axis=1)
        )
        return (
            x_next_nodes,
            u_next_nodes,
            dual_next_nodes,
            q_next,
            dq_next,
            tau0_next,
            jnp.ones((q_cur.shape[0],), dtype=q_cur.dtype),
            violation,
            finite,
        )

    warmup_start = time.perf_counter()
    warmup = closed_loop_step(
        x_nodes,
        u_nodes,
        dual_nodes,
        reference,
        jnp.asarray(q0),
        jnp.asarray(dq0),
        jnp.zeros((args.batch_size, HUMANOID_NTAU), dtype=jnp.dtype(dtype)),
    )
    jax.block_until_ready(warmup[3])
    warmup_s = time.perf_counter() - warmup_start
    print(f"warmup_compile_and_run={warmup_s:.3f}s", flush=True)

    q_history = [q0]
    step_lengths = []
    violations = []
    finite_flags = []
    step_total_s = []
    step_param_update_s = []
    step_mpx_sync_s = []
    q_measured = q0
    dq_measured = dq0
    tau0_measured = np.zeros((args.batch_size, HUMANOID_NTAU), dtype=dtype)
    params_np = plan_params0

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
        reference_np = update_lifted_reference(params_np, reference_np, lifted)
        reference = jnp.asarray(reference_np)
        param_update_s = time.perf_counter() - param_update_start

        mpx_start = time.perf_counter()
        (
            x_nodes,
            u_nodes,
            dual_nodes,
            q_next,
            dq_next,
            tau0_next,
            step_len,
            violation,
            finite,
        ) = closed_loop_step(
            x_nodes,
            u_nodes,
            dual_nodes,
            reference,
            jnp.asarray(q_measured),
            jnp.asarray(dq_measured),
            jnp.asarray(tau0_measured),
        )
        (
            q_next,
            dq_next,
            tau0_next,
            step_len,
            violation,
            finite,
        ) = jax.block_until_ready(
            (
                q_next,
                dq_next,
                tau0_next,
                step_len,
                violation,
                finite,
            )
        )
        mpx_sync_s = time.perf_counter() - mpx_start
        q_measured = np.asarray(jax.device_get(q_next))
        dq_measured = np.asarray(jax.device_get(dq_next))
        tau0_measured = np.asarray(jax.device_get(tau0_next))
        q_history.append(q_measured)
        step_lengths.append(np.asarray(jax.device_get(step_len)))
        violations.append(np.asarray(jax.device_get(violation)))
        finite_flags.append(np.asarray(jax.device_get(finite)))
        step_param_update_s.append(param_update_s)
        step_mpx_sync_s.append(mpx_sync_s)
        step_total_s.append(time.perf_counter() - step_start)
    elapsed_s = time.perf_counter() - start

    q_arr = np.stack(q_history, axis=0)
    step_lengths_arr = np.stack(step_lengths, axis=0)
    violations_arr = np.stack(violations, axis=0)
    finite_arr = np.stack(finite_flags, axis=0)
    step_total_s_arr = np.asarray(step_total_s, dtype=np.float64)
    step_param_update_s_arr = np.asarray(step_param_update_s, dtype=np.float64)
    step_mpx_sync_s_arr = np.asarray(step_mpx_sync_s, dtype=np.float64)
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
        sqp_sync_s=step_mpx_sync_s_arr,
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
            finite=finite_arr,
            step_total_s=step_total_s_arr,
            step_param_update_s=step_param_update_s_arr,
            step_mpx_sync_s=step_mpx_sync_s_arr,
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
        "warm_starting": True,
        "qp_solver": "mpx",
        "mpx_solver_mode": args.mpx_solver_mode,
        "mpx_equality_weight": args.mpx_equality_weight,
        "mpx_barrier_alpha": args.mpx_barrier_alpha,
        "mpx_barrier_sigma": args.mpx_barrier_sigma,
        "mpx_num_alpha": args.mpx_num_alpha,
        "mpx_limited_memory": bool(args.mpx_limited_memory),
        "n_variables": lifted.n_variables,
        "n_constraints": lifted.n_constraints,
        "nnz_p": "",
        "nnz_a": "",
        "elapsed_s": elapsed_s,
        "problem_build_s": problem_build_s,
        "compile_setup_s": compile_setup_s,
        "initialization_s": initialization_s,
        "warmup_compile_and_run_s": warmup_s,
        "total_rti_steps": args.batch_size * args.sim_steps,
        "rti_steps_per_s": throughput,
        "min_step_rti_steps_per_s": float(np.min(step_throughput)),
        "max_step_rti_steps_per_s": float(np.max(step_throughput)),
        "mean_step_rti_steps_per_s": float(np.mean(step_throughput)),
        "step_throughput_plot_path": "" if args.skip_plots else str(throughput_plot_path),
        "step_throughput_csv_path": str(throughput_csv_path),
        "mean_step": float(np.mean(step_lengths_arr)),
        "max_violation": float(np.max(violations_arr)),
        "finite_rate": float(np.mean(finite_arr)),
        "mean_qp_prim": float("nan"),
        "mean_qp_dual": float("nan"),
    }
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({throughput:.3g} MPX-RTI steps/s), "
        f"mean_step={summary['mean_step']:.3g}, "
        f"max_violation={summary['max_violation']:.3e}, "
        f"finite_rate={summary['finite_rate']:.2%}",
        flush=True,
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
    parser.add_argument("--mpx-solver-mode", choices=("primal_dual",), default="primal_dual")
    parser.add_argument("--mpx-equality-weight", type=float, default=1.0e4)
    parser.add_argument("--mpx-barrier-alpha", type=float, default=0.1)
    parser.add_argument("--mpx-barrier-sigma", type=float, default=1.0)
    parser.add_argument("--mpx-num-alpha", type=int, default=11)
    parser.add_argument("--mpx-limited-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-npz", type=pathlib.Path, default=None)
    parser.add_argument(
        "--plot-path",
        type=pathlib.Path,
        default=pathlib.Path("results/nonlinear_mpc/humanoid_mpx_rollout.png"),
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
