#!/usr/bin/env python3
"""Closed-loop Crazyflie SQP-RTI rollout using TurboMPC autodiff blocks."""

from __future__ import annotations

import argparse
import json
import os
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

from benchmarks.nonlinear_mpc.evaluate_crazyflie_sqp import (
    _plot_position_cloud,
    _plot_state_distribution,
    _sample_initial_states,
)
from benchmarks.nonlinear_mpc.turbompc_autodiff_adapter import (
    AutodiffDenseStageTurbompcProblem,
    initial_admm_state_batch,
    make_problem_params,
    make_solver,
    make_solver_params,
)
from benchmarks.problems.turbompc_autodiff_value_problems import (
    make_crazyflie_turbompc_autodiff_problem,
)
from benchmarks.problems.crazyflie_sqp import (
    CRAZYFLIE_NU,
    CRAZYFLIE_NX,
    crazyflie_initial_guess_and_params,
    crazyflie_jax_euler_step,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))

    x0 = _sample_initial_states(args.batch_size, args.seed, dtype)
    z0, params0 = crazyflie_initial_guess_and_params(
        x0,
        n_steps=args.horizon_steps,
        dtype=dtype,
    )

    print("building Crazyflie TurboMPC-autodiff dense-stage problem...", flush=True)
    problem_start = time.perf_counter()
    value_problem = make_crazyflie_turbompc_autodiff_problem(args.horizon_steps)
    problem = AutodiffDenseStageTurbompcProblem(
        value_problem,
        state_dim=CRAZYFLIE_NX,
        control_dim=CRAZYFLIE_NU,
        name="CrazyflieAutodiffDenseStageTurboMPC",
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

    states0, controls0 = jax.vmap(problem.split_packed_z)(jnp.asarray(z0))
    params = jnp.asarray(params0)
    x = jnp.asarray(x0)
    admm_state0 = initial_admm_state_batch(
        solver,
        states0,
        controls0,
        params,
        rho=args.rho,
    )
    jax.block_until_ready(admm_state0.x_blocks)

    control_dt = jnp.asarray(args.control_dt, dtype=jnp.dtype(dtype))

    def solve_one(states, controls, packed_params, admm_state):
        return solver._solve_impl(
            states,
            controls,
            make_problem_params(packed_params),
            admm_state,
        )

    solve_batch = jax.vmap(solve_one)

    @jax.jit
    def closed_loop_rollout(states, controls, packed_params, x_current, admm_state):
        def step(carry, _):
            states_cur, controls_cur, params_cur, x_cur, admm_cur = carry
            params_cur = params_cur.at[:, :CRAZYFLIE_NX].set(x_cur)
            solution = solve_batch(states_cur, controls_cur, params_cur, admm_cur)
            u0 = solution.controls[:, 0, :CRAZYFLIE_NU]
            x_next = crazyflie_jax_euler_step(x_cur, u0, control_dt)
            eq_v = solution.solver_stats.eq_constraints_violations[:, -1]
            ineq_v = solution.solver_stats.ineq_constraints_violations[:, -1]
            conv = solution.convergence_error
            step_len = solution.linesearch_alphas[:, -1]
            out = (x_cur, u0, step_len, eq_v + ineq_v, eq_v, conv)
            return (
                solution.states,
                solution.controls,
                params_cur.at[:, :CRAZYFLIE_NX].set(x_next),
                x_next,
                solution.admm_state,
            ), out

        return jax.lax.scan(
            step,
            (states, controls, packed_params, x_current, admm_state),
            xs=None,
            length=args.sim_steps,
        )

    warmup_start = time.perf_counter()
    warmup = closed_loop_rollout(states0, controls0, params, x, admm_state0)
    jax.block_until_ready(warmup[0][3])
    warmup_s = time.perf_counter() - warmup_start
    print(f"warmup_compile_and_run={warmup_s:.3f}s", flush=True)

    start = time.perf_counter()
    final_carry, outputs = closed_loop_rollout(states0, controls0, params, x, admm_state0)
    jax.block_until_ready(final_carry[3])
    elapsed_s = time.perf_counter() - start
    x_final = final_carry[3]
    states_before, controls_applied, step_lengths, violations, prim_res, dual_res = outputs
    state_history = jnp.concatenate([states_before, x_final[None, :, :]], axis=0)

    states_np = np.asarray(jax.device_get(state_history))
    step_lengths_np = np.asarray(jax.device_get(step_lengths))
    violations_np = np.asarray(jax.device_get(violations))
    prim_np = np.asarray(jax.device_get(prim_res))
    dual_np = np.asarray(jax.device_get(dual_res))
    controls_np = np.asarray(jax.device_get(controls_applied))
    time_grid = np.arange(args.sim_steps + 1, dtype=np.float64) * args.control_dt

    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_npz,
            time=time_grid,
            states=states_np,
            controls=controls_np,
            step_lengths=step_lengths_np,
            constraint_violations=violations_np,
            prim_res=prim_np,
            dual_res=dual_np,
        )

    if not args.skip_plots:
        _plot_state_distribution(args.plot_path, time_grid, states_np)
        _plot_position_cloud(args.position_plot_path, states_np)

    final_position = states_np[-1, :, :3]
    final_state = states_np[-1]
    throughput = args.batch_size * args.sim_steps / elapsed_s
    summary: dict[str, object] = {
        "batch_size": args.batch_size,
        "dtype": str(dtype),
        "qp_solver": "turbompc_autodiff",
        "linearization_backend": "turbompc_jax_autodiff",
        "turbompc_forward_backend": args.turbompc_forward_backend,
        "turbompc_backward_backend": args.turbompc_backward_backend,
        "horizon_steps": args.horizon_steps,
        "sim_steps": args.sim_steps,
        "control_dt": args.control_dt,
        "max_iter": args.max_iter,
        "turbompc_sqp_iterations": 1,
        "turbompc_eps_abs": args.turbompc_eps_abs,
        "turbompc_eps_rel": args.turbompc_eps_rel,
        "rho": args.rho,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "warm_starting": True,
        "n_variables": desc.n_variables,
        "n_constraints": desc.n_constraints,
        "nnz_p": desc.nnz_p,
        "nnz_a": desc.nnz_a,
        "dense_block_dim": desc.block_dim,
        "dense_inequality_dim": desc.inequality_dim,
        "problem_build_s": problem_build_s,
        "setup_s": setup_s,
        "compile_s": warmup_s,
        "warmup_compile_and_run_s": warmup_s,
        "elapsed_s": elapsed_s,
        "solve_s": elapsed_s,
        "total_rti_steps": args.batch_size * args.sim_steps,
        "rti_steps_per_s": throughput,
        "mean_step": float(np.mean(step_lengths_np)),
        "final_position_rms": float(np.sqrt(np.mean(final_position**2))),
        "final_state_rms": float(np.sqrt(np.mean(final_state**2))),
        "max_violation": float(np.max(violations_np)),
        "mean_qp_prim": float(np.mean(prim_np)),
        "mean_qp_dual": float(np.mean(dual_np)),
    }
    print(
        f"closed-loop elapsed={elapsed_s:.3f}s "
        f"({throughput:.3g} SQP-RTI steps/s), "
        f"final_position_rms={summary['final_position_rms']:.3g}, "
        f"max_violation={summary['max_violation']:.3e}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--sim-time", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--turbompc-eps-abs", type=float, default=1e-3)
    parser.add_argument("--turbompc-eps-rel", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
    parser.add_argument("--turbompc-forward-backend", default="admm_fused_cudss")
    parser.add_argument("--turbompc-backward-backend", default="direct_cudss_ffi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-npz", type=pathlib.Path, default=None)
    parser.add_argument("--plot-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/turbompc_autodiff_crazyflie_states.png"))
    parser.add_argument("--position-plot-path", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/turbompc_autodiff_crazyflie_positions.png"))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.sim_steps is None:
        args.sim_steps = int(np.ceil(args.sim_time / args.control_dt))
    summary = run(args)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
