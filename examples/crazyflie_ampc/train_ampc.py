#!/usr/bin/env python3
"""DAgger training example for a Crazyflie AMPC policy.

This script builds a small neural policy that imitates a selected target from
the jitted sparse SQP MPC solver.  Reusable AMPC pieces live in
``warpmpc.jax_ampc``;
this file keeps the Crazyflie SQP setup, dynamics, costs, plots, and experiment
orchestration.
"""

from __future__ import annotations

import argparse
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
    CRAZYFLIE_NU,
    CRAZYFLIE_NX,
    CRAZYFLIE_NZ,
    CRAZYFLIE_Q,
    CRAZYFLIE_R,
    crazyflie_initial_guess_and_params,
    crazyflie_jax_euler_step,
    make_crazyflie_sqp_problem,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    FilterLineSearchSettings,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)
from warpmpc.jax_ampc import (
    CostPercentileConfig,
    FilterBounds as Bounds,
    MLP as PolicyMLP,
    Normalization,
    StatePercentileConfig,
    TrainingConfig,
    append_dataset,
    calibrate_filter_bounds,
    compute_normalization,
    create_train_state,
    dagger_schedule,
    dataset_keep_per_step,
    device_tree_block,
    estimate_dataset_gb,
    filter_transition_valid_mask_host,
    filtered_rollout_valid_mask_host,
    format_filtered_rollout_plot_stats,
    format_rollout_metrics,
    infinite_filter_bounds,
    initial_normalization,
    load_filter_bounds_npz,
    make_output_spec,
    make_policy_rollout,
    make_sqp_collect_rollout,
    make_training_kernels,
    parse_int_tuple,
    plot_rollout_histogram_pair,
    plot_state_distribution_pair,
    prediction_target_name,
    quadratic_rollout_cost_host,
    rollout_qp_metrics,
    sample_scaled_unit_ball_noise_np,
    sample_scaled_unit_ball_noise,
    save_flax_checkpoint,
    save_dataset_increment,
    subsample_dataset_increment,
    summarize_rollout_host,
    train_supervised_policy,
    write_json,
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

PROCESS_NOISE_SIGMA = 0.05
INPUT_NOISE_ETA = 0.1
DEFAULT_DAGGER_MIXING = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0)
CRAZYFLIE_Q_DIAG = np.diag(CRAZYFLIE_Q).astype(np.float64)
CRAZYFLIE_R_DIAG = np.diag(CRAZYFLIE_R).astype(np.float64)
CRAZYFLIE_PROCESS_NOISE_SCALE = np.full((CRAZYFLIE_NX,), PROCESS_NOISE_SIGMA, dtype=np.float64)
CRAZYFLIE_PROCESS_NOISE_SCALE[3:6] /= 5.0
CRAZYFLIE_PROCESS_NOISE_SCALE[9:11] *= 100.0
CRAZYFLIE_INPUT_NOISE_SCALE = np.full((CRAZYFLIE_NU,), INPUT_NOISE_ETA, dtype=np.float64)
CRAZYFLIE_INITIAL_STATE_NOISE_SCALE = 10.0 * CRAZYFLIE_PROCESS_NOISE_SCALE


def sample_initial_states(
    batch_size: int,
    seed: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Sample the AMPC initial distribution requested for this example."""

    rng = np.random.default_rng(seed)
    x0 = sample_scaled_unit_ball_noise_np(
        rng,
        batch_size,
        CRAZYFLIE_INITIAL_STATE_NOISE_SCALE,
        dtype=dtype,
    )
    x0[:, :3] = rng.uniform(-1.0, 1.0, size=(batch_size, 3)).astype(dtype)
    x0[:, 5] = rng.uniform(
        -0.5 * np.pi,
        0.5 * np.pi,
        size=(batch_size,),
    ).astype(dtype)
    return x0


def make_crazyflie_dynamics_step(control_dt_value: float, substeps: int, dtype: np.dtype):
    jdtype = jnp.dtype(dtype)
    control_dt = jnp.asarray(control_dt_value, dtype=jdtype)
    process_scale = jnp.asarray(CRAZYFLIE_PROCESS_NOISE_SCALE, dtype=jdtype)
    input_scale = jnp.asarray(CRAZYFLIE_INPUT_NOISE_SCALE, dtype=jdtype)

    def dynamics_step(key, x, u, noise_scale):
        key_q, key_d = jax.random.split(key)
        scale = jnp.asarray(noise_scale, dtype=jdtype)
        q = scale * sample_scaled_unit_ball_noise(key_q, x.shape[0], process_scale, dtype=jdtype)
        d = scale * sample_scaled_unit_ball_noise(key_d, x.shape[0], input_scale, dtype=jdtype)
        step_dt = control_dt / jnp.asarray(substeps, dtype=control_dt.dtype)

        def integrate_body(_, x_cur):
            return crazyflie_jax_euler_step(x_cur, u + d, step_dt)

        return jax.lax.fori_loop(0, substeps, integrate_body, x) + q

    return dynamics_step


def _stage_cost_jax(x, u, control_dt):
    q_diag = jnp.asarray(np.diag(CRAZYFLIE_Q), dtype=x.dtype)
    r_diag = jnp.asarray(np.diag(CRAZYFLIE_R), dtype=x.dtype)
    state_cost = jnp.sum(q_diag[None, :] * x * x, axis=1)
    action_cost = jnp.sum(r_diag[None, :] * u * u, axis=1)
    return 0.5 * control_dt * (state_cost + action_cost)


def _terminal_cost_jax(x):
    q_diag = jnp.asarray(np.diag(CRAZYFLIE_Q), dtype=x.dtype)
    return 0.5 * jnp.sum(q_diag[None, :] * x * x, axis=1)


def _make_sqp(args: argparse.Namespace, dtype: np.dtype):
    settings = OSQPSettings(
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        max_iter=args.osqp_max_iter,
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
    problem = make_crazyflie_sqp_problem(args.horizon_steps)
    plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
    compile_kwargs = {
        "dtype": dtype,
        "osqp_settings": settings,
        "transpose_work": True,
        "segmented": True,
        "segment_budget": args.segment_budget,
        "segment_strategy": args.segment_strategy,
        "level_scheduled_solve": args.level_scheduled_solve,
        "level_scheduled_solve_threshold": args.level_scheduled_solve_threshold,
        "qdldl_backend": args.qdldl_backend,
        "qdldl_factor_backend": args.qdldl_factor_backend,
        "qdldl_solve_backend": args.qdldl_solve_backend,
        "line_search_settings": line_search_settings,
        "group_repeated_stages": args.group_repeated_stages,
    }
    sqp = compile_sparse_mpc_sqp(problem, plan, **compile_kwargs)
    return plan, sqp


def run_sqp_rollout_diagnostics(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    jdtype = jnp.dtype(dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    output_spec = make_output_spec(
        args.prediction_target,
        nx=CRAZYFLIE_NX,
        nu=CRAZYFLIE_NU,
        nz=CRAZYFLIE_NZ,
        horizon_steps=args.horizon_steps,
    )
    hidden_sizes = parse_int_tuple(args.hidden_sizes)
    model = PolicyMLP(
        hidden_sizes=hidden_sizes,
        output_dim=output_spec.target_dim,
        activation=args.activation,
        negative_slope=args.leaky_relu_slope,
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    state = create_train_state(
        model=model,
        key=init_key,
        input_dim=CRAZYFLIE_NX,
        dtype=dtype,
        learning_rate=args.learning_rate,
    )
    normalization = initial_normalization(CRAZYFLIE_NX, output_spec.target_dim, dtype)

    print("building Crazyflie SQP solver...", flush=True)
    plan, sqp = _make_sqp(args, dtype)
    print(
        "Crazyflie SQP rollout diagnostics:",
        f"dtype={dtype}",
        f"collect_batch={args.collect_batch_size:,}",
        f"eval_batch={args.eval_batch_size:,}",
        f"rollout_steps={args.rollout_steps}",
        f"sqp_iterations={args.sqp_iterations}",
        f"osqp_max_iter={args.osqp_max_iter}",
        f"qdldl_variant={sqp.osqp.variant}",
        f"qdldl_factor_backend={sqp.osqp.qdldl.factor_backend}",
        f"qdldl_solve_backend={sqp.osqp.qdldl.solve_backend}",
        f"control_dt={args.control_dt}",
        f"horizon_steps={args.horizon_steps}",
        f"n={plan.n_variables}",
        f"m={plan.n_constraints}",
        f"nnz_P={plan.p_pattern.nnz}",
        f"nnz_A={plan.a_pattern.nnz}",
        flush=True,
    )
    print(
        "SQP warm starts: initial trajectory is linear state interpolation to zero "
        "with zero controls; closed-loop RTI reuses the previous primal MPC solution "
        "without shifting and carries the OSQP ADMM (x, z, y) state across solves.",
        flush=True,
    )

    control_dt = jnp.asarray(args.control_dt, dtype=jdtype)
    collect_rollout = make_sqp_collect_rollout(
        sqp=sqp,
        model=model,
        output_spec=output_spec,
        nx=CRAZYFLIE_NX,
        rollout_steps=args.rollout_steps,
        sqp_iterations=args.sqp_iterations,
        dynamics_step=make_crazyflie_dynamics_step(args.control_dt, args.integrator_substeps, dtype),
        stage_cost=lambda x, u: _stage_cost_jax(x, u, control_dt),
        terminal_cost=_terminal_cost_jax,
        dtype=dtype,
        valid_max_constraint_violation=args.valid_max_constraint_violation,
        valid_max_qp_residual=args.valid_max_qp_residual,
        require_line_search_accepted=args.require_line_search_accepted,
    )
    inf_bounds = Bounds(
        lower=jnp.full((CRAZYFLIE_NX,), -jnp.inf, dtype=jdtype),
        upper=jnp.full((CRAZYFLIE_NX,), jnp.inf, dtype=jdtype),
        cost_threshold=jnp.asarray(jnp.inf, dtype=jdtype),
    )
    time_grid = np.arange(args.rollout_steps + 1, dtype=np.float64) * args.control_dt

    filter_batch_size = args.filter_batch_size or args.collect_batch_size
    print(f"running diagnostic noise-free SQP rollout batch={filter_batch_size:,}...", flush=True)
    x0_np = sample_initial_states(filter_batch_size, args.seed + 101, dtype)
    z0_np, params0_np = crazyflie_initial_guess_and_params(
        x0_np,
        n_steps=args.horizon_steps,
        dtype=dtype,
    )
    z0, params0, x0 = jnp.asarray(z0_np), jnp.asarray(params0_np), jnp.asarray(x0_np)
    key, rollout_key = jax.random.split(key)
    start = time.perf_counter()
    noise_free = collect_rollout(
        state.params,
        normalization,
        inf_bounds,
        z0,
        params0,
        x0,
        rollout_key,
        jnp.asarray(0.0, dtype=jdtype),
        jnp.asarray(0.0, dtype=jdtype),
    )
    device_tree_block(noise_free)
    noise_free_elapsed = time.perf_counter() - start
    noise_free_states = np.asarray(jax.device_get(noise_free.states))
    noise_free_actions = np.asarray(jax.device_get(noise_free.applied_u))
    noise_free_valid = np.asarray(jax.device_get(noise_free.valid_mask))
    noise_free_costs = quadratic_rollout_cost_host(
        noise_free_states,
        noise_free_actions,
        state_weights=CRAZYFLIE_Q_DIAG,
        action_weights=CRAZYFLIE_R_DIAG,
        control_dt=args.control_dt,
    )
    bounds_lower, bounds_upper, cost_threshold, bounds_stats, _ = calibrate_filter_bounds(
        noise_free_states,
        actions=noise_free_actions,
        rollout_costs=noise_free_costs,
        state_config=StatePercentileConfig(
            low_percentile=args.filter_low_percentile,
            high_percentile=args.filter_high_percentile,
            min_width=args.filter_min_width,
            margin_abs=args.filter_margin_abs,
            margin_scale=args.filter_margin_scale,
        ),
        cost_config=CostPercentileConfig(
            high_percentile=args.filter_cost_high_percentile,
            min_width=args.filter_cost_min_width,
            margin_abs=args.filter_cost_margin_abs,
            margin_scale=args.filter_cost_margin_scale,
        ),
    )
    bounds = Bounds(
        lower=jnp.asarray(bounds_lower, dtype=jdtype),
        upper=jnp.asarray(bounds_upper, dtype=jdtype),
        cost_threshold=jnp.asarray(cost_threshold, dtype=jdtype),
    )
    np.savez(
        args.output_dir / "outlier_bounds.npz",
        lower=bounds_lower,
        upper=bounds_upper,
        cost_threshold=np.asarray(cost_threshold, dtype=np.float64),
    )
    noise_free_valid, noise_free_plot_filter_stats = filtered_rollout_valid_mask_host(
        states=noise_free_states,
        rollout_costs=noise_free_costs,
        valid_mask=noise_free_valid,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        cost_threshold=cost_threshold,
    )
    print(format_filtered_rollout_plot_stats("diagnostic_sqp_noise_free", noise_free_plot_filter_stats), flush=True)
    plot_state_distribution_pair(
        args.output_dir / "plots" / "diagnostic_sqp_noise_free_state_distribution.pdf",
        time_grid,
        noise_free_states,
        state_labels=STATE_LABELS,
        title="Diagnostic noise-free SQP rollout",
        valid_mask=noise_free_valid,
    )
    plot_rollout_histogram_pair(
        args.output_dir / "plots" / "diagnostic_sqp_noise_free_histograms.pdf",
        noise_free_states,
        noise_free_costs,
        state_labels=STATE_LABELS,
        title="Diagnostic noise-free SQP histograms",
        valid_mask=noise_free_valid,
    )
    noise_free_metrics = summarize_rollout_host(
        name="diagnostic_sqp_noise_free",
        states=noise_free_states,
        actions=noise_free_actions,
        rollout_costs=noise_free_costs,
        bounds=(bounds_lower, bounds_upper),
        cost_threshold=cost_threshold,
        valid_datum_mask=noise_free_valid,
        converged_position_norm=args.converged_position_norm,
        converged_state_norm=args.converged_state_norm,
    )
    noise_free_metrics["elapsed_s"] = noise_free_elapsed
    noise_free_metrics.update(rollout_qp_metrics(noise_free))
    print(format_rollout_metrics(noise_free_metrics), flush=True)
    print(
        "outlier bounds abs max:",
        np.array2string(np.maximum(np.abs(bounds_lower), np.abs(bounds_upper)), precision=3),
        flush=True,
    )
    print(
        "outlier bound calibration filter:",
        f"finite_traj={int(bounds_stats['finite_trajectory_count']):,}/{int(bounds_stats['batch_size']):,}",
        f"nonfinite_filtered={int(bounds_stats['nonfinite_trajectory_count']):,}",
        f"nan_traj_filtered={int(bounds_stats['nan_trajectory_count']):,}",
        f"inf_traj_filtered={int(bounds_stats['inf_trajectory_count']):,}",
        f"cost_filtered={int(bounds_stats['bounds_cost_filtered_trajectory_count']):,}",
        f"bounds_traj={int(bounds_stats['bounds_trajectory_count']):,}",
        f"state_samples_used={int(bounds_stats['bounds_state_sample_count']):,}",
        flush=True,
    )
    print(
        "outlier cost threshold:",
        f"threshold={cost_threshold:.4g}",
        f"percentile={float(bounds_stats['filter_cost_high_percentile']):.3g}",
        f"percentile_value={float(bounds_stats['filter_cost_percentile_value']):.4g}",
        f"margin_abs={float(bounds_stats['filter_cost_margin_abs']):.4g}",
        f"margin_scale={float(bounds_stats['filter_cost_margin_scale']):.4g}",
        flush=True,
    )
    del noise_free, noise_free_states, noise_free_actions, noise_free_costs, noise_free_valid
    del z0, params0, x0

    print(f"running diagnostic noisy SQP rollout batch={args.collect_batch_size:,}...", flush=True)
    x0_np = sample_initial_states(args.collect_batch_size, args.seed + 1000, dtype)
    z0_np, params0_np = crazyflie_initial_guess_and_params(
        x0_np,
        n_steps=args.horizon_steps,
        dtype=dtype,
    )
    z0, params0, x0 = jnp.asarray(z0_np), jnp.asarray(params0_np), jnp.asarray(x0_np)
    key, rollout_key = jax.random.split(key)
    start = time.perf_counter()
    noisy = collect_rollout(
        state.params,
        normalization,
        bounds,
        z0,
        params0,
        x0,
        rollout_key,
        jnp.asarray(0.0, dtype=jdtype),
        jnp.asarray(1.0, dtype=jdtype),
    )
    device_tree_block(noisy)
    noisy_elapsed = time.perf_counter() - start
    noisy_states = np.asarray(jax.device_get(noisy.states))
    noisy_actions = np.asarray(jax.device_get(noisy.applied_u))
    noisy_valid = np.asarray(jax.device_get(noisy.valid_mask))
    noisy_costs = quadratic_rollout_cost_host(
        noisy_states,
        noisy_actions,
        state_weights=CRAZYFLIE_Q_DIAG,
        action_weights=CRAZYFLIE_R_DIAG,
        control_dt=args.control_dt,
    )
    noisy_valid, noisy_plot_filter_stats = filtered_rollout_valid_mask_host(
        states=noisy_states,
        rollout_costs=noisy_costs,
        valid_mask=noisy_valid,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        cost_threshold=cost_threshold,
    )
    print(format_filtered_rollout_plot_stats("diagnostic_sqp_noisy", noisy_plot_filter_stats), flush=True)
    plot_state_distribution_pair(
        args.output_dir / "plots" / "diagnostic_sqp_noisy_state_distribution.pdf",
        time_grid,
        noisy_states,
        state_labels=STATE_LABELS,
        title="Diagnostic noisy SQP rollout",
        valid_mask=noisy_valid,
    )
    plot_rollout_histogram_pair(
        args.output_dir / "plots" / "diagnostic_sqp_noisy_histograms.pdf",
        noisy_states,
        noisy_costs,
        state_labels=STATE_LABELS,
        title="Diagnostic noisy SQP histograms",
        valid_mask=noisy_valid,
    )
    noisy_metrics = summarize_rollout_host(
        name="diagnostic_sqp_noisy",
        states=noisy_states,
        actions=noisy_actions,
        rollout_costs=noisy_costs,
        bounds=(bounds_lower, bounds_upper),
        cost_threshold=cost_threshold,
        valid_datum_mask=noisy_valid,
        converged_position_norm=args.converged_position_norm,
        converged_state_norm=args.converged_state_norm,
    )
    noisy_metrics["elapsed_s"] = noisy_elapsed
    noisy_metrics.update(rollout_qp_metrics(noisy))
    print(format_rollout_metrics(noisy_metrics), flush=True)

    summary = {
        "mode": "sqp_rollout_diagnostics",
        "noise_free": noise_free_metrics,
        "noisy": noisy_metrics,
        "outlier_bounds_stats": bounds_stats,
        "valid_plot_filters": {
            "noise_free": noise_free_plot_filter_stats,
            "noisy": noisy_plot_filter_stats,
        },
        "outlier_bounds_npz": str(args.output_dir / "outlier_bounds.npz"),
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "sqp_rollout_diagnostics_summary.json", summary)
    print(f"Wrote {args.output_dir / 'sqp_rollout_diagnostics_summary.json'}", flush=True)
    return summary


def _run_noise_free_calibration(
    *,
    args: argparse.Namespace,
    dtype: np.dtype,
    jdtype,
    key,
    state,
    normalization: Normalization,
    collect_rollout,
    inf_bounds: Bounds,
    time_grid: np.ndarray,
) -> tuple[object, Bounds, np.ndarray, np.ndarray, float, dict[str, int | float], dict[str, object], float]:
    filter_batch_size = args.filter_batch_size or args.collect_batch_size
    print(f"running noise-free expert calibration batch={filter_batch_size:,}...", flush=True)
    x0_np = sample_initial_states(filter_batch_size, args.seed + 101, dtype)
    z0_np, params0_np = crazyflie_initial_guess_and_params(
        x0_np,
        n_steps=args.horizon_steps,
        dtype=dtype,
    )
    z0, params0, x0 = jnp.asarray(z0_np), jnp.asarray(params0_np), jnp.asarray(x0_np)
    key, rollout_key = jax.random.split(key)
    warmup_start = time.perf_counter()
    calib = collect_rollout(
        state.params,
        normalization,
        inf_bounds,
        z0,
        params0,
        x0,
        rollout_key,
        jnp.asarray(0.0, dtype=jdtype),
        jnp.asarray(0.0, dtype=jdtype),
    )
    device_tree_block(calib)
    warmup_compilation_elapsed = time.perf_counter() - warmup_start
    calib_states = np.asarray(jax.device_get(calib.states))
    calib_actions = np.asarray(jax.device_get(calib.applied_u))
    calib_rollout_costs = quadratic_rollout_cost_host(
        calib_states,
        calib_actions,
        state_weights=CRAZYFLIE_Q_DIAG,
        action_weights=CRAZYFLIE_R_DIAG,
        control_dt=args.control_dt,
    )
    bounds_lower, bounds_upper, cost_threshold, bounds_stats, _ = calibrate_filter_bounds(
        calib_states,
        actions=calib_actions,
        rollout_costs=calib_rollout_costs,
        state_config=StatePercentileConfig(
            low_percentile=args.filter_low_percentile,
            high_percentile=args.filter_high_percentile,
            min_width=args.filter_min_width,
            margin_abs=args.filter_margin_abs,
            margin_scale=args.filter_margin_scale,
        ),
        cost_config=CostPercentileConfig(
            high_percentile=args.filter_cost_high_percentile,
            min_width=args.filter_cost_min_width,
            margin_abs=args.filter_cost_margin_abs,
            margin_scale=args.filter_cost_margin_scale,
        ),
    )
    bounds = Bounds(
        lower=jnp.asarray(bounds_lower, dtype=jdtype),
        upper=jnp.asarray(bounds_upper, dtype=jdtype),
        cost_threshold=jnp.asarray(cost_threshold, dtype=jdtype),
    )
    np.savez(
        args.output_dir / "outlier_bounds.npz",
        lower=bounds_lower,
        upper=bounds_upper,
        cost_threshold=np.asarray(cost_threshold, dtype=np.float64),
    )
    calib_valid = np.asarray(jax.device_get(calib.valid_mask))
    calib_valid = filter_transition_valid_mask_host(
        states=calib_states,
        rollout_costs=calib_rollout_costs,
        valid_mask=calib_valid,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        cost_threshold=cost_threshold,
    )
    plot_state_distribution_pair(
        args.output_dir / "plots" / "calibration_noise_free_state_distribution.pdf",
        time_grid,
        calib_states,
        state_labels=STATE_LABELS,
        title="Noise-free SQP calibration rollout",
        valid_mask=calib_valid,
    )
    plot_rollout_histogram_pair(
        args.output_dir / "plots" / "calibration_noise_free_histograms.pdf",
        calib_states,
        calib_rollout_costs,
        state_labels=STATE_LABELS,
        title="Noise-free SQP calibration histograms",
        valid_mask=calib_valid,
    )
    calibration_metrics = summarize_rollout_host(
        name="sqp_noise_free_calibration",
        states=calib_states,
        actions=calib_actions,
        rollout_costs=calib_rollout_costs,
        bounds=(bounds_lower, bounds_upper),
        cost_threshold=cost_threshold,
        valid_datum_mask=calib_valid,
        converged_position_norm=args.converged_position_norm,
        converged_state_norm=args.converged_state_norm,
    )
    print(format_rollout_metrics(calibration_metrics), flush=True)
    print(
        "outlier bounds abs max:",
        np.array2string(np.maximum(np.abs(bounds_lower), np.abs(bounds_upper)), precision=3),
        flush=True,
    )
    print(
        "outlier bound calibration filter:",
        f"finite_traj={int(bounds_stats['finite_trajectory_count']):,}/{int(bounds_stats['batch_size']):,}",
        f"nonfinite_filtered={int(bounds_stats['nonfinite_trajectory_count']):,}",
        f"nan_traj_filtered={int(bounds_stats['nan_trajectory_count']):,}",
        f"inf_traj_filtered={int(bounds_stats['inf_trajectory_count']):,}",
        f"cost_filtered={int(bounds_stats['bounds_cost_filtered_trajectory_count']):,}",
        f"bounds_traj={int(bounds_stats['bounds_trajectory_count']):,}",
        f"state_samples_used={int(bounds_stats['bounds_state_sample_count']):,}",
        flush=True,
    )
    print(
        "outlier cost threshold:",
        f"threshold={cost_threshold:.4g}",
        f"percentile={float(bounds_stats['filter_cost_high_percentile']):.3g}",
        f"percentile_value={float(bounds_stats['filter_cost_percentile_value']):.4g}",
        f"margin_abs={float(bounds_stats['filter_cost_margin_abs']):.4g}",
        f"margin_scale={float(bounds_stats['filter_cost_margin_scale']):.4g}",
        flush=True,
    )
    return (
        key,
        bounds,
        bounds_lower,
        bounds_upper,
        cost_threshold,
        bounds_stats,
        calibration_metrics,
        warmup_compilation_elapsed,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    jdtype = jnp.dtype(dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    output_spec = make_output_spec(
        args.prediction_target,
        nx=CRAZYFLIE_NX,
        nu=CRAZYFLIE_NU,
        nz=CRAZYFLIE_NZ,
        horizon_steps=args.horizon_steps,
    )
    hidden_sizes = parse_int_tuple(args.hidden_sizes)
    model = PolicyMLP(
        hidden_sizes=hidden_sizes,
        output_dim=output_spec.target_dim,
        activation=args.activation,
        negative_slope=args.leaky_relu_slope,
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    state = create_train_state(
        model=model,
        key=init_key,
        input_dim=CRAZYFLIE_NX,
        dtype=dtype,
        learning_rate=args.learning_rate,
    )
    normalization = initial_normalization(CRAZYFLIE_NX, output_spec.target_dim, dtype)
    train_step, eval_loss_sums = make_training_kernels(model)
    run_start = time.perf_counter()

    print("building Crazyflie SQP solver...", flush=True)
    plan, sqp = _make_sqp(args, dtype)
    print(
        "Crazyflie AMPC setup:",
        f"dtype={dtype}",
        f"collect_batch={args.collect_batch_size:,}",
        f"eval_batch={args.eval_batch_size:,}",
        f"rollout_steps={args.rollout_steps}",
        f"sqp_iterations={args.sqp_iterations}",
        f"osqp_max_iter={args.osqp_max_iter}",
        f"qdldl_variant={sqp.osqp.variant}",
        f"qdldl_factor_backend={sqp.osqp.qdldl.factor_backend}",
        f"qdldl_solve_backend={sqp.osqp.qdldl.solve_backend}",
        f"control_dt={args.control_dt}",
        f"horizon_steps={args.horizon_steps}",
        f"prediction_target={prediction_target_name(output_spec)}",
        f"target_dim={output_spec.target_dim}",
        f"n={plan.n_variables}",
        f"m={plan.n_constraints}",
        f"nnz_P={plan.p_pattern.nnz}",
        f"nnz_A={plan.a_pattern.nnz}",
        flush=True,
    )
    print(
        "SQP warm starts: initial trajectory is linear state interpolation to zero "
        "with zero controls; closed-loop RTI reuses the previous primal MPC solution "
        "without shifting and carries the OSQP ADMM (x, z, y) state across solves.",
        flush=True,
    )

    control_dt = jnp.asarray(args.control_dt, dtype=jdtype)
    dynamics_step = make_crazyflie_dynamics_step(args.control_dt, args.integrator_substeps, dtype)
    collect_rollout = make_sqp_collect_rollout(
        sqp=sqp,
        model=model,
        output_spec=output_spec,
        nx=CRAZYFLIE_NX,
        rollout_steps=args.rollout_steps,
        sqp_iterations=args.sqp_iterations,
        dynamics_step=dynamics_step,
        stage_cost=lambda x, u: _stage_cost_jax(x, u, control_dt),
        terminal_cost=_terminal_cost_jax,
        dtype=dtype,
        valid_max_constraint_violation=args.valid_max_constraint_violation,
        valid_max_qp_residual=args.valid_max_qp_residual,
        require_line_search_accepted=args.require_line_search_accepted,
    )
    policy_rollout = make_policy_rollout(
        model=model,
        output_spec=output_spec,
        rollout_steps=args.rollout_steps,
        dynamics_step=dynamics_step,
        stage_cost=lambda x, u: _stage_cost_jax(x, u, control_dt),
        terminal_cost=_terminal_cost_jax,
        dtype=dtype,
    )

    generated_samples_per_iteration = args.collect_batch_size * args.rollout_steps
    keep_per_step = dataset_keep_per_step(args.collect_batch_size, args.dataset_keep_fraction)
    kept_samples_per_iteration = keep_per_step * args.rollout_steps
    max_samples = args.dagger_iterations * kept_samples_per_iteration
    print(
        f"dataset subsampling: generated_per_iter={generated_samples_per_iteration:,}, "
        f"keep_per_step={keep_per_step:,}/{args.collect_batch_size:,}, "
        f"kept_per_iter={kept_samples_per_iteration:,} "
        f"({kept_samples_per_iteration / max(1, generated_samples_per_iteration):.3%})",
        flush=True,
    )
    print(
        f"preallocating dataset buffers: samples={max_samples:,}, "
        f"estimated={estimate_dataset_gb(max_samples, CRAZYFLIE_NX, output_spec.target_dim, dtype):.2f} GB",
        flush=True,
    )
    dataset_x = jnp.zeros((max_samples, CRAZYFLIE_NX), dtype=jdtype)
    dataset_y = jnp.zeros((max_samples, output_spec.target_dim), dtype=jdtype)
    dataset_mask = jnp.zeros((max_samples,), dtype=jnp.bool_)
    jax.block_until_ready(dataset_x)

    inf_bounds = infinite_filter_bounds(CRAZYFLIE_NX, jdtype)
    time_grid = np.arange(args.rollout_steps + 1, dtype=np.float64) * args.control_dt

    if args.disable_filtering:
        bounds = inf_bounds
        bounds_lower = np.full((CRAZYFLIE_NX,), -np.inf, dtype=dtype)
        bounds_upper = np.full((CRAZYFLIE_NX,), np.inf, dtype=dtype)
        cost_threshold = float("inf")
        bounds_stats = {"mode": "disabled"}
        calibration_metrics = None
        warmup_compilation_elapsed = 0.0
        print("outlier filtering disabled; skipping noise-free calibration rollout.", flush=True)
    elif args.filter_bounds_npz is not None:
        bounds_lower, bounds_upper, cost_threshold = load_filter_bounds_npz(
            args.filter_bounds_npz,
            dtype=dtype,
        )
        bounds = Bounds(
            lower=jnp.asarray(bounds_lower, dtype=jdtype),
            upper=jnp.asarray(bounds_upper, dtype=jdtype),
            cost_threshold=jnp.asarray(cost_threshold, dtype=jdtype),
        )
        bounds_stats = {
            "mode": "from_npz",
            "filter_bounds_npz": str(args.filter_bounds_npz),
            "filter_cost_threshold": cost_threshold,
        }
        calibration_metrics = None
        warmup_compilation_elapsed = 0.0
        print(
            f"loaded hard-coded outlier bounds from {args.filter_bounds_npz}; "
            "skipping noise-free calibration rollout.",
            flush=True,
        )
    else:
        (
            key,
            bounds,
            bounds_lower,
            bounds_upper,
            cost_threshold,
            bounds_stats,
            calibration_metrics,
            warmup_compilation_elapsed,
        ) = _run_noise_free_calibration(
            args=args,
            dtype=dtype,
            jdtype=jdtype,
            key=key,
            state=state,
            normalization=normalization,
            collect_rollout=collect_rollout,
            inf_bounds=inf_bounds,
            time_grid=time_grid,
        )

    history: list[dict[str, object]] = []
    current_capacity = 0
    initial_noisy_mpc_metrics: dict[str, object] | None = None
    dagger_mixing = dagger_schedule(args.dagger_iterations, args.dagger_mixing)
    timing_totals: dict[str, float] = {
        "warmup_compilations_s": warmup_compilation_elapsed,
        "dataset_generation_s": 0.0,
        "training_s": 0.0,
    }

    for iteration, dagger_beta in enumerate(dagger_mixing):
        print(
            f"\n=== dagger iteration {iteration + 1}/{args.dagger_iterations} "
            f"(dagger_beta={dagger_beta:.3g}) ===",
            flush=True,
        )
        dataset_generation_start = time.perf_counter()
        x0_np = sample_initial_states(args.collect_batch_size, args.seed + 1000 + iteration, dtype)
        z0_np, params0_np = crazyflie_initial_guess_and_params(
            x0_np,
            n_steps=args.horizon_steps,
            dtype=dtype,
        )
        z0, params0, x0 = jnp.asarray(z0_np), jnp.asarray(params0_np), jnp.asarray(x0_np)
        key, rollout_key = jax.random.split(key)
        collect_start = time.perf_counter()
        rollout = collect_rollout(
            state.params,
            normalization,
            bounds,
            z0,
            params0,
            x0,
            rollout_key,
            jnp.asarray(dagger_beta, dtype=jdtype),
            jnp.asarray(1.0, dtype=jdtype),
        )
        device_tree_block(rollout)
        collect_elapsed = time.perf_counter() - collect_start

        raw_valid_inc_count = int(jax.device_get(jnp.sum(rollout.valid_mask)))
        key, subsample_key = jax.random.split(key)
        if keep_per_step < args.collect_batch_size:
            x_inc, y_inc, valid_inc, source_indices = subsample_dataset_increment(
                rollout.dataset_x,
                rollout.expert_y,
                rollout.valid_mask,
                subsample_key,
                keep_per_step=keep_per_step,
            )
        else:
            x_inc = rollout.dataset_x.reshape((kept_samples_per_iteration, CRAZYFLIE_NX))
            y_inc = rollout.expert_y.reshape((kept_samples_per_iteration, output_spec.target_dim))
            valid_inc = rollout.valid_mask.reshape((kept_samples_per_iteration,))
            source_indices = None
        x_inc = jnp.where(valid_inc[:, None], x_inc, jnp.zeros_like(x_inc))
        y_inc = jnp.where(valid_inc[:, None], y_inc, jnp.zeros_like(y_inc))
        valid_inc_count = int(jax.device_get(jnp.sum(valid_inc)))
        dataset_x, dataset_y, dataset_mask = append_dataset(
            dataset_x,
            dataset_y,
            dataset_mask,
            x_inc,
            y_inc,
            valid_inc,
            jnp.asarray(current_capacity, dtype=jnp.int32),
        )
        current_capacity += kept_samples_per_iteration
        total_valid_count = int(jax.device_get(jnp.sum(dataset_mask[:current_capacity])))
        dataset_generation_elapsed = time.perf_counter() - dataset_generation_start
        timing_totals["dataset_generation_s"] += dataset_generation_elapsed
        print(
            f"collection: elapsed={collect_elapsed:.2f}s, "
            f"raw_valid={raw_valid_inc_count:,}/{generated_samples_per_iteration:,} "
            f"({100.0 * raw_valid_inc_count / max(1, generated_samples_per_iteration):.2f}%), "
            f"kept_valid={valid_inc_count:,}/{kept_samples_per_iteration:,} "
            f"({100.0 * valid_inc_count / max(1, kept_samples_per_iteration):.2f}%), "
            f"dataset_valid={total_valid_count:,}/{current_capacity:,}",
            flush=True,
        )
        if args.save_dataset_increments:
            save_dataset_increment(
                args.output_dir / "dataset_increments" / f"iter_{iteration:02d}.npz",
                x=x_inc,
                y=y_inc,
                valid_mask=valid_inc,
                source_indices=source_indices,
                iteration=iteration,
                dagger_beta=dagger_beta,
                generated_samples=generated_samples_per_iteration,
                kept_samples=kept_samples_per_iteration,
                dataset_keep_fraction=args.dataset_keep_fraction,
            )
            print(f"saved dataset increment for iter={iteration}", flush=True)

        rollout_states = np.asarray(jax.device_get(rollout.states))
        rollout_actions = np.asarray(jax.device_get(rollout.applied_u))
        rollout_valid = np.asarray(jax.device_get(rollout.valid_mask))
        rollout_costs = quadratic_rollout_cost_host(
            rollout_states,
            rollout_actions,
            state_weights=CRAZYFLIE_Q_DIAG,
            action_weights=CRAZYFLIE_R_DIAG,
            control_dt=args.control_dt,
        )
        collect_metrics = summarize_rollout_host(
            name=f"sqp_dagger_collect_iter_{iteration:02d}",
            states=rollout_states,
            actions=rollout_actions,
            rollout_costs=rollout_costs,
            bounds=(bounds_lower, bounds_upper),
            cost_threshold=cost_threshold,
            valid_datum_mask=rollout_valid,
            converged_position_norm=args.converged_position_norm,
            converged_state_norm=args.converged_state_norm,
        )
        collect_metrics["elapsed_s"] = collect_elapsed
        collect_metrics["dataset_generation_elapsed_s"] = dataset_generation_elapsed
        collect_metrics.update(rollout_qp_metrics(rollout))
        collect_metrics["generated_samples"] = generated_samples_per_iteration
        collect_metrics["kept_samples"] = kept_samples_per_iteration
        collect_metrics["raw_valid_count"] = raw_valid_inc_count
        collect_metrics["kept_valid_count"] = valid_inc_count
        collect_metrics["dataset_keep_fraction"] = args.dataset_keep_fraction
        print(format_rollout_metrics(collect_metrics), flush=True)
        plot_state_distribution_pair(
            args.output_dir / "plots" / f"dataset_iter_{iteration:02d}_state_distribution.pdf",
            time_grid,
            rollout_states,
            state_labels=STATE_LABELS,
            title=f"DAgger collection iter {iteration} (beta={dagger_beta:.2f})",
            valid_mask=rollout_valid,
        )
        if iteration == 0 and abs(dagger_beta) < 1e-12:
            initial_noisy_mpc_metrics = dict(collect_metrics)
            initial_noisy_mpc_metrics["name"] = "sqp_noisy_initial_mpc"

        normalization = compute_normalization(
            dataset_x[:current_capacity],
            dataset_y[:current_capacity],
            dataset_mask[:current_capacity],
            jnp.asarray(args.normalization_std_floor, dtype=jdtype),
            jnp.asarray(args.policy_action_clip_std, dtype=jdtype),
        )
        device_tree_block(normalization)
        norm_host = jax.device_get(normalization)
        print(
            "normalization:",
            f"x_std_min={float(np.min(np.asarray(norm_host.x_std))):.3e}",
            f"y_std_min={float(np.min(np.asarray(norm_host.y_std))):.3e}",
            f"y_mean_first={np.array2string(np.asarray(norm_host.y_mean[:min(8, norm_host.y_mean.shape[0])]), precision=3)}",
            flush=True,
        )

        training_start = time.perf_counter()
        state, training_metrics, key = train_supervised_policy(
            config=TrainingConfig(
                epochs=args.train_epochs,
                batch_size=args.train_batch_size,
                train_fraction=args.train_fraction,
            ),
            state=state,
            normalization=normalization,
            train_step=train_step,
            eval_loss_sums=eval_loss_sums,
            dataset_x=dataset_x,
            dataset_y=dataset_y,
            dataset_mask=dataset_mask,
            capacity=current_capacity,
            key=key,
            iteration=iteration,
        )
        training_elapsed = time.perf_counter() - training_start
        training_metrics["elapsed_s"] = training_elapsed
        timing_totals["training_s"] += training_elapsed

        eval_metrics: dict[str, dict[str, object]] = {}
        for noise_name, noise_scale, seed_offset in (
            ("policy_noise_free", 0.0, 5000),
            ("policy_noisy", 1.0, 6000),
        ):
            x_eval = sample_initial_states(
                args.eval_batch_size,
                args.seed + seed_offset + iteration,
                dtype,
            )
            key, eval_key = jax.random.split(key)
            eval_start = time.perf_counter()
            policy_eval = policy_rollout(
                state.params,
                normalization,
                bounds,
                jnp.asarray(x_eval),
                eval_key,
                jnp.asarray(noise_scale, dtype=jdtype),
            )
            device_tree_block(policy_eval)
            eval_elapsed = time.perf_counter() - eval_start
            eval_states = np.asarray(jax.device_get(policy_eval.states))
            eval_actions = np.asarray(jax.device_get(policy_eval.applied_u))
            eval_valid = np.asarray(jax.device_get(policy_eval.valid_mask))
            eval_costs = quadratic_rollout_cost_host(
                eval_states,
                eval_actions,
                state_weights=CRAZYFLIE_Q_DIAG,
                action_weights=CRAZYFLIE_R_DIAG,
                control_dt=args.control_dt,
            )
            metrics = summarize_rollout_host(
                name=f"{noise_name}_iter_{iteration:02d}",
                states=eval_states,
                actions=eval_actions,
                rollout_costs=eval_costs,
                bounds=(bounds_lower, bounds_upper),
                cost_threshold=cost_threshold,
                valid_datum_mask=eval_valid,
                converged_position_norm=args.converged_position_norm,
                converged_state_norm=args.converged_state_norm,
            )
            metrics["elapsed_s"] = eval_elapsed
            eval_metrics[noise_name] = metrics
            print(format_rollout_metrics(metrics), flush=True)
            plot_state_distribution_pair(
                args.output_dir / "plots" / f"{noise_name}_iter_{iteration:02d}_state_distribution.pdf",
                time_grid,
                eval_states,
                state_labels=STATE_LABELS,
                title=f"{noise_name.replace('_', ' ')} iter {iteration}",
                valid_mask=eval_valid,
            )

        comparison = {
            "calibration": calibration_metrics,
            "initial_noisy_mpc": initial_noisy_mpc_metrics,
            "policy_noise_free": eval_metrics["policy_noise_free"],
            "policy_noisy": eval_metrics["policy_noisy"],
        }
        print("comparison anchor metrics:", flush=True)
        if calibration_metrics is not None:
            print("  " + format_rollout_metrics(calibration_metrics), flush=True)
        else:
            print("  calibration: skipped", flush=True)
        if initial_noisy_mpc_metrics is not None:
            print("  " + format_rollout_metrics(initial_noisy_mpc_metrics), flush=True)
        print("  " + format_rollout_metrics(eval_metrics["policy_noise_free"]), flush=True)
        print("  " + format_rollout_metrics(eval_metrics["policy_noisy"]), flush=True)
        wall_elapsed = time.perf_counter() - run_start
        timing_snapshot = {
            "iteration": iteration,
            "warmup_compilations_s": timing_totals["warmup_compilations_s"],
            "dataset_generation_s": timing_totals["dataset_generation_s"],
            "dataset_generation_iter_s": dataset_generation_elapsed,
            "sqp_collection_iter_s": collect_elapsed,
            "training_s": timing_totals["training_s"],
            "training_iter_s": training_elapsed,
            "wall_elapsed_s": wall_elapsed,
        }
        print(
            f"timing breakdown iter={iteration:02d}: "
            f"warmup_compilations_total={timing_snapshot['warmup_compilations_s']:.2f}s, "
            f"dataset_generation_total={timing_snapshot['dataset_generation_s']:.2f}s "
            f"(iter={timing_snapshot['dataset_generation_iter_s']:.2f}s), "
            f"training_total={timing_snapshot['training_s']:.2f}s "
            f"(iter={timing_snapshot['training_iter_s']:.2f}s), "
            f"wall_elapsed={timing_snapshot['wall_elapsed_s']:.2f}s",
            flush=True,
        )

        iter_payload = {
            "iteration": iteration,
            "dagger_beta": dagger_beta,
            "capacity": current_capacity,
            "dataset_valid_count": total_valid_count,
            "generated_samples": generated_samples_per_iteration,
            "kept_samples": kept_samples_per_iteration,
            "dataset_keep_fraction": args.dataset_keep_fraction,
            "collection": collect_metrics,
            "training": training_metrics,
            "evaluation": eval_metrics,
            "comparison": comparison,
            "timing": timing_snapshot,
        }
        ckpt_dir = save_flax_checkpoint(
            output_dir=args.output_dir,
            iteration=iteration,
            state=state,
            normalization=normalization,
            args=args,
            model_config={
                "hidden_sizes": hidden_sizes,
                "output_dim": output_spec.target_dim,
                "prediction_target": prediction_target_name(output_spec),
                "activation": args.activation,
                "negative_slope": args.leaky_relu_slope,
            },
            metrics=iter_payload,
        )
        iter_payload["checkpoint_dir"] = str(ckpt_dir)
        history.append(iter_payload)
        write_json(args.output_dir / "summary.json", {"history": history})
        print(f"checkpoint: {ckpt_dir}", flush=True)

    summary = {
        "calibration": calibration_metrics,
        "history": history,
        "outlier_bounds_stats": bounds_stats,
        "timing_totals": {
            **timing_totals,
            "wall_elapsed_s": time.perf_counter() - run_start,
        },
        "outlier_bounds_npz": str(args.output_dir / "outlier_bounds.npz"),
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/crazyflie_ampc"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dagger-iterations", type=int, default=10)
    parser.add_argument("--iterations", dest="dagger_iterations", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--collect-batch-size", type=int, default=100_000)
    parser.add_argument(
        "--dataset-keep-fraction",
        type=float,
        default=0.1,
        help="Randomly keep this fraction of generated transitions per rollout step for training.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=100_000)
    parser.add_argument("--filter-batch-size", type=int, default=0)
    parser.add_argument(
        "--sqp-rollout-diagnostics-only",
        action="store_true",
        help="Run only noise-free and noisy SQP rollouts, write plots/histograms, and exit.",
    )
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--integrator-substeps", type=int, default=5)
    parser.add_argument("--horizon-steps", type=int, default=25)
    parser.add_argument("--dagger-mixing", default=",".join(str(value) for value in DEFAULT_DAGGER_MIXING))
    parser.add_argument("--hidden-sizes", default="32,32,32")
    parser.add_argument(
        "--prediction-target",
        choices=("first_action", "action_sequence", "primal_solution"),
        default="first_action",
        help="Supervised target: first MPC action, open-loop MPC action sequence, or full primal vector.",
    )
    parser.add_argument(
        "--activation",
        choices=("leaky_relu", "relu", "tanh", "gelu", "elu", "silu", "swish"),
        default="leaky_relu",
    )
    parser.add_argument("--leaky-relu-slope", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=100_000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--normalization-std-floor", type=float, default=1e-3)
    parser.add_argument(
        "--policy-action-clip-std",
        type=float,
        default=8.0,
        help="Clip denormalized policy actions to mean +/- this many expert stds; <=0 disables clipping.",
    )
    parser.add_argument("--sqp-iterations", type=int, default=1)
    parser.add_argument("--osqp-max-iter", type=int, default=25)
    parser.add_argument("--max-iter", dest="osqp_max_iter", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--segment-budget", type=int, default=256)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="jax")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--line-search-step-min", type=float, default=0.1)
    parser.add_argument(
        "--no-group-repeated-stages",
        action="store_false",
        dest="group_repeated_stages",
    )
    parser.set_defaults(group_repeated_stages=True)
    parser.add_argument("--filter-low-percentile", type=float, default=10)
    parser.add_argument("--filter-high-percentile", type=float, default=90)
    parser.add_argument("--filter-margin-scale", type=float, default=2.0)
    parser.add_argument("--filter-margin-abs", type=float, default=0.25)
    parser.add_argument("--filter-min-width", type=float, default=0.1)
    parser.add_argument("--filter-cost-high-percentile", type=float, default=90.0)
    parser.add_argument("--filter-cost-margin-scale", type=float, default=2.0)
    parser.add_argument("--filter-cost-margin-abs", type=float, default=0.25)
    parser.add_argument("--filter-cost-min-width", type=float, default=0.1)
    parser.add_argument(
        "--filter-bounds-npz",
        type=pathlib.Path,
        default=None,
        help="Load hard-coded outlier bounds from an NPZ with lower, upper, and cost_threshold.",
    )
    parser.add_argument(
        "--disable-filtering",
        action="store_true",
        help="Skip noise-free filter calibration and do not apply state/cost outlier filtering.",
    )
    parser.add_argument("--valid-max-constraint-violation", type=float, default=1e2)
    parser.add_argument("--valid-max-qp-residual", type=float, default=1e4)
    parser.add_argument("--require-line-search-accepted", action="store_true")
    parser.add_argument("--converged-position-norm", type=float, default=0.1)
    parser.add_argument("--converged-state-norm", type=float, default=2.0)
    parser.add_argument(
        "--save-dataset-increments",
        action="store_true",
        help="Write each generated dataset increment as an NPZ file. Default is false.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.integrator_substeps < 1:
        raise ValueError("--integrator-substeps must be >= 1")
    if args.sqp_iterations < 1:
        raise ValueError("--sqp-iterations must be >= 1")
    if args.dagger_iterations < 1:
        raise ValueError("--dagger-iterations must be >= 1")
    if args.osqp_max_iter < 1:
        raise ValueError("--osqp-max-iter must be >= 1")
    if args.level_scheduled_solve_threshold < 1:
        raise ValueError("--level-scheduled-solve-threshold must be >= 1")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.disable_filtering and args.filter_bounds_npz is not None:
        raise ValueError("--disable-filtering and --filter-bounds-npz are mutually exclusive")
    if args.sqp_rollout_diagnostics_only:
        summary = run_sqp_rollout_diagnostics(args)
        print(f"\nWrote SQP rollout diagnostics to {summary['output_dir']}")
        return
    summary = run(args)
    print(f"\nWrote AMPC results to {summary['output_dir']}")


if __name__ == "__main__":
    main()
