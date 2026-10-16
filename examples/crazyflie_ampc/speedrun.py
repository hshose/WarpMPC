#!/usr/bin/env python3
"""Paper Crazyflie AMPC: fixed calibration, fresh data, original training loop.

The original example remains the reference for sampling, solver settings, RNG
consumption, masking, normalization and optimizer updates. No model/data files,
plots, evaluation rollouts or controller exports are produced by this example.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ASSETS = Path(__file__).resolve().parent / "speedrun_assets"

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.problems.crazyflie_sqp import (
    CRAZYFLIE_NX, CRAZYFLIE_NU, CRAZYFLIE_NZ, CRAZYFLIE_Q, CRAZYFLIE_R,
    crazyflie_initial_guess_and_params, crazyflie_jax_euler_step,
    make_crazyflie_sqp_problem,
)
from warpmpc.jax_ampc.filters import FilterBounds, load_filter_bounds_npz, valid_state_jax
from warpmpc.jax_ampc.models import MLP
from warpmpc.jax_ampc.noise import sample_scaled_unit_ball_noise, sample_scaled_unit_ball_noise_np
from warpmpc.jax_ampc.normalization import compute_normalization
from warpmpc.jax_ampc.training import (
    TrainingConfig, create_train_state, make_training_kernels, train_supervised_policy,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import FilterLineSearchSettings, build_sparse_mpc_plan, compile_sparse_mpc_sqp


class Dataset(NamedTuple):
    x: object
    y: object
    valid: object


def configuration(**overrides):
    """Load the saved paper arguments; only workload/seed overrides are allowed."""
    saved = (ASSETS / "paper_config.json").read_bytes()
    manifest = json.loads((ASSETS / "provenance.json").read_text())
    if hashlib.sha256(saved).hexdigest() != manifest["paper_config_sha256"]:
        raise ValueError("Paper configuration does not match its recorded checksum")
    values = json.loads(saved)
    allowed = {"seed", "collect_batch_size", "rollout_steps", "train_epochs", "train_batch_size"}
    if overrides.keys() - allowed:
        raise ValueError(f"Unsupported overrides: {overrides.keys() - allowed}")
    values.update({k: v for k, v in overrides.items() if v is not None})
    for name in allowed - {"seed"}:
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    return SimpleNamespace(**values)


def load_bounds():
    path = ASSETS / "outlier_bounds.npz"
    manifest = json.loads((ASSETS / "provenance.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["bounds_sha256"]:
        raise ValueError("Paper calibration bounds do not match their recorded checksum")
    lower, upper, threshold = load_filter_bounds_npz(path, dtype=np.float32)
    if lower.shape != (CRAZYFLIE_NX,) or upper.shape != lower.shape:
        raise ValueError("Unexpected calibration dimensions")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            and np.all(lower <= upper) and np.isfinite(threshold) and threshold > 0):
        raise ValueError("Invalid calibration bounds")
    return FilterBounds(jnp.asarray(lower), jnp.asarray(upper), jnp.asarray(threshold, dtype=jnp.float32))


def noise_scales():
    # Identical to crazyflie_ampc/train_ampc.py, including the angular scaling.
    process = np.full((CRAZYFLIE_NX,), 0.05, dtype=np.float64)
    process[3:6] /= 5.0
    process[9:11] *= 100.0
    return process, np.full((CRAZYFLIE_NU,), 0.1, dtype=np.float64)


def sample_initial_states(batch_size, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    process, _ = noise_scales()
    x = sample_scaled_unit_ball_noise_np(rng, batch_size, 10.0 * process, dtype=np.dtype(dtype))
    x[:, :3] = rng.uniform(-1.0, 1.0, size=(batch_size, 3)).astype(dtype)
    x[:, 5] = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=(batch_size,)).astype(dtype)
    return x


def make_solver(config):
    settings = OSQPSettings(
        rho=config.rho, sigma=config.sigma, alpha=config.alpha,
        max_iter=config.osqp_max_iter, scaling=0, adaptive_rho=False,
        rho_is_vec=False, check_termination=0, warm_starting=True, polishing=False,
    )
    problem = make_crazyflie_sqp_problem(config.horizon_steps)
    plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
    return compile_sparse_mpc_sqp(
        problem, plan, dtype=np.dtype(config.dtype), osqp_settings=settings,
        transpose_work=True, segmented=True, segment_budget=config.segment_budget,
        segment_strategy=config.segment_strategy,
        level_scheduled_solve=config.level_scheduled_solve,
        level_scheduled_solve_threshold=config.level_scheduled_solve_threshold,
        qdldl_backend=config.qdldl_backend,
        qdldl_factor_backend=config.qdldl_factor_backend,
        qdldl_solve_backend=config.qdldl_solve_backend,
        line_search_settings=FilterLineSearchSettings(line_search_step_min=config.line_search_step_min),
        group_repeated_stages=config.group_repeated_stages,
    )


def make_expert_collector(sqp, config):
    """Specialize beta=0, noise=1 and first_action; retain the original filters."""
    process, inputs = noise_scales()
    process, inputs = jnp.asarray(process, dtype=jnp.float32), jnp.asarray(inputs, dtype=jnp.float32)
    dt = jnp.asarray(config.control_dt, dtype=jnp.float32)
    step_dt = dt / jnp.asarray(config.integrator_substeps, dtype=jnp.float32)
    q = jnp.asarray(np.diag(CRAZYFLIE_Q), dtype=jnp.float32)
    r = jnp.asarray(np.diag(CRAZYFLIE_R), dtype=jnp.float32)

    @jax.jit
    def collect(bounds, z, params, x, key):
        def step(carry, _):
            z, params, x, key, solver_state = carry
            params = params.at[:, :CRAZYFLIE_NX].set(x)

            def sqp_iteration(inner, _):
                z_iter, state_iter = inner
                result, next_state = sqp.step(z_iter, params, state=state_iter)
                return (result.z_next, next_state), result

            (z_next, next_solver_state), results = jax.lax.scan(
                sqp_iteration, (z, solver_state), xs=None, length=config.sqp_iterations,
            )
            result = jax.tree_util.tree_map(lambda leaf: leaf[-1], results)
            u = result.z_next[:, CRAZYFLIE_NX:CRAZYFLIE_NZ]
            key, noise_key = jax.random.split(key)
            key_q, key_d = jax.random.split(noise_key)
            state_noise = sample_scaled_unit_ball_noise(key_q, x.shape[0], process, dtype=jnp.float32)
            input_noise = sample_scaled_unit_ball_noise(key_d, x.shape[0], inputs, dtype=jnp.float32)
            x_next = jax.lax.fori_loop(
                0, config.integrator_substeps,
                lambda _, cur: crazyflie_jax_euler_step(cur, u + input_noise, step_dt), x,
            ) + state_noise
            valid = valid_state_jax(x, bounds) & jnp.all(jnp.isfinite(u), axis=1)
            valid &= jnp.isfinite(result.line_search.constraint_violation)
            valid &= result.line_search.constraint_violation <= config.valid_max_constraint_violation
            valid &= jnp.isfinite(result.solve.prim_res) & jnp.isfinite(result.solve.dual_res)
            valid &= result.solve.prim_res <= config.valid_max_qp_residual
            valid &= result.solve.dual_res <= config.valid_max_qp_residual
            valid &= result.is_finite
            if config.require_line_search_accepted:
                valid &= result.line_search.accepted
            cost = 0.5 * dt * (jnp.sum(q[None, :] * x * x, axis=1)
                               + jnp.sum(r[None, :] * u * u, axis=1))
            return (z_next, params.at[:, :CRAZYFLIE_NX].set(x_next), x_next, key, next_solver_state), (x, u, valid, cost)

        final, (x_steps, y_steps, valid, costs) = jax.lax.scan(
            step, (z, params, x, key, sqp.init_state(z.shape[0])), xs=None, length=config.rollout_steps,
        )
        # Preserve the reference reduction order instead of accumulating in scan.
        total_cost = jnp.sum(costs, axis=0) + 0.5 * jnp.sum(q[None, :] * final[2] * final[2], axis=1)
        valid &= (jnp.isfinite(total_cost) & (total_cost <= bounds.cost_threshold))[None, :]
        return Dataset(x_steps, y_steps, valid)

    return collect


@jax.jit
def prepare_dataset(rollout):
    valid = rollout.valid.reshape((-1,))
    x, y = rollout.x.reshape((-1, CRAZYFLIE_NX)), rollout.y.reshape((-1, CRAZYFLIE_NU))
    # Keep invalid rows in their original positions: multiplication by a zero
    # mask alone would not protect the loss/gradient from NaN inputs.
    return Dataset(jnp.where(valid[:, None], x, 0.0), jnp.where(valid[:, None], y, 0.0), valid)


def train(config, state, normalization, kernels, dataset, key, epoch_callback=None):
    """Reuse the paper's complete timed routine, including its loss evaluations."""
    epoch_start = None
    epoch = 0

    def print_epoch(message, **kwargs):
        nonlocal epoch_start, epoch
        end = time.perf_counter_ns()
        if message.startswith("  epoch ") and epoch_start is not None:
            # These intervals bracket stdout calls and include epoch bookkeeping.
            # The unchanged routine also returns its own native elapsed_s values.
            if epoch_callback:
                epoch_callback(epoch, epoch_start, end)
            epoch += 1
        print(message, **kwargs)
        epoch_start = time.perf_counter_ns()

    state, metrics, key = train_supervised_policy(
        config=TrainingConfig(epochs=config.train_epochs, batch_size=config.train_batch_size,
                              train_fraction=config.train_fraction),
        state=state, normalization=normalization,
        train_step=kernels[0], eval_loss_sums=kernels[1],
        dataset_x=dataset.x, dataset_y=dataset.y, dataset_mask=dataset.valid,
        capacity=dataset.valid.shape[0], key=key, iteration=0, print_fn=print_epoch,
    )
    if metrics['train_count'] == 0:
        raise ValueError("No valid training samples")
    return state, key, metrics


def run(config=None, *, stage_callback=None, epoch_callback=None, paper_callback=None):
    """Return the trained policy in memory; callback records contain timings only."""
    config = configuration() if config is None else config
    if (config.dtype != "float32" or config.prediction_target != "first_action"
            or config.dagger_iterations != 1 or float(config.dagger_mixing) != 0.0
            or config.dataset_keep_fraction != 1.0 or config.disable_filtering):
        raise ValueError("Speedrun requires float32, one expert-only iteration, first-action targets and filtering with keep_fraction=1")
    jax.config.update("jax_enable_x64", False)
    timings = {}
    paper_intervals = {}

    @contextlib.contextmanager
    def paper_timer(name):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()
        paper_intervals[name] = dict(start_ns=start, end_ns=end, wall_s=(end-start)/1e9)
        if paper_callback:
            paper_callback(name, start, end)

    @contextlib.contextmanager
    def stage(name):
        start = time.perf_counter_ns()
        if stage_callback:
            stage_callback("begin", name, start, None)
        try:
            yield
        finally:
            jax.block_until_ready(jax.live_arrays())
            jax.effects_barrier()
            end = time.perf_counter_ns()
            timings[name] = (end - start) / 1e9
            if stage_callback:
                stage_callback("end", name, start, end)
            print(f"{name}: {timings[name]:.6f} s (includes any first-call compilation)", flush=True)

    run_start = time.perf_counter_ns()
    with stage("setup"):
        bounds = load_bounds()
        model = MLP(hidden_sizes=tuple(map(int, config.hidden_sizes.split(','))), output_dim=CRAZYFLIE_NU,
                    activation=config.activation, negative_slope=config.leaky_relu_slope)
        key = jax.random.PRNGKey(config.seed)
        key, init_key = jax.random.split(key)
        state = create_train_state(model=model, key=init_key, input_dim=CRAZYFLIE_NX,
                                   dtype=np.dtype(config.dtype), learning_rate=config.learning_rate)
        # Consume the skipped calibration split exactly as the paper run did.
        key, _ = jax.random.split(key)
        collector = make_expert_collector(make_solver(config), config)
        kernels = make_training_kernels(model)
        # The paper's calibration had already compiled the collector before its
        # dataset timer. Restore that placement without executing calibration.
        probe_z, probe_p = crazyflie_initial_guess_and_params(
            np.zeros((1, CRAZYFLIE_NX), dtype=np.float32),
            n_steps=config.horizon_steps, dtype=np.float32,
        )
        abstract = lambda shape: jax.ShapeDtypeStruct(shape, np.float32)
        collector_args = (bounds,
                          abstract((config.collect_batch_size, probe_z.shape[1])),
                          abstract((config.collect_batch_size, probe_p.shape[1])),
                          abstract((config.collect_batch_size, CRAZYFLIE_NX)),
                          jax.ShapeDtypeStruct(key.shape, key.dtype))

    with stage("collector_lowering"):
        lowered = collector.lower(*collector_args)
    with stage("collector_compile_load"):
        compiled_collector = lowered.compile()

    with stage("dataset_synthesis"):
        with paper_timer("dataset_generation"):
            x0 = sample_initial_states(config.collect_batch_size, config.seed + 1000)
            z0, params0 = crazyflie_initial_guess_and_params(x0, n_steps=config.horizon_steps, dtype=np.float32)
            key, rollout_key = jax.random.split(key)
            rollout = compiled_collector(bounds, jnp.asarray(z0), jnp.asarray(params0), jnp.asarray(x0), rollout_key)
            jax.block_until_ready(rollout)
            # The paper dataset timer includes sanitization and validity counts;
            # normalization is outside it. Keep the no-copy, one-iteration dataset.
            key, _ = jax.random.split(key)
            dataset = prepare_dataset(rollout)
            valid_count = int(jax.device_get(jnp.sum(dataset.valid)))
    with stage("normalization"):
        del rollout, x0, z0, params0
        normalization = compute_normalization(
            dataset.x, dataset.y, dataset.valid,
            jnp.asarray(config.normalization_std_floor, dtype=jnp.float32),
            jnp.asarray(config.policy_action_clip_std, dtype=jnp.float32),
        )
        print(f"valid samples: {valid_count:,}/{dataset.valid.size:,}", flush=True)
    with stage("training"):
        with paper_timer("training"):
            state, key, training_metrics = train(config, state, normalization, kernels, dataset, key, epoch_callback)
    run_end = time.perf_counter_ns()
    print(f"paper timing boundaries: dataset_generation_total={paper_intervals['dataset_generation']['wall_s']:.6f}s, "
          f"training_total={paper_intervals['training']['wall_s']:.6f}s", flush=True)
    return dict(params=state.params, normalization=normalization, model=model,
                config=vars(config), timings=timings, valid_count=valid_count, train_count=training_metrics['train_count'],
                training_metrics=training_metrics, paper_intervals=paper_intervals,
                training_workload='original_train_supervised_policy_with_epoch_losses',
                optimizer_steps=int(jax.device_get(state.step)),
                start_ns=run_start, end_ns=run_end, wall_s=(run_end-run_start)/1e9)


def configure_cache(path):
    import warp as wp
    from benchmarks.jax_cache import configure_jax_compilation_cache
    root = Path(path).expanduser().resolve()
    for subdir in ("jax", "warp", "cuda", "xdg"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_CACHE_PATH=str(root / "cuda"), XDG_CACHE_HOME=str(root / "xdg"),
                      JAX_COMPILATION_CACHE_DIR=str(root / "jax"), WARP_CACHE_PATH=str(root / "warp"))
    configure_jax_compilation_cache(root / "jax")
    wp.config.kernel_cache_dir = str(root / "warp")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("AMPC_CACHE_DIR", "~/.cache/warpmpc/ampc_speedrun")))
    for name in ("seed", "collect-batch-size", "rollout-steps", "train-epochs", "train-batch-size"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args()
    configure_cache(args.cache_dir)
    config = configuration(**{k: v for k, v in vars(args).items() if k != "cache_dir"})
    result = run(config)
    print(json.dumps({k: v for k, v in result.items() if k not in ("params", "normalization", "model", "config")}, indent=2))


if __name__ == "__main__":
    main()
