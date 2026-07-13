#!/usr/bin/env python3
"""Train a tiny AMPC policy on labels synthesized by the SQP minimal MPC."""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = pathlib.Path(__file__).resolve().parent
for path in (ROOT, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jax_sqp_minimal import (
    make_stage_functions,
    params_from_reference,
    sample_separated_reference,
    warm_start_from_reference,
)
from warpmpc.jax_ampc import (
    CodeExportOptions,
    MLP,
    TrainingConfig,
    append_dataset,
    apply_normalized_policy,
    compute_normalization,
    create_train_state,
    export_checkpoint,
    first_action_from_prediction,
    make_output_spec,
    make_training_kernels,
    prediction_target_name,
    save_flax_checkpoint,
    select_prediction_target,
    train_supervised_policy,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import SparseMPCProblem, build_sparse_mpc_plan, compile_sparse_mpc_sqp
from utils import plot_ampc_rollout_comparison, unpack_state_action


def sample_constant_reference_batch(
    *,
    batch_size: int,
    horizon: int,
    dtype: np.dtype,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample scalar initial states and one constant reference per problem."""

    rng = np.random.default_rng(seed)
    x_initial = rng.uniform(-1.0, 1.0, size=(batch_size, 1))
    reference = sample_separated_reference(
        rng,
        x_initial,
        lower=-1.0,
        upper=1.0,
        min_separation=0.35,
    )
    x_initial = x_initial.astype(dtype, copy=False)
    reference = reference.astype(dtype, copy=False)
    refs = np.repeat(reference, horizon + 1, axis=1).astype(dtype, copy=False)
    params = params_from_reference(refs, x_initial)
    features = np.concatenate([x_initial, reference], axis=1).astype(dtype, copy=False)
    return refs, x_initial, params, features


def solve_sqp_open_loop(
    *,
    solver,
    refs: np.ndarray,
    x_initial: np.ndarray,
    params: np.ndarray,
    sqp_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the minimal SQP solver and return the final primal and validity mask."""

    z0 = warm_start_from_reference(refs, x_initial)
    state = solver.init_state(refs.shape[0])
    z = jnp.asarray(z0)
    params_jax = jnp.asarray(params)
    result = None
    for _ in range(sqp_iterations):
        result, state = solver.step(z, params_jax, state=state)
        z = result.z_next
    assert result is not None

    primal, prim_res, dual_res = jax.device_get(
        (result.z_next, result.solve.prim_res, result.solve.dual_res)
    )
    valid = np.isfinite(primal).all(axis=1)
    valid &= np.isfinite(prim_res) & np.isfinite(dual_res)
    return primal.astype(refs.dtype, copy=False), valid


def rollout_trained_ampc(
    *,
    model: MLP,
    policy_params,
    normalization,
    output_spec,
    x_initial: np.ndarray,
    reference: np.ndarray,
    horizon: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the trained AMPC policy for one prediction horizon."""

    x = x_initial.astype(dtype, copy=True)
    states = [x[:, 0].copy()]
    actions = []
    for _ in range(horizon):
        features = np.concatenate([x, reference], axis=1).astype(dtype, copy=False)
        prediction = apply_normalized_policy(
            model.apply,
            policy_params,
            normalization,
            jnp.asarray(features),
        )
        u = np.asarray(
            jax.device_get(first_action_from_prediction(output_spec, prediction)),
            dtype=dtype,
        )
        actions.append(u[:, 0].copy())
        x = x + u + 0.1 * x**2
        states.append(x[:, 0].copy())
    return np.stack(states, axis=1), np.stack(actions, axis=1)


def synthesize_mpc_dataset(
    *,
    batch_size: int,
    horizon: int,
    dtype: np.dtype,
    sqp_iterations: int,
    qdldl_backend: str,
):
    first, stage, terminal = make_stage_functions()
    problem = SparseMPCProblem.from_stage_functions(
        horizon=horizon,
        first=first,
        intermediate=stage,
        terminal=terminal,
    )
    settings = OSQPSettings(max_iter=25, scaling=10, warm_starting=True)
    plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
    solver = compile_sparse_mpc_sqp(
        problem,
        plan,
        dtype=dtype,
        osqp_settings=settings,
        qdldl_backend=qdldl_backend,
        qdldl_factor_backend=qdldl_backend,
        qdldl_solve_backend=qdldl_backend,
        transpose_work=True,
        segmented=True,
        segment_budget=32,
        segment_strategy="optimal",
        level_scheduled_solve=True,
        level_scheduled_solve_threshold=2,
    )

    refs, x_initial, params, features = sample_constant_reference_batch(
        batch_size=batch_size,
        horizon=horizon,
        dtype=dtype,
        seed=23,
    )
    primal, valid = solve_sqp_open_loop(
        solver=solver,
        refs=refs,
        x_initial=x_initial,
        params=params,
        sqp_iterations=sqp_iterations,
    )
    return features, primal, valid, solver.osqp.qdldl.variant, solver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-target",
        choices=("first_action", "action_sequence", "primal_solution"),
        default="first_action",
    )
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sqp-iterations", type=int, default=5)
    parser.add_argument("--qdldl-backend", choices=("warp", "jax"), default="warp")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/jax_ampc_minimal"))
    args = parser.parse_args()
    if args.sqp_iterations < 1:
        parser.error("--sqp-iterations must be at least 1")

    dtype = np.dtype("float32")
    horizon = 10
    nx, nu, nz = 1, 1, 2
    spec = make_output_spec(args.prediction_target, nx=nx, nu=nu, nz=nz, horizon_steps=horizon)
    features, primal, valid_np, qdldl_variant, solver = synthesize_mpc_dataset(
        batch_size=args.samples,
        horizon=horizon,
        dtype=dtype,
        sqp_iterations=args.sqp_iterations,
        qdldl_backend=args.qdldl_backend,
    )
    y = select_prediction_target(spec, jnp.asarray(primal))
    valid = jnp.asarray(valid_np)
    feature_dim = features.shape[1]
    initial_tracking_error = features[:, 0] - features[:, 1]

    dataset_x = jnp.zeros((args.samples, feature_dim), dtype=jnp.float32)
    dataset_y = jnp.zeros((args.samples, spec.target_dim), dtype=jnp.float32)
    dataset_mask = jnp.zeros((args.samples,), dtype=jnp.bool_)
    dataset_x, dataset_y, dataset_mask = append_dataset(
        dataset_x,
        dataset_y,
        dataset_mask,
        jnp.asarray(features),
        y,
        valid,
        jnp.asarray(0, dtype=jnp.int32),
    )

    key = jax.random.PRNGKey(31)
    key, init_key, train_key = jax.random.split(key, 3)
    model = MLP(hidden_sizes=(32, 32), output_dim=spec.target_dim, activation="leaky_relu")
    state = create_train_state(
        model=model,
        key=init_key,
        input_dim=feature_dim,
        dtype=dtype,
        learning_rate=1e-3,
    )
    normalization = compute_normalization(
        dataset_x,
        dataset_y,
        dataset_mask,
        jnp.asarray(1e-3, dtype=jnp.float32),
        jnp.asarray(8.0, dtype=jnp.float32),
    )
    train_step, eval_loss_sums = make_training_kernels(model)
    state, metrics, _ = train_supervised_policy(
        config=TrainingConfig(epochs=args.epochs, batch_size=512, train_fraction=0.8),
        state=state,
        normalization=normalization,
        train_step=train_step,
        eval_loss_sums=eval_loss_sums,
        dataset_x=dataset_x,
        dataset_y=dataset_y,
        dataset_mask=dataset_mask,
        capacity=args.samples,
        key=train_key,
        iteration=0,
    )

    checkpoint_dir = save_flax_checkpoint(
        output_dir=args.output_dir,
        iteration=0,
        state=state,
        normalization=normalization,
        args=SimpleNamespace(
            prediction_target=prediction_target_name(spec),
            samples=args.samples,
            horizon_steps=horizon,
            feature_dim=feature_dim,
            sqp_iterations=args.sqp_iterations,
            qdldl_backend=args.qdldl_backend,
            qdldl_variant=qdldl_variant,
        ),
        model_config={
            "hidden_sizes": (32, 32),
            "activation": "leaky_relu",
            "negative_slope": 0.01,
        },
        metrics=metrics,
    )

    # Other available forward backends are "none", "cmsis", and "eigen".
    # The plain C backend below writes *_data.c and *_forward.c files.
    export_options = CodeExportOptions(
        prefix="jax_ampc_minimal_policy",
        backend="simple",
        precision="float32_t",
        generate_example_main=True,
    )
    written = export_checkpoint(
        checkpoint_dir,
        args.output_dir / "c_export",
        test_inputs=np.asarray(dataset_x[:16]),
        test_count=16,
        options=export_options,
        name="jax_ampc_minimal_policy",
    )

    rollout_refs, rollout_x_initial, rollout_params, rollout_features = (
        sample_constant_reference_batch(
            batch_size=6,
            horizon=horizon,
            dtype=dtype,
            seed=47,
        )
    )
    rollout_primal, rollout_valid = solve_sqp_open_loop(
        solver=solver,
        refs=rollout_refs,
        x_initial=rollout_x_initial,
        params=rollout_params,
        sqp_iterations=args.sqp_iterations,
    )
    mpc_states, _ = unpack_state_action(rollout_primal, horizon)
    ampc_states, _ = rollout_trained_ampc(
        model=model,
        policy_params=state.params,
        normalization=normalization,
        output_spec=spec,
        x_initial=rollout_x_initial,
        reference=rollout_features[:, 1:2],
        horizon=horizon,
        dtype=dtype,
    )
    rollout_plot = plot_ampc_rollout_comparison(
        args.output_dir,
        ampc_states,
        mpc_states,
        rollout_features[:, 1],
    )

    first_u = first_action_from_prediction(spec, y)
    final_epoch = metrics["epochs"][-1]
    print(
        "jax_ampc_minimal:",
        f"target={prediction_target_name(spec)}",
        f"feature_dim={feature_dim}",
        f"target_dim={spec.target_dim}",
        f"valid={int(np.sum(valid_np))}/{args.samples}",
        f"qdldl_variant={qdldl_variant}",
        f"initial_error_rms={float(np.sqrt(np.mean(initial_tracking_error**2))):.5g}",
        f"first_action_mean={float(jnp.mean(first_u)):.5g}",
        f"rollout_mpc_valid={int(np.sum(rollout_valid))}/{rollout_valid.shape[0]}",
        f"train_mse={final_epoch['train_loss']:.5e}",
        f"test_mse={final_epoch['test_loss']:.5e}",
    )
    print(f"checkpoint: {checkpoint_dir}")
    print(f"rollout plot: {rollout_plot}")
    print("C export:")
    for key_name, path in sorted(written.items()):
        print(f"  {key_name}: {path}")


if __name__ == "__main__":
    main()
