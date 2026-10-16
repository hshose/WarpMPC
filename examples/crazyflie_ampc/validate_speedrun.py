#!/usr/bin/env python3
"""Reference comparisons, run separately from speedrun measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from collections import namedtuple
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from examples.crazyflie_ampc import speedrun as speed
from warpmpc.jax_ampc.normalization import initial_normalization, compute_normalization
from warpmpc.jax_ampc.training import TrainingConfig, make_training_kernels, train_supervised_policy


def compare_trees(actual, expected, *, atol=2e-7, rtol=2e-6):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    largest = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        if a.size and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            largest = max(largest, float(np.max(np.abs(a.astype(float)-b.astype(float)))))
    return largest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector", action="store_true", help="Also compare the real Warp MPC collector on an allocated H100")
    parser.add_argument("--allow-other-gpu", action="store_true", help="Diagnostic reference comparison only; no H100 timing claim")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "ampc-speedrun-validation")
    args = parser.parse_args()
    speed.configure_cache(args.cache_dir)
    # The reference's cache helper honors the directory configured above.
    from examples.crazyflie_ampc import train_ampc as reference

    cfg = speed.configuration(collect_batch_size=64, rollout_steps=4, train_epochs=3, train_batch_size=8)
    report = dict(devices=[str(d) + ' ' + d.device_kind for d in jax.devices()])
    for seed in (0, 17):
        np.testing.assert_array_equal(speed.sample_initial_states(64, seed), reference.sample_initial_states(64, seed, np.dtype('float32')))
    report['initial_sampling'] = 'exact'
    bounds = speed.load_bounds()
    assert bounds.lower.shape == (12,)

    # A non-divisible capacity, excluded samples, invalid NaNs, nontrivial
    # normalization, AdamW state and three independently shuffled epochs.
    rng = np.random.default_rng(10)
    x = rng.normal(size=(23, 12)).astype(np.float32)
    y = rng.normal(size=(23, 4)).astype(np.float32)
    valid = np.ones(23, dtype=bool)
    valid[[0, 5, 19]] = False
    x[~valid], y[~valid] = np.nan, np.inf
    dataset = speed.prepare_dataset(speed.Dataset(jnp.asarray(x), jnp.asarray(y), jnp.asarray(valid)))
    np.testing.assert_array_equal(np.asarray(dataset.x)[~valid], 0.0)
    assert np.isfinite(np.asarray(dataset.x)).all() and np.isfinite(np.asarray(dataset.y)).all()
    norm = compute_normalization(*dataset, jnp.float32(.001), jnp.float32(8.0))
    model = speed.MLP(hidden_sizes=(32,32,32), output_dim=4, activation='leaky_relu', negative_slope=.01)
    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)
    state = speed.create_train_state(model=model, key=init_key, input_dim=12, dtype=np.dtype('float32'), learning_rate=.001)
    for _ in range(3):
        key, _ = jax.random.split(key)  # calibration, collection, subsampling
    original_step, evaluate = make_training_kernels(model)
    original, metrics, original_key = train_supervised_policy(
        config=TrainingConfig(epochs=cfg.train_epochs, batch_size=cfg.train_batch_size, train_fraction=cfg.train_fraction),
        state=state, normalization=norm, train_step=original_step, eval_loss_sums=evaluate,
        dataset_x=dataset.x, dataset_y=dataset.y, dataset_mask=dataset.valid, capacity=23,
        key=key, iteration=0, print_fn=lambda *a, **kw: None,
    )
    actual, actual_key, actual_metrics = speed.train(cfg, state, norm, (original_step, evaluate), dataset, key)
    report['training_max_abs_difference'] = compare_trees(
        (actual.params, actual.opt_state, actual.step), (original.params, original.opt_state, original.step))
    np.testing.assert_array_equal(actual_key, original_key)
    assert actual_metrics['train_count'] == metrics['train_count'] and int(actual.step) == 9
    for observed, expected in zip(actual_metrics['epochs'], metrics['epochs'], strict=True):
        for name in ('train_loss', 'test_loss', 'train_target_components', 'test_target_components', 'batches'):
            np.testing.assert_allclose(observed[name], expected[name], rtol=1e-3, atol=1e-3)
    report['original_per_epoch_training_and_test_metrics'] = 'passed'
    report['training_mask_rng_padding_and_updates'] = 'passed'

    # Compare the complete collection/filtering logic on CPU as well. The fake
    # expert deliberately produces residual, constraint and finiteness failures;
    # the physical noisy dynamics, RNG, cost calculation and both collectors are
    # real. The separate H100 check below uses the actual Warp solver.
    Search = namedtuple('Search', 'constraint_violation accepted step_length')
    Solve = namedtuple('Solve', 'prim_res dual_res')
    Result = namedtuple('Result', 'z_next line_search solve is_finite')
    def fake_step(z, params, *, state):
        index = jnp.arange(z.shape[0])
        actions = jnp.broadcast_to(jnp.array([0.0,0.0,0.0,.4],dtype=z.dtype),(z.shape[0],4))
        next_z = z.at[:,12:16].set(actions)
        result = Result(next_z, Search(jnp.where(index%4==0,200.0,0.0), index%2==1, jnp.ones(z.shape[0])),
                        Solve(jnp.where(index%4==1,20000.0,0.0),jnp.zeros(z.shape[0])),index%4!=2)
        return result, state+1
    fake_sqp = SimpleNamespace(step=fake_step,init_state=lambda batch:jnp.zeros(batch))
    def reference_collector(sqp):
        return reference.make_sqp_collect_rollout(
            sqp=sqp, model=model,
            output_spec=reference.make_output_spec('first_action',nx=12,nu=4,nz=16,horizon_steps=cfg.horizon_steps),
            nx=12,rollout_steps=cfg.rollout_steps,sqp_iterations=cfg.sqp_iterations,
            dynamics_step=reference.make_crazyflie_dynamics_step(cfg.control_dt,cfg.integrator_substeps,np.dtype('float32')),
            stage_cost=lambda x,u:reference._stage_cost_jax(x,u,jnp.float32(cfg.control_dt)),
            terminal_cost=reference._terminal_cost_jax,dtype=np.dtype('float32'),
            valid_max_constraint_violation=cfg.valid_max_constraint_violation,
            valid_max_qp_residual=cfg.valid_max_qp_residual,require_line_search_accepted=cfg.require_line_search_accepted)
    fake_reference, fake_actual = reference_collector(fake_sqp), speed.make_expert_collector(fake_sqp,cfg)
    tiny_x = jnp.asarray(speed.sample_initial_states(8,1000))
    tiny_z,tiny_p = map(jnp.asarray,speed.crazyflie_initial_guess_and_params(tiny_x,n_steps=cfg.horizon_steps,dtype=np.float32))
    normal0 = initial_normalization(12,4,np.dtype('float32'))
    filter_cases = (
        speed.FilterBounds(jnp.full((12,),-jnp.inf),jnp.full((12,),jnp.inf),jnp.float32(jnp.inf)),
        speed.FilterBounds(jnp.full((12,),-jnp.inf),jnp.full((12,),jnp.inf),jnp.float32(0)),
        speed.FilterBounds(jnp.zeros(12),jnp.zeros(12),jnp.float32(jnp.inf)),
    )
    for index,case_bounds in enumerate(filter_cases):
        expected = fake_reference(state.params,normal0,case_bounds,tiny_z,tiny_p,tiny_x,key,jnp.float32(0),jnp.float32(1))
        observed = fake_actual(case_bounds,tiny_z,tiny_p,tiny_x,key)
        compare_trees((observed.x,observed.y),(expected.dataset_x,expected.expert_y))
        np.testing.assert_array_equal(observed.valid,expected.valid_mask)
        if index==0:
            assert int(jnp.sum(observed.valid))==2*cfg.rollout_steps
        else:
            assert not np.asarray(observed.valid).any()
    report['collector_dynamics_cost_bounds_residual_and_finiteness_filters']='passed'

    if args.collector:
        assert any(d.platform=='gpu' and (args.allow_other_gpu or 'H100' in d.device_kind) for d in jax.devices()), report['devices']
        sqp = speed.make_solver(cfg)
        output = reference.make_output_spec('first_action', nx=12, nu=4, nz=16, horizon_steps=cfg.horizon_steps)
        collect_reference = reference.make_sqp_collect_rollout(
            sqp=sqp, model=model, output_spec=output, nx=12, rollout_steps=cfg.rollout_steps,
            sqp_iterations=cfg.sqp_iterations,
            dynamics_step=reference.make_crazyflie_dynamics_step(cfg.control_dt, cfg.integrator_substeps, np.dtype('float32')),
            stage_cost=lambda x,u: reference._stage_cost_jax(x,u,jnp.float32(cfg.control_dt)),
            terminal_cost=reference._terminal_cost_jax, dtype=np.dtype('float32'),
            valid_max_constraint_violation=cfg.valid_max_constraint_violation,
            valid_max_qp_residual=cfg.valid_max_qp_residual,
            require_line_search_accepted=cfg.require_line_search_accepted,
        )
        x0 = speed.sample_initial_states(64, cfg.seed+1000)
        z0, p0 = speed.crazyflie_initial_guess_and_params(x0, n_steps=cfg.horizon_steps, dtype=np.float32)
        norm0 = initial_normalization(12, 4, np.dtype('float32'))
        rng_key = jax.random.PRNGKey(cfg.seed)
        for _ in range(2):
            rng_key, _ = jax.random.split(rng_key)
        _, rollout_key = jax.random.split(rng_key)
        z0, p0, x0 = map(jnp.asarray, (z0,p0,x0))
        original_rollout = collect_reference(state.params, norm0, bounds, z0,p0,x0,rollout_key,jnp.float32(0),jnp.float32(1))
        actual_collect = speed.make_expert_collector(sqp,cfg)
        abstract_args = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape,x.dtype),
                                    (bounds,z0,p0,x0,rollout_key))
        actual_rollout = actual_collect.lower(*abstract_args).compile()(bounds,z0,p0,x0,rollout_key)
        jax.block_until_ready((original_rollout,actual_rollout))
        np.testing.assert_array_equal(actual_rollout.valid, original_rollout.valid_mask)
        report['collector_max_abs_difference'] = compare_trees(
            (actual_rollout.x,actual_rollout.y), (original_rollout.dataset_x,original_rollout.expert_y),
            # Practical agreement, not a high-precision/bitwise requirement.
            atol=1e-3,rtol=1e-3)
        assert np.isfinite(np.asarray(actual_rollout.x)).all()
        assert np.isfinite(np.asarray(actual_rollout.y)).all()
        report['collector_mask'] = 'exact'
        report['collector_valid_samples'] = int(np.sum(actual_rollout.valid))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
