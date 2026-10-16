"""Check the sampling adapters against the publication layout and time grid."""
from dataclasses import replace

import jax
import numpy as np
import pytest

from benchmarks.nonlinear_mpc.evaluate_sampling_mpc import _run_rollout, source_hash
from benchmarks.nonlinear_mpc.sampling_mpc import default_settings, make_model


@pytest.fixture(autouse=True)
def float32_benchmark_context():
    # Other solver tests enable float64 globally; sampling CLIs use float32.
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_cartpole_publication_grid():
    model = make_model('cartpole')
    assert model.horizon == 100
    assert model.dt == 0.1
    assert model.steps * model.dt == 2.0
    np.testing.assert_allclose(model.schedule, 0.1)


@pytest.mark.parametrize('system', ['cartpole', 'crazyflie'])
@pytest.mark.parametrize('method', ['mppi', 'cem'])
def test_sampling_adapter_smoke(system, method):
    settings = replace(default_settings(system, method), samples=4,
                       iterations=1, initial_iterations=1)
    result = _run_rollout(system, settings, batch_size=1, seed=101, chunk_size=1,
                 repeats=1, sim_steps=1)
    assert result['finite_rate'] == 1.0
    assert result['total_closed_loop_steps'] == 1
    assert len(source_hash()) == 64
