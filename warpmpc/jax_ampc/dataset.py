"""JAX dataset buffer utilities for AMPC imitation."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, donate_argnums=(0, 1, 2))
def append_dataset(dataset_x, dataset_y, dataset_mask, x_inc, y_inc, mask_inc, start):
    """Append one dense increment into preallocated dataset arrays."""

    start = jnp.asarray(start, dtype=jnp.int32)
    col0 = jnp.asarray(0, dtype=start.dtype)
    dataset_x = jax.lax.dynamic_update_slice(dataset_x, x_inc, (start, col0))
    dataset_y = jax.lax.dynamic_update_slice(dataset_y, y_inc, (start, col0))
    dataset_mask = jax.lax.dynamic_update_slice(dataset_mask, mask_inc, (start,))
    return dataset_x, dataset_y, dataset_mask


@functools.partial(jax.jit, static_argnames=("keep_per_step",))
def subsample_dataset_increment(x_steps, y_steps, valid_steps, key, *, keep_per_step: int):
    """Randomly keep a fixed number of batch elements at each rollout step."""

    scores = jax.random.uniform(key, valid_steps.shape, dtype=jnp.float32)
    _, batch_indices = jax.lax.top_k(scores, keep_per_step)
    x_inc = jnp.take_along_axis(x_steps, batch_indices[:, :, None], axis=1)
    y_inc = jnp.take_along_axis(y_steps, batch_indices[:, :, None], axis=1)
    valid_inc = jnp.take_along_axis(valid_steps, batch_indices, axis=1)
    time_indices = jnp.arange(valid_steps.shape[0], dtype=jnp.int32)[:, None]
    source_indices = time_indices * jnp.asarray(valid_steps.shape[1], dtype=jnp.int32) + batch_indices
    return (
        x_inc.reshape((-1, x_steps.shape[-1])),
        y_inc.reshape((-1, y_steps.shape[-1])),
        valid_inc.reshape((-1,)),
        source_indices.reshape((-1,)),
    )


def dataset_keep_per_step(batch_size: int, keep_fraction: float) -> int:
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1]")
    return max(1, min(batch_size, int(math.ceil(batch_size * keep_fraction))))


def estimate_dataset_gb(max_samples: int, x_dim: int, y_dim: int, dtype: np.dtype) -> float:
    itemsize = np.dtype(dtype).itemsize
    bytes_total = max_samples * (x_dim + y_dim) * itemsize
    bytes_total += max_samples * np.dtype(np.bool_).itemsize
    return bytes_total / 1e9
