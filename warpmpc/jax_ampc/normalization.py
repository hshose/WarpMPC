"""Input/target normalization helpers for AMPC imitation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Normalization(NamedTuple):
    x_mean: object
    x_std: object
    y_mean: object
    y_std: object
    y_clip_low: object
    y_clip_high: object

    @property
    def u_mean(self):
        return self.y_mean

    @property
    def u_std(self):
        return self.y_std

    @property
    def u_clip_low(self):
        return self.y_clip_low

    @property
    def u_clip_high(self):
        return self.y_clip_high


@jax.jit
def compute_normalization(dataset_x, dataset_y, dataset_mask, std_floor, target_clip_std):
    dtype = dataset_x.dtype
    weights = dataset_mask.astype(dtype)
    count = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=dtype))
    x_mean = jnp.sum(jnp.where(dataset_mask[:, None], dataset_x, 0.0), axis=0) / count
    y_mean = jnp.sum(jnp.where(dataset_mask[:, None], dataset_y, 0.0), axis=0) / count
    x_var = jnp.sum(
        jnp.where(dataset_mask[:, None], (dataset_x - x_mean[None, :]) ** 2, 0.0),
        axis=0,
    ) / count
    y_var = jnp.sum(
        jnp.where(dataset_mask[:, None], (dataset_y - y_mean[None, :]) ** 2, 0.0),
        axis=0,
    ) / count
    floor = jnp.asarray(std_floor, dtype=dtype)
    x_std = jnp.maximum(jnp.sqrt(x_var), floor)
    y_std = jnp.maximum(jnp.sqrt(y_var), floor)
    clip_std = jnp.asarray(target_clip_std, dtype=dtype)
    finite_clip_low = y_mean - clip_std * y_std
    finite_clip_high = y_mean + clip_std * y_std
    y_clip_low = jnp.where(clip_std > 0.0, finite_clip_low, -jnp.inf * jnp.ones_like(y_mean))
    y_clip_high = jnp.where(clip_std > 0.0, finite_clip_high, jnp.inf * jnp.ones_like(y_mean))
    return Normalization(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        y_clip_low=y_clip_low,
        y_clip_high=y_clip_high,
    )


def initial_normalization(x_dim: int, y_dim: int, dtype: np.dtype) -> Normalization:
    jdtype = jnp.dtype(dtype)
    return Normalization(
        x_mean=jnp.zeros((x_dim,), dtype=jdtype),
        x_std=jnp.ones((x_dim,), dtype=jdtype),
        y_mean=jnp.zeros((y_dim,), dtype=jdtype),
        y_std=jnp.ones((y_dim,), dtype=jdtype),
        y_clip_low=jnp.full((y_dim,), -jnp.inf, dtype=jdtype),
        y_clip_high=jnp.full((y_dim,), jnp.inf, dtype=jdtype),
    )


def normalize_inputs(x, normalization: Normalization):
    return (x - normalization.x_mean[None, :]) / normalization.x_std[None, :]


def normalize_targets(y, normalization: Normalization):
    return (y - normalization.y_mean[None, :]) / normalization.y_std[None, :]


def denormalize_targets(y_norm, normalization: Normalization, *, clip: bool = True):
    y = y_norm * normalization.y_std[None, :] + normalization.y_mean[None, :]
    if clip:
        y = jnp.clip(y, normalization.y_clip_low[None, :], normalization.y_clip_high[None, :])
    return y


def apply_normalized_policy(apply_fn, params, normalization: Normalization, x, *, clip: bool = True):
    x_norm = normalize_inputs(x, normalization)
    y_norm = apply_fn({"params": params}, x_norm)
    return denormalize_targets(y_norm, normalization, clip=clip)
