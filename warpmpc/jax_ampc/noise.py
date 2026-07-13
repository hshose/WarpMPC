"""Reusable JAX noise samplers for AMPC rollouts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def sample_unit_ball_np(
    rng: np.random.Generator,
    batch_size: int,
    dim: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Sample uniformly from the unit ball in R^dim using NumPy."""

    z = rng.standard_normal((batch_size, dim)).astype(dtype)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norm, np.asarray(1e-12, dtype=dtype))
    r = rng.random((batch_size, 1), dtype=dtype) ** (1.0 / dim)
    return r * z


def sample_scaled_unit_ball_noise_np(
    rng: np.random.Generator,
    batch_size: int,
    scale,
    dtype: np.dtype,
) -> np.ndarray:
    """Sample NumPy unit-ball noise with a per-coordinate scaling vector."""

    scale = np.asarray(scale, dtype=dtype)
    return sample_unit_ball_np(rng, batch_size, scale.shape[0], dtype) * scale[None, :]


def sample_unit_ball(key, batch_size: int, dim: int, dtype):
    """Sample uniformly from the unit ball in R^dim."""

    key_z, key_r = jax.random.split(key)
    z = jax.random.normal(key_z, (batch_size, dim), dtype=dtype)
    norm = jnp.linalg.norm(z, axis=1, keepdims=True)
    z = z / jnp.maximum(norm, jnp.asarray(1e-12, dtype=dtype))
    r = jax.random.uniform(key_r, (batch_size, 1), dtype=dtype) ** (1.0 / dim)
    return r * z


def sample_scaled_unit_ball_noise(key, batch_size: int, scale, dtype=None):
    """Sample additive noise from a unit ball with a per-coordinate scale."""

    scale = jnp.asarray(scale, dtype=dtype)
    dtype = scale.dtype if dtype is None else jnp.dtype(dtype)
    return sample_unit_ball(key, batch_size, scale.shape[0], dtype) * scale[None, :]


def sample_scaled_gaussian_noise(key, batch_size: int, scale, dtype=None):
    """Sample additive independent Gaussian noise with per-coordinate scale."""

    scale = jnp.asarray(scale, dtype=dtype)
    dtype = scale.dtype if dtype is None else jnp.dtype(dtype)
    return jax.random.normal(key, (batch_size, scale.shape[0]), dtype=dtype) * scale[None, :]
