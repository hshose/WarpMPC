"""Sequential pose warm starts inside independent, equally sized chunks."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np


def chunk_indices(frame_count, clip_starts, chunk_size):
    """Drop incomplete tails; never join frames from different motion clips."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    starts = sorted(set(int(x) for x in clip_starts))
    if not starts or starts[0] != 0 or starts[-1] >= frame_count:
        raise ValueError("clip_starts must begin at zero and refer to input frames")
    chunks = []
    for begin, end in zip(starts, starts[1:] + [frame_count]):
        for first in range(begin, end - chunk_size + 1, chunk_size):
            chunks.append(np.arange(first, first + chunk_size))
    if not chunks:
        raise ValueError(
            f"No complete {chunk_size}-frame chunk; prepare longer clips or decrease chunk size"
        )
    return np.stack(chunks)


def rollout_chunks(
    cold,
    warm,
    preprocess,
    initial,
    bank,
    lanes,
    *,
    collect=False,
    qindices=None,
    lower=None,
    upper=None,
):
    """Each lane reads a compact chunk bank and advances its own robot state.

    The timing path retains only final poses and online validity diagnostics,
    so memory does not grow with the product batch_size * chunk_size. All
    frames are actually solved; this is not extrapolated one-frame timing.
    """
    q = jnp.broadcast_to(initial, (lanes.shape[0], initial.shape[0]))
    q = cold(q, preprocess(bank[lanes, 0]))

    def diagnostics(q):
        finite = jnp.all(jnp.isfinite(q))
        if qindices is None:
            violation = jnp.asarray(0.0, dtype=q.dtype)
        else:
            joints = q[:, qindices]
            violation = jnp.maximum(
                0.0, jnp.maximum(jnp.max(lower - joints), jnp.max(joints - upper))
            )
        return finite, violation

    finite, violation = diagnostics(q)

    def step(carry, t):
        previous, valid, worst = carry
        current = warm(previous, preprocess(bank[lanes, t]))
        ok, bound = diagnostics(current)
        return (current, valid & ok, jnp.maximum(worst, bound)), current if collect else None

    (final, finite, violation), rest = jax.lax.scan(
        step, (q, finite, violation), jnp.arange(1, bank.shape[1])
    )
    if collect:
        trajectory = jnp.concatenate((q[None], rest), axis=0).transpose(1, 0, 2)
        return trajectory, finite, violation
    return final, finite, violation


def compile_chunks(model, settings, initial_sqp_iterations=10, *, collect=False):
    if initial_sqp_iterations < 1:
        raise ValueError("initial_sqp_iterations must be positive")
    cold, _ = model.compile(replace(settings, sqp_iterations=initial_sqp_iterations))
    warm, _ = model.compile(settings)
    initial = jnp.asarray(model.mj.qpos0, dtype=jnp.float32)
    qindices = jnp.asarray(model.qindices)
    lower, upper = jnp.asarray(model.lower, dtype=jnp.float32), jnp.asarray(
        model.upper, dtype=jnp.float32
    )

    def run(bank, lanes):
        return rollout_chunks(
            cold,
            warm,
            model.preprocess,
            initial,
            bank,
            lanes,
            collect=collect,
            qindices=qindices,
            lower=lower,
            upper=upper,
        )

    return jax.jit(run)
