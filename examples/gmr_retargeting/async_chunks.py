"""Independent GMR stage/frame progress per lane, with one QP per round.

Lanes share batched GPU kernels, but never wait at a stage or frame boundary.
Only lanes that have finished their entire chunk become idle. The compact
input bank is gathered using each lane's own frame index. Timing retains
O(batch) solver state; collecting O(batch * frames) poses is validation-only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rollout_async(
    advance,
    preprocess,
    initial,
    bank,
    lanes,
    *,
    initial_cap=11,
    warm_cap=11,
    collect=False,
    qindices=None,
    lower=None,
    upper=None,
):
    """advance(q, target, stage) returns q_next, error_before, error_after.

    Each live lane accepts one SQP step, then independently decides whether
    to continue its stage, enter stage 2, or finish the frame. ADMM is cold
    started for each QP, exactly as in the frame-synchronous implementation.
    """
    if min(initial_cap, warm_cap) < 1:
        raise ValueError("SQP caps must be positive")
    batch, frames = len(lanes), bank.shape[1]
    q = jnp.broadcast_to(initial, (batch, len(initial)))
    zeros = jnp.zeros(batch, dtype=jnp.int32)
    trajectory = jnp.zeros((batch, frames, len(initial)), q.dtype) if collect else None
    completion_rounds = jnp.zeros((batch, frames), dtype=jnp.int32) if collect else None
    # q, frame, stage, iteration within stage, accepted QPs, round,
    # all-frame finite flag, maximum final-pose limit violation, optional poses.
    carry = (
        q,
        zeros,
        zeros,
        zeros,
        zeros,
        jnp.int32(0),
        jnp.bool_(True),
        jnp.float32(0),
        trajectory,
        completion_rounds,
    )
    lane_index = jnp.arange(batch)

    def body(carry):
        (
            q,
            frame,
            stage,
            iteration,
            work,
            rounds,
            finite,
            violation,
            trajectory,
            completion_rounds,
        ) = carry
        live = frame < frames
        safe_frame = jnp.minimum(frame, frames - 1)
        targets = preprocess(bank[lanes, safe_frame])
        target = targets[lane_index, stage]
        candidate, before, after = advance(q, target, stage)
        q = jnp.where(live[:, None], candidate, q)
        iteration = iteration + live.astype(jnp.int32)
        cap = jnp.where(frame == 0, initial_cap, warm_cap)
        end_stage = live & ((before - after <= 0.001) | (iteration >= cap))
        end_frame = end_stage & (stage == 1)
        # Mark nonfinite intermediate solves, even if a later stage recovers.
        valid = (
            jnp.all(jnp.isfinite(candidate), axis=1) & jnp.isfinite(before) & jnp.isfinite(after)
        )
        finite = finite & jnp.all(~live | valid)
        if qindices is not None:
            bound = jnp.max(jnp.maximum(lower - q[:, qindices], q[:, qindices] - upper), axis=1)
            violation = jnp.maximum(violation, jnp.max(jnp.where(end_frame, bound, 0.0)))
        if collect:
            # Dropped out-of-range writes leave unfinished-frame slots untouched.
            write_frame = jnp.where(end_frame, safe_frame, frames)
            trajectory = trajectory.at[lane_index, write_frame].set(q, mode="drop")
            completion_rounds = completion_rounds.at[lane_index, write_frame].set(
                rounds + 1, mode="drop"
            )
        return (
            q,
            frame + end_frame.astype(jnp.int32),
            jnp.where(end_stage, 1 - stage, stage),
            jnp.where(end_stage, 0, iteration),
            work + live.astype(jnp.int32),
            rounds + 1,
            finite,
            violation,
            trajectory,
            completion_rounds,
        )

    (
        final,
        frame,
        stage,
        iteration,
        work,
        rounds,
        finite,
        violation,
        trajectory,
        completion_rounds,
    ) = jax.lax.while_loop(lambda c: jnp.any(c[1] < frames), body, carry)
    # int32 sums can overflow at 300k long trajectories; use float32 for totals.
    useful = jnp.sum(work.astype(jnp.float32))
    utilization = useful / (jnp.float32(batch) * rounds)
    stats = dict(
        rounds=rounds,
        useful_qps=useful,
        lane_utilization=utilization,
        end_idle_fraction=1 - utilization,
        min_lane_qps=jnp.min(work),
        max_lane_qps=jnp.max(work),
        mean_lane_qps=jnp.mean(work.astype(jnp.float32)),
        completed_frames=jnp.sum(frame.astype(jnp.float32)),
    )
    if collect:
        stats["lane_qps"] = work
        stats["frame_completion_rounds"] = completion_rounds
    return trajectory if collect else final, finite, violation, stats


def compile_async_chunks(model, settings, initial_sqp_iterations=11, *, collect=False):
    if not settings.adaptive_stopping:
        raise ValueError("Independent lane advancement requires --adaptive-stopping")
    step, errors, active, _ = model.compile_primitives(settings)

    def advance(q, target, stage):
        before = jnp.linalg.norm(errors(q, target) * active[stage], axis=1)
        candidate = step(q, target, stage)
        after = jnp.linalg.norm(errors(candidate, target) * active[stage], axis=1)
        return candidate, before, after

    def run(bank, lanes):
        return rollout_async(
            advance,
            model.preprocess,
            jnp.asarray(model.mj.qpos0, dtype=jnp.float32),
            bank,
            lanes,
            initial_cap=initial_sqp_iterations,
            warm_cap=settings.sqp_iterations,
            collect=collect,
            qindices=jnp.asarray(model.qindices),
            lower=jnp.asarray(model.lower, dtype=jnp.float32),
            upper=jnp.asarray(model.upper, dtype=jnp.float32),
        )

    return jax.jit(run)
