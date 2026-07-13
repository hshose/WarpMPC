"""Jitted rollout builders for AMPC imitation."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .filters import FilterBounds, valid_state_jax
from .normalization import Normalization, apply_normalized_policy
from .outputs import OutputSpec, first_action_from_prediction, select_prediction_target


class SQPRolloutOutput(NamedTuple):
    dataset_x: object
    expert_y: object
    applied_u: object
    valid_mask: object
    states: object
    costs: object
    step_lengths: object
    constraint_violations: object
    line_search_accepted: object
    prim_res: object
    dual_res: object


class PolicyRolloutOutput(NamedTuple):
    applied_u: object
    valid_mask: object
    states: object
    costs: object


def make_sqp_collect_rollout(
    *,
    sqp,
    model,
    output_spec: OutputSpec,
    nx: int,
    rollout_steps: int,
    sqp_iterations: int,
    dynamics_step: Callable,
    stage_cost: Callable,
    terminal_cost: Callable,
    set_initial_state: Callable | None = None,
    dtype: np.dtype,
    valid_max_constraint_violation: float,
    valid_max_qp_residual: float,
    require_line_search_accepted: bool,
) -> Callable:
    """Build a jitted SQP expert collection rollout with DAgger action mixing."""

    jdtype = jnp.dtype(dtype)
    if sqp.init_state is None:
        raise RuntimeError("compiled SQP solver does not expose a solver-state initializer")
    if set_initial_state is None:
        set_initial_state = lambda params, x: params.at[:, :nx].set(x)

    @jax.jit
    def collect(
        policy_params,
        normalization: Normalization,
        bounds: FilterBounds,
        z,
        params,
        x,
        key,
        dagger_beta,
        noise_scale,
    ) -> SQPRolloutOutput:
        dagger_beta = jnp.asarray(dagger_beta, dtype=jdtype)
        noise_scale = jnp.asarray(noise_scale, dtype=jdtype)

        def step(carry, _):
            z, params, x, key, solver_state = carry
            params = set_initial_state(params, x)

            def sqp_iteration(iter_carry, _):
                z_iter, solver_state_iter = iter_carry
                result_iter, solver_state_iter_next = sqp.step(
                    z_iter,
                    params,
                    state=solver_state_iter,
                )
                return (result_iter.z_next, solver_state_iter_next), result_iter

            (z_next, solver_state_next), sqp_results = jax.lax.scan(
                sqp_iteration,
                (z, solver_state),
                xs=None,
                length=sqp_iterations,
            )
            result = jax.tree_util.tree_map(lambda leaf: leaf[-1], sqp_results)
            expert_y = select_prediction_target(output_spec, result.z_next)
            expert_u = first_action_from_prediction(output_spec, expert_y)
            policy_y = apply_normalized_policy(model.apply, policy_params, normalization, x)
            policy_u = first_action_from_prediction(output_spec, policy_y)
            applied_u = (1.0 - dagger_beta) * expert_u + dagger_beta * policy_u
            key, noise_key = jax.random.split(key)
            x_next = dynamics_step(noise_key, x, applied_u, noise_scale)
            params_next = set_initial_state(params, x_next)
            valid = valid_state_jax(x, bounds)
            valid = valid & jnp.all(jnp.isfinite(expert_y), axis=1)
            valid = valid & jnp.all(jnp.isfinite(expert_u), axis=1)
            valid = valid & jnp.isfinite(result.line_search.constraint_violation)
            valid = valid & (result.line_search.constraint_violation <= valid_max_constraint_violation)
            valid = valid & jnp.isfinite(result.solve.prim_res)
            valid = valid & jnp.isfinite(result.solve.dual_res)
            valid = valid & (result.solve.prim_res <= valid_max_qp_residual)
            valid = valid & (result.solve.dual_res <= valid_max_qp_residual)
            valid = valid & result.is_finite
            if require_line_search_accepted:
                valid = valid & result.line_search.accepted
            output = (
                x,
                expert_y,
                applied_u,
                valid,
                stage_cost(x, applied_u),
                result.line_search.step_length,
                result.line_search.constraint_violation,
                result.line_search.accepted,
                result.solve.prim_res,
                result.solve.dual_res,
            )
            return (
                z_next,
                params_next,
                x_next,
                key,
                solver_state_next,
            ), output

        solver_state = sqp.init_state(z.shape[0])
        final_carry, outputs = jax.lax.scan(
            step,
            (z, params, x, key, solver_state),
            xs=None,
            length=rollout_steps,
        )
        _, _, x_final, _, _ = final_carry
        (
            dataset_x,
            expert_y,
            applied_u,
            valid_mask,
            costs,
            step_lengths,
            constraint_violations,
            line_search_accepted,
            prim_res,
            dual_res,
        ) = outputs
        states = jnp.concatenate([dataset_x, x_final[None, :, :]], axis=0)
        rollout_cost = jnp.sum(costs, axis=0) + terminal_cost(x_final)
        cost_valid = jnp.isfinite(rollout_cost) & (rollout_cost <= bounds.cost_threshold)
        valid_mask = valid_mask & cost_valid[None, :]
        return SQPRolloutOutput(
            dataset_x=dataset_x,
            expert_y=expert_y,
            applied_u=applied_u,
            valid_mask=valid_mask,
            states=states,
            costs=costs,
            step_lengths=step_lengths,
            constraint_violations=constraint_violations,
            line_search_accepted=line_search_accepted,
            prim_res=prim_res,
            dual_res=dual_res,
        )

    return collect


def make_policy_rollout(
    *,
    model,
    output_spec: OutputSpec,
    rollout_steps: int,
    dynamics_step: Callable,
    stage_cost: Callable,
    terminal_cost: Callable,
    dtype: np.dtype,
):
    """Build a jitted rollout for a normalized policy prediction target."""

    jdtype = jnp.dtype(dtype)

    @jax.jit
    def rollout(policy_params, normalization: Normalization, bounds: FilterBounds, x, key, noise_scale):
        noise_scale = jnp.asarray(noise_scale, dtype=jdtype)

        def step(carry, _):
            x, key = carry
            policy_y = apply_normalized_policy(model.apply, policy_params, normalization, x)
            applied_u = first_action_from_prediction(output_spec, policy_y)
            key, noise_key = jax.random.split(key)
            x_next = dynamics_step(noise_key, x, applied_u, noise_scale)
            valid = valid_state_jax(x, bounds) & jnp.all(jnp.isfinite(applied_u), axis=1)
            return (x_next, key), (x, applied_u, valid, stage_cost(x, applied_u))

        (x_final, _), outputs = jax.lax.scan(step, (x, key), xs=None, length=rollout_steps)
        states_before, applied_u, valid_mask, costs = outputs
        states = jnp.concatenate([states_before, x_final[None, :, :]], axis=0)
        rollout_cost = jnp.sum(costs, axis=0) + terminal_cost(x_final)
        cost_valid = jnp.isfinite(rollout_cost) & (rollout_cost <= bounds.cost_threshold)
        valid_mask = valid_mask & cost_valid[None, :]
        return PolicyRolloutOutput(
            applied_u=applied_u,
            valid_mask=valid_mask,
            states=states,
            costs=costs,
        )

    return rollout
