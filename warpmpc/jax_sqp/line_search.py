"""Static-shape filter line search utilities for SQP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


LINE_SEARCH_MAX_ITER = 0
LINE_SEARCH_CONSTRAINT = 1
LINE_SEARCH_COST = 2
LINE_SEARCH_COST_OR_CONSTRAINT = 3


@dataclass(frozen=True)
class FilterLineSearchSettings:
    """Parameters for the fixed-candidate filter line search."""

    line_search_g_max: float = 1e-1
    line_search_g_min: float = 1e-3
    line_search_gamma_c: float = 1e-3
    line_search_armijo_factor: float = 1e-4
    line_search_step_decay: float = 0.5
    line_search_step_min: float = 0.1
    line_search_constraint_scale: float = 0.1
    line_search_cost_accept_uses_trial_cost: bool = True


class FilterLineSearchResult(NamedTuple):
    """Selected step and diagnostics for each batch element."""

    step_length: object
    cost: object
    constraint_violation: object
    reason: object
    accepted: object
    candidate_costs: object
    candidate_constraint_violations: object
    candidate_acceptance: object


def make_step_lengths(settings: FilterLineSearchSettings) -> np.ndarray:
    """Create the static line-search step ladder."""

    if not (0.0 < settings.line_search_step_decay < 1.0):
        raise ValueError("line_search_step_decay must be in (0, 1)")
    if settings.line_search_step_min <= 0.0:
        raise ValueError("line_search_step_min must be positive")
    steps: list[float] = []
    step = 1.0
    tolerance = max(1e-12, 1e-12 * settings.line_search_step_min)
    while step + tolerance >= settings.line_search_step_min:
        steps.append(step)
        step *= settings.line_search_step_decay
    if not steps:
        steps.append(1.0)
    return np.asarray(steps, dtype=np.float64)


def constraint_violation(lower, g, upper, scale: float):
    """Scaled bound violation ``sqrt(||max(l-g,0)||^2 + ||min(u-g,0)||^2)``."""

    lower_gap = lower - g
    upper_gap = upper - g
    violation_sq = jnp.sum(jnp.maximum(lower_gap, 0.0) ** 2, axis=1)
    violation_sq = violation_sq + jnp.sum(jnp.minimum(upper_gap, 0.0) ** 2, axis=1)
    return jnp.asarray(scale, dtype=jnp.asarray(g).dtype) * jnp.sqrt(violation_sq)


def filter_line_search_from_evaluations(
    *,
    settings: FilterLineSearchSettings,
    step_lengths,
    baseline_cost,
    baseline_constraint_violation,
    armijo_descent_metric,
    candidate_costs,
    candidate_constraint_violations,
) -> FilterLineSearchResult:
    """Select the first accepted step from precomputed candidate evaluations.

    ``candidate_costs`` and ``candidate_constraint_violations`` have shape
    ``(n_steps, batch)``. The returned step is one per batch element.
    """

    dtype = jnp.asarray(candidate_costs).dtype
    step_lengths = jnp.asarray(step_lengths, dtype=dtype)
    baseline_cost = jnp.asarray(baseline_cost, dtype=dtype)
    baseline_constraint_violation = jnp.asarray(
        baseline_constraint_violation, dtype=dtype
    )
    armijo_descent_metric = jnp.asarray(armijo_descent_metric, dtype=dtype)
    candidate_costs = jnp.asarray(candidate_costs, dtype=dtype)
    candidate_constraint_violations = jnp.asarray(
        candidate_constraint_violations, dtype=dtype
    )

    g_max = jnp.asarray(settings.line_search_g_max, dtype=dtype)
    g_min = jnp.asarray(settings.line_search_g_min, dtype=dtype)
    gamma_c = jnp.asarray(settings.line_search_gamma_c, dtype=dtype)
    armijo_factor = jnp.asarray(settings.line_search_armijo_factor, dtype=dtype)

    constraint_reason = candidate_constraint_violations > g_max
    constraint_accept = candidate_constraint_violations < (
        (1.0 - gamma_c) * baseline_constraint_violation[None, :]
    )

    cost_reason = (
        (candidate_constraint_violations < g_min)
        & (baseline_constraint_violation[None, :] < g_min)
        & (armijo_descent_metric[None, :] < 0.0)
    )
    if settings.line_search_cost_accept_uses_trial_cost:
        cost_accept_lhs = candidate_costs
    else:
        cost_accept_lhs = jnp.broadcast_to(baseline_cost[None, :], candidate_costs.shape)
    cost_accept = cost_accept_lhs < (
        baseline_cost[None, :] + armijo_factor * armijo_descent_metric[None, :]
    )

    filter_accept = (
        (
            candidate_costs
            < (baseline_cost[None, :] - gamma_c * baseline_constraint_violation[None, :])
        )
        | (
            candidate_constraint_violations
            < ((1.0 - gamma_c) * baseline_constraint_violation[None, :])
        )
    )

    accepted = jnp.where(
        constraint_reason,
        constraint_accept,
        jnp.where(cost_reason, cost_accept, filter_accept),
    )
    reason = jnp.where(
        constraint_reason,
        LINE_SEARCH_CONSTRAINT,
        jnp.where(cost_reason, LINE_SEARCH_COST, LINE_SEARCH_COST_OR_CONSTRAINT),
    )

    any_accepted = jnp.any(accepted, axis=0)
    first_index = jnp.argmax(accepted.astype(jnp.int32), axis=0)
    fallback_index = jnp.full_like(first_index, step_lengths.shape[0] - 1)
    selected_index = jnp.where(any_accepted, first_index, fallback_index)
    batch_index = jnp.arange(candidate_costs.shape[1], dtype=jnp.int32)
    selected_cost = candidate_costs[selected_index, batch_index]
    selected_violation = candidate_constraint_violations[selected_index, batch_index]
    selected_reason = jnp.where(
        any_accepted,
        reason[selected_index, batch_index],
        jnp.full_like(selected_index, LINE_SEARCH_MAX_ITER),
    )
    selected_step = step_lengths[selected_index]

    return FilterLineSearchResult(
        step_length=selected_step,
        cost=selected_cost,
        constraint_violation=selected_violation,
        reason=selected_reason,
        accepted=any_accepted,
        candidate_costs=candidate_costs,
        candidate_constraint_violations=candidate_constraint_violations,
        candidate_acceptance=accepted,
    )


__all__ = [
    "FilterLineSearchResult",
    "FilterLineSearchSettings",
    "LINE_SEARCH_CONSTRAINT",
    "LINE_SEARCH_COST",
    "LINE_SEARCH_COST_OR_CONSTRAINT",
    "LINE_SEARCH_MAX_ITER",
    "constraint_violation",
    "filter_line_search_from_evaluations",
    "make_step_lengths",
]
