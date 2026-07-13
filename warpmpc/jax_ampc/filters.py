"""Composable rollout filters for AMPC datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class FilterBounds(NamedTuple):
    """JAX-friendly state and cost bounds used inside rollout kernels."""

    lower: object
    upper: object
    cost_threshold: object


@dataclass(frozen=True)
class StatePercentileConfig:
    low_percentile: float = 10.0
    high_percentile: float = 90.0
    min_width: float = 0.1
    margin_abs: float = 0.25
    margin_scale: float = 2.0


@dataclass(frozen=True)
class CostPercentileConfig:
    high_percentile: float = 90.0
    min_width: float = 0.1
    margin_abs: float = 0.25
    margin_scale: float = 2.0


@dataclass(frozen=True)
class StateBoundsFilter:
    lower: np.ndarray
    upper: np.ndarray

    def transition_mask(self, states: np.ndarray, rollout_costs: np.ndarray | None = None) -> np.ndarray:
        del rollout_costs
        state_samples = np.asarray(states)[:-1]
        return np.all(
            np.isfinite(state_samples)
            & (state_samples >= self.lower[None, None, :])
            & (state_samples <= self.upper[None, None, :]),
            axis=2,
        )

    def rollout_mask(self, states: np.ndarray, rollout_costs: np.ndarray | None = None) -> np.ndarray:
        del rollout_costs
        states = np.asarray(states)
        return np.all(
            np.isfinite(states)
            & (states >= self.lower[None, None, :])
            & (states <= self.upper[None, None, :]),
            axis=(0, 2),
        )


@dataclass(frozen=True)
class CostThresholdFilter:
    threshold: float

    def transition_mask(self, states: np.ndarray, rollout_costs: np.ndarray | None = None) -> np.ndarray:
        if rollout_costs is None:
            raise ValueError("CostThresholdFilter requires rollout_costs")
        costs = np.asarray(rollout_costs, dtype=np.float64).reshape((np.asarray(states).shape[1],))
        valid = np.isfinite(costs) & (costs <= self.threshold)
        return np.broadcast_to(valid[None, :], (np.asarray(states).shape[0] - 1, valid.size)).copy()

    def rollout_mask(self, states: np.ndarray, rollout_costs: np.ndarray | None = None) -> np.ndarray:
        if rollout_costs is None:
            raise ValueError("CostThresholdFilter requires rollout_costs")
        costs = np.asarray(rollout_costs, dtype=np.float64).reshape((np.asarray(states).shape[1],))
        return np.isfinite(costs) & (costs <= self.threshold)


@dataclass(frozen=True)
class CompositeFilter:
    components: tuple[StateBoundsFilter | CostThresholdFilter, ...] = ()

    def transition_mask(self, states: np.ndarray, rollout_costs: np.ndarray | None = None) -> np.ndarray:
        states = np.asarray(states)
        mask = np.ones((states.shape[0] - 1, states.shape[1]), dtype=bool)
        for component in self.components:
            mask &= component.transition_mask(states, rollout_costs)
        return mask

    def rollout_mask(self, states: np.ndarray, rollout_costs: np.ndarray | None = None) -> np.ndarray:
        states = np.asarray(states)
        mask = np.ones((states.shape[1],), dtype=bool)
        for component in self.components:
            mask &= component.rollout_mask(states, rollout_costs)
        return mask

    def to_bounds(self, nx: int, dtype: np.dtype) -> FilterBounds:
        lower = np.full((nx,), -np.inf, dtype=dtype)
        upper = np.full((nx,), np.inf, dtype=dtype)
        cost_threshold = np.asarray(np.inf, dtype=dtype)
        for component in self.components:
            if isinstance(component, StateBoundsFilter):
                lower = np.maximum(lower, np.asarray(component.lower, dtype=dtype))
                upper = np.minimum(upper, np.asarray(component.upper, dtype=dtype))
            elif isinstance(component, CostThresholdFilter):
                cost_threshold = np.minimum(cost_threshold, np.asarray(component.threshold, dtype=dtype))
        return FilterBounds(
            lower=jnp.asarray(lower),
            upper=jnp.asarray(upper),
            cost_threshold=jnp.asarray(cost_threshold),
        )


def infinite_filter_bounds(nx: int, dtype) -> FilterBounds:
    jdtype = jnp.dtype(dtype)
    return FilterBounds(
        lower=jnp.full((nx,), -jnp.inf, dtype=jdtype),
        upper=jnp.full((nx,), jnp.inf, dtype=jdtype),
        cost_threshold=jnp.asarray(jnp.inf, dtype=jdtype),
    )


def valid_state_jax(x, bounds: FilterBounds):
    finite = jnp.all(jnp.isfinite(x), axis=1)
    inside = jnp.all((x >= bounds.lower[None, :]) & (x <= bounds.upper[None, :]), axis=1)
    return finite & inside


def trajectory_finiteness(
    states: np.ndarray,
    actions: np.ndarray | None = None,
    rollout_costs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    states = np.asarray(states)
    batch = int(states.shape[1])
    finite = np.all(np.isfinite(states), axis=(0, 2))
    has_nan = np.any(np.isnan(states), axis=(0, 2))
    has_inf = np.any(np.isinf(states), axis=(0, 2))

    if actions is not None:
        actions = np.asarray(actions)
        batch_axis = 1 if actions.ndim >= 2 and actions.shape[1] == batch else 0
        axes = tuple(axis for axis in range(actions.ndim) if axis != batch_axis)
        finite = finite & np.all(np.isfinite(actions), axis=axes)
        has_nan = has_nan | np.any(np.isnan(actions), axis=axes)
        has_inf = has_inf | np.any(np.isinf(actions), axis=axes)

    if rollout_costs is not None:
        rollout_costs = np.asarray(rollout_costs).reshape((batch,))
        finite = finite & np.isfinite(rollout_costs)
        has_nan = has_nan | np.isnan(rollout_costs)
        has_inf = has_inf | np.isinf(rollout_costs)

    stats: dict[str, int | float] = {
        "batch_size": batch,
        "finite_trajectory_count": int(np.sum(finite)),
        "nonfinite_trajectory_count": int(batch - np.sum(finite)),
        "nan_trajectory_count": int(np.sum(has_nan)),
        "inf_trajectory_count": int(np.sum(has_inf)),
        "finite_trajectory_pct": 100.0 * float(np.mean(finite)) if batch else float("nan"),
    }
    return finite, stats


def _finite_action_by_trajectory(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    batch_axis = 1 if actions.ndim >= 3 else 0
    axes = tuple(axis for axis in range(actions.ndim) if axis != batch_axis)
    return np.all(np.isfinite(actions), axis=axes)


def calibrate_filter_bounds(
    states: np.ndarray,
    actions: np.ndarray | None,
    rollout_costs: np.ndarray,
    *,
    state_config: StatePercentileConfig | None,
    cost_config: CostPercentileConfig | None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, int | float], CompositeFilter]:
    """Calibrate composable state and cost filters from rollout data."""

    states = np.asarray(states, dtype=np.float64)
    nx = int(states.shape[-1])
    rollout_costs = np.asarray(rollout_costs, dtype=np.float64).reshape((states.shape[1],))
    finite_trajectory_mask, stats = trajectory_finiteness(
        states,
        actions=actions,
        rollout_costs=rollout_costs,
    )
    if actions is not None:
        finite_trajectory_mask &= _finite_action_by_trajectory(actions)
    if not np.any(finite_trajectory_mask):
        raise RuntimeError("calibration produced no finite trajectories")

    components: list[StateBoundsFilter | CostThresholdFilter] = []
    cost_threshold = float("inf")
    cost_percentile = float("nan")
    cost_filtered_trajectory_mask = finite_trajectory_mask
    if cost_config is not None:
        finite_costs = rollout_costs[finite_trajectory_mask]
        cost_percentile = float(np.percentile(finite_costs, cost_config.high_percentile))
        cost_width = max(abs(cost_percentile), float(cost_config.min_width))
        cost_margin = cost_config.margin_abs + cost_config.margin_scale * cost_width
        cost_threshold = float(cost_percentile + cost_margin)
        cost_filter = CostThresholdFilter(cost_threshold)
        components.append(cost_filter)
        cost_filtered_trajectory_mask = finite_trajectory_mask & cost_filter.rollout_mask(
            states,
            rollout_costs,
        )
        if not np.any(cost_filtered_trajectory_mask):
            raise RuntimeError("cost filter rejected every finite calibration trajectory")

    if state_config is not None:
        filtered_states = states[:, cost_filtered_trajectory_mask, :]
        flat = filtered_states.reshape((-1, nx))
        finite_rows = np.all(np.isfinite(flat), axis=1)
        if not np.any(finite_rows):
            raise RuntimeError("calibration produced no finite state samples")
        flat = flat[finite_rows]
        low = np.percentile(flat, state_config.low_percentile, axis=0)
        high = np.percentile(flat, state_config.high_percentile, axis=0)
        width = np.maximum(high - low, state_config.min_width)
        margin = state_config.margin_abs + state_config.margin_scale * width
        lower = low - margin
        upper = high + margin
        components.insert(0, StateBoundsFilter(lower=lower, upper=upper))
    else:
        flat = states[:, cost_filtered_trajectory_mask, :].reshape((-1, nx))
        flat = flat[np.all(np.isfinite(flat), axis=1)]
        lower = np.full((nx,), -np.inf, dtype=np.float64)
        upper = np.full((nx,), np.inf, dtype=np.float64)

    stats.update(
        {
            "bounds_state_sample_count": int(flat.shape[0]),
            "bounds_filtered_state_sample_count": int(states.shape[0] * states.shape[1] - flat.shape[0]),
            "bounds_trajectory_count": int(np.sum(cost_filtered_trajectory_mask)),
            "bounds_cost_filtered_trajectory_count": int(
                np.sum(finite_trajectory_mask) - np.sum(cost_filtered_trajectory_mask)
            ),
            "filter_low_percentile": float(state_config.low_percentile) if state_config else float("nan"),
            "filter_high_percentile": float(state_config.high_percentile) if state_config else float("nan"),
            "filter_min_width": float(state_config.min_width) if state_config else float("nan"),
            "filter_margin_abs": float(state_config.margin_abs) if state_config else float("nan"),
            "filter_margin_scale": float(state_config.margin_scale) if state_config else float("nan"),
            "filter_cost_high_percentile": (
                float(cost_config.high_percentile) if cost_config else float("nan")
            ),
            "filter_cost_percentile_value": cost_percentile,
            "filter_cost_min_width": float(cost_config.min_width) if cost_config else float("nan"),
            "filter_cost_margin_abs": float(cost_config.margin_abs) if cost_config else float("nan"),
            "filter_cost_margin_scale": float(cost_config.margin_scale) if cost_config else float("nan"),
            "filter_cost_threshold": cost_threshold,
            "filter_high_cost_trajectory_count": (
                int(np.sum(rollout_costs[finite_trajectory_mask] > cost_threshold))
                if np.isfinite(cost_threshold)
                else 0
            ),
        }
    )
    return lower, upper, cost_threshold, stats, CompositeFilter(tuple(components))


def filter_transition_valid_mask_host(
    *,
    states: np.ndarray,
    rollout_costs: np.ndarray,
    valid_mask: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    cost_threshold: float,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    rollout_costs = np.asarray(rollout_costs, dtype=np.float64).reshape((states.shape[1],))
    valid_mask = np.asarray(valid_mask, dtype=bool)
    bounds_lower = np.asarray(bounds_lower, dtype=np.float64)
    bounds_upper = np.asarray(bounds_upper, dtype=np.float64)
    if valid_mask.shape[0] != states.shape[0] - 1:
        raise ValueError(
            f"valid mask time dimension {valid_mask.shape[0]} is incompatible with {states.shape[0]} states"
        )
    state_filter = StateBoundsFilter(bounds_lower, bounds_upper)
    cost_filter = CostThresholdFilter(float(cost_threshold))
    return valid_mask & state_filter.transition_mask(states) & cost_filter.transition_mask(states, rollout_costs)


def filtered_rollout_valid_mask_host(
    *,
    states: np.ndarray,
    rollout_costs: np.ndarray,
    valid_mask: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    cost_threshold: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    states = np.asarray(states, dtype=np.float64)
    rollout_costs = np.asarray(rollout_costs, dtype=np.float64).reshape((states.shape[1],))
    valid_mask = np.asarray(valid_mask, dtype=bool)
    bounds_lower = np.asarray(bounds_lower, dtype=np.float64)
    bounds_upper = np.asarray(bounds_upper, dtype=np.float64)

    finite_states = np.all(np.isfinite(states), axis=(0, 2))
    finite_costs = np.isfinite(rollout_costs)
    within_bounds = finite_states & np.all(
        (states >= bounds_lower[None, None, :]) & (states <= bounds_upper[None, None, :]),
        axis=(0, 2),
    )
    within_cost = finite_costs & (rollout_costs <= cost_threshold)
    valid_sqp_rollout = np.all(valid_mask, axis=0)
    valid_rollout = valid_sqp_rollout & within_bounds & within_cost
    filtered_mask = np.broadcast_to(valid_rollout[None, :], valid_mask.shape).copy()
    stats = {
        "plot_valid_rollout_count": int(np.sum(valid_rollout)),
        "plot_invalid_rollout_count": int(valid_rollout.size - np.sum(valid_rollout)),
        "plot_sqp_invalid_rollout_count": int(np.sum(~valid_sqp_rollout)),
        "plot_nonfinite_state_rollout_count": int(np.sum(~finite_states)),
        "plot_state_outlier_rollout_count": int(np.sum(finite_states & ~within_bounds)),
        "plot_nonfinite_cost_rollout_count": int(np.sum(~finite_costs)),
        "plot_high_cost_rollout_count": int(np.sum(finite_costs & ~within_cost)),
        "plot_valid_rollout_pct": 100.0 * float(np.mean(valid_rollout)) if valid_rollout.size else float("nan"),
    }
    return filtered_mask, stats


def load_filter_bounds_npz(path, *, dtype=np.float32) -> tuple[np.ndarray, np.ndarray, float]:
    payload = np.load(path)
    return (
        np.asarray(payload["lower"], dtype=dtype),
        np.asarray(payload["upper"], dtype=dtype),
        float(np.asarray(payload["cost_threshold"]).reshape(())),
    )
