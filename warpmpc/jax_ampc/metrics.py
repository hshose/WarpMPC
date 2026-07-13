"""Host-side rollout metrics for AMPC examples."""

from __future__ import annotations

import jax
import numpy as np

from .filters import trajectory_finiteness


def device_tree_block(value):
    leaves = jax.tree_util.tree_leaves(value)
    if leaves:
        jax.block_until_ready(leaves[0])
    return value


def quadratic_rollout_cost_host(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    state_weights,
    action_weights,
    control_dt: float,
) -> np.ndarray:
    """Compute finite-horizon quadratic rollout costs on the host."""

    q_diag = np.asarray(state_weights, dtype=np.float64)
    r_diag = np.asarray(action_weights, dtype=np.float64)
    x = np.asarray(states[:-1], dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        state_cost = np.sum(q_diag[None, None, :] * x * x, axis=2)
        action_cost = np.sum(r_diag[None, None, :] * u * u, axis=2)
        running = np.sum(0.5 * control_dt * (state_cost + action_cost), axis=0)
        terminal = 0.5 * np.sum(q_diag[None, :] * np.asarray(states[-1], dtype=np.float64) ** 2, axis=1)
    return running + terminal


def summarize_rollout_host(
    *,
    name: str,
    states: np.ndarray,
    actions: np.ndarray,
    rollout_costs: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    cost_threshold: float | None,
    valid_datum_mask: np.ndarray | None,
    converged_position_norm: float,
    converged_state_norm: float,
) -> dict[str, float | int | str]:
    """Summarize rollout cost, divergence, convergence, and finiteness."""

    states = np.asarray(states)
    actions = np.asarray(actions)
    batch = states.shape[1]
    costs = np.asarray(rollout_costs, dtype=np.float64).reshape((batch,))
    finite, finite_stats = trajectory_finiteness(states, actions=actions, rollout_costs=costs)
    if bounds is None:
        within = np.ones((batch,), dtype=bool)
    else:
        lower, upper = bounds
        within = np.all((states >= lower[None, None, :]) & (states <= upper[None, None, :]), axis=(0, 2))
    finite_cost = np.isfinite(costs)
    if cost_threshold is None:
        below_cost_threshold = np.ones((batch,), dtype=bool)
        high_cost = np.zeros((batch,), dtype=bool)
    else:
        cost_threshold = float(cost_threshold)
        below_cost_threshold = finite_cost & (costs <= cost_threshold)
        high_cost = finite_cost & (costs > cost_threshold)
    diverged = ~(finite & within & below_cost_threshold)
    final_pos_norm = np.linalg.norm(states[-1, :, :3], axis=1)
    final_state_norm = np.linalg.norm(states[-1], axis=1)
    converged = (~diverged) & (final_pos_norm <= converged_position_norm) & (
        final_state_norm <= converged_state_norm
    )
    nondiverged = finite_cost & (~diverged)
    finite_final_pos = np.all(np.isfinite(states[-1, :, :3]), axis=1)
    finite_final_state = np.all(np.isfinite(states[-1]), axis=1)
    final_pos_values = states[-1, finite_final_pos, :3].astype(np.float64)
    final_state_values = states[-1, finite_final_state].astype(np.float64)
    final_pos_nondiv = states[-1, ~diverged, :3].astype(np.float64)
    final_state_nondiv = states[-1, ~diverged].astype(np.float64)
    finite_state_values = states[np.isfinite(states)]
    metrics: dict[str, float | int | str] = {
        "name": name,
        "batch_size": int(batch),
        "mean_cost": float(np.mean(costs[finite_cost])) if np.any(finite_cost) else float("nan"),
        "mean_cost_nondiverged": float(np.mean(costs[nondiverged])) if np.any(nondiverged) else float("nan"),
        "median_cost": float(np.median(costs[finite_cost])) if np.any(finite_cost) else float("nan"),
        "converged_pct": 100.0 * float(np.mean(converged)),
        "diverged_pct": 100.0 * float(np.mean(diverged)),
        "final_position_rms": (
            float(np.sqrt(np.mean(final_pos_values**2))) if final_pos_values.size else float("nan")
        ),
        "final_state_rms": (
            float(np.sqrt(np.mean(final_state_values**2))) if final_state_values.size else float("nan")
        ),
        "final_position_rms_nondiverged": (
            float(np.sqrt(np.mean(final_pos_nondiv**2))) if final_pos_nondiv.size else float("nan")
        ),
        "final_state_rms_nondiverged": (
            float(np.sqrt(np.mean(final_state_nondiv**2))) if final_state_nondiv.size else float("nan")
        ),
        "finite_final_position_count": int(np.sum(finite_final_pos)),
        "finite_final_state_count": int(np.sum(finite_final_state)),
        "max_abs_state": float(np.max(np.abs(finite_state_values))) if finite_state_values.size else float("nan"),
        "high_cost_trajectory_count": int(np.sum(high_cost)),
        "high_cost_trajectory_pct": 100.0 * float(np.mean(high_cost)) if batch else float("nan"),
    }
    if cost_threshold is not None:
        metrics["cost_threshold"] = cost_threshold
    metrics.update(finite_stats)
    if valid_datum_mask is not None:
        metrics["valid_datum_pct"] = 100.0 * float(np.mean(valid_datum_mask))
        metrics["valid_datum_count"] = int(np.sum(valid_datum_mask))
    return metrics


def format_rollout_metrics(metrics: dict[str, float | int | str]) -> str:
    text = (
        f"{metrics['name']}: "
        f"mean_cost={float(metrics['mean_cost']):.4g}, "
        f"mean_cost_nondiv={float(metrics['mean_cost_nondiverged']):.4g}, "
        f"conv={float(metrics['converged_pct']):.2f}%, "
        f"div={float(metrics['diverged_pct']):.2f}%, "
        f"final_pos_rms_finite={float(metrics['final_position_rms']):.3g}, "
        f"nan_traj={int(metrics['nan_trajectory_count'])}, "
        f"nonfinite_traj={int(metrics['nonfinite_trajectory_count'])}/{int(metrics['batch_size'])}"
    )
    if "cost_threshold" in metrics:
        text += (
            f", high_cost_traj={int(metrics['high_cost_trajectory_count'])}, "
            f"cost_threshold={float(metrics['cost_threshold']):.4g}"
        )
    return text


def rollout_qp_metrics(rollout) -> dict[str, float]:
    return {
        "mean_step_length": float(np.mean(np.asarray(jax.device_get(rollout.step_lengths)))),
        "max_constraint_violation": float(
            np.max(np.asarray(jax.device_get(rollout.constraint_violations)))
        ),
        "mean_qp_prim_res": float(np.mean(np.asarray(jax.device_get(rollout.prim_res)))),
        "mean_qp_dual_res": float(np.mean(np.asarray(jax.device_get(rollout.dual_res)))),
    }


def format_filtered_rollout_plot_stats(name: str, stats: dict[str, int | float]) -> str:
    return " ".join(
        (
            f"{name} valid plot filter:",
            f"valid_rollouts={int(stats['plot_valid_rollout_count']):,}/"
            f"{int(stats['plot_valid_rollout_count']) + int(stats['plot_invalid_rollout_count']):,}",
            f"sqp_invalid={int(stats['plot_sqp_invalid_rollout_count']):,}",
            f"state_outlier={int(stats['plot_state_outlier_rollout_count']):,}",
            f"high_cost={int(stats['plot_high_cost_rollout_count']):,}",
            f"nonfinite_state={int(stats['plot_nonfinite_state_rollout_count']):,}",
            f"nonfinite_cost={int(stats['plot_nonfinite_cost_rollout_count']):,}",
        )
    )
