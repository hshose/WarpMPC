"""Reusable plotting utilities for AMPC rollout diagnostics."""

from __future__ import annotations

import os
import pathlib

import numpy as np


def state_plot_valid_mask(valid_mask: np.ndarray, n_state_steps: int) -> np.ndarray:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"expected valid mask with shape (time, batch), got {valid_mask.shape}")
    if valid_mask.shape[0] == n_state_steps:
        return valid_mask
    if valid_mask.shape[0] == n_state_steps - 1:
        final_mask = valid_mask[-1:] if valid_mask.shape[0] else np.zeros_like(valid_mask[:1])
        return np.concatenate([valid_mask, final_mask], axis=0)
    raise ValueError(
        f"valid mask time dimension {valid_mask.shape[0]} is incompatible with {n_state_steps} states"
    )


def plot_state_distribution(
    path: pathlib.Path,
    time_grid: np.ndarray,
    states: np.ndarray,
    *,
    state_labels: tuple[str, ...],
    title: str,
    sample_mask: np.ndarray | None = None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    states = np.asarray(states, dtype=np.float64)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != states.shape[:2]:
            raise ValueError(
                f"sample_mask shape {mask.shape} does not match state samples {states.shape[:2]}"
            )
        states = np.where(mask[:, :, None], states, np.nan)

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(np.ceil(len(state_labels) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(12.0, max(3.0, 2.5 * rows)), sharex=True)
    axes = np.asarray(axes).ravel()
    for idx, label in enumerate(state_labels):
        ax = axes[idx]
        values = states[:, :, idx]
        values = np.where(np.isfinite(values), values, np.nan)
        median = np.full((values.shape[0],), np.nan, dtype=np.float64)
        low = np.full_like(median, np.nan)
        high = np.full_like(median, np.nan)
        min_value = np.full_like(median, np.nan)
        max_value = np.full_like(median, np.nan)
        for time_idx in range(values.shape[0]):
            finite_values = values[time_idx, np.isfinite(values[time_idx])]
            if finite_values.size:
                median[time_idx] = np.median(finite_values)
                low[time_idx] = np.percentile(finite_values, 10.0)
                high[time_idx] = np.percentile(finite_values, 90.0)
                min_value[time_idx] = np.min(finite_values)
                max_value[time_idx] = np.max(finite_values)
        ax.fill_between(time_grid, min_value, max_value, color="tab:blue", alpha=0.10, linewidth=0.0)
        ax.fill_between(time_grid, low, high, color="tab:blue", alpha=0.25, linewidth=0.0)
        ax.plot(time_grid, median, color="tab:blue", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(state_labels) :]:
        ax.axis("off")
    for ax in axes[-3:]:
        ax.set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote plot {path}", flush=True)


def plot_state_distribution_pair(
    base_path: pathlib.Path,
    time_grid: np.ndarray,
    states: np.ndarray,
    *,
    state_labels: tuple[str, ...],
    title: str,
    valid_mask: np.ndarray,
) -> None:
    all_path = base_path.with_name(f"{base_path.stem}_all{base_path.suffix}")
    valid_path = base_path.with_name(f"{base_path.stem}_valid{base_path.suffix}")
    plot_state_distribution(
        all_path,
        time_grid,
        states,
        state_labels=state_labels,
        title=f"{title} (all trajectories)",
    )
    plot_state_distribution(
        valid_path,
        time_grid,
        states,
        state_labels=state_labels,
        title=f"{title} (valid samples only)",
        sample_mask=state_plot_valid_mask(valid_mask, np.asarray(states).shape[0]),
    )


def plot_rollout_histograms(
    path: pathlib.Path,
    states: np.ndarray,
    rollout_costs: np.ndarray,
    *,
    state_labels: tuple[str, ...],
    bins: int = 50,
    title: str = "Rollout histograms",
    sample_mask: np.ndarray | None = None,
    cost_mask: np.ndarray | None = None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    states = np.asarray(states, dtype=np.float64)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != states.shape[:2]:
            raise ValueError(
                f"sample_mask shape {mask.shape} does not match state samples {states.shape[:2]}"
            )
        states = np.where(mask[:, :, None], states, np.nan)

    path.parent.mkdir(parents=True, exist_ok=True)
    n_axes = len(state_labels) + 1
    cols = 4
    rows = int(np.ceil(n_axes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.0, max(3.0, 2.5 * rows)))
    axes = np.asarray(axes).ravel()
    for idx, label in enumerate(state_labels):
        values = states[:, :, idx].reshape(-1)
        values = values[np.isfinite(values)]
        ax = axes[idx]
        if values.size:
            ax.hist(values, bins=bins, color="tab:blue", alpha=0.82)
        else:
            ax.text(0.5, 0.5, "no finite samples", ha="center", va="center")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)

    costs = np.asarray(rollout_costs, dtype=np.float64).reshape(-1)
    if cost_mask is not None:
        mask = np.asarray(cost_mask, dtype=bool).reshape(-1)
        if mask.shape != costs.shape:
            raise ValueError(f"cost_mask shape {mask.shape} does not match costs {costs.shape}")
        costs = np.where(mask, costs, np.nan)
    costs = costs[np.isfinite(costs)]
    ax = axes[len(state_labels)]
    if costs.size:
        ax.hist(costs, bins=bins, color="tab:orange", alpha=0.82)
    else:
        ax.text(0.5, 0.5, "no finite samples", ha="center", va="center")
    ax.set_title("total cost")
    ax.grid(True, alpha=0.25)

    for ax in axes[len(state_labels) + 1 :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote plot {path}", flush=True)


def plot_rollout_histogram_pair(
    base_path: pathlib.Path,
    states: np.ndarray,
    rollout_costs: np.ndarray,
    *,
    state_labels: tuple[str, ...],
    title: str,
    valid_mask: np.ndarray,
    bins: int = 50,
) -> None:
    all_path = base_path.with_name(f"{base_path.stem}_all{base_path.suffix}")
    valid_path = base_path.with_name(f"{base_path.stem}_valid{base_path.suffix}")
    valid_mask = np.asarray(valid_mask, dtype=bool)
    plot_rollout_histograms(
        all_path,
        states,
        rollout_costs,
        state_labels=state_labels,
        bins=bins,
        title=f"{title} (all trajectories)",
    )
    plot_rollout_histograms(
        valid_path,
        states,
        rollout_costs,
        state_labels=state_labels,
        bins=bins,
        title=f"{title} (valid samples only)",
        sample_mask=state_plot_valid_mask(valid_mask, np.asarray(states).shape[0]),
        cost_mask=np.all(valid_mask, axis=0),
    )
