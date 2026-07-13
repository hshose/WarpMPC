"""Shared plotting helpers for the runnable examples."""

from __future__ import annotations

import os
import pathlib

import numpy as np


def unpack_state_action(z: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    stages = z.reshape((z.shape[0], horizon + 1, 2))
    return stages[:, :, 0], stages[:, :horizon, 1]


def _selected_iteration_indices(iterate_count: int) -> list[int]:
    candidates = [0, 1, 2, iterate_count - 1]
    return sorted({idx for idx in candidates if 0 <= idx < iterate_count})


def _plot_band(ax, x_grid, values, *, color: str, label: str | None) -> None:
    median = np.median(values, axis=0)
    low = np.percentile(values, 10.0, axis=0)
    high = np.percentile(values, 90.0, axis=0)
    min_value = np.min(values, axis=0)
    max_value = np.max(values, axis=0)
    ax.fill_between(x_grid, min_value, max_value, color=color, alpha=0.10, linewidth=0.0)
    ax.fill_between(x_grid, low, high, color=color, alpha=0.24, linewidth=0.0)
    ax.plot(x_grid, median, color=color, linewidth=1.8, label=label)


def plot_sqp_error_distribution(
    plot_dir: pathlib.Path,
    iterates: np.ndarray,
    refs: np.ndarray,
) -> pathlib.Path:
    """Plot tracking-error distributions over the prediction horizon."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    horizon = refs.shape[1] - 1
    node_grid = np.arange(horizon + 1)
    colors = ["tab:gray", "tab:blue", "tab:green", "tab:red"]
    selected = _selected_iteration_indices(iterates.shape[0])

    path = plot_dir / "reference_error_distribution.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for color, iteration in zip(colors, selected, strict=False):
        x_pred, _ = unpack_state_action(iterates[iteration], horizon)
        _plot_band(
            ax,
            node_grid,
            x_pred - refs,
            color=color,
            label=f"SQP iter {iteration}",
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("prediction step")
    ax.set_ylabel("predicted state minus reference")
    ax.set_title("Tracking-error distribution over the prediction horizon")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sqp_timeseries_samples(
    plot_dir: pathlib.Path,
    iterates: np.ndarray,
    refs: np.ndarray,
    *,
    sample_count: int = 6,
) -> pathlib.Path:
    """Plot reference and predicted open-loop state sequences for a few samples."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    horizon = refs.shape[1] - 1
    node_grid = np.arange(horizon + 1)
    batch_size = refs.shape[0]
    count = min(sample_count, batch_size)
    order = np.argsort(refs[:, -1])
    positions = np.linspace(0, batch_size - 1, count, dtype=np.int64)
    sample_indices = order[positions]
    selected_iterations = _selected_iteration_indices(iterates.shape[0])
    colors = ["0.55", "tab:blue", "tab:green", "tab:red"]
    initial_states, _ = unpack_state_action(iterates[0], horizon)

    path = plot_dir / "sample_timeseries_predictions.png"
    fig, axes = plt.subplots(
        count,
        1,
        figsize=(7.4, max(2.2 * count, 3.0)),
        sharex=True,
        squeeze=False,
    )
    for row, sample_index in enumerate(sample_indices):
        ax = axes[row, 0]
        ax.plot(
            node_grid,
            refs[sample_index],
            color="black",
            linewidth=2.0,
            label="reference",
        )
        for color, iteration in zip(colors, selected_iterations, strict=False):
            x_pred, _ = unpack_state_action(iterates[iteration], horizon)
            linestyle = "--" if iteration == 0 else "-"
            ax.plot(
                node_grid,
                x_pred[sample_index],
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                alpha=0.9,
                label=f"predicted iter {iteration}",
            )
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.set_ylabel("state")
        ax.set_title(
            f"sample {int(sample_index)}: "
            f"x0={float(initial_states[sample_index, 0]):.3g}, "
            f"ref_0={float(refs[sample_index, 0]):.3g}, "
            f"ref_N={float(refs[sample_index, -1]):.3g}"
        )
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(loc="best", fontsize=8, ncols=2)
    axes[-1, 0].set_xlabel("prediction step")
    fig.suptitle("Reference and predicted open-loop state sequences")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ampc_rollout_comparison(
    plot_dir: pathlib.Path,
    ampc_states: np.ndarray,
    mpc_states: np.ndarray,
    references: np.ndarray,
) -> pathlib.Path:
    """Plot AMPC closed-loop states against MPC open-loop predictions."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    ampc_states = np.asarray(ampc_states)
    mpc_states = np.asarray(mpc_states)
    references = np.asarray(references).reshape((-1,))
    if ampc_states.ndim != 2:
        raise ValueError("ampc_states must have shape (batch, horizon + 1)")
    if mpc_states.shape != ampc_states.shape:
        raise ValueError("mpc_states must have the same shape as ampc_states")
    if references.shape[0] != ampc_states.shape[0]:
        raise ValueError("references must contain one reference per sample")

    plot_dir.mkdir(parents=True, exist_ok=True)
    horizon = ampc_states.shape[1] - 1
    node_grid = np.arange(horizon + 1)
    count = ampc_states.shape[0]
    path = plot_dir / "ampc_rollout_vs_mpc_open_loop.png"

    fig, axes = plt.subplots(
        count,
        1,
        figsize=(7.4, max(2.2 * count, 3.0)),
        sharex=True,
        squeeze=False,
    )
    for row in range(count):
        ax = axes[row, 0]
        ax.plot(
            node_grid,
            np.full_like(node_grid, references[row], dtype=np.float64),
            color="black",
            linewidth=2.0,
            linestyle=":",
            label="reference",
        )
        ax.plot(
            node_grid,
            mpc_states[row],
            color="tab:blue",
            linewidth=1.6,
            label="MPC open-loop",
        )
        ax.plot(
            node_grid,
            ampc_states[row],
            color="tab:orange",
            linewidth=1.6,
            label="AMPC rollout",
        )
        ax.set_ylabel("state")
        ax.set_title(
            f"sample {row}: "
            f"x0={float(ampc_states[row, 0]):.3g}, "
            f"ref={float(references[row]):.3g}"
        )
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(loc="best", fontsize=8, ncols=3)
    axes[-1, 0].set_xlabel("prediction step")
    fig.suptitle("AMPC closed-loop rollout vs MPC open-loop prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
