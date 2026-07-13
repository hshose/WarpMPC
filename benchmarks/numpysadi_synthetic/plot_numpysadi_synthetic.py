"""Plot synthetic benchmark results from benchmark_numpysadi_synthetic.py JSONL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROVIDERS = ("numpysadi", "jaxadi", "cusadi")
COLORS = {
    "numpysadi": "#1f77b4",
    "jaxadi": "#d62728",
    "cusadi": "#9467bd",
}
MARKERS = {
    "numpysadi": "o",
    "jaxadi": "s",
    "cusadi": "D",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ok_rows(rows: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row.get("provider") == provider and row.get("status") == "ok"),
        key=lambda row: row["instructions"],
    )


def timeout_rows(rows: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row.get("provider") == provider and row.get("status") == "timeout"),
        key=lambda row: row["instructions"],
    )


def values(rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    filtered = [row for row in rows if metric in row]
    x = np.array([row["instructions"] for row in filtered], dtype=float)
    y = np.array([row[metric] for row in filtered], dtype=float)
    return x, y


def plot_metric(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    *,
    show_timeouts: bool = False,
) -> None:
    for provider in PROVIDERS:
        provider_rows = ok_rows(rows, provider)
        x, y = values(provider_rows, metric)
        if len(x):
            axis.plot(
                x,
                y,
                marker=MARKERS[provider],
                color=COLORS[provider],
                linewidth=2,
                label=provider,
            )

        if show_timeouts:
            timeout_x = []
            timeout_y = []
            for row in timeout_rows(rows, provider):
                timeout_x.append(row["instructions"])
                timeout_y.append(row.get("compile_timeout_seconds", np.nan))
            if timeout_x:
                axis.scatter(
                    timeout_x,
                    timeout_y,
                    marker="x",
                    s=100,
                    color=COLORS[provider],
                    linewidths=2,
                    label=f"{provider} timeout",
                )

    axis.set_title(title)
    axis.set_xlabel("CasADi instructions")
    axis.set_ylabel(ylabel)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)


def make_plot(results_path: Path, output_path: Path, *, title: str) -> None:
    rows = load_rows(results_path)
    if not rows:
        raise ValueError(f"No benchmark rows found in {results_path}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    plot_metric(
        axes[0],
        rows,
        "translation_seconds",
        "Translation",
        "seconds",
    )
    plot_metric(
        axes[1],
        rows,
        "jax_compile_seconds",
        "JAX compile",
        "seconds",
        show_timeouts=True,
    )
    plot_metric(
        axes[2],
        rows,
        "execution_mean_seconds",
        "Evaluation",
        "seconds / call",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4))
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/numpysadi_synthetic/benchmark_results.png"))
    parser.add_argument("--title", default="numpysadi vs jaxadi synthetic benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plot(args.results, args.output, title=args.title)
    print(args.output)


if __name__ == "__main__":
    main()
