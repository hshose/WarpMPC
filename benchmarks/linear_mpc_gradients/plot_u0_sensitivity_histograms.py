#!/usr/bin/env python3
"""Debug all entries of du_0/dx_0: one 12x4 histogram page per iteration pair.

Writes signed-error and backward-versus-FD distribution PDFs. Each cell contains
all QPs for one Jacobian entry; no clipping or sample exclusions. Bin edges and
axis limits are fixed for each entry across all pages, including all outliers.
Signed-logarithmic bins resolve near-zero values and tails; the center is linear.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-u0-histograms")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales, entry_statistics, active_set_filter

BLUE = "#08519C"
ORANGE = "#D55E00"


def signed_edges(extent, threshold, bins_per_side):
    extent = max(float(extent) * 1.02, threshold * 10)
    positive = np.geomspace(threshold, extent, bins_per_side + 1)
    return np.concatenate((-positive[::-1], [0.0], positive))


def fixed_ticks(extent, threshold):
    outer = 10.0 ** np.floor(np.log10(extent))
    inner = max(threshold, outer / 1000.0)
    return sorted(set([-outer, -inner, 0.0, inner, outer]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS_DIR/figures/histograms")
    parser.add_argument("--linear-threshold", type=float, default=1e-6,
                        help="Half-width of the linear core of the signed-log x axis")
    parser.add_argument("--bins-per-side", type=int, default=30)
    parser.add_argument("--normalization", choices=("none", "fd-entry-max"), default="none")
    parser.add_argument("--active-set-filter", type=Path, help="active_sets.npz audit; use a fixed stable-active-set population")
    parser.add_argument("--bin-scale", choices=("both", "log", "linear"), default="both",
                        help="Generate geometric/symlog bins, uniform/linear bins, or both; y remains log counts")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.results_dir / "figures" / "histograms"
    if not np.isfinite(args.linear_threshold) or args.linear_threshold <= 0 or args.bins_per_side < 2:
        parser.error("linear threshold must be finite and positive; bins per side must be >= 2")
    metadata = json.loads((args.results_dir / "metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["objective"] != "u0-jacobian":
        raise ValueError("Expected a completed u0-jacobian study")
    with np.load(args.results_dir / "reference.npz") as archive:
        reference = archive["fd_half"]
        reference_coarse = archive["fd"]
    batch, nu, nx = reference.shape
    if [batch, nu, nx] != metadata["sensitivity_shape"] or (nu, nx) != (4, 12):
        raise ValueError("Expected (batch, 4, 12) Jacobians matching metadata")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(reference_coarse)):
        raise ValueError("Nonfinite finite-difference reference")
    normalized = args.normalization == "fd-entry-max"
    scales = fd_entry_scales(reference) if normalized else np.ones((nu, nx))
    keep, exclusion_label = active_set_filter(args.active_set_filter, reference)
    original_shape = reference.shape
    original_batch = batch
    batch = int(keep.sum())
    reference = reference[keep] / scales
    reference_coarse = reference_coarse[keep] / scales
    pairs = [(f, r) for f in metadata["forward_iterations"] for r in metadata["refinements"]]
    jacobians = {}
    error_extent = np.zeros((nu, nx))
    value_extent = np.max(np.abs(reference), axis=0)
    records = []
    for forward, refinement in pairs:
        with np.load(args.results_dir / f"grid_f{forward}_r{refinement}.npz") as archive:
            jacobian = archive["jacobian"]
        if jacobian.shape != original_shape or not np.all(np.isfinite(jacobian)):
            raise ValueError(f"Invalid Jacobians for ({forward}, {refinement})")
        jacobian = jacobian[keep] / scales
        jacobians[forward, refinement] = jacobian
        error = jacobian - reference
        error_extent = np.maximum(error_extent, np.max(np.abs(error), axis=0))
        value_extent = np.maximum(value_extent, np.max(np.abs(jacobian), axis=0))
        for state in range(nx):
            for action in range(nu):
                values = error[:, action, state]
                abs_values = np.abs(values)
                records.append(dict(forward_iterations=forward, refinement_iterations=refinement,
                                    x0_dimension=state, u0_dimension=action, samples=batch, excluded_percent=100*(1-batch/original_batch),
                                    normalization_scale=float(scales[action, state]),
                                    **{f"absolute_{k}": v for k, v in entry_statistics(abs_values).items() if k in ("p90", "p95", "p99")},
                                    signed_median=float(np.median(values)), signed_mean=float(np.mean(values)),
                                    absolute_median=float(np.median(abs_values)),
                                    absolute_mean=float(np.mean(abs_values)), absolute_max=float(np.max(abs_values)),
                                    fd_absolute_median=float(np.median(np.abs(reference[:, action, state]))),
                                    fd_near_zero_count=int(np.count_nonzero(np.abs(reference[:, action, state]) < 1e-8)),
                                    fd_step_absolute_max=float(np.max(np.abs(reference_coarse[:, action, state] - reference[:, action, state])))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "u0_jacobian_component_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 8,
                         "axes.titlesize": 10, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": 0.5, "pdf.fonttype": 42, "legend.frameon": False})
    formats = [
        ("signed_error", "Signed action-Jacobian error", error_extent,
         r"Signed entry error $\Delta = J_{j,i} - J^{\mathrm{FD}}_{j,i}$"),
        ("sensitivity_comparison", "Action-Jacobian sensitivity distributions", value_extent,
         r"Sensitivity entry $J_{j,i}=\partial u_{0,j}/\partial x_{0,i}$"),
    ]
    if normalized:
        formats = [("normalized_absolute_error", "FD-max-normalized action-Jacobian entry errors", error_extent,
                    r"Normalized absolute entry error $|J_{j,i}-J^{\mathrm{FD}}_{j,i}|/\max_b|J^{\mathrm{FD}}_{b,j,i}|$")]
    index = []
    for page, (forward, refinement) in enumerate(pairs, 1):
        index.append(dict(page=page, forward_iterations=forward, refinement_iterations=refinement))
    (args.output_dir / "histogram_pages.json").write_text(json.dumps(dict(
        rows="x0 dimensions 0..11", columns="u0 dimensions 0..3", samples_per_cell=batch,
        bin_scale=args.bin_scale, xscale="symlog for log bins; linear for uniform bins", linear_threshold=args.linear_threshold, yscale="log QP counts",
        bins=("nonnegative geometric tail and one linear center bin; fixed per entry across pages" if normalized
              else "symmetric geometric tails and two linear center bins; fixed per entry across pages"),
        clipping="none", excluded_samples=original_batch-batch, excluded_percent=100*(1-batch/original_batch), normalization=args.normalization, scales=scales.tolist(),
        quantiles=[0.9, 0.95, 0.99], pages=index), indent=2) + "\n")
    bin_scales = ["log", "linear"] if args.bin_scale == "both" else [args.bin_scale]
    for bin_scale, (kind, title, extents, xlabel) in product(bin_scales, formats):
        bins = {(action, state): signed_edges(extents[action, state], args.linear_threshold, args.bins_per_side)
                for action in range(nu) for state in range(nx)}
        if normalized:
            bins = {key: edges[edges >= 0] for key, edges in bins.items()}
        if bin_scale == "linear":
            bins = {key: np.linspace(edges[0], edges[-1], 2 * args.bins_per_side + 1)
                    for key, edges in bins.items()}
        target = args.output_dir / f"u0_jacobian_{kind}_histograms_{bin_scale}_bins.pdf"
        with PdfPages(target, metadata={"Title": title, "Subject": f"12 state rows x 4 action columns; {batch} QPs per histogram"}) as pdf:
            for page, (forward, refinement) in enumerate(pairs, 1):
                jacobian = jacobians[forward, refinement]
                error = jacobian - reference
                fig, axes = plt.subplots(nx, nu, figsize=(11.7, 16.5))
                for state in range(nx):
                    for action in range(nu):
                        ax = axes[state, action]
                        edges = bins[action, state]
                        def histogram(values, color, label, fill=False):
                            counts, _ = np.histogram(values, bins=edges)
                            if counts.sum() != batch:
                                raise ValueError("Histogram dropped samples")
                            ax.stairs(counts, edges, baseline=0.8, color=color, linewidth=0.85,
                                      fill=fill, alpha=0.45 if fill else 0.95, label=label)
                        if kind == "normalized_absolute_error":
                            histogram(np.abs(error[:, action, state]), BLUE, "Normalized absolute error", fill=True)
                        elif kind == "signed_error":
                            histogram(error[:, action, state], BLUE, "Backward minus FD", fill=True)
                        else:
                            histogram(reference[:, action, state], ORANGE, "Finite differences", fill=True)
                            histogram(jacobian[:, action, state], BLUE, "Backward sensitivity")
                        ax.axvline(0, color="0.4", linewidth=0.4, linestyle="--")
                        if bin_scale == "log":
                            ax.set_xscale("symlog", linthresh=args.linear_threshold, linscale=0.8)
                        else:
                            ax.set_xscale("linear")
                        ax.set_yscale("log")
                        ax.set_ylim(0.8, batch * 1.25)
                        ax.set_xlim(edges[0], edges[-1])
                        ax.xaxis.set_major_locator(FixedLocator(fixed_ticks(edges[-1], args.linear_threshold))
                                                       if bin_scale == "log" else matplotlib.ticker.MaxNLocator(3))
                        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
                        ax.xaxis.set_minor_locator(NullLocator())
                        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
                        ax.yaxis.set_minor_locator(NullLocator())
                        ax.tick_params(length=2, width=0.4, pad=1)
                        ax.grid(axis="y", color="0.9", linewidth=0.4)
                        absolute = np.abs(error[:, action, state])
                        ax.text(0.985, 0.965,
                                f"med {np.median(absolute):.2g}; mean {np.mean(absolute):.2g}\np95 {np.quantile(absolute, .95):.2g}; p99 {np.quantile(absolute, .99):.2g}\nmax {np.max(absolute):.2g}",
                                transform=ax.transAxes, va="top", ha="right", fontsize=5.5,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
                        if state == 0:
                            ax.set_title(rf"$u_{{0,{action}}}$", pad=5)
                        if action == 0:
                            ax.set_ylabel(rf"$x_{{0,{state}}}$", rotation=0, labelpad=20, va="center")
                        else:
                            ax.tick_params(labelleft=False)
                paper = (forward == metadata["paper_forward_iterations"] and refinement == metadata["paper_refinement_iterations"])
                fig.suptitle((metadata.get("configuration_label", "") + "\n" if metadata.get("configuration_label") else "")
                             + f"{title} ({bin_scale} bins)\nOSQP iterations: {forward}    Refinement steps: {refinement}"
                             + ("    [paper setting]" if paper else ""), fontsize=12, y=0.988)
                if kind == "sensitivity_comparison":
                    fig.legend([Line2D([0], [0], color=BLUE), Line2D([0], [0], color=ORANGE)],
                               ["Backward sensitivity", "Central finite differences (h/2)"],
                               loc="upper center", bbox_to_anchor=(0.55, 0.956), ncol=2, fontsize=8)
                else:
                    fig.text(0.55, 0.95, ("Per-entry FD max normalization; annotations summarize absolute normalized errors" if normalized
                                            else "Paired signed errors; annotations summarize absolute entry errors"),
                             ha="center", fontsize=8)
                if exclusion_label:
                    fig.text(0.55, 0.937, exclusion_label, ha="center", fontsize=8)
                fig.text(0.012, 0.50, "QP count per bin (log scale)", rotation=90, va="center", fontsize=9)
                fig.text(0.55, 0.022, xlabel, ha="center", fontsize=10)
                fig.text(0.55, 0.009,
                         f"{batch} QPs/cell; zero-based indices; {bin_scale} bins and x scale; y: log counts; "
                         f"fixed bins across pages; no clipping. Page {page}/{len(pairs)}", ha="center", fontsize=7)
                fig.subplots_adjust(left=0.09, right=0.99, bottom=0.05, top=0.925, hspace=0.42, wspace=0.14)
                pdf.savefig(fig)
                if paper or page == len(pairs):
                    fig.savefig(args.output_dir / f"u0_jacobian_{kind}_osqp_{forward}_refinement_{refinement}_{bin_scale}_bins.png", dpi=120)
                plt.close(fig)
                print(f"{kind} ({bin_scale} bins): page {page}/{len(pairs)} (OSQP {forward}, refinement {refinement})", flush=True)
        print(f"Wrote {target}", flush=True)


if __name__ == "__main__":
    main()
