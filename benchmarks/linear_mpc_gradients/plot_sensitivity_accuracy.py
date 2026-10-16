#!/usr/bin/env python3
"""Plot both families of slices through the full MPC sensitivity iteration grid."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from benchmarks.linear_mpc_gradients import plot_style as style
from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales, entry_statistics, active_set_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS_DIR/figures")
    parser.add_argument("--metric", choices=("relative_l2", "absolute_l2", "component_absolute", "fd_entry_max"), default="relative_l2")
    upper = parser.add_mutually_exclusive_group()
    upper.add_argument("--upper-quantile", type=float, choices=(0.90, 0.95), default=None,
                       help="Use a percentile instead of the maximum")
    upper.add_argument("--maximum", action="store_true", help="Show maximum error with median and mean")
    parser.add_argument("--aggregation", choices=("qp", "action-rows"), default="qp",
                        help="Aggregate one full-Jacobian error per QP, or pool the four row errors")
    parser.add_argument("--no-tex", action="store_true", help="Preview without the paper's LaTeX fonts")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--active-set-filter", type=Path, help="active_sets.npz audit; exclude QPs changing active set in the FD stencil")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.results_dir / "figures"
    metadata = json.loads((args.results_dir / "metadata.json").read_text())
    if metadata["status"] != "complete" and not args.allow_partial:
        raise ValueError("Run is incomplete; use --allow-partial only for a preview")
    with (args.results_dir / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(f, r) for f in metadata["forward_iterations"] for r in metadata["refinements"]}
    actual = [(int(r["forward_iterations"]), int(r["refinement_iterations"])) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) - expected:
        raise ValueError("Duplicate or unexpected iteration pairs in summary.csv")
    if expected - set(actual) and not args.allow_partial:
        raise ValueError(f"Missing grid points: {sorted(expected-set(actual))}")
    if any(r["sweep"] != "grid" for r in rows):
        raise ValueError("Expected artifacts from the full grid benchmark")
    is_action = metadata["objective"] == "u0-jacobian"
    normalized = args.metric == "fd_entry_max"
    if normalized and not is_action:
        parser.error("fd_entry_max requires a u0-jacobian study")
    if args.aggregation == "action-rows" and (not is_action or args.metric in ("component_absolute", "fd_entry_max")):
        parser.error("action-rows requires a u0-jacobian run and an L2 metric")
    use_maximum = args.maximum or (is_action and args.upper_quantile is None)
    quantile = args.upper_quantile if args.upper_quantile is not None else 0.95
    upper_label = "Maximum" if use_maximum else f"{100 * quantile:.0f}th percentile"
    metric = f"action_{args.metric}" if args.aggregation == "action-rows" else args.metric
    reference = None
    if args.metric in ("component_absolute", "fd_entry_max"):
        with np.load(args.results_dir / "reference.npz") as archive:
            reference = archive["fd_half"]
    if args.active_set_filter and not normalized:
        parser.error("active-set filtering requires --metric fd_entry_max")
    scales = fd_entry_scales(reference) if normalized else None
    keep, exclusion_label = active_set_filter(args.active_set_filter, reference) if normalized else (None, None)
    for row in rows:
        forward, refinement = int(row["forward_iterations"]), int(row["refinement_iterations"])
        with np.load(args.results_dir / f"grid_f{forward}_r{refinement}.npz") as archive:
            errors = (np.abs(archive["gradient"] - reference) if reference is not None
                      else archive[metric])
        if normalized:
            errors = (errors / scales)[keep]
            row.update({f"{metric}_{key}": value for key, value in entry_statistics(errors).items()})
        if not np.all(np.isfinite(errors)):
            raise ValueError(f"Nonfinite per-sample errors at forward={forward}, refinement={refinement}")
        row[f"{metric}_upper_statistic"] = float(np.max(errors) if use_maximum else np.quantile(errors, quantile, method="linear"))
    style._apply_paper_rc(plt)
    if args.no_tex:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif"})
    styles = [("median", "Median", style.OURS_BLUE_DARK, "o"),
              ("mean", "Mean", "#7E57C2", "s"), ("upper_statistic", upper_label, "#D55E00", "^")]
    if normalized:
        styles = [("median", "Median", style.OURS_BLUE_DARK, "o"),
                  ("mean", "Mean", "#7E57C2", "s"),
                  ("p90", "90th percentile", "#009E73", "v"),
                  ("p95", "95th percentile", "#E69F00", "D"),
                  ("p99", "99th percentile", "#D55E00", "^"),
                  ("max", "Maximum", "0.15", "x")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ylabel = {"relative_l2": r"Relative Sensitivity Error [$\ell_2$]",
              "absolute_l2": r"Absolute Sensitivity Error [$\ell_2$]",
              "component_absolute": "Absolute Component Error",
              "fd_entry_max": "Normalized Absolute Entry Error"}[args.metric]
    if is_action:
        norm = "F" if args.aggregation == "qp" else r"\ell_2"
        if args.metric in ("relative_l2", "absolute_l2"):
            label = "Relative" if args.metric == "relative_l2" else "Absolute"
            ylabel = rf"{label} $u_0$ Jacobian Error [$\|\cdot\|_{{{norm}}}$]"
    suffix = "" if args.metric == "relative_l2" else f"_{args.metric}"
    if args.aggregation == "action-rows":
        suffix += "_action_rows"
    if use_maximum and not is_action:
        suffix += "_max"
    elif not use_maximum and (is_action or quantile != 0.95):
        suffix += f"_p{100 * quantile:.0f}"
    filter_suffix = "_stable_active_set" if args.active_set_filter else ""
    suffix += filter_suffix
    prefix = "qp_u0_sensitivity" if is_action else "qp_sensitivity"
    if normalized:
        with (args.output_dir / "fd_entry_max_scales.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["u0_dimension", "x0_dimension", "max_abs_fd"])
            writer.writerows((j, i, scales[j, i]) for j in range(4) for i in range(12))
        with (args.output_dir / f"qp_u0_normalized_entry_statistics{filter_suffix}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["forward_iterations", "refinement_iterations", "entry_count", "excluded_percent", "median", "mean", "p90", "p95", "p99", "max"])
            writer.writerows([r["forward_iterations"], r["refinement_iterations"], int(keep.sum())*48, float(100*(1-keep.mean())),
                              *[r[f"{metric}_{k}"] for k in ("median", "mean", "p90", "p95", "p99", "max")]] for r in rows)
    statistics_path = args.output_dir / f"{prefix}_error_statistics{suffix}.csv"
    with statistics_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["objective", "forward_iterations", "refinement_iterations", "metric",
                         "median", "mean", "upper_statistic", "upper_statistic_error"])
        for row in rows:
            writer.writerow([metadata["objective"], row["forward_iterations"], row["refinement_iterations"],
                             metric, row[f"{metric}_median"], row[f"{metric}_mean"],
                             upper_label, row[f"{metric}_upper_statistic"]])

    def draw(ax, *, key, fixed_key, fixed, ticks, highlight, fixed_paper):
        selected = sorted((r for r in rows if int(r[fixed_key]) == fixed), key=lambda r: int(r[key]))
        if any(int(r["nonfinite_count"]) for r in selected):
            raise ValueError(f"Nonfinite sensitivities at {fixed_key}={fixed}; inspect artifacts")
        x = [int(r[key]) for r in selected]
        for stat, label, color, marker in styles:
            y = np.asarray([float(r[f"{metric}_{stat}"]) for r in selected])
            if not np.all(np.isfinite(y)) or np.any(y < 0):
                raise ValueError(f"Invalid {stat} errors at {fixed_key}={fixed}")
            if np.any(y == 0):
                warnings.warn("Zero errors shown at 1e-16 on the log scale")
            ax.plot(x, np.maximum(y, 1e-16), label=label, color=color, marker=marker,
                    markersize=2.7, markeredgewidth=0.5, linewidth=0.75, alpha=style.PAPER_LINE_ALPHA)
        if not selected:
            ax.text(0.5, 0.5, "Pending", transform=ax.transAxes, ha="center")
        ax.axvline(highlight, color="0.2", linestyle=(0, (2, 1.5)), linewidth=0.65,
                   label=f"Paper setting ({highlight})", zorder=1)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, labels=[str(v) for v in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(min(ticks)*0.9, max(ticks)*1.1)
        ax.set_xlabel("Backward Refinement Iterations" if key == "refinement_iterations" else "Forward OSQP Iterations", labelpad=0.2)
        ax.grid(True, which="major", color="0.86", linewidth=0.35)
        ax.grid(True, which="minor", color="0.92", linewidth=0.25)
        ax.tick_params(axis="both", which="major", length=1.6, width=0.35, pad=0.7)
        ax.tick_params(axis="both", which="minor", length=0.8, width=0.25)
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)
        title = (f"{fixed} Forward OSQP Iterations" if fixed_key == "forward_iterations" else f"{fixed} Refinement Steps")
        if fixed == fixed_paper:
            title += " (paper)"
        ax.set_title(style._paper_title(title) if not args.no_tex else title, pad=2)
        ax.set_ylabel(ylabel, labelpad=0.2)

    def save(fig, stem):
        if metadata.get("configuration_label"):
            fig.text(0.5, 1.025, metadata["configuration_label"], ha="center", va="bottom", fontsize=5)
        if exclusion_label:
            label = exclusion_label if args.no_tex else exclusion_label.replace("%", r"\%")
            fig.text(0.5, -0.025, label, ha="center", va="top", fontsize=4)
        target = args.output_dir / f"{stem}{suffix}"
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.005)
        fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.005)
        plt.close(fig)
        print(f"Wrote {target}.pdf")

    families = [
        ("refinement_iterations", "forward_iterations", metadata["forward_iterations"], metadata["refinements"],
         metadata["paper_refinement_iterations"], metadata["paper_forward_iterations"], f"{prefix}_refinement_error", "osqp"),
        ("forward_iterations", "refinement_iterations", metadata["refinements"], metadata["forward_iterations"],
         metadata["paper_forward_iterations"], metadata["paper_refinement_iterations"], f"{prefix}_forward_error", "refinement"),
    ]
    for key, fixed_key, fixed_values, ticks, highlight, fixed_paper, stem, tag in families:
        count = math.ceil(len(fixed_values)/2)
        fig, axes = plt.subplots(count, 2, figsize=(3.4, 1.15*count+0.25), squeeze=False, sharey=True)
        for index, fixed in enumerate(fixed_values):
            kwargs = dict(key=key, fixed_key=fixed_key, fixed=fixed, ticks=ticks,
                          highlight=highlight, fixed_paper=fixed_paper)
            ax = axes.flat[index]
            draw(ax, **kwargs)
            if index % 2:
                ax.set_ylabel("")
            individual, single_ax = plt.subplots(figsize=(3.4, 1.35))
            draw(single_ax, **kwargs)
            single_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.2, labelspacing=1)
            individual.subplots_adjust(left=0.14, right=0.73, bottom=0.22, top=0.84)
            save(individual, f"{stem}_{tag}_{fixed}")
        for ax in list(axes.flat)[len(fixed_values):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1), ncol=4,
                   handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.12, top=0.88, wspace=0.2, hspace=0.65)
        save(fig, stem)


if __name__ == "__main__":
    main()
