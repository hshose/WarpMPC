#!/usr/bin/env python3
"""Summarize the real-AMASS local iteration sweep as tables and heatmaps."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gmr")

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()
    from benchmarks.linear_mpc_gradients.plot_style import _apply_paper_rc
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.backends.backend_pdf import PdfPages

    summary, tables = [], {}
    for subset in ("running", "kicks"):
        for policy in ("fixed", "adaptive"):
            directory = args.results_dir / f"{subset}_{policy}"
            with (directory / "accuracy.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            keyed = {
                (int(row["sqp_iterations"]), int(row["admm_iterations"]), row["mode"]): row
                for row in rows
            }
            counts = sorted({key[0] for key in keyed})
            tables[subset, policy] = counts, keyed
            for sqp in counts:
                for admm in (25, 50, 100):
                    original = keyed[sqp, admm, "chunk_vs_original_gmr"]
                    matched = keyed[sqp, admm, "chunk_vs_matched_daqp"]
                    row = dict(
                        subset=subset,
                        policy=policy,
                        initial_sqp_iterations=int(original["initial_sqp_iterations"]),
                        sqp_iterations=sqp,
                        admm_iterations=admm,
                        rho=float(original["rho"]),
                        original_tangent_max=float(original["q_tangent_max"]),
                        original_tangent_rms=float(original["q_tangent_rms"]),
                        matched_tangent_max=float(matched["q_tangent_max"]),
                        original_cold_max=float(
                            keyed[sqp, admm, "chunk_vs_original_gmr_cold"]["q_tangent_max"]
                        ),
                        original_warm_max=float(
                            keyed[sqp, admm, "chunk_vs_original_gmr_warm"]["q_tangent_max"]
                        ),
                        body_position_rms_m=float(original["body_position_rms_m"]),
                        joint_limit_violation_rad=float(original["joint_limit_violation_rad"]),
                        fraction_within_1e2=float(original["fraction_q_within_1e2"]),
                        fraction_within_1e3=float(original["fraction_q_within_1e3"]),
                        stage1_cost_relative_difference=float(
                            original["stage1_cost_relative_difference"]
                        ),
                        stage2_cost_relative_difference=float(
                            original["stage2_cost_relative_difference"]
                        ),
                        tracking_position_rms_m=float(original["stage2_tracking_position_rms_m"]),
                        reference_tracking_position_rms_m=float(
                            original["reference_stage2_tracking_position_rms_m"]
                        ),
                        tracking_rotation_rms_rad=float(
                            original["stage2_tracking_rotation_rms_rad"]
                        ),
                        reference_tracking_rotation_rms_rad=float(
                            original["reference_stage2_tracking_rotation_rms_rad"]
                        ),
                    )
                    summary.append(row)
    with (args.results_dir / "accuracy_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _apply_paper_rc(plt)
    if args.no_tex:
        plt.rcParams["text.usetex"] = False
    candidates = []
    for subset, suffix in (
        ("running", "adaptive"),
        ("kicks", "adaptive"),
        ("female_running", "heldout"),
        ("general", "heldout"),
    ):
        for directory in (
            args.results_dir / f"{subset}_{suffix}",
            args.results_dir / f"{subset}_rho008",
        ):
            if not (directory / "accuracy.csv").exists():
                continue
            with (directory / "accuracy.csv").open() as stream:
                for row in csv.DictReader(stream):
                    if row["mode"] == "chunk_vs_original_gmr" and int(row["sqp_iterations"]) == 11:
                        candidates.append(dict(subset=subset, **row))
    if candidates:
        with (args.results_dir / "candidate_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
    with PdfPages(args.results_dir / "accuracy_sweep.pdf") as pdf:
        for mode, title, filename in (
            (
                "chunk_vs_original_gmr",
                "Maximum pose difference from original GMR/DAQP",
                "accuracy_sweep.png",
            ),
            (
                "chunk_vs_matched_daqp",
                "Maximum pose difference from DAQP at the same SQP policy",
                "accuracy_matched.png",
            ),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), layout="constrained")
            for row, subset in enumerate(("running", "kicks")):
                for col, policy in enumerate(("fixed", "adaptive")):
                    counts, keyed = tables[subset, policy]
                    values = np.array(
                        [
                            [
                                float(keyed[sqp, admm, mode]["q_tangent_max"])
                                for admm in (25, 50, 100)
                            ]
                            for sqp in counts
                        ]
                    )
                    ax = axes[row, col]
                    plot = ax.imshow(
                        np.maximum(values, 1e-7),
                        cmap="viridis",
                        norm=LogNorm(1e-6, 1.0),
                        aspect="auto",
                    )
                    for i in range(len(counts)):
                        for j in range(3):
                            ax.text(
                                j,
                                i,
                                f"{values[i,j]:.1e}",
                                ha="center",
                                va="center",
                                color="black" if values[i, j] > 0.015 else "white",
                                fontsize=6,
                            )
                    initial = int(keyed[counts[0], 25, mode]["initial_sqp_iterations"])
                    policy_title = "Fixed iterations" if policy == "fixed" else "GMR stopping rule"
                    ax.set_title(f"ACCAD {subset}: {policy_title} (initial {initial})")
                    ax.set_xticks(range(3), [25, 50, 100])
                    ax.set_yticks(range(len(counts)), counts)
                    ax.set_xlabel("ADMM iterations")
                    ax.set_ylabel(
                        "Subsequent SQP iterations/stage"
                        if policy == "fixed"
                        else "Subsequent SQP cap/stage"
                    )
                    ax.tick_params(which="both", length=0)
            fig.suptitle(title + "\n64-frame chunks; local RTX 5060 Ti; scaled QPs, rho = 0.1")
            fig.colorbar(plot, ax=axes, label="Maximum tangent difference (m / rad)", shrink=0.85)
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(args.results_dir / filename, dpi=250, bbox_inches="tight")
            plt.close(fig)
        if any(row["subset"] == "general" for row in candidates):
            fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
            for subset, label, marker in (
                ("running", "Male running", "o"),
                ("kicks", "Kicks", "s"),
                ("female_running", "Female running", "^"),
                ("general", "General motions", "D"),
            ):
                selected = sorted(
                    (r for r in candidates if r["subset"] == subset and float(r["rho"]) == 0.1),
                    key=lambda r: int(r["admm_iterations"]),
                )
                for ax, metric in zip(axes, ("q_tangent_max", "joint_limit_violation_rad")):
                    ax.plot(
                        [int(r["admm_iterations"]) for r in selected],
                        [max(1e-8, float(r[metric])) for r in selected],
                        marker=marker,
                        label=label,
                    )
                    ax.set_yscale("log")
                    ax.set_xticks([25, 50, 100])
                    ax.set_xlabel("ADMM iterations")
                    ax.grid(True, which="major", alpha=0.25)
            for tolerance in (1e-2, 1e-3):
                axes[0].axhline(tolerance, color="grey", linestyle="--", linewidth=0.7)
            axes[0].set_ylabel("Max. pose difference from GMR (m / rad)")
            axes[1].set_ylabel("Max. joint-limit violation (rad)")
            axes[0].legend(fontsize=6)
            fig.suptitle(
                "All 16 real ACCAD clips: GMR stopping rule, SQP cap 11/stage\n"
                "64-frame chunks; local RTX 5060 Ti; scaled QPs, rho = 0.1"
            )
            axes[1].text(
                0.02,
                0.02,
                "Zero violations displayed at $10^{-8}$",
                transform=axes[1].transAxes,
                fontsize=6,
            )
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(args.results_dir / "accuracy_candidates.png", dpi=250, bbox_inches="tight")
            plt.close(fig)
    for tolerance in (1e-2, 1e-3):
        candidates = []
        for policy in ("fixed", "adaptive"):
            for sqp in sorted(
                {row["sqp_iterations"] for row in summary if row["policy"] == policy}
            ):
                for admm in (25, 50, 100):
                    selected = [
                        r
                        for r in summary
                        if (r["policy"], r["sqp_iterations"], r["admm_iterations"])
                        == (policy, sqp, admm)
                    ]
                    worst = max(r["original_tangent_max"] for r in selected)
                    if len(selected) == 2 and worst <= tolerance:
                        work = (selected[0]["initial_sqp_iterations"] + 63 * sqp) * admm
                        candidates.append((work, policy, sqp, admm, worst))
        print(
            f"Settings meeting {tolerance:g} maximum pose difference on both subsets (iteration-count proxy first):"
        )
        for row in sorted(candidates):
            print(row)


if __name__ == "__main__":
    main()
