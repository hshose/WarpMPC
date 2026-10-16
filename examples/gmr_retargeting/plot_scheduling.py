#!/usr/bin/env python3
"""Plot actual lane progress, final idle work, and measured local throughput."""

from __future__ import annotations

import argparse
import csv
import json
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
    from matplotlib.backends.backend_pdf import PdfPages

    _apply_paper_rc(plt)
    if args.no_tex:
        plt.rcParams["text.usetex"] = False
    with PdfPages(args.results_dir / "independent_scheduling.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
        for ax, folder, title in zip(
            axes, ("dancedb_2048", "bmlrub_512"), ("DanceDB: 2,048 frames", "BMLrub: 512 frames")
        ):
            stats = json.loads(
                (args.results_dir / folder / "schedule_sqp11_admm50.json").read_text()
            )
            completion = np.array(stats["frame_completion_rounds"])
            for lane, events in enumerate(completion):
                ax.plot(np.r_[0, events], np.arange(len(events) + 1), label=f"Trajectory {lane+1}")
                ax.plot(
                    [events[-1], stats["rounds"]], [len(events)] * 2, color="grey", linestyle="--"
                )
            ax.set(
                title=(
                    f"{title}\nFinal idle slots: {100*stats['end_idle_fraction']:.2f}\\%"
                    if not args.no_tex
                    else f"{title}\nFinal idle slots: {100*stats['end_idle_fraction']:.2f}%"
                ),
                xlabel="Batched QP round",
                ylabel="Completed frames",
            )
            ax.legend(fontsize=6)
            ax.grid(alpha=0.25)
        fig.suptitle("Independent trajectory progress; GMR stopping, 50 ADMM iterations")
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(args.results_dir / "independent_progress.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

        table = args.results_dir / "throughput_summary.csv"
        if table.exists():
            rows = list(csv.DictReader(table.open()))
            fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
            for ax, dataset in zip(axes, ("dancedb_2048", "bmlrub_512")):
                title = {
                    "dancedb_2048": "DanceDB: 2,048 frames",
                    "bmlrub_512": "BMLrub: 512 frames",
                }[dataset]
                selected = [r for r in rows if r["dataset"] == dataset and r["status"] == "ok"]
                batches = sorted({int(r["batch_size"]) for r in selected})
                if len(batches) == 1:
                    selected.sort(key=lambda r: r["scheduler"] == "independent")
                    bars = ax.bar(
                        range(len(selected)),
                        [float(r["frames_per_s"]) for r in selected],
                        color=[
                            "#1f77b4" if r["scheduler"] == "independent" else "#888888"
                            for r in selected
                        ],
                    )
                    ax.set_xticks(
                        range(len(selected)),
                        [
                            (
                                "Independent lanes"
                                if r["scheduler"] == "independent"
                                else "Frame-synchronous"
                            )
                            for r in selected
                        ],
                    )
                    ax.bar_label(
                        bars,
                        labels=[f"{float(r['frames_per_s']):,.0f}" for r in selected],
                        padding=3,
                        fontsize=7,
                    )
                    ax.set(
                        title=title + f"; batch {batches[0]:,}",
                        ylabel="Retargeted frames/s",
                        ylim=(0, 1.2 * max(float(r["frames_per_s"]) for r in selected)),
                    )
                    ax.grid(axis="y", alpha=0.25)
                    continue
                for scheduler, label, marker in (
                    ("frame", "Frame-synchronous", "s"),
                    ("independent", "Independent lanes", "o"),
                ):
                    group = sorted(
                        (r for r in selected if r["scheduler"] == scheduler),
                        key=lambda r: int(r["batch_size"]),
                    )
                    if not group:
                        continue
                    ax.plot(
                        [int(r["batch_size"]) for r in group],
                        [float(r["frames_per_s"]) for r in group],
                        marker=marker,
                        label=label,
                    )
                ax.set(title=title, xlabel="Parallel trajectories", ylabel="Retargeted frames/s")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=6)
            fig.suptitle(
                "Local RTX 5060 Ti throughput; 50 ADMM iterations\nOne timed complete rollout after warmup per configuration"
            )
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(
                args.results_dir / "independent_throughput.png", dpi=250, bbox_inches="tight"
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
