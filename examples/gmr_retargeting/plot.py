#!/usr/bin/env python3
"""Paper-style GMR throughput plot with the original DAQP baseline."""

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=pathlib.Path, required=True)
    p.add_argument(
        "--reference-dir",
        type=pathlib.Path,
        required=True,
        help="CPU GMR baseline measured on the same host as throughput",
    )
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--title", help="Optional figure title identifying the motion subset")
    p.add_argument("--no-tex", action="store_true")
    args = p.parse_args()
    from benchmarks.linear_mpc_gradients.plot_style import (
        OURS_BLUE_DARK, PAPER_LINE_ALPHA, _apply_paper_rc,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    meta = json.loads((args.reference_dir / "metadata.json").read_text())
    with (args.results_dir / "throughput.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    if not rows:
        raise ValueError("No successful throughput measurements")
    if any(r["host"] != meta["provenance"]["host"] for r in rows):
        raise ValueError("CPU baseline and GPU measurements must come from the same host")
    _apply_paper_rc(plt)
    if args.no_tex:
        plt.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize=(3.4, 1.55))

    def variant(row):
        adaptive = str(row.get("adaptive_stopping", "false")).lower() in ("true", "1")
        return (
            int(row["sqp_iterations"]),
            int(row["admm_iterations"]),
            adaptive,
            row.get("scheduler", "frame"),
        )

    variants = sorted({variant(r) for r in rows})
    for i, (sqp, admm, adaptive, scheduler) in enumerate(variants):
        group = sorted(
            (r for r in rows if variant(r) == (sqp, admm, adaptive, scheduler)),
            key=lambda r: int(r["batch_size"]),
        )
        policy = "SQP cap/stage" if adaptive else "SQP/stage"
        initial = int(group[0].get("initial_sqp_iterations", sqp))
        schedule = str(sqp) if initial == sqp else f"{initial} then {sqp}"
        ax.plot(
            [int(r["batch_size"]) for r in group],
            [float(r["frames_per_s"]) for r in group],
            color=OURS_BLUE_DARK,
            marker=["X", "o", ">"][i % 3],
            markersize=2,
            markeredgewidth=0.3,
            linewidth=0.7,
            alpha=PAPER_LINE_ALPHA,
            label=f"WarpMPC ({schedule} {policy}, {admm} ADMM)",
        )
    baseline = (
        meta["cpu_frames_per_s"]
        if meta.get("chunk_size")
        else (meta.get("cpu_warm_frames_per_s") or meta["cpu_frames_per_s"])
    )
    ax.axhline(
        baseline,
        color="#D55E00",
        linestyle="--",
        linewidth=0.8,
        label="Original GMR (CPU, DAQP)",
    )
    title = (
        "Motion retargeting (G1)"
        if meta["source"].get("amass")
        else "Motion retargeting (G1, synthetic test)"
    )
    if args.title:
        title = args.title if meta["source"].get("amass") else f"{args.title} (synthetic test)"
    xlabel = (
        f"Parallel chunks ({meta['chunk_size']} frames each)"
        if meta.get("chunk_size")
        else "Batch size"
    )
    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="Retargeted frames/s", title=title)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
    ax.grid(True, which="major", linewidth=0.3, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    if args.output.suffix.lower() != ".png":
        fig.savefig(args.output.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
