#!/usr/bin/env python3
"""Measure original GMR's SQP counts on a saved CPU reference trajectory."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mink
import mujoco
import numpy as np

from examples.gmr_retargeting.model import RetargetModel, load_gmr
from examples.gmr_retargeting.reporting import write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-root", type=pathlib.Path, default=ROOT / "resources/GMR")
    args = parser.parse_args()
    metadata = json.loads((args.reference_dir / "metadata.json").read_text())
    model = RetargetModel(load_gmr(args.gmr_root, metadata["height"]))
    with np.load(args.reference_dir / "reference.npz", allow_pickle=False) as data:
        data = dict(data)
    starts = set(map(int, data["starts"]))
    original = mink.solve_ik
    rows = []
    for i, seed in enumerate(data["seeds"]):
        counts = [0, 0]
        last_decreases = [0.0, 0.0]

        def counted(configuration, tasks, *args, **kwargs):
            stage = 0 if tasks is model.gmr.tasks1 else 1
            counts[stage] += 1
            velocity = original(configuration, tasks, *args, **kwargs)
            target = model.targets()[stage]
            active = np.repeat(model.active[stage], 6)
            before = np.linalg.norm(
                np.asarray(model.error_function(configuration.q, target)).ravel() * active
            )
            next_q = configuration.q.copy()
            mujoco.mj_integratePos(model.mj, next_q, velocity, args[0])
            after = np.linalg.norm(
                np.asarray(model.error_function(next_q, target)).ravel() * active
            )
            last_decreases[stage] = float(before - after)
            return velocity

        human = {
            name: (data["human"][i, j, :3], data["human"][i, j, 3:])
            for j, name in enumerate(model.human_names)
        }
        model.gmr.configuration.update(seed)
        try:
            mink.solve_ik = counted
            result = model.gmr.retarget(human)
        finally:
            mink.solve_ik = original
        np.testing.assert_allclose(result, data["qpos"][i], atol=2e-10, rtol=0)
        for stage, count in enumerate(counts, start=1):
            rows.append(
                dict(
                    frame=i,
                    stage=stage,
                    cold=i in starts,
                    iterations=count,
                    last_error_decrease=last_decreases[stage - 1],
                )
            )
    summary = []
    for cold in (True, False):
        for stage in (1, 2):
            values = np.array(
                [r["iterations"] for r in rows if r["cold"] == cold and r["stage"] == stage]
            )
            if not len(values):
                continue
            summary.append(
                dict(
                    cold=cold,
                    stage=stage,
                    frames=len(values),
                    minimum=int(values.min()),
                    median=float(np.median(values)),
                    p95=float(np.quantile(values, 0.95)),
                    maximum=int(values.max()),
                    mean=float(values.mean()),
                    histogram={str(i): int(np.sum(values == i)) for i in np.unique(values)},
                )
            )
    write_csv(args.output_dir / "iterations.csv", rows)
    write_json(args.output_dir / "iterations_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
