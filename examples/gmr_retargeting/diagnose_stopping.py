#!/usr/bin/env python3
"""Compare GPU and original GMR stopping decisions on a saved chunk run."""

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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import mink
import mujoco
import numpy as np

from examples.gmr_retargeting.model import RetargetModel, SolverSettings, load_gmr
from examples.gmr_retargeting.reporting import write_json


def cpu_decreases(model, human, seed, expected):
    """Replay one original-GMR frame and record its exact stopping inputs."""
    original = mink.solve_ik
    history = [[], []]

    def record(configuration, tasks, dt, *args, **kwargs):
        stage = 0 if tasks is model.gmr.tasks1 else 1
        before = np.linalg.norm(
            np.concatenate([task.compute_error(configuration) for task in tasks])
        )
        velocity = original(configuration, tasks, dt, *args, **kwargs)
        saved = configuration.q.copy()
        candidate = saved.copy()
        mujoco.mj_integratePos(model.mj, candidate, velocity, dt)
        try:
            configuration.update(candidate)
            after = np.linalg.norm(
                np.concatenate([task.compute_error(configuration) for task in tasks])
            )
        finally:
            configuration.update(saved)
        history[stage].append(float(before - after))
        return velocity

    model.gmr.configuration.update(seed)
    targets = {name: (human[j, :3], human[j, 3:]) for j, name in enumerate(model.human_names)}
    try:
        mink.solve_ik = record
        result = model.gmr.retarget(targets)
    finally:
        mink.solve_ik = original
    np.testing.assert_allclose(result, expected, atol=2e-10, rtol=0)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=pathlib.Path, required=True)
    parser.add_argument("--counts-dir", type=pathlib.Path, required=True)
    parser.add_argument("--admm-iterations", type=int, required=True)
    parser.add_argument("--sqp-iterations", type=int, default=11)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--gmr-root", type=pathlib.Path, default=ROOT / "resources/GMR")
    args = parser.parse_args()
    meta = json.loads((args.results_dir / "metadata.json").read_text())
    model = RetargetModel(load_gmr(args.gmr_root, meta["height"]))
    initial = meta["initial_sqp_iterations"]
    cold, _ = model.compile(
        SolverSettings(initial, args.admm_iterations, args.rho, adaptive_stopping=True),
        diagnostics=True,
    )
    warm, _ = model.compile(
        SolverSettings(args.sqp_iterations, args.admm_iterations, args.rho, adaptive_stopping=True),
        diagnostics=True,
    )
    with np.load(args.results_dir / "reference.npz") as data:
        human, seeds, reference = data["human"], data["seeds"], data["qpos"]
        bank = jnp.asarray(
            data["human"].reshape(meta["chunks"], meta["chunk_size"], *data["human"].shape[1:]),
            dtype=jnp.float32,
        )

    @jax.jit
    def run(bank):
        q0 = jnp.broadcast_to(jnp.asarray(model.mj.qpos0, dtype=jnp.float32), (len(bank), model.nq))
        first = cold(q0, model.preprocess(bank[:, 0]))

        def step(q, t):
            result = warm(q, model.preprocess(bank[:, t]))
            return result[0], result

        _, rest = jax.lax.scan(step, first[0], jnp.arange(1, bank.shape[1]))
        return tuple(
            jnp.concatenate((a[None], b), axis=0).transpose(1, 0, 2).reshape(-1, a.shape[-1])
            for a, b in zip(first, rest)
        )

    q, counts, decreases = [np.asarray(x) for x in jax.block_until_ready(run(bank))]
    with np.load(
        args.results_dir
        / f"chunks_adaptive_initial{initial}_sqp{args.sqp_iterations}_admm{args.admm_iterations}.npz"
    ) as data:
        saved = data["gpu"]
    # The diagnostic path must describe the actual saved numerical trajectory.
    np.testing.assert_array_equal(q, saved)
    cpu_rows = list(csv.DictReader((args.counts_dir / "iterations.csv").open()))
    differences = []
    histories = {}
    for row in cpu_rows:
        frame, stage = int(row["frame"]), int(row["stage"]) - 1
        if counts[frame, stage] != int(row["iterations"]):
            if frame not in histories:
                histories[frame] = cpu_decreases(
                    model, human[frame], seeds[frame], reference[frame]
                )
            differences.append(
                dict(
                    frame=frame,
                    stage=stage + 1,
                    cpu_iterations=int(row["iterations"]),
                    gpu_iterations=int(counts[frame, stage]),
                    cpu_last_error_decrease=float(row["last_error_decrease"]),
                    gpu_last_error_decrease=float(decreases[frame, stage]),
                    cpu_error_decreases=histories[frame][stage],
                )
            )
    tangent = np.empty((len(q), model.nv))
    for i in range(len(q)):
        mujoco.mj_differentiatePos(model.mj, tangent[i], 1.0, reference[i], q[i].astype(np.float64))
    frame, coordinate = np.unravel_index(np.argmax(np.abs(tangent)), tangent.shape)
    report = dict(
        admm_iterations=args.admm_iterations,
        diagnostic_trajectory_bitwise_equal=True,
        maximum_pose_difference=dict(
            frame=int(frame),
            coordinate=int(coordinate),
            value=float(abs(tangent[frame, coordinate])),
        ),
        differing_decisions=differences,
    )
    write_json(args.results_dir / f"stopping_admm{args.admm_iterations}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
