#!/usr/bin/env python3
"""Compare every frame of independent lanes with the synchronous GPU and GMR."""

from __future__ import annotations

import argparse
import dataclasses
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
import numpy as np

from examples.gmr_retargeting.async_chunks import compile_async_chunks
from examples.gmr_retargeting.chunks import compile_chunks
from examples.gmr_retargeting.model import RetargetModel, SolverSettings, load_gmr
from examples.gmr_retargeting.reporting import provenance, write_json
from examples.gmr_retargeting.validate import metrics, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-root", type=pathlib.Path, default=ROOT / "resources/GMR")
    parser.add_argument("--admm-iterations", type=int, default=50)
    parser.add_argument("--rho", type=float, default=0.1)
    args = parser.parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Scheduler parity requires the local GPU")
    meta = json.loads((args.reference_dir / "metadata.json").read_text())
    with np.load(args.reference_dir / "reference.npz", allow_pickle=False) as data:
        human, cpu = data["human"], data["qpos"]
    model = RetargetModel(load_gmr(args.gmr_root, meta["height"]))
    bank = jnp.asarray(
        human.reshape(meta["chunks"], meta["chunk_size"], *human.shape[1:]), dtype=jnp.float32
    )
    lanes = jnp.arange(len(bank), dtype=jnp.int32)
    settings = SolverSettings(11, args.admm_iterations, args.rho, adaptive_stopping=True)
    baseline = compile_chunks(model, settings, 11, collect=True)
    independent = compile_async_chunks(model, settings, 11, collect=True)
    synchronous = jax.block_until_ready(baseline(bank, lanes))
    asynchronous = jax.block_until_ready(independent(bank, lanes))
    frame_q, independent_q = [
        np.asarray(result[0]).reshape(-1, model.nq) for result in (synchronous, asynchronous)
    ]
    if not bool(synchronous[1]) or not bool(asynchronous[1]):
        raise FloatingPointError("Nonfinite scheduler trajectory")
    report = dict(
        settings=dataclasses.asdict(settings),
        provenance=provenance(),
        source_sha256=source_hashes(args.gmr_root),
        reference_dir=str(args.reference_dir.resolve()),
        frames=len(cpu),
        chunks=meta["chunks"],
        chunk_size=meta["chunk_size"],
        bitwise_equal=bool(np.array_equal(frame_q, independent_q)),
        independent_vs_frame=metrics(model, independent_q, frame_q),
        independent_vs_original_gmr=metrics(model, independent_q, cpu),
        schedule={key: np.asarray(value).tolist() for key, value in asynchronous[3].items()},
    )
    write_json(args.output_dir / "comparison.json", report)
    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        frame=frame_q,
        independent=independent_q,
        original_gmr=cpu,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("source_sha256", "schedule")
            }
        ),
        flush=True,
    )
    # Scheduling alone must preserve the existing numerical trajectory on this GPU.
    np.testing.assert_array_equal(independent_q, frame_q)


if __name__ == "__main__":
    main()
