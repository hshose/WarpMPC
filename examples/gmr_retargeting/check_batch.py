#!/usr/bin/env python3
"""Check all poses at throughput batch size against the small reference pool."""

from __future__ import annotations

import argparse
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
from examples.gmr_retargeting.model import RetargetModel, SolverSettings, load_gmr
from examples.gmr_retargeting.reporting import provenance, write_json
from examples.gmr_retargeting.validate import metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=pathlib.Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Batch parity requires a GPU")
    meta = json.loads((args.reference_dir / "metadata.json").read_text())
    if args.batch_size < meta["chunks"]:
        p.error("Batch must contain every reference chunk")
    model = RetargetModel(load_gmr(ROOT / "resources/GMR", meta["height"]))
    with np.load(args.reference_dir / "reference.npz") as data:
        human, cpu = data["human"], data["qpos"]
    with np.load(
        args.reference_dir / "chunks_independent_adaptive_initial11_sqp11_admm50.npz"
    ) as data:
        expected = data["gpu"].reshape(meta["chunks"], meta["chunk_size"], model.nq)
    bank = jnp.asarray(
        human.reshape(meta["chunks"], meta["chunk_size"], *human.shape[1:]), dtype=jnp.float32
    )
    indices = np.arange(args.batch_size, dtype=np.int32) % len(bank)
    run = compile_async_chunks(
        model, SolverSettings(11, 50, 0.1, adaptive_stopping=True), 11, collect=True
    )
    result = jax.block_until_ready(run(bank, jnp.asarray(indices)))
    actual = np.asarray(result[0])
    if not bool(result[1]):
        raise FloatingPointError("Nonfinite batch trajectory")
    # Check lane indexing, stage transitions and termination on every pose.
    # Comparing one lane at a time avoids another batch*frames allocation.
    for lane, source in enumerate(indices):
        np.testing.assert_array_equal(actual[lane], actual[source])
    selected = actual[: len(bank)].reshape(-1, model.nq)
    bitwise = np.array_equal(actual[: len(bank)], expected)
    report = dict(
        provenance=provenance(),
        batch_size=args.batch_size,
        chunk_size=meta["chunk_size"],
        total_compared_frames=args.batch_size * meta["chunk_size"],
        repeated_lanes_bitwise_equal=True,
        small_pool_bitwise_equal=bool(bitwise),
        versus_small_pool=metrics(model, selected, expected.reshape(-1, model.nq)),
        versus_original_gmr=metrics(model, selected, cpu),
        schedule={
            key: np.asarray(value).item() for key, value in result[3].items() if np.ndim(value) == 0
        },
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
