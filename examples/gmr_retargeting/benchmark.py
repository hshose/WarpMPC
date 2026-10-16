#!/usr/bin/env python3
"""Retargeting throughput with frame-synchronous or independent lane progress."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-gmr-cache")

DEFAULT_BATCHES = "512,2048,10000,20000,50000,100000,200000,300000"


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from examples.gmr_retargeting.model import RetargetModel, SolverSettings, load_gmr
    from examples.gmr_retargeting.chunks import chunk_indices, compile_chunks

    if args.scheduler == "independent":
        from examples.gmr_retargeting.async_chunks import compile_async_chunks as compile_chunks
    from examples.gmr_retargeting.reporting import provenance, write_json

    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Throughput benchmark requires a GPU")
    metadata = json.loads((args.reference_dir / "metadata.json").read_text())
    if not metadata["source"].get("amass") and not args.allow_synthetic:
        raise ValueError(
            "Paper throughput requires AMASS; --allow-synthetic is only for smoke testing"
        )
    model = RetargetModel(load_gmr(args.gmr_root, metadata["height"]))
    settings = SolverSettings(
        args.sqp_iterations,
        args.admm_iterations,
        args.rho,
        backend=args.backend,
        derivative_backend=args.derivative_backend,
        adaptive_stopping=args.adaptive_stopping,
    )
    with np.load(args.reference_dir / "reference.npz") as data:
        np.testing.assert_array_equal(data["weights"], model.weights)
        if data["seeds"].shape[1] != model.nq:
            raise ValueError("Reference robot dimension differs from this model")
        chunks = chunk_indices(len(data["human"]), data["starts"], args.chunk_size)
        if metadata.get("chunk_size") != args.chunk_size:
            raise ValueError(
                "CPU reference must use the same chunk boundaries; rerun validate.py --chunk-size"
            )
        bank = jnp.asarray(data["human"][chunks], dtype=jnp.float32)
        lanes = jnp.arange(args.worker_batch, dtype=jnp.int32) % len(chunks)
    begin = time.perf_counter()
    complete = compile_chunks(model, settings, args.initial_sqp_iterations)
    executable = complete.lower(bank, lanes).compile()
    compile_s = time.perf_counter() - begin
    print(
        f"Compiled {args.scheduler}, batch={args.worker_batch}, frames={args.chunk_size} in {compile_s:.2f}s",
        flush=True,
    )
    for i in range(args.warmups):
        print(f"Warmup {i+1}/{args.warmups}", flush=True)
        result = jax.block_until_ready(executable(bank, lanes))
    timings = []
    for i in range(args.repeats):
        print(f"Measurement {i+1}/{args.repeats}", flush=True)
        begin = time.perf_counter()
        result = jax.block_until_ready(executable(bank, lanes))
        timings.append(time.perf_counter() - begin)
        print(f"Elapsed: {timings[-1]:.6f}s", flush=True)
    # Host copy/check is deliberately outside the timed region.
    values = np.asarray(result[0])
    if not bool(result[1]) or not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite retargeted configurations")
    elapsed = float(np.median(timings))
    memory = jax.devices()[0].memory_stats() or {}
    row = (
        dataclasses.asdict(settings)
        | provenance()
        | {
            "status": "ok",
            "batch_size": args.worker_batch,
            "elapsed_s": elapsed,
            "frames_per_s": args.worker_batch * args.chunk_size / elapsed,
            "chunks_per_s": args.worker_batch / elapsed,
            "timings_s": timings,
            "compile_s": compile_s,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "chunk_size": args.chunk_size,
            "initial_sqp_iterations": args.initial_sqp_iterations,
            "scheduler": args.scheduler,
            "total_frames": args.worker_batch * args.chunk_size,
            "joint_limit_violation_rad": float(result[2]),
            "peak_bytes_in_use": memory.get("peak_bytes_in_use"),
            "pool_size": len(chunks),
            "dtype": "float32",
            "nnz_p": model.pattern.nnz,
            "unique_chunks_used": min(args.worker_batch, len(chunks)),
            "mean_copies_per_input_chunk": args.worker_batch / len(chunks),
            "input_bank_bytes": int(bank.size * bank.dtype.itemsize),
            "pool_sampling": "Deterministic cycling over the real, nonoverlapping chunk bank; no claim of batch_size unique recordings",
            "timing_scope": "Device human target scaling/offsets + both IK stages, including FK, linearization, factorization, ADMM, integration; excludes setup/transfers",
            "initialization": "Independent chunks reset to model default; subsequent frames use preceding GPU pose",
            "source": metadata["source"],
            "reference_dir": str(args.reference_dir.resolve()),
            "reference_source_sha256": metadata.get("source_sha256", {}),
        }
    )
    if args.scheduler == "independent":
        row.update({key: np.asarray(value).item() for key, value in result[3].items()})
    else:
        row["rounds"] = 2 * (
            args.initial_sqp_iterations + (args.chunk_size - 1) * args.sqp_iterations
        )
    write_json(args.output_dir / f"batch_{args.worker_batch}.json", row)
    print(json.dumps(row), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=pathlib.Path, required=True)
    p.add_argument("--gmr-root", type=pathlib.Path, default=ROOT / "resources/GMR")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--batch-sizes", default=DEFAULT_BATCHES)
    p.add_argument(
        "--sqp-iterations", type=int, required=True, help="Per stage; select after accuracy review"
    )
    p.add_argument("--admm-iterations", type=int, required=True)
    p.add_argument("--initial-sqp-iterations", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--backend", choices=["warp", "jax"], default="warp")
    p.add_argument("--derivative-backend", choices=["warp", "jax"], default="warp")
    p.add_argument("--adaptive-stopping", action="store_true")
    p.add_argument("--scheduler", choices=["frame", "independent"], default="frame")
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--worker-batch", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if (
        min(
            args.warmups,
            args.repeats,
            args.sqp_iterations,
            args.admm_iterations,
            args.initial_sqp_iterations,
            args.chunk_size,
        )
        < 1
    ):
        p.error("Iteration counts, warmups and repeats must be positive")
    if args.scheduler == "independent" and not args.adaptive_stopping:
        p.error("--scheduler independent requires --adaptive-stopping")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_batch is not None:
        worker(args)
        return
    from examples.gmr_retargeting.reporting import write_csv

    rows = []
    for batch in map(int, args.batch_sizes.split(",")):
        if not 1 <= batch <= 300_000:
            p.error("batch sizes must be in [1, 300000]")
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker-batch",
            str(batch),
        ]
        log = args.output_dir / f"batch_{batch}.log"
        result_path = args.output_dir / f"batch_{batch}.json"
        # Avoid accidentally reading stale success after a failed repeat.
        result_path.unlink(missing_ok=True)
        print(f"Running batch={batch}; log={log}", flush=True)
        with log.open("w") as stream:
            try:
                proc = subprocess.run(
                    cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout
                )
                if proc.returncode == 0 and result_path.exists():
                    row = json.loads(result_path.read_text())
                else:
                    tail = log.read_text()[-3000:]
                    status = (
                        "oom"
                        if any(
                            x in tail.lower()
                            for x in (
                                "out of memory",
                                "resource_exhausted",
                                "cuda_error_out_of_memory",
                            )
                        )
                        else "failed"
                    )
                    row = {
                        "batch_size": batch,
                        "status": status,
                        "returncode": proc.returncode,
                        "error": tail,
                    }
            except subprocess.TimeoutExpired:
                row = {"batch_size": batch, "status": "timeout"}
        rows.append(row)
        write_csv(args.output_dir / "throughput.csv", rows)
        print(f"batch={batch}: {row['status']} {row.get('frames_per_s', '')}", flush=True)
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit("Some batch sizes failed; see throughput.csv and per-batch logs")


if __name__ == "__main__":
    main()
