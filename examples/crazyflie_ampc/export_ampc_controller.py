#!/usr/bin/env python3
"""Export a trained Crazyflie AMPC policy checkpoint to standalone C/C++ code."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from warpmpc.jax_ampc import (
    CodeExportOptions,
    QuantizationConfig,
    export_mlp_policy,
    load_mlp_checkpoint,
    make_random_test_cases,
    make_test_cases,
    sample_scaled_unit_ball_noise_np,
)


CRAZYFLIE_NX = 12
PROCESS_NOISE_SIGMA = 0.05
CRAZYFLIE_PROCESS_NOISE_SCALE = np.full((CRAZYFLIE_NX,), PROCESS_NOISE_SIGMA, dtype=np.float64)
CRAZYFLIE_PROCESS_NOISE_SCALE[3:6] /= 5.0
CRAZYFLIE_PROCESS_NOISE_SCALE[9:11] *= 100.0
CRAZYFLIE_INITIAL_STATE_NOISE_SCALE = 10.0 * CRAZYFLIE_PROCESS_NOISE_SCALE


def sample_initial_states(batch_size: int, seed: int, dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = sample_scaled_unit_ball_noise_np(
        rng,
        batch_size,
        CRAZYFLIE_INITIAL_STATE_NOISE_SCALE,
        dtype=dtype,
    )
    x0[:, :3] = rng.uniform(-1.0, 1.0, size=(batch_size, 3)).astype(dtype)
    x0[:, 5] = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=(batch_size,)).astype(dtype)
    return x0


def _latest_checkpoint(results_dir: pathlib.Path) -> pathlib.Path:
    checkpoint_root = results_dir / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"results directory has no checkpoints folder: {checkpoint_root}")
    candidates = sorted(path for path in checkpoint_root.glob("iter_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no iter_* checkpoint directories found in {checkpoint_root}")
    return candidates[-1]


def _load_npz_inputs(path: pathlib.Path, key: str, count: int) -> np.ndarray:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain key {key!r}; available keys: {sorted(data.files)}")
        x = np.asarray(data[key], dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape((-1, x.shape[-1]))
        if x.ndim != 2:
            raise ValueError(f"expected NPZ key {key!r} to contain a 2D array, got {x.shape}")
        if "valid_mask" in data:
            mask = np.asarray(data["valid_mask"], dtype=bool).reshape(-1)
            if mask.shape[0] == x.shape[0]:
                x = x[mask]
        if x.shape[0] < 1:
            raise ValueError(f"no usable input samples found in {path}")
        return x[:count]


def _numpy_dtype_from_precision(precision: str) -> np.dtype:
    if precision in ("float", "float32", "float32_t"):
        return np.dtype("float32")
    if precision in ("double", "float64"):
        return np.dtype("float64")
    raise ValueError(f"unsupported precision {precision!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        help="Checkpoint directory containing checkpoint.json, params.msgpack, and normalization.npz.",
    )
    source.add_argument(
        "--results-dir",
        type=pathlib.Path,
        help="Training results directory; the newest checkpoints/iter_* directory is exported.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for generated code. Defaults to <results>/code_export or <checkpoint>/code_export.",
    )
    parser.add_argument("--prefix", default="ampc_policy", help="C symbol/file prefix for generated code.")
    parser.add_argument(
        "--backend",
        choices=("none", "simple", "cmsis", "eigen"),
        default="simple",
        help="Optional forward-pass backend to generate in addition to data files.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float32_t", "float", "double", "float64"),
        default="float32_t",
        help="Scalar precision for generated code.",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "q31", "q15", "q7"),
        default="none",
        help="Post-training quantize exported weights/activations using the selected CMSIS-style dtype.",
    )
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-source",
        choices=("initial_distribution", "random", "npz"),
        default="initial_distribution",
        help="Where exported test inputs come from.",
    )
    parser.add_argument("--test-scale", type=float, default=1.0, help="Scale for random normalization-based tests.")
    parser.add_argument("--test-npz", type=pathlib.Path, default=None, help="NPZ file used when --test-source=npz.")
    parser.add_argument("--test-key", default="x", help="Input key inside --test-npz.")
    parser.add_argument("--test-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--generate-example-main",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate an example main for simple and Eigen backends.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else _latest_checkpoint(args.results_dir)
    if args.output_dir is None:
        base = args.results_dir if args.results_dir is not None else checkpoint_dir
        output_dir = base / "code_export"
    else:
        output_dir = args.output_dir

    spec = load_mlp_checkpoint(checkpoint_dir, name="crazyflie_ampc_policy")
    if args.test_source == "initial_distribution":
        inputs = sample_initial_states(args.test_count, args.seed, _numpy_dtype_from_precision(args.precision))
        tests = make_test_cases(spec, inputs)
    elif args.test_source == "npz":
        if args.test_npz is None:
            raise ValueError("--test-npz is required when --test-source=npz")
        tests = make_test_cases(spec, _load_npz_inputs(args.test_npz, args.test_key, args.test_count))
    else:
        tests = make_random_test_cases(spec, count=args.test_count, seed=args.seed, scale=args.test_scale)

    options = CodeExportOptions(
        prefix=args.prefix,
        backend=args.backend,
        precision=args.precision,
        generate_example_main=args.generate_example_main,
        test_tolerance=args.test_tolerance,
        quantization=None if args.quantize == "none" else QuantizationConfig(qtype=args.quantize),
    )
    written = export_mlp_policy(spec, output_dir, test_cases=tests, options=options)

    print("Crazyflie AMPC code export")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  output:     {output_dir}")
    print(f"  backend:    {args.backend}")
    print(f"  quantize:   {args.quantize}")
    print(f"  test cases: {tests.count} ({args.test_source})")
    for key, path in sorted(written.items()):
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
