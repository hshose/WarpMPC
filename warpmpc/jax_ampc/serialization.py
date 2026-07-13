"""Small serialization helpers shared by AMPC examples."""

from __future__ import annotations

import json
import math
import pathlib

import jax
import numpy as np
from flax import serialization


def jsonable(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
    return value


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def save_flax_checkpoint(
    *,
    output_dir: pathlib.Path,
    iteration: int,
    state,
    normalization,
    args,
    model_config: dict[str, object],
    metrics: dict[str, object],
) -> pathlib.Path:
    ckpt_dir = output_dir / "checkpoints" / f"iter_{iteration:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    params_host = jax.device_get(state.params)
    (ckpt_dir / "params.msgpack").write_bytes(serialization.to_bytes(params_host))
    norm_host = jax.device_get(normalization)
    np.savez(
        ckpt_dir / "normalization.npz",
        x_mean=np.asarray(norm_host.x_mean),
        x_std=np.asarray(norm_host.x_std),
        y_mean=np.asarray(norm_host.y_mean),
        y_std=np.asarray(norm_host.y_std),
        y_clip_low=np.asarray(norm_host.y_clip_low),
        y_clip_high=np.asarray(norm_host.y_clip_high),
        u_mean=np.asarray(norm_host.y_mean),
        u_std=np.asarray(norm_host.y_std),
        u_clip_low=np.asarray(norm_host.y_clip_low),
        u_clip_high=np.asarray(norm_host.y_clip_high),
    )
    payload = {
        "iteration": iteration,
        "model_config": model_config,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "metrics": metrics,
        "files": {
            "params": str(ckpt_dir / "params.msgpack"),
            "normalization": str(ckpt_dir / "normalization.npz"),
        },
    }
    write_json(ckpt_dir / "checkpoint.json", payload)
    return ckpt_dir


def save_dataset_increment(
    path: pathlib.Path,
    *,
    x,
    y,
    valid_mask,
    source_indices,
    iteration: int,
    dagger_beta: float,
    generated_samples: int,
    kept_samples: int,
    dataset_keep_fraction: float,
) -> None:
    """Save one generated dataset increment as an NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": np.asarray(jax.device_get(x)),
        "y": np.asarray(jax.device_get(y)),
        "valid_mask": np.asarray(jax.device_get(valid_mask)),
        "iteration": np.asarray(iteration, dtype=np.int32),
        "dagger_beta": np.asarray(dagger_beta, dtype=np.float32),
        "generated_samples": np.asarray(generated_samples, dtype=np.int64),
        "kept_samples": np.asarray(kept_samples, dtype=np.int64),
        "dataset_keep_fraction": np.asarray(dataset_keep_fraction, dtype=np.float32),
    }
    if source_indices is not None:
        payload["source_indices"] = np.asarray(jax.device_get(source_indices))
    np.savez(path, **payload)
