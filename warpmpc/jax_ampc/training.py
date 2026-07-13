"""Jitted supervised training kernels for AMPC imitation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import optax

from .normalization import Normalization, normalize_inputs, normalize_targets


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 100_000
    train_fraction: float = 0.8


def create_train_state(*, model, key, input_dim: int, dtype: np.dtype, learning_rate: float):
    jdtype = jnp.dtype(dtype)
    params = model.init(key, jnp.zeros((1, input_dim), dtype=jdtype))["params"]
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_training_kernels(model):
    def loss_sums(params, normalization: Normalization, x, y, sample_mask):
        x_norm = normalize_inputs(x, normalization)
        target_norm = normalize_targets(y, normalization)
        pred_norm = model.apply({"params": params}, x_norm)
        err_sq = (pred_norm - target_norm) ** 2
        weights = sample_mask.astype(err_sq.dtype)
        sse = jnp.sum(err_sq * weights[:, None])
        count = jnp.sum(weights) * err_sq.shape[1]
        loss = sse / jnp.maximum(count, jnp.asarray(1.0, dtype=err_sq.dtype))
        return loss, sse, count

    @jax.jit
    def train_step(state, normalization: Normalization, x, y, sample_mask):
        def loss_fn(params):
            loss, _, _ = loss_sums(params, normalization, x, y, sample_mask)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        _, sse, count = loss_sums(state.params, normalization, x, y, sample_mask)
        return state, {"loss": loss, "sse": sse, "count": count}

    @jax.jit
    def eval_loss_sums(params, normalization: Normalization, x, y, sample_mask):
        return loss_sums(params, normalization, x, y, sample_mask)

    return train_step, eval_loss_sums


def index_batch(order, start: int, batch_size: int):
    end = min(start + batch_size, int(order.shape[0]))
    idx = order[start:end]
    valid = np.ones((end - start,), dtype=bool)
    if end - start < batch_size:
        pad = batch_size - (end - start)
        repeats = int(math.ceil(pad / max(1, int(order.shape[0]))))
        pad_idx = jnp.tile(order, repeats)[:pad]
        idx = jnp.concatenate([idx, pad_idx], axis=0)
        valid = np.concatenate([valid, np.zeros((pad,), dtype=bool)], axis=0)
    return idx, jnp.asarray(valid)


def evaluate_masked_loss(
    *,
    eval_loss_sums,
    params,
    normalization: Normalization,
    dataset_x,
    dataset_y,
    sample_mask,
    batch_size: int,
) -> dict[str, float]:
    capacity = int(sample_mask.shape[0])
    if capacity == 0:
        return {"loss": float("nan"), "count": 0.0}
    order = jnp.arange(capacity, dtype=jnp.int32)
    sse_total = jnp.asarray(0.0, dtype=dataset_x.dtype)
    count_total = jnp.asarray(0.0, dtype=dataset_x.dtype)
    for start in range(0, capacity, batch_size):
        idx, pad_mask = index_batch(order, start, batch_size)
        mask_batch = sample_mask[idx] & pad_mask
        _, sse, count = eval_loss_sums(
            params,
            normalization,
            dataset_x[idx],
            dataset_y[idx],
            mask_batch,
        )
        sse_total = sse_total + sse
        count_total = count_total + count
    loss = sse_total / jnp.maximum(count_total, jnp.asarray(1.0, dtype=dataset_x.dtype))
    return {
        "loss": float(jax.device_get(loss)),
        "count": float(jax.device_get(count_total)),
    }


def train_supervised_policy(
    *,
    config: TrainingConfig,
    state: train_state.TrainState,
    normalization: Normalization,
    train_step,
    eval_loss_sums,
    dataset_x,
    dataset_y,
    dataset_mask,
    capacity: int,
    key,
    iteration: int,
    print_fn=print,
) -> tuple[train_state.TrainState, dict[str, object], object]:
    dataset_x_current = dataset_x[:capacity]
    dataset_y_current = dataset_y[:capacity]
    dataset_mask_current = dataset_mask[:capacity]
    key, split_key = jax.random.split(key)
    split_draw = jax.random.uniform(split_key, (capacity,), dtype=jnp.float32)
    train_mask = dataset_mask_current & (split_draw < config.train_fraction)
    test_mask = dataset_mask_current & (~train_mask)
    train_count = int(jax.device_get(jnp.sum(train_mask)))
    test_count = int(jax.device_get(jnp.sum(test_mask)))
    valid_count = int(jax.device_get(jnp.sum(dataset_mask_current)))
    print_fn(
        f"training iter={iteration}: valid={valid_count:,}, "
        f"train={train_count:,}, test={test_count:,}, epochs={config.epochs}",
        flush=True,
    )
    if train_count == 0:
        return state, {
            "valid_count": valid_count,
            "train_count": train_count,
            "test_count": test_count,
            "epochs": [],
        }, key

    epoch_metrics = []
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        key, perm_key = jax.random.split(key)
        order = jax.random.permutation(
            perm_key,
            jnp.arange(capacity, dtype=jnp.int32),
        )
        sse_total = jnp.asarray(0.0, dtype=dataset_x.dtype)
        count_total = jnp.asarray(0.0, dtype=dataset_x.dtype)
        n_batches = int(math.ceil(capacity / config.batch_size))
        for start in range(0, capacity, config.batch_size):
            idx, pad_mask = index_batch(order, start, config.batch_size)
            mask_batch = train_mask[idx] & pad_mask
            state, metrics = train_step(
                state,
                normalization,
                dataset_x_current[idx],
                dataset_y_current[idx],
                mask_batch,
            )
            sse_total = sse_total + metrics["sse"]
            count_total = count_total + metrics["count"]
        train_loss = sse_total / jnp.maximum(count_total, jnp.asarray(1.0, dtype=dataset_x.dtype))
        test_metrics = evaluate_masked_loss(
            eval_loss_sums=eval_loss_sums,
            params=state.params,
            normalization=normalization,
            dataset_x=dataset_x_current,
            dataset_y=dataset_y_current,
            sample_mask=test_mask,
            batch_size=config.batch_size,
        )
        elapsed = time.perf_counter() - epoch_start
        train_loss_host = float(jax.device_get(train_loss))
        epoch_payload = {
            "epoch": epoch,
            "train_loss": train_loss_host,
            "test_loss": test_metrics["loss"],
            "train_target_components": float(jax.device_get(count_total)),
            "test_target_components": test_metrics["count"],
            "batches": n_batches,
            "elapsed_s": elapsed,
        }
        epoch_metrics.append(epoch_payload)
        print_fn(
            f"  epoch {epoch + 1:02d}/{config.epochs}: "
            f"train_mse={train_loss_host:.5e}, "
            f"test_mse={test_metrics['loss']:.5e}, "
            f"batches={n_batches}, elapsed={elapsed:.2f}s",
            flush=True,
        )
    return state, {
        "valid_count": valid_count,
        "train_count": train_count,
        "test_count": test_count,
        "epochs": epoch_metrics,
    }, key
