"""Prepare large rollout shards without keeping duplicate datasets in RAM."""
from pathlib import Path
import json
import numpy as np
from dataset import split_trajectories


def output_baseline(stage_times):
    """Fixed affine baseline: extrapolate q with measured dq, retain dq and tau."""
    if len(stage_times)!=4:raise ValueError('Four native plan-node times required')
    matrix=np.zeros((40,84),np.float32)
    for k,time in enumerate(stage_times):
        for j in range(7):
            matrix[6+j,21*k+j]=1.;matrix[13+j,21*k+j]=time
            matrix[13+j,21*k+7+j]=1.;matrix[23+j,21*k+14+j]=1.
    return matrix


def prepare(dataset, cache, trajectories, seed, baseline=None):
    dataset=Path(dataset)
    manifest=json.loads((dataset/'manifest.json').read_text()) if (dataset/'manifest.json').exists() else {}
    shards=[(dataset / r['path'], int(r['trajectory_offset'])) for r in manifest['shards']] if 'shards' in manifest else [(p,0) for p in sorted(dataset.glob('shard_*.npz'))]
    if not shards:
        raise ValueError('No rollout shards')
    splits = split_trajectories(trajectories, seed)
    assignment = np.empty(trajectories, np.uint8)
    for code, name in enumerate(('train', 'validation', 'test')):
        assignment[splits[name]] = code
    total = count = 0
    mean = np.zeros(124, np.float64)
    m2 = np.zeros(124, np.float64)
    # Merge centered per-shard moments; validation/test never affect scaling.
    for path, trajectory_offset in shards:
        with np.load(path) as shard:
            x, y, ids = shard['inputs'], shard['targets'], shard['trajectory_id'] + trajectory_offset
            if x.ndim != 2 or y.ndim != 2 or x.shape != (len(ids), 40) or y.shape != (len(ids), 84):
                raise ValueError(f'Invalid sample dimensions: {path}')
            if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= trajectories)):
                raise ValueError(f'Invalid trajectory IDs: {path}')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'Nonfinite uncensored data: {path}')
            if baseline is not None:y=y.astype(np.float64)-x.astype(np.float64)@baseline
            total += len(ids)
            take = assignment[ids] == 0
            if take.any():
                values = np.concatenate((x[take], y[take]), axis=1).astype(np.float64)
                n = len(values)
                shard_mean = values.mean(axis=0)
                delta = shard_mean - mean
                m2 += np.sum((values - shard_mean) ** 2, axis=0) + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
    if not count:
        raise ValueError('No valid training rows')
    scale = np.maximum(np.sqrt(m2 / count), 1e-5)
    # Statistics accumulate in float64, but training and deployed inference
    # must use identical float32 constants and input normalization arithmetic.
    norm = dict(input_mean=mean[:40].astype(np.float32), input_scale=scale[:40].astype(np.float32),
                output_mean=mean[40:].astype(np.float32), output_scale=scale[40:].astype(np.float32))
    if baseline is not None:norm['output_baseline']=np.asarray(baseline,np.float32)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(cache / 'inputs.npy', mode='w+', dtype=np.float32, shape=(total, 40))
    y = np.lib.format.open_memmap(cache / 'targets.npy', mode='w+', dtype=np.float32, shape=(total, 84))
    partition = np.lib.format.open_memmap(cache / 'partition.npy', mode='w+', dtype=np.uint8, shape=(total,))
    offset = 0
    for path, trajectory_offset in shards:
        with np.load(path) as shard:
            n = len(shard['trajectory_id'])
            target = slice(offset, offset + n)
            x[target] = (shard['inputs'] - norm['input_mean']) / norm['input_scale']
            labels=shard['targets'].astype(np.float64)
            if baseline is not None:labels-=shard['inputs'].astype(np.float64)@baseline
            y[target] = (labels - norm['output_mean']) / norm['output_scale']
            partition[target] = assignment[shard['trajectory_id'] + trajectory_offset]
            offset += n
    x.flush(); y.flush(); partition.flush()
    masks = {name: partition == code for code, name in enumerate(('train', 'validation', 'test'))}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError('A whole-trajectory split has no valid rows')
    return x, y, masks, norm, splits
