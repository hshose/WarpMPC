"""Run the actual collector unchanged for a bounded allocator experiment."""
import bootstrap
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import jax
import warp as wp
from fast_rollout import run
from gpu_dataset import GPUDataset

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seconds', type=float, default=.10)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
batch = 60000
store = GPUDataset(batch, math.floor(3*2**30/(batch*47*4)))
print(json.dumps(dict(stage='original_collector_start', pid=os.getpid(),
    environment={key: os.environ.get(key) for key in (
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_ALLOCATOR')})), flush=True)
params = SimpleNamespace(factors=args.root/'results/juggling_reference/gpu_factors',
    source=args.root/'live_source', calibration=args.root/'live_source/runs/20261006_sim_quality/model.json',
    output=args.output/'collection', batch=batch, seconds=args.seconds, seed=2026,
    multiplier=1., arm_multiplier=1.5, policy=None)
run(params, sink=store)
result = dict(stage='original_collector_complete', dataset=store.summary(),
    cuda_free=wp.get_device('cuda:0').free_memory,
    cuda_total=wp.get_device('cuda:0').total_memory,
    jax=jax.devices()[0].memory_stats())
(args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
