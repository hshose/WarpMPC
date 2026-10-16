"""Exercise the full 60k Newton + MPC + GPU reservoir footprint before the sweep."""
import bootstrap
import argparse,json
from types import SimpleNamespace
from gpu_dataset import GPUDataset
from rollout import run
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--multiplier',type=float,default=1.5);p.add_argument('--arm-multiplier',type=float);a=p.parse_args();r=a.root
store=GPUDataset(60000,285)
args=SimpleNamespace(factors=r/'results/juggling_reference/gpu_factors',observer=r/'results/observer_claix/libjuggling_batch_observer.so',
 source=r/'live_source',calibration=r/'live_source/runs/20261006_sim_quality/model.json',output=r/'results/full_batch_memory_smoke',
 batch=60000,seconds=.04,seed=2026,multiplier=a.multiplier,arm_multiplier=a.arm_multiplier,shard_rows=1000000,policy=None,collect=True,smoke=True,
 diagnostic=False,native_expert=False,native_workers=1,native_compiled=False,expert_qualification=None)
report=run(args,sink=store)
print(json.dumps(dict(full_batch_memory_passed=True,dataset=store.summary(),timing=report['timing'])),flush=True)
