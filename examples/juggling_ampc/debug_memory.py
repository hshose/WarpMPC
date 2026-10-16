"""Full-batch allocation reproducer, with optional one-epoch training/export.

Compare XLA_PYTHON_CLIENT_ALLOCATOR=bfc and cuda_async in separate processes,
with XLA_PYTHON_CLIENT_PREALLOCATE=false. --plain preserves collector dispatch.
No robot access. Diagnostic checkpoints are not hardware-qualified policies.
"""
import bootstrap
import argparse,json,time,subprocess,os
from pathlib import Path
from types import SimpleNamespace
import jax
from gpu_dataset import GPUDataset
import fast_rollout

p=argparse.ArgumentParser();p.add_argument('--plain',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--root',type=Path,required=True);p.add_argument('--seconds',type=float,default=.06);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
print(json.dumps({'environment':{k:os.environ.get(k) for k in ('XLA_PYTHON_CLIENT_PREALLOCATE','XLA_PYTHON_CLIENT_ALLOCATOR','TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC','XLA_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_MEM_FRACTION')},'batch':60000,'seconds':a.seconds}),flush=True)
def memory(label):
    data=dict(label=label,time=time.time(),jax=jax.devices()[0].memory_stats(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip())
    print(json.dumps(data),flush=True)
    with (a.output/'memory.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
Original=fast_rollout.MPC
class Instrumented(Original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);original=self.solve
        def solve(*args):
            memory('before_solve')
            if hasattr(original,'lower') and not hasattr(self,'executable'):
                self.executable=original.lower(*args).compile()
                print('COMPILED_MEMORY',self.executable.memory_analysis(),flush=True)
                memory('after_compile')
            result=(self.executable if hasattr(self,'executable') else original)(*args)
            jax.block_until_ready(result);memory('after_solve')
            return result
        self.solve=solve
if not a.plain:fast_rollout.MPC=Instrumented
root=Path(__file__).resolve().parent
store=GPUDataset(60000,285,2026)
memory('dataset')
args=SimpleNamespace(factors=a.root/'gpu_factors',source=root/'reference',calibration=root/'calibration/model.json',batch=60000,seconds=a.seconds,seed=2026,multiplier=1.,arm_multiplier=1.5,policy=None,output=a.output/'collection',smoke=False,diagnostic=False,observer=None)
fast_rollout.run(args,sink=store)
memory('complete');print(json.dumps(store.summary()),flush=True)

if a.train:
    import gc
    from train_gpu import train
    from export_policy import export
    gc.collect();jax.clear_caches();memory('before_training')
    options=SimpleNamespace(seed=2026,train_batch=8192,epochs=1,learning_rate=.001)
    checkpoint=train(store,options,a.output/'policy')
    export(checkpoint,a.output/'policy/eigen');memory('after_export')
