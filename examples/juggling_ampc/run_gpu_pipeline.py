"""One allocation/process: full-batch Newton collection -> VRAM training -> evaluation."""
import bootstrap
import argparse,gc,json,math,time,contextlib,os,hashlib
from pathlib import Path
from types import SimpleNamespace
import jax
from gpu_dataset import GPUDataset
from fast_rollout import run
from train_gpu import train
from export_policy import export
from compare_rollouts import compare


def main(a):
    if a.batch!=60000 and not a.smoke:raise ValueError('Production collection always uses all 60,000 worlds')
    if a.seconds not in (5,10,20):raise ValueError('Collection sweep is 5, 10, 20 seconds')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'run_configuration.json').write_text(json.dumps(dict(arguments=vars(a),
        environment={k:os.environ.get(k) for k in ('XLA_PYTHON_CLIENT_PREALLOCATE','XLA_PYTHON_CLIENT_ALLOCATOR','TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC','XLA_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_MEM_FRACTION','JAX_COMPILATION_CACHE_DIR','WARP_CACHE_PATH','CUDA_CACHE_PATH','SLURM_JOB_ID')},
        factor_metadata_sha256=hashlib.sha256((a.factors/'metadata.json').read_bytes()).hexdigest()),indent=2,default=str)+'\n')
    @contextlib.contextmanager
    def timing(name):
        start=time.perf_counter_ns();wall=time.time_ns();status='failed'
        try:
            yield
            jax.effects_barrier();status='complete'
        finally:
            record=dict(phase=name,start_ns=start,end_ns=time.perf_counter_ns(),start_unix_ns=wall,status=status,includes_compilation=True)
            with (a.output/'pipeline_timings.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    slots=min(round(a.seconds/.02),math.floor(a.dataset_gib*2**30/(a.batch*47*4)))
    if slots<1:raise ValueError('VRAM budget must retain at least one sample per trajectory')
    store=GPUDataset(a.batch,slots,a.seed)
    common=dict(factors=a.factors,observer=a.observer,calibration=a.calibration,source=a.source,batch=a.batch,seconds=a.seconds,seed=a.seed,
                multiplier=a.collection_multiplier,arm_multiplier=a.collection_arm_multiplier,shard_rows=1000000,policy=None,collect=True,smoke=a.smoke,diagnostic=False,native_expert=False,native_workers=1,native_compiled=False,expert_qualification=a.expert_qualification)
    with timing('collection'):run(SimpleNamespace(**common,output=a.output/'collection'),sink=store)
    print(json.dumps(store.summary()),flush=True)
    (a.output/'dataset_summary.json').write_text(json.dumps(store.summary(),indent=2)+'\n')
    gc.collect();jax.clear_caches()
    with timing('training'):checkpoint=train(store,a,a.output/'policy')
    del store;gc.collect();jax.clear_caches()
    with timing('export'):export(checkpoint,a.output/'policy/eigen')
    for kind in (('student','expert') if a.compare_expert else ('student',)):
        params=dict(common,batch=a.eval_batch,seconds=30,seed=1002026,collect=False,policy=checkpoint if kind=='student' else None,
                    multiplier=a.validation_multiplier,arm_multiplier=a.validation_arm_multiplier,
                    expert_qualification=a.validation_expert_qualification or a.expert_qualification)
        with timing('evaluation_'+kind):run(SimpleNamespace(**params,output=a.output/('evaluation_'+kind)))
        gc.collect();jax.clear_caches()
    result=compare(a.output/'evaluation_expert/rollout.json',a.output/'evaluation_student/rollout.json',checkpoint,a.output/'policy/eigen') if a.compare_expert else None
    # Research outcome: retain/export the policy regardless of relative performance.
    print(json.dumps(dict(status='complete',policy=str(checkpoint),comparison=result)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('factors','calibration','source','eigen'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path,default=bootstrap.ROOT/'results'/'juggling_ampc'/time.strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--compare-expert',action='store_true',help='Optionally run a separate optimization-MPC baseline; disabled by default')
    p.add_argument('--observer',type=Path)  # Ignored compatibility option; no observer is loaded.
    p.add_argument('--expert-qualification',type=Path);p.add_argument('--authoritative-audit',type=Path)
    p.add_argument('--collection-multiplier',type=float,default=1.);p.add_argument('--collection-arm-multiplier',type=float,default=1.5)
    p.add_argument('--validation-multiplier',type=float,default=1.);p.add_argument('--validation-arm-multiplier',type=float,default=1.)
    p.add_argument('--validation-expert-qualification',type=Path)
    p.add_argument('--seconds',type=int,choices=(5,10,20),default=5);p.add_argument('--batch',type=int,default=60000);p.add_argument('--eval-batch',type=int,default=1024)
    p.add_argument('--dataset-gib',type=float,default=3.);p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--train-batch',type=int,default=8192);p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
