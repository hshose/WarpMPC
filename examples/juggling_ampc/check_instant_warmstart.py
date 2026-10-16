"""Check native warm starts and frozen event state on CPU.

Use the same generated factors/library as check_factors.py. This offline check
does not run the robot or load a simulation observer.
"""
import bootstrap
import argparse
import json
import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from factors import Factors
from warmstart import WarmStart
from cpp.python_controller import NativeController


def check(root):
    trace=root/'warmstart_event_trace.jsonl';trace.unlink(missing_ok=True)
    os.environ['JUGGLING_FIRST_EVENT_CENTER_TRACE']=str(trace)
    f=Factors(root/'gpu_factors');warm=WarmStart(f,backend='jax')
    native=NativeController(library=root/'controller/libjuggling_controller.so',factors=root/'gpu_factors',compiled=False)
    x=jnp.asarray(np.r_[[.01,-.01,.65,0,0,-.1],f.arrays['home'],np.zeros(10)][None],jnp.float32)
    tau=warm.torque(x[:,6:13],x[:,13:20],jnp.zeros((1,7),jnp.float32)).reshape(1,7)
    z,d=warm.initialize(x,tau);native.initialize(np.asarray(x[0]),np.asarray(tau[0]))
    results=[]
    for k in range(2):
        fallback=False
        if k:
            x=warm.free(x,.02,jnp.zeros((1,7),jnp.float32))
            z,d,failed=warm.shift(z,d,x,tau);fallback=bool(np.asarray(failed).any())
            native.shift(.02,np.asarray(x[0]),np.asarray(tau[0]))
        event=json.loads(trace.read_text().splitlines()[-1])
        expected=np.r_[event['active'],event['relative_event_time_s'],event['stage'],event['target']]
        actual=np.asarray(warm.event_parameters[0])
        np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-3)
        Y=np.asarray(z).reshape(-1,34)*f.arrays['scale']+f.arrays['offset']
        hybrid=f.meta['spec'].get('kinematic_tail')
        if hybrid:
            for i in range(hybrid['dynamic_prefix']+1,f.cfg['horizon']):
                h=f.meta['stage_steps'][i];a=.5*(Y[i,23:30]+Y[i+1,23:30])
                np.testing.assert_allclose(Y[i+1,13:20],Y[i,13:20]+h*a,atol=3e-5)
                np.testing.assert_allclose(Y[i+1,6:13],Y[i,6:13]+h*Y[i,13:20]+.5*h*h*a,atol=3e-5)
        nz,nd=native.warmstart()
        np.testing.assert_allclose(np.asarray(z[0]),nz,atol=1e-3,rtol=1e-3)
        results.append(dict(step=k,guarded_ball_seed_fallback=fallback,event=actual.tolist(),native_event=expected.tolist(),
            max_scaled_seed_difference=float(np.max(np.abs(np.asarray(z[0])-nz))),
            max_dual_difference=float(np.max(np.abs(np.asarray(d[0])-nd)))))
    report=dict(passed=True,checks=results,tail_endpoint_dynamics='passed' if hybrid else 'not applicable: full dynamic horizon',event_tolerance=1e-3)
    (root/'warmstart_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();check(a.directory.resolve())
