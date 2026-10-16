"""Two MPC calls exercising event parameters and the mixed-control warm shift.

Run on a workstation GPU or inside a Slurm GPU allocation, never a CLAIX login.
This is a finite-output smoke check, not a native controller accuracy claim.
"""
import bootstrap
import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from factors import Factors
from warmstart import WarmStart
from mpc import MPC
from projector import GPUProjectors


def check(directory,output):
    f=Factors(directory);warm=WarmStart(f);mpc=MPC(f,action_only=True);project=GPUProjectors(f)
    x=jnp.asarray(np.tile(np.r_[[.01,-.01,.65,0,0,-.1],f.arrays['home'],np.zeros(10)],(2,1)),jnp.float32)
    tau=warm.torque(x[:,6:13],x[:,13:20],jnp.zeros((2,7),jnp.float32)).reshape(2,7)
    z,d=warm.initialize(x,tau)
    posture=jnp.asarray(np.tile(f.arrays['home'],(2,1)),jnp.float32)
    p=jnp.concatenate((x,tau,jnp.zeros((2,3)),posture),axis=1);projector=project(posture)
    results=[]
    for k in range(2):
        if k:z,d,failed=warm.shift(z,d,x,tau)
        start=time.perf_counter();z,d,knot=mpc.solve(z,d,p,projector,warm.event_parameters)
        knot.block_until_ready()
        assert all(np.isfinite(np.asarray(v)).all() for v in (z,d,knot))
        results.append(dict(step=k,solve_including_compile_s=time.perf_counter()-start,
            maximum_absolute_knot=float(jnp.max(jnp.abs(knot))),event=np.asarray(warm.event_parameters).tolist()))
        print(json.dumps(results[-1]),flush=True)
    report=dict(passed=True,devices=[str(v) for v in jax.devices()],checks=results)
    if output:output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();check(a.factors,a.output)
