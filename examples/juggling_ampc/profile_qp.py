"""Measure fixed-budget QP components on an actual native MPC linearization."""
import bootstrap
import argparse,json,time,os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from factors import Factors
from qp import NativeQP
from warpmpc.jax_osqp.solver import values_for_compiled_qdldl,rhs_for_compiled_qdldl

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--batch',type=int,default=128);p.add_argument('--repeat',type=int,default=3);p.add_argument('--thresholds',default='');a=p.parse_args()
f=Factors(a.directory/'factors');fixture=np.load(a.directory/'native_fixtures.npz');z=fixture['z'][0];param=fixture['p'][0]
h,g,matrix,c,cost=f.numpy(z,param);qp=NativeQP(f.P,f.A,f.cfg['rho'],f.cfg['osqp_iters'])
batch=lambda x:jnp.asarray(np.broadcast_to(x,(a.batch,len(x))).copy(),dtype=jnp.float32)
hp,ap,q=map(batch,(h.data,matrix.data,g-h@z))
def measure(call):
    jax.block_until_ready(call());times=[]
    for _ in range(a.repeat):
        start=time.monotonic();jax.block_until_ready(call());times.append(time.monotonic()-start)
    return times
r=dict(batch=a.batch,scale_s=measure(lambda:qp.scale(hp,ap,q)))
hp,ap,q,d,e,c=qp.scale(hp,ap,q);plan=qp.plan;kernel=qp.compiled.qdldl
k=batch(plan.kkt_upper.data).at[:,plan.p_to_kkt].set(hp+jnp.asarray(plan.p_diag_sigma,dtype=jnp.float32))
k=k.at[:,plan.a_to_kkt].set(ap).at[:,plan.rho_to_kkt].set(-1000.)
values=values_for_compiled_qdldl(kernel,k)
r['factor_s']=measure(lambda:kernel.factor(values));lx,_,di=kernel.factor(values)
rhs=rhs_for_compiled_qdldl(kernel,jnp.ones((a.batch,plan.n+plan.m),dtype=jnp.float32))
r['triangular_solve_s']=measure(lambda:kernel.solve(lx,di,rhs))
r.update(kkt_n=plan.n+plan.m,nnz_l=kernel.plan.nnz_l if hasattr(kernel,'plan') else None)
print(json.dumps(r,indent=2),flush=True)

if a.thresholds:
    for threshold in map(int,a.thresholds.split(',')):
        os.environ['JUGGLING_SOLVE_LEVEL_THRESHOLD']=str(threshold)
        variant=NativeQP(f.P,f.A,f.cfg['rho'],f.cfg['osqp_iters']).compiled.qdldl
        lv,_,dv=variant.factor(values_for_compiled_qdldl(variant,k))
        vrhs=rhs_for_compiled_qdldl(variant,jnp.ones((a.batch,plan.n+plan.m),dtype=jnp.float32))
        print(json.dumps(dict(batch=a.batch,level_threshold=threshold,triangular_solve_s=measure(lambda:variant.solve(lv,dv,vrhs)))),flush=True)
