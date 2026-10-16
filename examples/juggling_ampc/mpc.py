"""Fixed-budget Franka SQP with batched sparse WarpMPC solves.

The caller supplies the shifted native-compatible primal/dual seed. This keeps
seed parity testable independently of factor assembly and QP arithmetic.
"""
import numpy as np
import jax
import jax.numpy as jnp
from factors import Factors
from qp import NativeQP

class MPC:
    def __init__(self,directory,backend='warp',dtype=np.float32,action_only=False):
        self.factors=f=directory if isinstance(directory,Factors) else Factors(directory)
        self.qp=qp=NativeQP(f.P,f.A,f.cfg['rho'],f.cfg['osqp_iters'],backend,dtype)
        assemble=f.compile(backend);self.assemble=assemble;merit_factors=f.compile_merit(backend)
        self.merit_factors=merit_factors
        low=jnp.asarray(f.arrays['all_lower'],dtype=dtype);high=jnp.asarray(f.arrays['all_upper'],dtype=dtype)
        cl=jnp.asarray(f.arrays['constraint_lower'],dtype=dtype);cu=jnp.asarray(f.arrays['constraint_upper'],dtype=dtype)
        scale=jnp.asarray(f.arrays['scale'],dtype=dtype);offset=jnp.asarray(f.arrays['offset'],dtype=dtype)
        pr=jnp.asarray(qp.plan.p_full_rows);pc=jnp.asarray(qp.plan.p_full_cols);pv=jnp.asarray(qp.plan.p_full_value_map)
        def hx(h,x):return jnp.zeros_like(x).at[:,pr].add(h[:,pv]*x[:,pc])
        unlimited=np.isin(np.arange(f.n)%34,list(range(13,20))+list(range(23,30)))
        def violation(c,z):return jnp.sum(jnp.maximum(cl-c,0)+jnp.maximum(c-cu,0),axis=1)+jnp.sum(jnp.maximum(low-z,0)+jnp.maximum(z-high,0),axis=1)
        @jax.jit
        def solve(z,dual,p,projector,event=None):
            if event is not None:p=jnp.concatenate((p,event),axis=1)
            retry_state=None;retry_mask=jnp.zeros(z.shape[0],bool)
            for _ in range(f.cfg['sqp_iters']):
                h,g,a,c,cost=assemble(z,p,projector);q=qp.gradient_residual(h,z,g) if qp.gradient_residual is not None else g-hx(h,z);az=qp.matvec(a,z)[:,:f.m]
                center=jnp.clip(z,low,high)
                lb=jnp.where(unlimited,low,jnp.maximum(low,center-f.cfg['trust']))
                ub=jnp.where(unlimited,high,jnp.minimum(high,center+f.cfg['trust']))
                measured=(p[:,:30]-offset[:30])/scale[:30]
                lb=lb.at[:,:30].set(jnp.minimum(lb[:,:30],measured));ub=ub.at[:,:30].set(jnp.maximum(ub[:,:30],measured))
                l=jnp.concatenate([cl-c+az,lb],axis=1);u=jnp.concatenate([cu-c+az,ub],axis=1)
                if qp.incremental:
                    candidate,yd,state=qp.solve(h,a,q,l,u,z,dual,gradient=g,
                        relative_lower=jnp.concatenate([cl-c,lb-z],axis=1),
                        relative_upper=jnp.concatenate([cu-c,ub-z],axis=1),
                        retry_state=retry_state,retry_mask=retry_mask,return_state=True)
                else:candidate,yd,state=qp.solve(h,a,q,l,u,z,dual,retry_state=retry_state,retry_mask=retry_mask,return_state=True)
                good=jnp.all(jnp.isfinite(candidate),axis=1)&(jnp.max(jnp.abs(candidate),axis=1)<1e6)
                merit0=cost+1000*f.cfg['objective_scale']*violation(c,z)
                chosen=z;accepted=jnp.zeros(z.shape[0],bool)
                for alpha in (1.,.5,.25,.125,.0625):
                    def try_step(carry):
                        chosen,accepted=carry
                        trial=z+alpha*(candidate-z);ct,objective=merit_factors(trial,p,projector)
                        merit=objective+1000*f.cfg['objective_scale']*violation(ct,trial)
                        take=good&~accepted&(merit<=merit0+1e-8)
                        return jnp.where(take[:,None],trial,chosen),accepted|take
                    chosen,accepted=jax.lax.cond(jnp.any(good&~accepted),try_step,lambda carry:carry,(chosen,accepted))
                z=chosen;dual=jnp.where(accepted[:,None],yd,dual)
                if f.meta['spec'].get('solver_options',{}).get('qp_retry_warmstart',0)==2:
                    retry_mask=good&~accepted&jnp.all(jnp.isfinite(yd),axis=1)&(jnp.max(jnp.abs(yd),axis=1)<1e20)
                    retry_state=state
            if action_only:
                columns=np.r_[6:20,23:30]
                knot=z[:,34+columns]*scale[columns]+offset[columns]
                return z,dual,knot
            return z,dual,z.reshape(-1,f.cfg['horizon']+1,34)*scale+offset
        self.solve=solve
