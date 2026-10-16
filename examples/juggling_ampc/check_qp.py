"""Independent OSQP fixed-budget regression (run on CPU or CUDA)."""
import bootstrap
import argparse,json
import numpy as np
import scipy.sparse as sp
import osqp
import jax
from qp import NativeQP


def check(backend,dtype):
    jax.config.update('jax_enable_x64',dtype=='float64')
    rng=np.random.default_rng(27);n=12;m=21
    a=rng.normal(size=(m,n));a=np.vstack([a,np.eye(n)])
    v=rng.normal(size=(n,n));p=sp.triu(sp.csc_matrix(v.T@v+.01*np.eye(n)),format='csc');a=sp.csc_matrix(a)
    b=4;ps=[];av=[];qs=[];ls=[];us=[];xs=[];ys=[];expected=[];gs=[];relative_l=[];relative_u=[]
    solver=NativeQP(p,a,backend=backend,dtype=np.dtype(dtype))
    for i in range(b):
        pp=p.copy();aa=a.copy();pp.data*=1+i*.3;aa.data*=1+i*.2
        q=rng.normal(size=n);l=np.full(m+n,-1.);u=np.ones(m+n);l[:4]=u[:4]=0.;l[4]=-np.inf;u[4]=np.inf
        x=rng.normal(size=n)*.1;y=rng.normal(size=m+n)*.1
        full=pp+pp.T-sp.diags(pp.diagonal());gs.append(full@x+q);relative_l.append(l-aa@x);relative_u.append(u-aa@x)
        ref=osqp.OSQP();ref.setup(P=pp,A=aa,q=q,l=l,u=u,rho=.001,max_iter=40,check_termination=0,adaptive_rho=False,polishing=False,verbose=False,eps_abs=1e-4,eps_rel=1e-4)
        ref.warm_start(x=x,y=y);out=ref.solve();expected.append(out.x)
        for dst,src in zip([ps,av,qs,ls,us,xs,ys],[pp.data,aa.data,q,l,u,x,y]):dst.append(src)
    extra={k:np.asarray(v,dtype=dtype) for k,v in dict(gradient=gs,relative_lower=relative_l,relative_upper=relative_u).items()} if solver.incremental else {}
    actual,_=solver.solve(*[np.asarray(v,dtype=dtype) for v in [ps,av,qs,ls,us,xs,ys]],**extra)
    error=float(np.max(np.abs(np.asarray(actual)-expected)))
    result=dict(backend=backend,dtype=dtype,batch=b,max_abs_error=error,passed=error<(2e-4 if dtype=='float32' else 1e-9))
    print(json.dumps(result,indent=2));assert result['passed'],result
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',default='jax',choices=['jax','warp']);p.add_argument('--dtype',default='float64');a=p.parse_args();check(a.backend,a.dtype)
