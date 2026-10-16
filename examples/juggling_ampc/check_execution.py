"""Check generated callback/reference against the copied native equations."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import casadi as ca
import jax.numpy as jnp
from factors import Factors
from warmstart import WarmStart
from execution import Execution
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
f=Factors(a.directory/'gpu_factors');w=WarmStart(f,backend='jax');e=Execution(f,w,backend='jax')
data=np.load(a.directory/'native_fixtures.npz');plan=jnp.asarray(data['plan'],dtype=jnp.float32);q=plan[:,0,6:13];dq=plan[:,0,13:20];initial=jnp.zeros_like(q)
ramp,raw=e.project(plan,q,dq,initial)
reference=e.reference(ramp,.017)
q0,v0,a0,j=map(np.asarray,ramp);qr,vr,ar=map(np.asarray,reference)
np.testing.assert_allclose(qr,q0+v0*.017+a0*.017**2/2+j*.017**3/6,atol=2e-7)
velocity,acc,violation=e.callback(reference,q,dq,jnp.zeros_like(q),jnp.zeros_like(q))
desired=ar+e.kp*(qr-np.asarray(q))+e.kd*(vr-np.asarray(dq));bound=e.jerk_fraction*np.asarray(e.jerk_limit)*.001
expected=np.clip(np.clip(desired,-.95*np.asarray(e.acc_limit),.95*np.asarray(e.acc_limit)),-bound,bound)
np.testing.assert_allclose(acc,expected,atol=1e-5)
np.testing.assert_allclose(velocity,expected*.001,atol=1e-7)
g=np.asarray(e.geometry(q,dq)).reshape(len(q),48)
assert np.isfinite(g).all()
fk=ca.Function.load(str(f.directory/'panda_contact_fk.casadi'))
mass=ca.Function.load(str(f.directory/'panda_mass.casadi'))
for i in range(len(q)):
    pos,n,J,W,R=fk(np.asarray(q[i],dtype=float),ca.DM(f.arrays['factor_parameters']))
    spatial=np.vstack([J,W]);inv=spatial@np.linalg.solve(np.asarray(mass(np.asarray(q[i],dtype=float))),spatial.T)
    np.testing.assert_allclose(g[i,12:].reshape(6,6),inv,rtol=1e-5,atol=2e-5)
print(json.dumps(dict(passed=True,batch=len(q),geometry_shape=list(g.shape),max_projection_ratio=float(np.max(raw)))))
