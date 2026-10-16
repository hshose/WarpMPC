"""Native CPU vs batched GPU SQP from identical physical warm starts."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import jax
from mpc import MPC
from factors import Factors
from plan_parity import metrics
from cpp.python_controller import NativeController

def main(a):
    jax.config.update('jax_enable_x64',a.dtype=='float64')
    f=Factors(a.directory/'factors')
    states=[];zs=[];duals=[];parameters=[];projectors=[];expected=[]
    for i in range(a.batch):
        native=NativeController(library=a.directory/'controller/libjuggling_controller.so',factors=a.directory/'factors',compiled=False)
        ref=np.r_[[.01*i,-.005*i,-.03],f.arrays['home']]
        x=np.r_[[ref[0],ref[1],1.05],[0.,0.,-.1],f.arrays['home'],np.zeros(10)];tau=np.zeros(7)
        native.set_reference(ref[:3],ref[3:]);native.initialize(x,tau);z,y=native.warmstart()
        zs.append(z);duals.append(y);parameters.append(np.r_[x,tau,ref]);projectors.append(f.projector(ref).ravel(order='F'))
        expected.append(native.step(x,tau,0.)[0]);native.close()
    np.savez(a.directory/'native_fixtures.npz',z=np.asarray(zs),dual=np.asarray(duals),p=np.asarray(parameters),projector=np.asarray(projectors),plan=np.asarray(expected))
    if a.fixtures_only:return
    mpc=MPC(a.directory/'gpu_factors',a.backend,np.dtype(a.dtype))
    z,y,plan=mpc.solve(*[np.asarray(v,dtype=a.dtype) for v in [zs,duals,parameters,projectors]])
    result=metrics(f,np.asarray(plan),np.asarray(expected));result.update(backend=a.backend,dtype=a.dtype,batch=a.batch)
    print(json.dumps(result,indent=2));(a.directory/'mpc_parity.json').write_text(json.dumps(result,indent=2)+'\n');assert result['passed']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--backend',default='warp');p.add_argument('--dtype',default='float32');p.add_argument('--batch',type=int,default=4);p.add_argument('--fixtures-only',action='store_true');a=p.parse_args();a.directory=a.directory.resolve();main(a)
