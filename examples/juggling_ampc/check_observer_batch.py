"""Check independently evolving native EKFs against their ballistic limit."""
import bootstrap
import argparse, json
from pathlib import Path
import numpy as np
from observer_batch import ObserverBatch
p=argparse.ArgumentParser();p.add_argument('library',type=Path);a=p.parse_args()
n=4
cfg=dict(gravity_x=0.,gravity_y=0.,gravity_z=-9.81,drag=.34,mass=.0039,radius=.02,restitution=.55,friction=.3,inertia_ratio=2/3,paddle_radius=.075)
g=np.zeros((n,48));g[:,2]=-10.;g[:,5]=1.
g[:,12:]=np.tile(np.eye(6).ravel(),(n,1))
pos=np.tile([0.,0.,1.],(n,1));pos[:,0]=np.arange(n)*.1
vel=np.zeros((n,3));vel[:,0]=np.arange(n)*.05
observer=ObserverBatch(a.library,cfg,pos,vel,g)
maximum=0.
for k in range(1,41):
    t=k*.002;decay=np.exp(-cfg['drag']*t);gravity=np.array([0.,0.,-9.81])
    v=vel*decay+gravity/cfg['drag']*(1-decay)
    x=pos+vel*(1-decay)/cfg['drag']+gravity/cfg['drag']*(t-(1-decay)/cfg['drag'])
    capture=np.full(n,t if k%5==0 else np.nan)
    result=observer.step(t,g,capture,x)
    maximum=max(maximum,float(np.max(abs(result[:,:6]-np.c_[x,v]))))
    assert np.isfinite(result[:,:9]).all() and not result[:,11].any()
observer.close()
print(json.dumps(dict(batch=n,seconds=t,max_state_error=maximum,passed=maximum<1e-6)))
assert maximum<1e-6
