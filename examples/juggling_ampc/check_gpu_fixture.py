import bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np
from mpc import MPC
from plan_parity import metrics
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
f=np.load(a.directory/'native_fixtures.npz');t=time.monotonic();mpc=MPC(a.directory/'gpu_factors')
print('Compiled functions after',time.monotonic()-t,flush=True)
z,y,plan=mpc.solve(*[np.asarray(f[k],np.float32) for k in ['z','dual','p','projector']])
r=metrics(mpc.factors,np.asarray(plan),f['plan']);r['elapsed_s']=time.monotonic()-t
r['repeat_solve_seconds']=[]
for _ in range(a.repeat):
    begin=time.monotonic();mpc.solve(*[np.asarray(f[k],np.float32) for k in ['z','dual','p','projector']])[2].block_until_ready();r['repeat_solve_seconds'].append(time.monotonic()-begin)
print(json.dumps(r,indent=2));(a.directory/'gpu_mpc_parity.json').write_text(json.dumps(r,indent=2)+'\n');assert r['passed']
