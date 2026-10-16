import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from factors import Factors
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
f=Factors(a.directory/'gpu_factors');data=np.load(a.directory/'native_fixtures.npz');evaluate=f.compile_merit()
c,v=evaluate(*[np.asarray(data[k],np.float32) for k in ['z','p','projector']])
expected=[f.numpy(z,p)[3:] for z,p in zip(data['z'],data['p'])]
ce=float(np.max(abs(np.asarray(c)-np.array([x[0] for x in expected]))));ve=float(np.max(abs(np.asarray(v)-np.array([x[1] for x in expected]))))
r=dict(max_constraint_error=ce,max_objective_error=ve,passed=ce<2e-4 and ve<1e-5)
print(json.dumps(r));assert r['passed']
