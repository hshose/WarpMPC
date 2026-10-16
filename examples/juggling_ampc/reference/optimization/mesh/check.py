"""Check uniform-grid equivalence and mesh invariants before plant trials."""
import json
from pathlib import Path
import numpy as np
from arm.model import Panda,HOME
from arm.solver_spin import SpinSolver
from arm.spin_formulation import Config as BaseConfig
from optimization.mesh.formulation import Config,grid
from optimization.mesh.solver import MeshSolver

def main():
    robot=Panda();cfg=Config(codegen=False,actuator_margin=.98)
    baseline=SpinSolver(BaseConfig(codegen=False,actuator_margin=.98),robot)
    mesh=MeshSolver(cfg,robot)
    x=np.r_[[0,0,.67],np.zeros(3),HOME,np.zeros(10)]
    tau=np.array(robot.bias(HOME,np.zeros(7))).ravel()
    baseline.initialize(x,tau);mesh.initialize(x,tau)
    result={'uniform_initial_seed_max_error':float(np.max(abs(baseline.z-mesh.z)))}
    p=np.r_[x,tau]
    result['uniform_data_max_error']=max(float(np.max(abs(np.array(a)-np.array(b))))
      for a,b in zip(baseline.data(baseline.z,p),mesh.data(mesh.z,p)))
    rng=np.random.default_rng(9);size=len(baseline.problem['cl'])+baseline.n
    baseline.dual=rng.normal(size=size);mesh.dual=baseline.dual.copy()
    baseline.shift(.025,x,tau);mesh.shift(.025,x,tau)
    result['uniform_shift_seed_max_error']=float(np.max(abs(baseline.z-mesh.z)))
    result['uniform_shift_dual_max_error']=float(np.max(abs(baseline.dual-mesh.dual)))
    result['mesh']={}
    for n in (20,24,28,32):
      h,t,g=grid(Config(horizon=n));result['mesh'][n]={'growth':g,'steps':h.tolist(),'times':t.tolist()}
      assert abs(t[-1]-1.2)<1e-12 and h[0]==.03 and np.all(np.diff(h)>0)
    assert max(result[k] for k in result if k!='mesh')<1e-10,result
    result['passed']=True
    Path('optimization/mesh/checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='mesh'},indent=2))

if __name__=='__main__':main()
