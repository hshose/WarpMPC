"""Check generic native mesh/pruning assembly and fixed-budget updates."""
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from arm.model import Panda
from cpp.python_controller import NativeController


def main():
    p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('library',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.factors/'metadata.json').read_text());spec=meta['spec'];robot=Panda(assembly=spec.get('assembly'),**spec.get('geometry',{}))
    if spec.get('drop'):
        if spec.get('mesh'):
            from optimization.pruning.combined_solver import CombinedSolver as Solver
            from optimization.pruning.combined_formulation import Config
        else:
            from optimization.pruning.solver import PruningSolver as Solver
            from optimization.pruning.formulation import Config
        cfg=Config(**dict(meta['config'],codegen=False,pruning_prefix=spec['prefix'],pruning_groups=tuple(spec['drop']),deduplicate_endpoints=spec.get('dedup',False)))
    else:
        from optimization.mesh.solver import MeshSolver as Solver
        from optimization.mesh.formulation import Config
        cfg=Config(**dict(meta['config'],codegen=False))
    ref=Solver(cfg,robot);ref.guard_ball_rollout=True;native=NativeController(library=a.library,factors=a.factors)
    saved=np.load('runs/20260924T165001.832420Z_observer_wind/arrays.npz');plant=np.load('runs/20260924T165001.832420Z_observer_wind/plant.npz')
    x=saved['x'][0];tau=np.asarray(robot.bias(x[6:13],x[13:20])).ravel();ref.initialize(x,tau);native.initialize(x,tau)
    errors=dict(initial=float(np.max(abs(native.warmstart()[0]-ref.z))),H=0.,A=0.,gradient=0.,constraints=0.,cost=0.,plan=0.,dual=0.)
    for k in range(12):
        x=saved['x'][k];ix=round(saved['t'][k]/.0005);tau=plant['tau'][ix]-.00025*plant['torque_rate'][ix]
        dt=float(saved['t'][k]-saved['t'][k-1]) if k else 0.
        if k:ref.shift(dt,x,tau);native.shift(dt,x,tau)
        z=ref.z;parameter=np.r_[x,tau];H,g,A,c,cost=native.linearize(z,parameter);h0,g0,a0,c0,cost0=ref.data(z,parameter)
        for name,delta in [('H',H-h0.sparse()),('A',A-sp.vstack([a0.sparse(),sp.eye(native.n,format='csc')]))]:errors[name]=max(errors[name],float(np.max(abs(delta.data),initial=0)))
        for name,actual,expected in [('gradient',g,g0),('constraints',c,c0),('cost',cost,cost0)]:errors[name]=max(errors[name],float(np.max(abs(np.asarray(actual).ravel()-np.asarray(expected).ravel()))))
        Y,info=ref.solve(x,tau);CY,target,stats=native.step(x,tau,0.)
        errors['plan']=max(errors['plan'],float(np.max(abs(CY-Y))))
        errors['dual']=max(errors['dual'],float(np.max(abs(native.warmstart()[1]-ref.dual))))
        assert int(stats[12])==50 and len(info['iterations'])==1
    result=dict(passed=max(errors.values())<1e-5,errors=errors,variables=native.n,constraints=native.m,hessian_nnz=native.nh,constraint_plus_bound_nnz=native.na,spec=spec)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed'],result


if __name__=='__main__':main()
