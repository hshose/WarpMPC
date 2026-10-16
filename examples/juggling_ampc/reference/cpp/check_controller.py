"""Compare native factors, assembly, warmstarts and a fixed-budget RTI step."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from arm.model import Panda
from arm.spin_formulation import Config,SCALE,OFFSET
from arm.solver_spin import SpinSolver
from .python_controller import NativeController


def main():
    robot=Panda();cfg=Config(actuator_margin=.98,codegen=False);reference=SpinSolver(cfg,robot);reference.guard_ball_rollout=True
    cpp=NativeController();run=Path('runs/20260924T165001.832420Z_observer_wind');samples=np.load(run/'arrays.npz');pred=np.load(run/'predictions.npz')['Y']
    errors={name:0. for name in ['H','gradient','A','constraints','cost','initialize','shift','dual_shift','one_rti_plan','one_rti_dual','successive_updates']}
    for k in [0,5,20,40,60,80]:
        Y=pred[k];z=((Y-OFFSET)/SCALE).ravel();p=Y[0,:30]
        H,g,A,c,cost=cpp.linearize(z,p);rh,rg,ra,rc,rf=reference.data(z,p)
        for name,delta in [('H',H-rh.sparse()),('A',A-sp.vstack([ra.sparse(),sp.eye(cpp.n,format='csc')]))]:
            errors[name]=max(errors[name],float(np.max(abs(delta.data),initial=0)))
        for name,actual,expected in [('gradient',g,rg),('constraints',c,rc),('cost',cost,rf)]:
            errors[name]=max(errors[name],float(np.max(abs(np.asarray(actual).ravel()-np.asarray(expected).ravel()))))
    assert max(errors.values())<1e-8,errors
    x=samples['x'][0];tau=np.asarray(robot.bias(x[6:13],x[13:20])).ravel()
    reference.initialize(x,tau);cpp.initialize(x,tau);z,d=cpp.warmstart();errors['initialize']=float(max(abs(z-reference.z)));assert errors['initialize']<1e-7,errors
    # Match precisely the same scaled seed/dual, then compare one fixed-budget QP.
    cpp.set_warmstart(reference.z,np.zeros(cpp.n+cpp.m));Y,info=reference.solve(x,tau);CY,target,stats=cpp.step(x,tau,0.)
    errors['one_rti_plan']=float(np.max(abs(CY-Y)));_,dual=cpp.warmstart();errors['one_rti_dual']=float(max(abs(dual-reference.dual)))
    assert errors['one_rti_plan']<1e-5 and errors['one_rti_dual']<1e-5,errors
    assert int(stats[12])==50 and len(info['iterations'])==1 and info['iterations'][0]['iter']==50
    plant=np.load(run/'plant.npz')
    for k in range(1,12):
        x=samples['x'][k];index=round(samples['t'][k]/.0005)
        tau=plant['tau'][index]-.00025*plant['torque_rate'][index]
        dt=samples['t'][k]-samples['t'][k-1]
        reference.shift(dt,x,tau);RY,ri=reference.solve(x,tau);CY,_,ci=cpp.step(x,tau,dt)
        delta=float(np.max(abs(CY-RY)));errors['successive_updates']=max(errors['successive_updates'],delta)
        assert delta<1e-5,(k,delta,errors)
    # Shift the identical optimized seed, preserving both primal and dual arrays.
    for i,dt in [(10,.02),(30,.025),(50,.03)]:
        reference.z=((pred[i]-OFFSET)/SCALE).ravel();reference.dual=np.linspace(-.1,.1,cpp.n+cpp.m)
        x=samples['x'][i*10];tau=pred[i,0,23:30];cpp.set_warmstart(reference.z,reference.dual)
        reference.shift(dt,x,tau);cpp.shift(dt,x,tau);z,d=cpp.warmstart()
        errors['shift']=max(errors['shift'],float(max(abs(z-reference.z))))
        errors['dual_shift']=max(errors['dual_shift'],float(max(abs(d-reference.dual))))
    assert errors['shift']<1e-6 and errors['dual_shift']<1e-12,errors
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),passed=True,errors=errors,
        variables=cpp.n,nonlinear_constraints=cpp.m,hessian_nnz=cpp.nh,constraint_plus_bounds_nnz=cpp.na,
        first_cpp_step_ms=float(stats[0]),exactly_one_sqp_and_50_osqp=True)
    Path('docs/cpp_controller_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
