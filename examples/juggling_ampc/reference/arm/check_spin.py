"""Focused derivative, locality, momentum and dissipation checks for spin MPC."""
import argparse,json
from time import perf_counter
from pathlib import Path
import numpy as np
import pinocchio as pin
from .model import Panda,HOME
from .spin_formulation import Config,NX,NZ,NP,SCALE,OFFSET
from .solver_spin import SpinSolver
from .spin_contact import impact


def main():
    p=argparse.ArgumentParser();p.add_argument('--compiled',action='store_true');p.add_argument('--optimization',choices=['O0','O1'],default='O0');args=p.parse_args()
    robot=Panda();cfg=Config(codegen=args.compiled,codegen_optimization=args.optimization);start=perf_counter();solver=SpinSolver(cfg,robot);build_s=perf_counter()-start
    x=np.r_[0,0,.67,0,0,0,HOME,np.zeros(10)];tau=np.asarray(robot.bias(HOME,np.zeros(7))).ravel()
    solver.initialize(x,tau);z=solver.z.copy();p=np.r_[x,tau]
    native=solver.data(z,p);reference=solver.problem['data'](z,p)
    native_error=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(native,reference))
    rng=np.random.default_rng(27);direction=rng.normal(size=len(z));direction/=np.linalg.norm(direction);eps=1e-6
    for _ in range(6):
        trial=z.reshape(-1,NZ).copy()
        trial[:,6:23]+=rng.normal(0,.015,trial[:,6:23].shape)
        trial[:,30]=np.maximum(trial[:,30],.005)
        outputs=solver.data(trial.ravel(),p);references=solver.problem['data'](trial.ravel(),p)
        native_error=max(native_error,max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(outputs,references)))
    op,cp=solver.value(z+eps*direction,p);om,cm=solver.value(z-eps*direction,p)
    derivative_error=float(np.max(abs((np.asarray(cp).ravel()-np.asarray(cm).ravel())/(2*eps)-np.asarray(native[2])@direction)))
    objective_error=float(abs(float(op-om)/(2*eps)-np.asarray(native[1]).ravel()@direction))
    assert native_error<1e-9 and derivative_error<1e-6 and objective_error<1e-8,(native_error,derivative_error,objective_error)
    rows,cols=solver.data.sparsity_out(2).get_triplet();nf=solver.problem['ncon_first'];nc=solver.problem['ncon_stage']
    for row,col in zip(rows,cols):
        node=col//NZ
        if row<nf:assert node==0
        elif row<nf+cfg.horizon*nc:assert node in ((row-nf)//nc,(row-nf)//nc+1)
        else:assert node==cfg.horizon
    errors=dict(normal_restitution=0.,joint_momentum=0.,ball_angular_momentum=0.,angular_jacobian=0.,friction_cone=0.,energy_gain_J=0.)
    for _ in range(100):
        q=HOME+rng.normal(0,.1,7);dq=rng.normal(0,.2,7);omega=rng.normal(0,20,3)
        pos,n,Jp,W,R=robot.contact_fk(q);n=np.asarray(n).ravel();R=np.asarray(R);W=np.asarray(W)
        pinJ=pin.computeFrameJacobian(robot.pin,robot.data,q,robot.frame,pin.LOCAL_WORLD_ALIGNED)
        errors['angular_jacobian']=max(errors['angular_jacobian'],float(np.max(abs(W-pinJ[3:]))))
        tangent=R[:,:2]@rng.normal(0,.01,2);J=np.asarray(Jp)-pin.skew(tangent)@W;M=np.asarray(robot.mass(q))
        relative=R[:,:2]@rng.normal(0,.5,2)-rng.uniform(.5,5)*n
        v=relative+np.cross(omega,cfg.radius*n)+J@dq
        u,dw,du,jn=impact(n,R[:,:2],J,M,relative)
        after_relative=v+u-np.cross(omega+dw,cfg.radius*n)-J@(dq+du)
        inertia=cfg.inertia_ratio*cfg.mass*cfg.radius**2
        errors['normal_restitution']=max(errors['normal_restitution'],abs(float(n@after_relative+cfg.restitution*n@relative)))
        errors['joint_momentum']=max(errors['joint_momentum'],float(np.max(abs(M@du+cfg.mass*J.T@u))))
        errors['ball_angular_momentum']=max(errors['ball_angular_momentum'],float(np.max(abs(inertia*dw-np.cross(-cfg.radius*n,cfg.mass*u)))))
        errors['friction_cone']=max(errors['friction_cone'],float(np.linalg.norm(u-jn*n)-cfg.friction*jn))
        before=.5*cfg.mass*(v@v)+.5*inertia*(omega@omega)+.5*dq@M@dq
        after=.5*cfg.mass*((v+u)@(v+u))+.5*inertia*((omega+dw)@(omega+dw))+.5*(dq+du)@M@(dq+du)
        errors['energy_gain_J']=max(errors['energy_gain_J'],float(after-before))
    assert max(errors.values())<1e-9,errors
    result=dict(compiled=args.compiled,optimization=args.optimization,build_s=build_s,compiled_reference_samples=7,
        compiled_reference_error=native_error,constraint_derivative_error=derivative_error,
        objective_derivative_error=objective_error,adjacent_node_locality=True,impact_samples=100,impact_errors=errors,
        qp_variables=solver.n,constraint_rows=len(solver.problem['cl']),hessian_nnz=native[0].nnz(),jacobian_nnz=native[2].nnz())
    path=Path('docs')/('spin_compiled_checks.json' if args.compiled else 'spin_formulation_checks.json')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
