"""Focused dynamics, derivatives, locality and finite-arm impact verification."""
from pathlib import Path
import json
import numpy as np
import pinocchio as pin
from .model import Panda,HOME
from .formulation import Config,NX,NZ,NP,SCALE,OFFSET
from .solver import Solver
from .simulator import Simulator,SimConfig


def main():
    robot=Panda();cfg=Config();solver=Solver(cfg,robot)
    x=np.r_[0,0,.67,0,0,0,HOME,np.zeros(7)]
    torque=pin.computeGeneralizedGravity(robot.pin,robot.data,HOME).copy()
    solver.initialize(x,torque);p=np.r_[x,torque];z=solver.z.copy()
    native=solver.data(z,p);reference=solver.problem['data'](z,p)
    errors=[float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(native,reference)]
    assert max(errors)<1e-9,errors
    rng=np.random.default_rng(17);direction=rng.normal(size=len(z));direction/=np.linalg.norm(direction)
    eps=1e-6
    objective_plus,cplus=solver.value(z+eps*direction,p);objective_minus,cminus=solver.value(z-eps*direction,p)
    A=np.asarray(native[2]);gradient=np.asarray(native[1]).ravel()
    fd_constraint=(np.asarray(cplus).ravel()-np.asarray(cminus).ravel())/(2*eps)
    derivative_error=float(np.max(abs(fd_constraint-A@direction)))
    objective_error=float(abs(float(objective_plus-objective_minus)/(2*eps)-gradient@direction))
    assert derivative_error<1e-6 and objective_error<1e-8,(derivative_error,objective_error)
    row,col=solver.problem['data'].sparsity_out(2).get_triplet();nf=solver.problem['ncon_first'];nc=solver.problem['ncon_stage']
    for r,c in zip(row,col):
        node=c//NZ
        if r<nf:assert node==0
        elif r<nf+cfg.horizon*nc:assert node in ((r-nf)//nc,(r-nf)//nc+1)
        else:assert node==cfg.horizon
    sim=Simulator(robot,SimConfig(restitution=.86,drag=0))
    state=x.copy();state[2]=.025;state[5]=-1.
    _,contacts=sim.step(state,torque,torque,.01)
    assert len(contacts)==1
    event=contacts[0];M=pin.crba(robot.pin,robot.data,HOME).copy();_,n,J,_=robot.numeric_fk(HOME);Jn=n@J
    jump=np.array(event['joint_velocity_jump']);ball=np.array(event['ball_velocity_after']);incoming=event['incoming_normal_speed']
    restitution_error=float(abs(n@ball-Jn@jump+.86*incoming))
    momentum_error=float(np.max(abs(M@jump+Jn*event['impulse_Ns'])))
    mass=.0027;before=.5*mass*incoming**2;after=.5*mass*(ball@ball)+.5*jump@M@jump
    effective=1/mass+Jn@np.linalg.solve(M,Jn)
    dissipation_error=float(abs(before-after-.5*(1-.86**2)*incoming**2/effective))
    assert max(restitution_error,momentum_error,dissipation_error)<1e-9
    result=dict(compiled_vs_interpreted_max_errors=errors,constraint_directional_derivative_error=derivative_error,
                objective_directional_derivative_error=objective_error,adjacent_node_locality=True,
                finite_arm_impact=dict(restitution_error_m_s=restitution_error,joint_momentum_error=momentum_error,
                                      energy_loss_error_J=dissipation_error),
                qp_variables=solver.n,constraint_rows=len(solver.problem['cl']),
                hessian_nnz=native[0].nnz(),constraint_jacobian_nnz=native[2].nnz())
    Path('docs/arm_formulation_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
