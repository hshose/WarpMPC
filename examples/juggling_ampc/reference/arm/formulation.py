"""Adjacent-node ball/Panda contact-implicit midpoint dynamics; no qddot variable."""
from dataclasses import dataclass
import casadi as ca
import numpy as np
from .model import Panda,HOME,TORQUE_LIMIT,VELOCITY_LIMIT,ACCEL_LIMIT,TORQUE_RATE_LIMIT,JERK_LIMIT

NX,NZ,NP=20,31,27
SCALE=np.r_[[.5,.5,1.,4.,4.,4.],np.ones(7),np.full(7,2.),[40.]*4+[6.]*3,8.,1.,.03,1.]
OFFSET=np.r_[np.zeros(6),HOME,np.zeros(18)]

@dataclass
class Config:
    horizon:int=40
    h:float=.03
    control_dt:float=.02
    sim_dt:float=.0005
    sqp_iters:int=1
    osqp_iters:int=50
    mass:float=.0027
    radius:float=.02
    paddle_radius:float=.075
    restitution:float=.88
    apex:float=.75
    comp_weight:float=200.
    energy_weight:float=12.
    lateral_weight:float=8.
    pose_weight:float=2.
    velocity_weight:float=1.
    accel_weight:float=.4
    torque_weight:float=.02
    rate_weight:float=.15
    posture_weight:float=.3
    actuator_margin:float=1.
    rate_margin:float=1.
    objective_scale:float=.0001
    proximal:float=.00001
    trust:float=.5
    rho:float=.001
    codegen:bool=True
    codegen_optimization:str='O1'


def create_factors(cfg,robot):
    z=ca.MX.sym('z',NZ);zn=ca.MX.sym('zn',NZ);p=ca.MX.sym('p',NP)
    y=z*ca.DM(SCALE)+ca.DM(OFFSET);yn=zn*ca.DM(SCALE)+ca.DM(OFFSET)
    b,v,q,dq,tau,j,f,h=y[:3],y[3:6],y[6:13],y[13:20],y[20:27],y[27],y[28],y[29]
    qn,dqn,taun=yn[6:13],yn[13:20],yn[20:27]
    qm=.5*(q+qn);vm=.5*(dq+dqn);acc=(dqn-dq)/h;tc=f*h
    qc=q+tc*dq+.5*tc*tc*acc;vc=dq+tc*acc
    pc,n,Jc,Nc,_=robot.fk(qc);pc=pc-ca.DM(robot.origin)
    pos,normal,J,N,_=robot.fk(q);pos=pos-ca.DM(robot.origin)
    posn,nn,_,_,_=robot.fk(qn);posn=posn-ca.DM(robot.origin)
    bc=b+tc*v+.5*tc*tc*ca.DM([0,0,-9.81]);ballvc=v+tc*ca.DM([0,0,-9.81])
    rel=bc-pc;gap=ca.dot(n,rel)-cfg.radius
    # Surface velocity includes rotation at the contact offset.
    contact_J=n.T@Jc-rel.T@Nc
    vn=ca.dot(n,ballvc)-contact_J@vc
    effective=ca.solve(robot.mass(qc),contact_J.T)
    restitution=j*(1+cfg.mass*(contact_J@effective))+(1+cfg.restitution)*vn
    tangent=rel-n*ca.dot(n,rel)
    bnext=b+h*v+.5*h*h*ca.DM([0,0,-9.81])+(h-tc)*j*n
    vnext=v+h*ca.DM([0,0,-9.81])+j*n
    # Midpoint force balance includes the equal/opposite generalized impulse.
    torque=robot.rnea(qm,vm,acc)+cfg.mass*j/h*contact_J.T
    endgap=ca.dot(nn,yn[:3]-posn)-cfg.radius
    contactgap=j/8*gap/.05;contactrest=j/8*restitution/8
    accel0=ca.solve(robot.mass(q),tau-robot.bias(q,dq))
    accel1=ca.solve(robot.mass(qn),taun-robot.bias(qn,dqn))
    constraints=[];lo=[];hi=[];groups={}
    def add(name,c,l,u):
        start=len(lo);count=int(c.numel());constraints.append(c)
        lo.extend(np.broadcast_to(l,(count,)));hi.extend(np.broadcast_to(u,(count,)))
        groups[name]=(start,start+count)
    add('ball',ca.vertcat(yn[:3]-bnext,yn[3:6]-vnext)/ca.DM(SCALE[:6]),0,0)
    add('position',qn-q-h*vm,0,0)
    # Precondition force balance into a velocity defect. Torque scaling alone
    # barely constrains the low-inertia wrist at a fixed, small QP budget.
    add('dynamics',h*ca.solve(robot.mass(qm),torque-.5*(tau+taun))/2.,0,0)
    add('clock',yn[30]-y[30]-h,0,0)
    add('contact',ca.vertcat(gap/.5,endgap/.5,j/8*(ca.dot(tangent,tangent)/(cfg.paddle_radius-cfg.radius)**2-1),j/8*vn/4),[0,0,-np.inf,-np.inf],[np.inf,np.inf,0,0])
    add('workspace',pos/ca.DM([.4,.4,.18]),-1,1)
    add('tilt',normal[:2]/np.sin(.4),-1,1)
    add('upward',normal[2],np.cos(.4),np.inf)
    add('paddle_velocity',J@dq/1.2,-1,1)
    add('acceleration',acc/ca.DM(ACCEL_LIMIT),-cfg.actuator_margin,cfg.actuator_margin)
    add('endpoint_acceleration',ca.vertcat(accel0,accel1)/ca.DM(np.tile(ACCEL_LIMIT,2)),-cfg.actuator_margin,cfg.actuator_margin)
    add('torque_rate',(taun-tau)/h/ca.DM(TORQUE_RATE_LIMIT),-cfg.actuator_margin*cfg.rate_margin,cfg.actuator_margin*cfg.rate_margin)
    add('jerk',(accel1-accel0)/h/ca.DM(JERK_LIMIT),-cfg.actuator_margin,cfg.actuator_margin)
    energy=b[2]-cfg.radius+v[2]**2/(2*9.81)
    residual=ca.vertcat(cfg.energy_weight*(energy-cfg.apex),cfg.lateral_weight*b[:2]/.2,2*v[:2],
        cfg.pose_weight*pos/ca.DM([.3,.3,.15]),.3*normal[:2]/.3,
        cfg.velocity_weight*(J@dq),.5*cfg.velocity_weight*(N@dq),
        cfg.accel_weight*acc/ca.DM(ACCEL_LIMIT),cfg.torque_weight*tau/ca.DM(TORQUE_LIMIT),
        cfg.rate_weight*(taun-tau)/ca.DM([10.]*4+[2.]*3),cfg.posture_weight*(q-ca.DM(HOME)),.3*dq,
        cfg.comp_weight*contactgap,cfg.comp_weight*contactrest,.001*j/8)
    terminal=ca.vertcat(2*cfg.energy_weight*(energy-cfg.apex),2*cfg.lateral_weight*b[:2]/.2,4*v[:2])
    first=ca.Function('arm_first',[z,p],[ca.vertcat((y[:NX]-p[:NX])/ca.DM(SCALE[:NX]),
                            (tau-p[NX:])/ca.DM(SCALE[20:27]),y[30])])
    stage=ca.Function('arm_stage',[z,zn,p],[residual,ca.vertcat(*constraints)])
    last=ca.Function('arm_last',[z,p],[terminal,y[30]])
    diagnostic=ca.Function('arm_contact',[z,zn],[gap,restitution,ca.norm_2(tangent),vn])
    return first,stage,last,diagnostic,np.array(lo),np.array(hi),groups


def build_problem(cfg,robot):
    first,stage,last,diag,lo,hi,groups=create_factors(cfg,robot)
    Z=ca.MX.sym('Z',NZ,cfg.horizon+1);p=ca.MX.sym('p',NP)
    rr=[];cc=[first(Z[:,0],p)]
    for k in range(cfg.horizon):
        r,c=stage(Z[:,k],Z[:,k+1],p);rr.append(r);cc.append(c)
    r,c=last(Z[:,-1],p);rr.append(r);cc.append(c)
    residual=np.sqrt(cfg.objective_scale)*ca.vertcat(*rr);con=ca.vertcat(*cc);flat=ca.vec(Z)
    jr=ca.jacobian(residual,flat);jc=ca.jacobian(con,flat)
    H=ca.triu(jr.T@jr+cfg.proximal*ca.MX.eye(flat.numel()))
    data=ca.Function('arm_sqp_data',[flat,p],[H,jr.T@residual,jc,con,.5*ca.dot(residual,residual)])
    value=ca.Function('arm_nlp_value',[flat,p],[.5*ca.dot(residual,residual),con])
    return dict(data=data,value=value,first=first,stage=stage,last=last,diagnostics=diag,
                cl=np.r_[np.zeros(NP+1),np.tile(lo,cfg.horizon),cfg.horizon*cfg.h],
                ch=np.r_[np.zeros(NP+1),np.tile(hi,cfg.horizon),cfg.horizon*cfg.h],
                ncon_first=NP+1,ncon_stage=len(lo),ncon_last=1,groups=groups)


def bounds(cfg,robot):
    low=np.r_[[-.6,-.6,-.15],[-4,-4,-6],robot.q_min,-VELOCITY_LIMIT,-TORQUE_LIMIT,0.,0.,cfg.h,0.]
    high=np.r_[[.6,.6,1.5],[4,4,6],robot.q_max,VELOCITY_LIMIT,TORQUE_LIMIT,12.,1.,cfg.h,cfg.horizon*cfg.h]
    return (low-OFFSET)/SCALE,(high-OFFSET)/SCALE
