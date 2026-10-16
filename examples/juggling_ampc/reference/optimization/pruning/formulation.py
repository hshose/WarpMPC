"""Contact-implicit Panda MPC with hollow-ball spin and Coulomb friction."""
from dataclasses import dataclass
import casadi as ca
from arm.ball_flight import flight
import numpy as np
from arm.model import Panda,HOME,TORQUE_LIMIT,VELOCITY_LIMIT,ACCEL_LIMIT,TORQUE_RATE_LIMIT,JERK_LIMIT

NX,NZ,NP=23,34,30
SCALE=np.r_[[.5,.5,1.,4.,4.,4.],np.ones(7),np.full(7,2.),np.full(3,20.),[40.]*4+[6.]*3,8.,1.,.03,1.]
OFFSET=np.r_[np.zeros(6),HOME,np.zeros(21)]

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
    workspace_forward:float=.4
    workspace_side:float=.4
    landing_forward:float=.35
    landing_side:float=.35
    fixture_landing_forward:float=.12
    fixture_landing_side:float=.12
    return_to_nominal_s:float=0.
    contact_edge_margin:float=.02
    contact_center_weight:float=0.
    contact_center_grip_offset:float=0.  # Positive shifts the preferred impact toward CAD -X (grip).
    restitution:float=.54
    drag:float=.34
    gravity_x:float=0.
    gravity_y:float=0.
    gravity_z:float=-9.81
    friction:float=.3
    inertia_ratio:float=2/3
    apex:float=.75
    comp_weight:float=200.
    energy_weight:float=12.
    lateral_weight:float=8.
    ball_velocity_weight:float=1.
    paddle_lateral_weight:float=1.
    pose_weight:float=2.
    velocity_weight:float=1.
    accel_weight:float=.4
    torque_weight:float=.02
    rate_weight:float=.15
    posture_weight:float=.3
    actuator_margin:float=1.
    velocity_margin:float=1.
    rate_margin:float=1.
    jerk_margin:float=1.
    warmstart_feedback_time:float=0.
    initial_acceleration_margin:float=0.
    objective_scale:float=.0001
    proximal:float=.00001
    trust:float=.5
    rho:float=.001
    codegen:bool=True
    codegen_optimization:str='O0'
    pruning_prefix:int=5
    pruning_groups:tuple=('workspace','tilt','upward','paddle_velocity','jerk')
    deduplicate_endpoints:bool=False


def create_factors(cfg,robot,drop=(),left_endpoint_only=False,suffix='',runtime=None):
    if runtime is None:
        if not 0 <= cfg.contact_edge_margin < cfg.paddle_radius:raise ValueError('Contact margin must be smaller than paddle radius')
        if not np.isfinite(cfg.contact_center_weight) or cfg.contact_center_weight < 0:raise ValueError('Contact center weight must be finite and nonnegative')
        if not np.isfinite(cfg.contact_center_grip_offset) or not 0 <= cfg.contact_center_grip_offset < cfg.paddle_radius:raise ValueError('Preferred contact offset must lie inside the paddle')
        if not np.isfinite([cfg.gravity_x,cfg.gravity_y,cfg.gravity_z]).all() or cfg.gravity_z>=0:raise ValueError('Ball gravity must be finite and downward')
    z=ca.MX.sym('z',NZ);zn=ca.MX.sym('zn',NZ);p=ca.MX.sym('p',NP)
    y=z*ca.DM(SCALE)+ca.DM(OFFSET);yn=zn*ca.DM(SCALE)+ca.DM(OFFSET)
    b,v,q,dq,omega,tau,j,f,h=y[:3],y[3:6],y[6:13],y[13:20],y[20:23],y[23:30],y[30],y[31],y[32]
    qn,dqn,taun=yn[6:13],yn[13:20],yn[23:30]
    qm=.5*(q+qn);vm=.5*(dq+dqn);acc=(dqn-dq)/h;tc=f*h
    qc=q+tc*dq+.5*tc*tc*acc;vc=dq+tc*acc
    pc,n,Jc,Wc,Rc=robot.contact_fk(qc);pc=pc-ca.DM(robot.origin)
    pos,normal,J,N,_=robot.fk(q);pos=pos-ca.DM(robot.origin)
    posn,nn,_,_,_=robot.fk(qn);posn=posn-ca.DM(robot.origin)
    bc,ballvc=flight(b,v,tc,cfg.drag,[cfg.gravity_x,cfg.gravity_y,cfg.gravity_z])
    rel=bc-pc;gap=ca.dot(n,rel)-cfg.radius
    tangent=rel-n*ca.dot(n,rel)
    Jcontact=Jc-ca.skew(tangent)@Wc
    # Relative surface velocity, including the hollow ball's spin.
    relative=ballvc-ca.cross(omega,cfg.radius*n)-Jcontact@vc
    vn=ca.dot(n,relative);T=Rc[:,:2]
    K=ca.MX.eye(3)+(ca.MX.eye(3)-n@n.T)/cfg.inertia_ratio+cfg.mass*Jcontact@ca.solve(robot.mass(qc),Jcontact.T)
    desired_tangent=-ca.solve(T.T@K@T,T.T@(relative+K@n*j))
    # Smooth Coulomb saturation: zero impulse at j=0; approaches sticking
    # for small slip and the friction cone for large slip.
    capacity=cfg.friction*j
    jt=desired_tangent*capacity/ca.sqrt(capacity**2+ca.dot(desired_tangent,desired_tangent)+1e-10)
    impulse=j*n+T@jt
    restitution=ca.dot(n,K@impulse)+(1+cfg.restitution)*vn
    bnext,vnext=flight(bc,ballvc+impulse,h-tc,cfg.drag,[cfg.gravity_x,cfg.gravity_y,cfg.gravity_z])
    omeganext=omega-ca.cross(n,impulse)/(cfg.inertia_ratio*cfg.radius)
    torque=robot.rnea(qm,vm,acc)+cfg.mass/h*Jcontact.T@impulse
    endgap=ca.dot(nn,yn[:3]-posn)-cfg.radius
    contactgap=j/8*gap/.05;contactrest=j/8*restitution/8
    accel0=ca.solve(robot.mass(q),tau-robot.bias(q,dq))
    accel1=ca.solve(robot.mass(qn),taun-robot.bias(qn,dqn))
    constraints=[];lo=[];hi=[];groups={}
    def add(name,c,l,u):
        if name in drop:return
        start=len(lo);count=int(c.numel());constraints.append(c)
        lo.extend(np.broadcast_to(l,(count,)));hi.extend(np.broadcast_to(u,(count,)))
        groups[name]=(start,start+count)
    add('ball',ca.vertcat(yn[:3]-bnext,yn[3:6]-vnext)/ca.DM(SCALE[:6]),0,0)
    add('spin',(yn[20:23]-omeganext)/20.,0,0)
    add('position',qn-q-h*vm,0,0)
    # Precondition force balance into a velocity defect. Torque scaling alone
    # barely constrains the low-inertia wrist at a fixed, small QP budget.
    add('dynamics',h*ca.solve(robot.mass(qm),torque-.5*(tau+taun))/2.,0,0)
    add('clock',yn[33]-y[33]-h,0,0)
    add('contact',ca.vertcat(gap/.5,endgap/.5,j/8*(ca.dot(tangent,tangent)/(cfg.paddle_radius-cfg.contact_edge_margin)**2-1),j/8*vn/4),[0,0,-np.inf,-np.inf],[np.inf,np.inf,0,0])
    add('workspace',pos/ca.DM([.4,.4,.18]),
        [-1,-cfg.workspace_side/.4,-1],[cfg.workspace_forward/.4,cfg.workspace_side/.4,1])
    add('tilt',normal[:2]/np.sin(.4),-1,1)
    add('upward',normal[2],np.cos(.4),np.inf)
    add('paddle_velocity',J@dq/1.2,-1,1)
    add('acceleration',acc/ca.DM(ACCEL_LIMIT),-cfg.actuator_margin,cfg.actuator_margin)
    endpoint=accel0/ca.DM(ACCEL_LIMIT) if left_endpoint_only else ca.vertcat(accel0,accel1)/ca.DM(np.tile(ACCEL_LIMIT,2))
    add('endpoint_acceleration',endpoint,-cfg.actuator_margin,cfg.actuator_margin)
    add('torque_rate',(taun-tau)/h/ca.DM(TORQUE_RATE_LIMIT),-cfg.actuator_margin*cfg.rate_margin,cfg.actuator_margin*cfg.rate_margin)
    add('jerk',(accel1-accel0)/h/ca.DM(JERK_LIMIT),-cfg.actuator_margin*cfg.jerk_margin,cfg.actuator_margin*cfg.jerk_margin)
    energy=b[2]-cfg.radius+v[2]**2/(-2*cfg.gravity_z)
    residual=ca.vertcat(cfg.energy_weight*(energy-cfg.apex),cfg.lateral_weight*b[:2]/.2,2*cfg.ball_velocity_weight*v[:2],
        cfg.pose_weight*pos*ca.vertcat(cfg.paddle_lateral_weight/.3,cfg.paddle_lateral_weight/.3,1/.15),.3*normal[:2]/.3,
        cfg.velocity_weight*(J@dq),.5*cfg.velocity_weight*(N@dq),
        cfg.accel_weight*acc/ca.DM(ACCEL_LIMIT),cfg.torque_weight*tau/ca.DM(TORQUE_LIMIT),
        cfg.rate_weight*(taun-tau)/ca.DM([10.]*4+[2.]*3),cfg.posture_weight*(q-ca.DM(HOME)),.3*dq,
        cfg.comp_weight*contactgap,cfg.comp_weight*contactrest,.001*j/8)
    # Discrete impact penalty: do not scale these appended rows by mesh dt.
    if runtime is not None or cfg.contact_center_weight:
        center_error=tangent
        if runtime is not None or cfg.contact_center_grip_offset:
            # CAD head extends along +X; grip is -X. Express its projection
            # in the corrected contact-plane axes, including the mount yaw.
            grip=robot.paddle_normal_rotation.T@ca.DM([-1.,0.,0.])
            grip=ca.vertcat(grip[0],grip[1],0);grip=grip/ca.norm_2(grip)
            preferred=cfg.contact_center_grip_offset*(Rc@grip)
            center_error=tangent-preferred
        residual=ca.vertcat(residual,cfg.contact_center_weight*(j/8)*center_error/cfg.paddle_radius)
    terminal=ca.vertcat(2*cfg.energy_weight*(energy-cfg.apex),2*cfg.lateral_weight*b[:2]/.2,4*cfg.ball_velocity_weight*v[:2])
    extra=[] if runtime is None else [runtime]
    first=ca.Function('arm_first',[z,p]+extra,[ca.vertcat((y[:NX]-p[:NX])/ca.DM(SCALE[:NX]),
                            (tau-p[NX:])/ca.DM(SCALE[23:30]),y[33])])
    stage=ca.Function('arm_stage'+suffix,[z,zn,p]+extra,[residual,ca.vertcat(*constraints)])
    last=ca.Function('arm_last',[z,p]+extra,[terminal,y[33]])
    diagnostic=ca.Function('arm_contact',[z,zn]+extra,[gap,restitution,ca.norm_2(tangent),vn])
    return first,stage,last,diagnostic,np.array(lo),np.array(hi),groups


def build_problem(cfg,robot):
    first,stage,last,diag,lo,hi,groups=create_factors(cfg,robot)
    _,reduced,_,_,lr,ur,gr=create_factors(cfg,robot,drop=cfg.pruning_groups,
        left_endpoint_only=cfg.deduplicate_endpoints,suffix='_reduced')
    _,tail,_,_,lt,ut,gt=create_factors(cfg,robot,drop=cfg.pruning_groups,suffix='_tail')
    Z=ca.MX.sym('Z',NZ,cfg.horizon+1);p=ca.MX.sym('p',NP)
    rr=[];cc=[first(Z[:,0],p)];lower=[np.zeros(NP+1)];upper=[np.zeros(NP+1)]
    kept_rows=[];offsets=[NP+1]
    for k in range(cfg.horizon):
        if k<cfg.pruning_prefix:f,l,u,g=stage,lo,hi,groups
        elif k==cfg.horizon-1:f,l,u,g=tail,lt,ut,gt
        else:f,l,u,g=reduced,lr,ur,gr
        r,c=f(Z[:,k],Z[:,k+1],p);rr.append(r);cc.append(c)
        lower.append(l);upper.append(u);offsets.append(offsets[-1]+len(l))
        rows=[]
        for group,(start,end) in g.items():
            original_start=groups[group][0]
            rows.extend(range(original_start,original_start+end-start))
        kept_rows.append(np.array(rows,dtype=int))
    r,c=last(Z[:,-1],p);rr.append(r);cc.append(c)
    lower.append([cfg.horizon*cfg.h]);upper.append([cfg.horizon*cfg.h])
    residual=np.sqrt(cfg.objective_scale)*ca.vertcat(*rr);con=ca.vertcat(*cc);flat=ca.vec(Z)
    jr=ca.jacobian(residual,flat);jc=ca.jacobian(con,flat)
    H=ca.triu(jr.T@jr+cfg.proximal*ca.MX.eye(flat.numel()))
    data=ca.Function('arm_sqp_data',[flat,p],[H,jr.T@residual,jc,con,.5*ca.dot(residual,residual)])
    value=ca.Function('arm_nlp_value',[flat,p],[.5*ca.dot(residual,residual),con])
    return dict(data=data,value=value,first=first,stage=stage,last=last,diagnostics=diag,
                cl=np.concatenate(lower),ch=np.concatenate(upper),
                ncon_first=NP+1,ncon_stage=len(lo),ncon_last=1,groups=groups,
                kept_rows=kept_rows,stage_offsets=np.array(offsets),reduced_stage=reduced,tail_stage=tail)


def bounds(cfg,robot):
    if not 0 < cfg.velocity_margin <= 1:raise ValueError('Velocity reserve must be in (0, 1]')
    low=np.r_[[-.6,-.6,-.15],[-4,-4,-6],robot.q_min,-cfg.velocity_margin*VELOCITY_LIMIT,np.full(3,-100.),-TORQUE_LIMIT,0.,0.,cfg.h,0.]
    high=np.r_[[.6,.6,1.5],[4,4,6],robot.q_max,cfg.velocity_margin*VELOCITY_LIMIT,np.full(3,100.),TORQUE_LIMIT,12.,1.,cfg.h,cfg.horizon*cfg.h]
    return (low-OFFSET)/SCALE,(high-OFFSET)/SCALE
