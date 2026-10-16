"""Offline structural variant matching the native acceleration-ramp reference.

The legacy model is retained as ``create_legacy_factors``. An exporter can choose
ramp factors for its first K stages and legacy factors afterwards, or choose ramp
factors everywhere. Selection is structural, with no new runtime parameter, and
all residual rows, constraint groups, scalings and function signatures are kept.
Optional prediction parameters adjust only the predicted contact plane and
radial restitution; they do not move the robot pose reference or physical plant.

Endpoint torques define endpoint accelerations through the same rigid-body model
as ``tracking.rehearsal.executable_trajectory``. Joint acceleration varies linearly
between these endpoints; contact position and velocity use the exact integrated
ramp. This predicts the reference, not the full calibrated servo: lag, 6 ms dead
time, feedback, acceleration clipping and jerk clipping remain plant mismatch.

The empirical plant does not apply ball-impulse recoil to the servo-controlled
arm. Accordingly there is no synthetic recoil in the joint velocity equality.
The existing finite-arm effective-mass term in the BALL impact law is retained,
matching that plant. This is not a momentum-conserving coupled impact model; it
is an intentionally consistent approximation for the 3.9 g ball and velocity
interface. Do not deploy it as a torque-controlled rigid-body collision model.
"""
import casadi as ca
import numpy as np
from arm.ball_flight import flight
from arm.model import (Panda, HOME, TORQUE_LIMIT, VELOCITY_LIMIT, ACCEL_LIMIT,
                       TORQUE_RATE_LIMIT, JERK_LIMIT)
from optimization.pruning.formulation import (Config, NX, NZ, NP, SCALE, OFFSET,
                                               bounds, create_factors as create_legacy_factors)


def ramp_kinematics(q, dq, acceleration0, acceleration1, h, time):
    """Exact integral of a linear acceleration ramp; NumPy/CasADi compatible."""
    jerk=(acceleration1-acceleration0)/h
    return (q+time*dq+.5*time*time*acceleration0+time**3*jerk/6.,
            dq+time*acceleration0+.5*time*time*jerk)


def create_ramp_factors(cfg,robot,drop=(),left_endpoint_only=False,suffix='',runtime=None,
                        *,prediction_parameters=None):
    """Build ramp factors with optional scalar or symbolic contact corrections.

    ``plane_offset_m`` moves the predicted face along its corrected normal;
    ``edge_loss`` subtracts a quartic radial restitution loss at radius 75 mm.
    Symbolic corrections must belong to the supplied ``runtime`` input. Omitting
    the mapping retains the original ramp factor graph exactly.
    """
    if runtime is None:
        if not 0 <= cfg.contact_edge_margin < cfg.paddle_radius:raise ValueError('Contact margin must be smaller than paddle radius')
        if not np.isfinite(cfg.contact_center_weight) or cfg.contact_center_weight < 0:raise ValueError('Contact center weight must be finite and nonnegative')
        if not np.isfinite(cfg.contact_center_grip_offset) or not 0 <= cfg.contact_center_grip_offset < cfg.paddle_radius:raise ValueError('Preferred contact offset must lie inside the paddle')
        if not np.isfinite([cfg.gravity_x,cfg.gravity_y,cfg.gravity_z]).all() or cfg.gravity_z>=0:raise ValueError('Ball gravity must be finite and downward')
    z=ca.MX.sym('z',NZ);zn=ca.MX.sym('zn',NZ);p=ca.MX.sym('p',NP)
    y=z*ca.DM(SCALE)+ca.DM(OFFSET);yn=zn*ca.DM(SCALE)+ca.DM(OFFSET)
    b,v,q,dq,omega,tau,j,f,h=y[:3],y[3:6],y[6:13],y[13:20],y[20:23],y[23:30],y[30],y[31],y[32]
    qn,dqn,taun=yn[6:13],yn[13:20],yn[23:30]
    acc=(dqn-dq)/h;tc=f*h
    accel0=ca.solve(robot.mass(q),tau-robot.bias(q,dq))
    accel1=ca.solve(robot.mass(qn),taun-robot.bias(qn,dqn))
    qc,vc=ramp_kinematics(q,dq,accel0,accel1,h,tc)
    qend,vend=ramp_kinematics(q,dq,accel0,accel1,h,h)
    pc,n,Jc,Wc,Rc=robot.contact_fk(qc);pc=pc-ca.DM(robot.origin)
    if prediction_parameters is not None:
        plane_offset=prediction_parameters.get('plane_offset_m',0.)
        offset=plane_offset*n
        pc=pc+offset
        # The displaced face center rotates with the body. Keep the nominal
        # FK below unchanged for pose costs and workspace/velocity constraints.
        Jc=Jc-ca.skew(offset)@Wc
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
    contact_restitution=cfg.restitution
    if prediction_parameters is not None:
        edge_loss=prediction_parameters.get('edge_loss',0.)
        radial_quartic=(ca.dot(tangent,tangent)/(.075**2))**2
        contact_restitution=ca.fmin(1.5,ca.fmax(0.,cfg.restitution-edge_loss*radial_quartic))
    restitution=ca.dot(n,K@impulse)+(1+contact_restitution)*vn
    bnext,vnext=flight(bc,ballvc+impulse,h-tc,cfg.drag,[cfg.gravity_x,cfg.gravity_y,cfg.gravity_z])
    omeganext=omega-ca.cross(n,impulse)/(cfg.inertia_ratio*cfg.radius)
    endgap=ca.dot(nn,yn[:3]-posn)-cfg.radius
    if prediction_parameters is not None:endgap-=plane_offset
    contactgap=j/8*gap/.05;contactrest=j/8*restitution/8
    constraints=[];lo=[];hi=[];groups={}
    def add(name,c,l,u):
        if name in drop:return
        start=len(lo);count=int(c.numel());constraints.append(c)
        lo.extend(np.broadcast_to(l,(count,)));hi.extend(np.broadcast_to(u,(count,)))
        groups[name]=(start,start+count)
    add('ball',ca.vertcat(yn[:3]-bnext,yn[3:6]-vnext)/ca.DM(SCALE[:6]),0,0)
    add('spin',(yn[20:23]-omeganext)/20.,0,0)
    add('position',qn-qend,0,0)
    # Same seven rows and velocity scaling as the legacy force balance.
    # Match the commanded ramp: the empirical velocity-servo arm does not
    # receive synthetic ball recoil. Ball effective mass above stays intact.
    add('dynamics',(dqn-vend)/2.,0,0)
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


# Same import convention as pruning/formulation.py. Exporters selecting a mixed
# horizon should call create_ramp_factors and create_legacy_factors separately.
create_factors=create_ramp_factors


def sanity_check(samples=8):
    """Small no-build check against the real executable-reference projection.

    Produces feasible four-knot ramp references, then checks executable_trajectory
    and the new stage position/dynamics rows. All accelerations stay below clip
    bounds, and node zero is anchored at its generated measured state.
    """
    from tracking.rehearsal import executable_trajectory

    rng=np.random.default_rng(4206)
    robot=Panda()
    cfg=Config(mass=.0039,contact_center_weight=8.)
    _,stage,_,_,lo,hi,groups=create_ramp_factors(cfg,robot)
    legacy=create_legacy_factors(cfg,robot)
    assert groups==legacy[-1]
    assert stage.size1_out(0)==legacy[1].size1_out(0)==57
    assert stage.size1_out(1)==legacy[1].size1_out(1)
    np.testing.assert_array_equal(lo,legacy[4])
    np.testing.assert_array_equal(hi,legacy[5])
    maxima=dict(position_m_rad=0.,velocity_rad_s=0.,acceleration_rad_s2=0.,
                factor_position=0.,factor_velocity=0.,contact_gap_m=0.)
    for sample in range(samples):
        steps=rng.uniform(.015,.105,3)
        a=rng.uniform(-.08,.08,(4,7))*ACCEL_LIMIT
        Y=np.zeros((4,NZ))
        Y[0,6:13]=HOME+rng.uniform(-.025,.025,7)
        Y[0,13:20]=rng.uniform(-.06,.06,7)
        for k in range(4):
            Y[k,:3]=[0.,0.,.7]
            Y[k,23:30]=np.asarray(robot.rnea(Y[k,6:13],Y[k,13:20],a[k])).ravel()
            Y[k,31]=rng.uniform(.1,.9)
            Y[k,32]=steps[min(k,2)]
            if k<3:
                Y[k+1,6:13],Y[k+1,13:20]=ramp_kinematics(
                    Y[k,6:13],Y[k,13:20],a[k],a[k+1],steps[k],steps[k])
        meta=dict(arrays=dict(acceleration_limit=ACCEL_LIMIT),stage_steps=steps)
        trajectory,ratio,projection=executable_trajectory(
            robot,meta,Y,Y[0,6:13],Y[0,13:20],a[0])
        assert ratio<.1 and projection==0.
        maxima['position_m_rad']=max(maxima['position_m_rad'],float(np.max(abs(trajectory[:,:7]-Y[:,6:13]))))
        maxima['velocity_rad_s']=max(maxima['velocity_rad_s'],float(np.max(abs(trajectory[:,7:14]-Y[:,13:20]))))
        maxima['acceleration_rad_s2']=max(maxima['acceleration_rad_s2'],float(np.max(abs(trajectory[:,21:28]-a))))
        for k in range(3):
            _,c=stage((Y[k]-OFFSET)/SCALE,(Y[k+1]-OFFSET)/SCALE,np.zeros(NP))
            c=np.asarray(c).ravel()
            for group,metric in [('position','factor_position'),('dynamics','factor_velocity')]:
                begin,end=groups[group]
                maxima[metric]=max(maxima[metric],float(np.max(abs(c[begin:end]))))
            # Independent cubic Hermite position / derivative reconstructs the
            # same ramp using only its two q/dq endpoints.
            h=steps[k];f=Y[k,31];q0=Y[k,6:13];q1=Y[k+1,6:13]
            v0=Y[k,13:20];v1=Y[k+1,13:20]
            qc=(2*f**3-3*f*f+1)*q0+(f**3-2*f*f+f)*h*v0+(-2*f**3+3*f*f)*q1+(f**3-f*f)*h*v1
            vc=(6*f*f-6*f)/h*q0+(3*f*f-4*f+1)*v0+(-6*f*f+6*f)/h*q1+(3*f*f-2*f)*v1
            qr,vr=ramp_kinematics(q0,v0,a[k],a[k+1],h,f*h)
            np.testing.assert_allclose(qc,qr,rtol=0.,atol=1e-12)
            np.testing.assert_allclose(vc,vr,rtol=0.,atol=1e-11)
            pc,n,*_=robot.contact_fk(qc)
            pc=np.asarray(pc).ravel()-robot.origin;n=np.asarray(n).ravel()
            bc,_=flight(ca.DM(Y[k,:3]),ca.DM(Y[k,3:6]),f*h,cfg.drag,
                        [cfg.gravity_x,cfg.gravity_y,cfg.gravity_z])
            gap=float(n@(np.asarray(bc).ravel()-pc)-cfg.radius)
            maxima['contact_gap_m']=max(maxima['contact_gap_m'],abs(.5*c[groups['contact'][0]]-gap))
    assert max(maxima.values())<1e-10,maxima
    return dict(samples=samples,stages_checked=3*samples,maximum_errors=maxima,
                residual_rows=stage.size1_out(0),constraint_rows=stage.size1_out(1),
                physical_robot_used=False,compiled=False)


if __name__=='__main__':
    import json
    print(json.dumps(sanity_check(),indent=2))
