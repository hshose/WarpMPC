"""Optional prediction-only simplifications of the legacy arm/contact factors.

These options do not change the empirical plant or hardware defaults. Removing
ball reaction treats the paddle as prescribed during an impulse: the arm's
inverse effective mass is omitted from K AND the reaction wrench is omitted from
midpoint force balance. Paddle motion, contact-point velocity, ball spin and
Coulomb friction remain. Ball mass/payload parameters are not changed.

The constant midpoint preconditioner is the complete inverse of M(HOME), computed
from the supplied robot including its configured payload and mounting rotation.
It replaces only the left preconditioner of the midpoint force-balance defect;
RNEA itself and BOTH endpoint inverse-mass calculations stay configuration
dependent. Since this constant matrix is positive definite, the exact equality
zero set is unchanged. Defect units remain rad/s, but relative row scaling and
Gauss--Newton curvature away from feasible points change. In particular, this
retains wrist inertia scaling rather than using uniform torque normalization.

All-off delegates to the original builders. Shapes, ordering, group IDs, costs
and bounds remain legacy compatible. The separately loaded plane correction must
use the SAME switches; edge-restitution and private ball-bias helpers are purely
kinematic and are unchanged. No graph-level options are added to runtime tuning.
"""

import casadi as ca
import numpy as np

from arm.ball_flight import flight
from arm.model import HOME, TORQUE_LIMIT, ACCEL_LIMIT, TORQUE_RATE_LIMIT, JERK_LIMIT
from optimization.pruning.formulation import (
    NX, NZ, NP, SCALE, OFFSET, create_factors as legacy_create_factors)
from optimization.contact_plane_prediction import (
    RESIDUAL_ROWS, CONSTRAINT_ROWS,
    create_contact_plane_correction as legacy_plane_correction)
from optimization.runtime_parameters import SIZE


def nominal_midpoint_mass(robot):
    """Return the exact nominal HOME mass matrix with this robot's payload.

    ``runtime_parameters.Robot`` forwards mass to its numeric base robot, so this
    is independent of symbolic ball/cost/normal-rotation runtime parameters.
    """
    matrix = np.asarray(robot.mass(HOME), dtype=float)
    if matrix.shape != (7, 7) or not np.isfinite(matrix).all():
        raise ValueError('Nominal arm mass must be a finite numeric 7 by 7 matrix')
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError('Nominal arm mass must be symmetric')
    np.linalg.cholesky(matrix)
    return matrix


def nominal_midpoint_preconditioner(robot):
    """Return M(HOME)^-1 without diagonalization, retuning, or regularization."""
    return np.linalg.solve(nominal_midpoint_mass(robot), np.eye(7))


def create_factors(cfg,robot,drop=(),left_endpoint_only=False,suffix='',runtime=None,
                   *,neglect_ball_reaction=False,constant_midpoint_preconditioner=False):
    """Build legacy-shaped factors with independently optional simplifications.

    Endpoint accelerations and jerk retain configuration-dependent full-arm
    dynamics. Neither option changes residual order, constraint groups or bounds.
    The default delegates to the original builder without changing its graph.
    """
    if not neglect_ball_reaction and not constant_midpoint_preconditioner:
        return legacy_create_factors(cfg,robot,drop,left_endpoint_only,suffix,runtime)
    P_home=ca.DM(nominal_midpoint_preconditioner(robot)) if constant_midpoint_preconditioner else None
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
    K=ca.MX.eye(3)+(ca.MX.eye(3)-n@n.T)/cfg.inertia_ratio
    if not neglect_ball_reaction:
        K+=cfg.mass*Jcontact@ca.solve(robot.mass(qc),Jcontact.T)
    desired_tangent=-ca.solve(T.T@K@T,T.T@(relative+K@n*j))
    # Smooth Coulomb saturation: zero impulse at j=0; approaches sticking
    # for small slip and the friction cone for large slip.
    capacity=cfg.friction*j
    jt=desired_tangent*capacity/ca.sqrt(capacity**2+ca.dot(desired_tangent,desired_tangent)+1e-10)
    impulse=j*n+T@jt
    restitution=ca.dot(n,K@impulse)+(1+cfg.restitution)*vn
    bnext,vnext=flight(bc,ballvc+impulse,h-tc,cfg.drag,[cfg.gravity_x,cfg.gravity_y,cfg.gravity_z])
    omeganext=omega-ca.cross(n,impulse)/(cfg.inertia_ratio*cfg.radius)
    torque=robot.rnea(qm,vm,acc)
    if not neglect_ball_reaction:
        torque+=cfg.mass/h*Jcontact.T@impulse
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
    force_defect=torque-.5*(tau+taun)
    velocity_defect=P_home@force_defect if constant_midpoint_preconditioner else ca.solve(robot.mass(qm),force_defect)
    add('dynamics',h*velocity_defect/2.,0,0)
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



def create_contact_plane_correction(cfg, robot, runtime, plane_offset=None, *,
                                    neglect_ball_reaction=False,
                                    constant_midpoint_preconditioner=False):
    """Return data/value functions using ``[z68,p30,tuning35,plane_offset]``.

    Data outputs are ``[dr2,dJr2x68,dc18,dJc18x68]``; value outputs are
    ``[dr2,dc18]``. ``dr`` already includes objective/contact-weight scaling;
    ``dc`` already uses the original constraint normalization. Row destinations
    are RESIDUAL_ROWS and CONSTRAINT_ROWS in the *full* original stage layout.
    Map constraint IDs when assembling pruned stage types.

    ``cfg`` and ``robot`` must use the supplied symbolic runtime vector. Native
    callers should bypass the helper at zero offset to preserve baseline floating
    point execution exactly. The runtime rotation must remain a proper rotation,
    as required by the existing factor ABI.
    """
    if not neglect_ball_reaction and not constant_midpoint_preconditioner:
        return legacy_plane_correction(cfg, robot, runtime, plane_offset)
    P_home = ca.DM(nominal_midpoint_preconditioner(robot)) if constant_midpoint_preconditioner else None
    if runtime.shape != (SIZE, 1):
        raise ValueError('Contact plane requires the unchanged runtime vector')
    if plane_offset is None:
        plane_offset = ca.MX.sym('plane_offset')
    if plane_offset.shape != (1, 1):
        raise ValueError('Contact plane offset must be a scalar input symbol')

    z = ca.MX.sym('z', 2 * NZ)
    p = ca.MX.sym('p', NP)
    y = z[:NZ] * ca.DM(SCALE) + ca.DM(OFFSET)
    yn = z[NZ:] * ca.DM(SCALE) + ca.DM(OFFSET)
    q, dq, qn, dqn = y[6:13], y[13:20], yn[6:13], yn[13:20]
    omega, j, fraction, h = y[20:23], y[30], y[31], y[32]
    contact_time = fraction * h
    acceleration = (dqn-dq)/h
    contact_q = q+contact_time*dq+.5*contact_time**2*acceleration
    contact_dq = dq+contact_time*acceleration
    position, normal, jacobian, angular_jacobian, rotation = robot.contact_fk(contact_q)
    ball_position, ball_velocity = flight(
        y[:3], y[3:6], contact_time, cfg.drag,
        [cfg.gravity_x, cfg.gravity_y, cfg.gravity_z])
    relative_position = ball_position-position+ca.DM(robot.origin)
    tangent = relative_position-normal*ca.dot(normal, relative_position)
    J0 = jacobian-ca.skew(tangent) @ angular_jacobian
    Jdelta = J0-ca.skew(plane_offset*normal) @ angular_jacobian
    # A displacement along n leaves the tangent and normal surface speed
    # unchanged. Its tangential velocity and effective mass still change.
    if not neglect_ball_reaction:
        responses = ca.solve(robot.mass(contact_q), ca.horzcat(J0.T, Jdelta.T))
    ball_inverse_inertia = ca.MX.eye(3)+(ca.MX.eye(3)-normal @ normal.T)/cfg.inertia_ratio
    tangents = rotation[:, :2]
    impulse_maps = []
    for index, J in enumerate((J0, Jdelta)):
        K = ball_inverse_inertia
        if not neglect_ball_reaction:
            K += cfg.mass*J @ responses[:, 3*index:3*index+3]
        relative_velocity = ball_velocity-ca.cross(omega, cfg.radius*normal)-J @ contact_dq
        desired = -ca.solve(tangents.T @ K @ tangents,
                            tangents.T @ (relative_velocity+K @ normal*j))
        capacity = cfg.friction*j
        tangential_impulse = desired*capacity/ca.sqrt(capacity**2+ca.dot(desired, desired)+1e-10)
        impulse = j*normal+tangents @ tangential_impulse
        impulse_maps.append((impulse, K))
    u0, K0 = impulse_maps[0]
    udelta, Kdelta = impulse_maps[1]
    delta_impulse = udelta-u0
    delta_wrench = Jdelta.T @ udelta-J0.T @ u0
    delta_position, delta_velocity = flight(
        ca.MX.zeros(3), delta_impulse, h-contact_time, cfg.drag, [0., 0., 0.])
    delta_ball = -ca.vertcat(delta_position, delta_velocity)/ca.DM(SCALE[:6])
    delta_spin = ca.cross(normal, delta_impulse)/(cfg.inertia_ratio*cfg.radius*20.)
    if neglect_ball_reaction:
        delta_dynamics = ca.MX.zeros(7)
    elif constant_midpoint_preconditioner:
        delta_dynamics = cfg.mass/2.*(P_home @ delta_wrench)
    else:
        delta_dynamics = cfg.mass/2.*ca.solve(robot.mass(.5*(q+qn)), delta_wrench)
    delta_constraints = ca.vertcat(delta_ball, delta_spin, delta_dynamics,
                                   -plane_offset/.5, -plane_offset/.5)
    # (1+e)*vn cancels exactly, including the independently applied edge law.
    delta_restitution = ca.dot(normal, Kdelta @ udelta-K0 @ u0)
    delta_residuals = ca.sqrt(cfg.objective_scale)*cfg.comp_weight*ca.vertcat(
        -(j/8.)*plane_offset/.05, (j/64.)*delta_restitution)
    inputs = [z, p, runtime, plane_offset]
    names = ['z', 'p', 'tuning', 'plane_offset']
    data = ca.Function('contact_plane_prediction_data', inputs,
                       [delta_residuals, ca.jacobian(delta_residuals, z),
                        delta_constraints, ca.jacobian(delta_constraints, z)],
                       names, ['residual_correction', 'residual_jacobian',
                               'constraint_correction', 'constraint_jacobian'])
    value = ca.Function('contact_plane_prediction_value', inputs,
                        [delta_residuals, delta_constraints], names,
                        ['residual_correction', 'constraint_correction'])
    return data, value
