"""Adjacent-node CasADi factors; all optimizer variables are dimensionless."""
from dataclasses import dataclass, asdict
import casadi as ca
import numpy as np
from .specs import BALL_MASS_KG, BALL_RADIUS_M, PADDLE_RADIUS_M

NX, NU, NZ = 16, 5, 25
# [ball xyz, ball vxyz, paddle xyz/roll/pitch, paddle vxyz/rollrate/pitchrate,
#  paddle accelerations, impulse/mass, event fraction, interval duration]
SCALE = np.array([.5,.5,1.,4.,4.,4., .3,.3,.2,.3,.3, 1.,1.,1.,2.,2.,
                  10.,10.,10.,20.,20., 8.,1.,.03,1.])

@dataclass
class Config:
    horizon: int = 40
    h_min: float = .01
    h_max: float = .05
    h_nom: float = .03
    variable_time: bool = False
    mass: float = BALL_MASS_KG
    radius: float = BALL_RADIUS_M
    paddle_radius: float = PADDLE_RADIUS_M
    restitution: float = .88
    apex: float = .75
    sqp_iters: int = 2
    osqp_iters: int = 100
    comp_weight: float = 200.
    energy_weight: float = 12.
    lateral_weight: float = 8.
    accel_weight: float = 1.
    velocity_weight: float = 1.
    pose_weight: float = 2.
    jerk_weight: float = .4
    objective_scale: float = .0001
    min_horizon: float = .9
    proximal: float = .00001
    trust: float = .5
    rho: float = .001
    codegen: bool = False


def normal(a):
    roll, pitch = a[0], a[1]
    return ca.vertcat(ca.sin(pitch)*ca.cos(roll), -ca.sin(roll),
                      ca.cos(pitch)*ca.cos(roll))


def create_factors(cfg):
    z, zn, p = ca.SX.sym('z', NZ), ca.SX.sym('zn', NZ), ca.SX.sym('p', NX)
    y, yn = z*ca.DM(SCALE), zn*ca.DM(SCALE)
    b,v,pose,w,u,j,f,h = y[:3],y[3:6],y[6:11],y[11:16],y[16:21],y[21],y[22],y[23]
    tau = h*f
    grav = ca.DM([0,0,-9.81])
    pc = pose + tau*w + .5*tau*tau*u
    wc = w + tau*u
    bc = b + tau*v + .5*tau*tau*grav
    vc = v + tau*grav
    n = normal(pc[3:5])
    aa = ca.SX.sym('aa',2)
    dn = ca.substitute(ca.jacobian(normal(aa),aa),aa,pc[3:5]) @ wc[3:5]
    rel = bc-pc[:3]
    gap = ca.dot(n,rel)-cfg.radius
    vn = ca.dot(n,vc-wc[:3])+ca.dot(dn,rel)
    bnext = b+h*v+.5*h*h*grav+(h-tau)*j*n
    vnext = v+h*grav+j*n
    posenext = pose+h*w+.5*h*h*u
    wnext = w+h*u
    predicted = ca.vertcat(bnext,vnext,posenext,wnext)
    dyn = (yn[:NX]-predicted)/ca.DM(SCALE[:NX])
    endgap = ca.dot(normal(yn[9:11]),yn[:3]-yn[6:9])-cfg.radius
    tangent = rel-n*ca.dot(n,rel)
    # Contact can be created and removed by SQP: products enter a penalty, not
    # a fixed active-contact schedule. Physical defects are reported explicitly.
    contact_gap = (j/8.)*(gap/.05)
    restitution = (j/8.)*((j+(1+cfg.restitution)*vn)/8.)
    reach = (j/8.)*(ca.dot(tangent,tangent)/(cfg.paddle_radius-cfg.radius)**2-1.)
    closing = (j/8.)*vn/4.
    c = ca.vertcat(dyn,yn[24]-y[24]-h,gap/.5,endgap/.5,reach,closing)
    lo = np.r_[np.zeros(NX+1),0.,0.,-np.inf,-np.inf]
    hi = np.r_[np.zeros(NX+1),np.inf,np.inf,0.,0.]
    energy = b[2]-cfg.radius+v[2]**2/(2*9.81)
    flight = ca.sqrt(h/cfg.h_nom)
    r = ca.vertcat(
        flight*cfg.energy_weight*(energy-cfg.apex),
        flight*cfg.lateral_weight*b[:2]/.2,
        flight*2.*v[:2],
        flight*cfg.pose_weight*pose[:3]/ca.DM([.3,.3,.15]),
        flight*.3*pose[3:5]/.3,
        flight*cfg.velocity_weight*w/ca.DM([1.,1.,1.,2.,2.]),
        flight*cfg.accel_weight*u/ca.DM([10.,10.,10.,20.,20.]),
        cfg.jerk_weight*(yn[16:21]-u)/ca.DM([10.,10.,10.,20.,20.]),
        cfg.comp_weight*contact_gap,
        cfg.comp_weight*restitution,
        .2*(h-cfg.h_nom)/cfg.h_nom,
        .001*j/8.)
    terminal_energy = y[2]-cfg.radius+y[5]**2/(2*9.81)
    rt = ca.vertcat(2*cfg.energy_weight*(terminal_energy-cfg.apex),
                    2*cfg.lateral_weight*y[:2]/.2,4*y[3:5])
    first = ca.Function('first',[z,p],[ca.vertcat((y[:NX]-p)/ca.DM(SCALE[:NX]),y[24])])
    stage = ca.Function('stage',[z,zn,p],[r,c])
    last = ca.Function('last',[z,p],[rt,y[24]])
    diagnostics = ca.Function('contact_diagnostics',[z],[gap,j+(1+cfg.restitution)*vn,
                                  ca.sqrt(ca.dot(tangent,tangent)),vn])
    return first,stage,last,diagnostics,lo,hi


def variable_bounds(cfg):
    lo=np.r_[[-.6,-.6,-.15],[-4,-4,-6],[-.4,-.4,-.18,-.4,-.4],
             [-1.5,-1.5,-1.5,-3,-3],[-18,-18,-18,-35,-35],0.,0.,cfg.h_min,0.]
    hi=np.r_[[.6,.6,1.5],[4,4,6],[.4,.4,.18,.4,.4],
             [1.5,1.5,1.5,3,3],[18,18,18,35,35],12.,1.,cfg.h_max,cfg.horizon*cfg.h_max]
    if not cfg.variable_time:
        lo[23]=hi[23]=cfg.h_nom
    return lo/SCALE,hi/SCALE


def build_problem(cfg):
    first,stage,last,diagnostics,cl,ch=create_factors(cfg)
    Z=ca.MX.sym('Z',NZ,cfg.horizon+1)
    x=ca.MX.sym('measured',NX)
    rr=[]; cc=[first(Z[:,0],x)]
    for k in range(cfg.horizon):
        r,c=stage(Z[:,k],Z[:,k+1],x)
        rr.append(r); cc.append(c)
    rt,ct=last(Z[:,-1],x)
    rr.append(rt);cc.append(ct)
    r=np.sqrt(cfg.objective_scale)*ca.vertcat(*rr); c=ca.vertcat(*cc)
    flat=ca.vec(Z)
    jr=ca.jacobian(r,flat); jc=ca.jacobian(c,flat)
    H=ca.triu(jr.T@jr+cfg.proximal*ca.MX.eye(flat.numel()))
    grad=jr.T@r
    data=ca.Function('sqp_data',[flat,x],[H,grad,jc,c,.5*ca.dot(r,r)])
    value=ca.Function('nlp_value',[flat,x],[.5*ca.dot(r,r),c])
    return dict(data=data,value=value,first=first,stage=stage,last=last,
                diagnostics=diagnostics,cl=np.r_[np.zeros(NX+1),np.tile(cl,cfg.horizon),min(cfg.min_horizon,cfg.horizon*cfg.h_nom)],
                ch=np.r_[np.zeros(NX+1),np.tile(ch,cfg.horizon),cfg.horizon*cfg.h_max],
                ncon_first=NX+1,ncon_stage=len(cl),ncon_last=1)
