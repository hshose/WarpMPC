"""Fixed exponential shooting mesh with matching continuous running costs.

Running costs use h_k / 30 ms quadrature, with torque differences converted
back into a torque derivative. Contact residuals are discrete event penalties.
"""
from dataclasses import dataclass
import casadi as ca
import numpy as np
from scipy.optimize import brentq
from arm import spin_formulation as base
from arm.spin_formulation import NX, NZ, NP, SCALE, OFFSET

@dataclass
class Config(base.Config):
    mesh_horizon: float = 1.2


def grid(cfg):
    n=cfg.horizon
    if n*cfg.h > cfg.mesh_horizon+1e-12:
        raise ValueError('First mesh dt is too large for the requested horizon')
    growth=1. if np.isclose(n*cfg.h,cfg.mesh_horizon) else brentq(
        lambda g: cfg.h*np.sum(g**np.arange(n))-cfg.mesh_horizon,1.,2.,xtol=1e-14)
    steps=cfg.h*growth**np.arange(n)
    return steps,np.r_[0.,np.cumsum(steps)],growth


def create_factors(cfg,robot):
    return base.create_factors(cfg,robot)


def bounds(cfg,robot):
    low,high=base.bounds(cfg,robot)
    high[33]=cfg.mesh_horizon
    return low,high


def build_problem(cfg,robot):
    first,stage,last,diag,lo,hi,groups=create_factors(cfg,robot)
    steps,times,growth=grid(cfg)
    Z=ca.MX.sym('Z',NZ,cfg.horizon+1);p=ca.MX.sym('p',NP)
    rr=[];cc=[first(Z[:,0],p)]
    for k,h in enumerate(steps):
        r,c=stage(Z[:,k],Z[:,k+1],p)
        weights=np.ones(int(r.numel()))
        assert len(weights) in (54,57)
        weights[:51]=np.sqrt(h/.03)
        weights[30:37]*=.03/h
        rr.append(ca.DM(weights)*r);cc.append(c)
    r,c=last(Z[:,-1],p);rr.append(r);cc.append(c)
    residual=np.sqrt(cfg.objective_scale)*ca.vertcat(*rr);con=ca.vertcat(*cc);flat=ca.vec(Z)
    jr=ca.jacobian(residual,flat);jc=ca.jacobian(con,flat)
    H=ca.triu(jr.T@jr+cfg.proximal*ca.MX.eye(flat.numel()))
    data=ca.Function('arm_sqp_data',[flat,p],[H,jr.T@residual,jc,con,.5*ca.dot(residual,residual)])
    value=ca.Function('arm_nlp_value',[flat,p],[.5*ca.dot(residual,residual),con])
    return dict(data=data,value=value,first=first,stage=stage,last=last,diagnostics=diag,
                cl=np.r_[np.zeros(NP+1),np.tile(lo,cfg.horizon),cfg.mesh_horizon],
                ch=np.r_[np.zeros(NP+1),np.tile(hi,cfg.horizon),cfg.mesh_horizon],
                ncon_first=NP+1,ncon_stage=len(lo),ncon_last=1,groups=groups,
                mesh_steps=steps,mesh_times=times,mesh_growth=growth)
