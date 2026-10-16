"""Geometric grid plus conservative late-row removal and endpoint deduplication."""
from dataclasses import dataclass
import casadi as ca
import numpy as np
from . import formulation as prune
from .formulation import NX,NZ,NP,SCALE,OFFSET,create_factors
from optimization.mesh.formulation import grid


@dataclass
class Config(prune.Config):
    mesh_horizon:float=1.2
    deduplicate_endpoints:bool=True


def bounds(cfg,robot):
    lower,upper=prune.bounds(cfg,robot);upper[33]=cfg.mesh_horizon
    return lower,upper


def build_problem(cfg,robot):
    first,stage,last,diag,lo,hi,groups=create_factors(cfg,robot)
    _,reduced,_,_,lr,ur,gr=create_factors(cfg,robot,drop=cfg.pruning_groups,
        left_endpoint_only=True,suffix='_reduced')
    _,tail,_,_,lt,ut,gt=create_factors(cfg,robot,drop=cfg.pruning_groups,suffix='_tail')
    steps,times,growth=grid(cfg)
    Z=ca.MX.sym('Z',NZ,cfg.horizon+1);p=ca.MX.sym('p',NP)
    rr=[];cc=[first(Z[:,0],p)];lower=[np.zeros(NP+1)];upper=[np.zeros(NP+1)]
    kept_rows=[];offsets=[NP+1]
    for k,h in enumerate(steps):
        if k<cfg.pruning_prefix:f,l,u,g=stage,lo,hi,groups
        elif k==cfg.horizon-1:f,l,u,g=tail,lt,ut,gt
        else:f,l,u,g=reduced,lr,ur,gr
        r,c=f(Z[:,k],Z[:,k+1],p)
        weights=np.ones(int(r.numel()));assert len(weights) in (54,57)
        weights[:51]=np.sqrt(h/.03);weights[30:37]*=.03/h
        rr.append(ca.DM(weights)*r);cc.append(c)
        if k==0 and cfg.initial_acceleration_margin>0 and 'endpoint_acceleration' in g:
            l=l.copy();u=u.copy();begin=g['endpoint_acceleration'][0]
            l[begin:begin+7]=-cfg.initial_acceleration_margin;u[begin:begin+7]=cfg.initial_acceleration_margin
        lower.append(l);upper.append(u);offsets.append(offsets[-1]+len(l))
        rows=[]
        for group,(start,end) in g.items():
            original_start=groups[group][0]
            rows.extend(range(original_start,original_start+end-start))
        kept_rows.append(np.array(rows,dtype=int))
    r,c=last(Z[:,-1],p);rr.append(r);cc.append(c)
    lower.append([cfg.mesh_horizon]);upper.append([cfg.mesh_horizon])
    residual=np.sqrt(cfg.objective_scale)*ca.vertcat(*rr);con=ca.vertcat(*cc);flat=ca.vec(Z)
    jr=ca.jacobian(residual,flat);jc=ca.jacobian(con,flat)
    H=ca.triu(jr.T@jr+cfg.proximal*ca.MX.eye(flat.numel()))
    data=ca.Function('arm_sqp_data',[flat,p],[H,jr.T@residual,jc,con,.5*ca.dot(residual,residual)])
    value=ca.Function('arm_nlp_value',[flat,p],[.5*ca.dot(residual,residual),con])
    return dict(data=data,value=value,first=first,stage=stage,last=last,diagnostics=diag,
        cl=np.concatenate(lower),ch=np.concatenate(upper),ncon_first=NP+1,ncon_stage=len(lo),ncon_last=1,groups=groups,
        kept_rows=kept_rows,stage_offsets=np.array(offsets),reduced_stage=reduced,tail_stage=tail,
        mesh_steps=steps,mesh_times=times,mesh_growth=growth)
