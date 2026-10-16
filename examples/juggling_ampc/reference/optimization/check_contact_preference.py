"""Brief symbolic check: grip-side cost minimum and physical-center edge inset."""
import json
import argparse
from pathlib import Path
import numpy as np
import casadi as ca
from arm.model import Panda, HOME
from arm.spin_formulation import Config, SCALE, OFFSET
from optimization.pruning.formulation import create_factors
from optimization.runtime_parameters import values as tuning_values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--factors',type=Path);args=parser.parse_args()
    spec=json.loads(Path('config/cpp_ball_spec.json').read_text())
    cfg=Config(**spec['config']);robot=Panda(assembly=spec['assembly'],**spec['geometry'])
    _,stage,_,_,_,_,groups=create_factors(cfg,robot)
    compiled=None
    if args.factors:
        compiled=ca.external('stage_data',str(args.factors/'stage_data.so'))
    z=ca.MX.sym('z',68);param=ca.MX.sym('p',30)
    residual,_=stage(z[:34],z[34:],param)
    jac=ca.Function('center_jac',[z,param],[ca.jacobian(residual[-3:],z)])
    gravity=np.array([cfg.gravity_x,cfg.gravity_y,cfg.gravity_z]);errors=[]
    for q in [HOME, HOME+np.array([.2,.1,-.1,.1,.05,-.1,.1])]:
        p,n,_,_,R=map(np.asarray,robot.contact_fk(q));p=p.ravel()-robot.origin;n=n.ravel()
        grip=robot.paddle_normal_rotation.T@np.array([-1.,0.,0.]);grip[2]=0;grip/=np.linalg.norm(grip)
        direction=R@grip;preferred=cfg.contact_center_grip_offset*direction
        assert np.dot(direction,-(R@robot.paddle_normal_rotation.T)[:,0])>.999
        assert abs(n@preferred)<1e-12 and abs(np.linalg.norm(preferred)-cfg.contact_center_grip_offset)<1e-12
        h=.03;tc=h*.5;velocity=np.array([0.,0.,-3.]);a=-np.expm1(-cfg.drag*tc)/cfg.drag;b=(tc-a)/cfg.drag
        y=np.zeros(34);y[3:6]=velocity;y[6:13]=q;y[23:30]=np.asarray(robot.rnea(q,np.zeros(7),np.zeros(7))).ravel();y[30:]=[4.,.5,h,0.]
        yn=y.copy();yn[33]=h
        def at(contact):
            initial=y.copy();initial[:3]=p+cfg.radius*n+contact-a*velocity-b*gravity
            return np.r_[(initial-OFFSET)/SCALE,(yn-OFFSET)/SCALE]
        def evaluate(v):
            rr,cc=stage(v[:34],v[34:],np.zeros(30));return np.asarray(rr).ravel(),np.asarray(cc).ravel()
        wanted=at(preferred);rr,_=evaluate(wanted);assert np.linalg.norm(rr[-3:])<1e-10
        assert np.linalg.norm(evaluate(at(np.zeros(3)))[0][-3:])>.1
        edge_row=groups['contact'][0]+2
        assert evaluate(at(.064*direction))[1][edge_row]<0
        assert evaluate(at(.066*direction))[1][edge_row]>0
        exact=np.asarray(jac(wanted,np.zeros(30)));numeric=np.zeros_like(exact);eps=1e-6
        if compiled is not None:
            scale=np.sqrt(cfg.objective_scale)/(1. if spec.get("runtime_parameters") else cfg.contact_center_weight)
            for v in [wanted,at(np.zeros(3)),at(.066*direction)]:
                cr,cj,cc,_=compiled(v,np.zeros(30),*([tuning_values(cfg,spec["geometry"])] if spec.get("runtime_parameters") else []));sr,sc=evaluate(v)
                assert np.allclose(np.asarray(cr).ravel()[-3:],scale*sr[-3:],atol=1e-10)
                assert np.allclose(np.asarray(cj)[-3:],scale*np.asarray(jac(v,np.zeros(30))),atol=1e-9)
                assert np.allclose(np.asarray(cc).ravel(),sc,atol=1e-9)
        for i in range(68):
            delta=np.zeros(68);delta[i]=eps
            numeric[:,i]=(evaluate(wanted+delta)[0][-3:]-evaluate(wanted-delta)[0][-3:])/(2*eps)
        errors.append(float(np.max(abs(exact-numeric))))
    assert max(errors)<1e-6,errors
    print(json.dumps(dict(passed=True,physical_robot_used=False,compiled_factor_checked=compiled is not None,preferred_grip_offset_m=cfg.contact_center_grip_offset,
        usable_radius_m=cfg.paddle_radius-cfg.contact_edge_margin,max_jacobian_error=max(errors))))


if __name__=='__main__':main()
