"""Compare parametric factors with independently rebuilt numeric factors."""
import argparse,copy,json,tempfile
from pathlib import Path
from dataclasses import replace
import casadi as ca
import numpy as np
from arm.model import Panda,HOME
from arm.spin_formulation import Config,SCALE,OFFSET
from optimization.pruning.formulation import create_factors
from optimization.native_export import export
from optimization.runtime_parameters import values

def check(spec,path,compiled=False):
    cfg=Config(**spec['config']);worst=0.;cases=[]
    for variant in range(3):
        s=copy.deepcopy(spec)
        if variant:
            s['config'].update(contact_center_grip_offset=.014 if variant==1 else 0.,contact_edge_margin=.012,
                mass=.0035,drag=.21 if variant==1 else 0.,gravity_x=-.15,gravity_y=.04,gravity_z=-9.79,
                restitution=.62,friction=.23,inertia_ratio=.7,apex=.55,energy_weight=24.,comp_weight=330.,
                contact_center_weight=5. if variant==1 else 0.,paddle_radius=.072,radius=.019,
                lateral_weight=12.,ball_velocity_weight=1.3,paddle_lateral_weight=.8,pose_weight=3.,
                velocity_weight=1.1,accel_weight=.6,torque_weight=.04,rate_weight=.2,posture_weight=.4,objective_scale=.0002)
            s['geometry']['paddle_normal_rotvec_deg']=[-.8,2.2,.3]
        c=Config(**s['config']);robot=Panda(assembly=s['assembly'],**s['geometry']);t=values(c,s['geometry'])
        first,stage,last,*_=create_factors(c,robot)
        rng=np.random.default_rng(872)
        for name,f in [('first',first),('stage',stage),('last',last)]:
            n=68 if name=='stage' else 34;z=ca.MX.sym('z',n);p=ca.MX.sym('p',30)
            if name=='first':r=ca.MX.zeros(0,1);con=f(z,p)
            elif name=='stage':r,con=f(z[:34],z[34:],p)
            else:r,con=f(z,p)
            if name=="stage" and r.numel()==54:r=ca.vertcat(r,ca.MX.zeros(3,1))
            r=np.sqrt(c.objective_scale)*r
            ref=ca.Function('reference',[z,p],[r,ca.jacobian(r,z),con,ca.jacobian(con,z)])
            fn=ca.external(name+'_data',str(path/(name+'_data.so'))) if compiled else ca.Function.load(str(path/(name+'_data.casadi')))
            for k in range(2):
                v=rng.normal(0,.03,n);v[30:34]=[.35,.5,1.,.1]
                if n==68:v[64:68]=[.2,.5,1.,.13]
                initial=np.r_[np.zeros(6),HOME,np.zeros(17)]
                expected=ref(v,initial);actual=fn(v,initial,t)
                for a,b in zip(actual,expected):
                    a,b=np.asarray(a),np.asarray(b)
                    error=float(np.max(np.abs(a-b),initial=0));worst=max(worst,error)
                    np.testing.assert_allclose(a,b,rtol=2e-9,atol=2e-9)
        for name,reference in [('panda_fk',robot.fk),('panda_contact_fk',robot.contact_fk)]:
            fn=ca.external(name,str(path/(name+'.so'))) if compiled else ca.Function.load(str(path/(name+'.casadi')))
            for a,b in zip(fn(HOME,t),reference(HOME)):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
        cases.append({'variant':variant,'passed':True})
    return dict(passed=True,compiled=compiled,max_absolute_error=worst,cases=cases,physical_robot_used=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,default=Path('config/cpp_ball_spec.json'));p.add_argument('--factors',type=Path,required=True);p.add_argument('--compiled',action='store_true');a=p.parse_args()
    print(json.dumps(check(json.loads(a.spec.read_text()),a.factors,a.compiled),indent=2))
