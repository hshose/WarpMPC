"""Stable parameter ABI for generated factors; numeric tuning never enters C code."""
from types import SimpleNamespace
from dataclasses import asdict
import hashlib,json
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation

FIELDS = ('mass','radius','paddle_radius','contact_edge_margin','contact_center_weight',
 'contact_center_grip_offset','restitution','drag','gravity_x','gravity_y','gravity_z',
 'friction','inertia_ratio','apex','comp_weight','energy_weight','lateral_weight',
 'ball_velocity_weight','paddle_lateral_weight','pose_weight','velocity_weight',
 'accel_weight','torque_weight','rate_weight','posture_weight','objective_scale')
SIZE=len(FIELDS)+9

def values(cfg, geometry):
    R=Rotation.from_rotvec(np.deg2rad(geometry.get('paddle_normal_rotvec_deg',[0,0,0]))).as_matrix()
    return np.r_[[getattr(cfg,n) for n in FIELDS],R.ravel(order='F')]

def symbolic(cfg, t):
    fields=asdict(cfg);fields.update({n:t[i] for i,n in enumerate(FIELDS)})
    return SimpleNamespace(**fields)

class Robot:
    """Rotate the contact normal about the physical face center, as in Panda."""
    def __init__(self,base,t):
        self.base=base;self.origin=base.origin
        self.mass=base.mass;self.bias=base.bias;self.rnea=base.rnea
        self.paddle_normal_rotation=ca.reshape(t[len(FIELDS):],3,3)
    def fk(self,q):
        p,n,J,N,R=self.base.fk(q);R=R@self.paddle_normal_rotation;n=R[:,2]
        _,_,_,W,_=self.base.contact_fk(q)
        return p,n,J,-ca.skew(n)@W,R
    def contact_fk(self,q):
        p,n,J,W,R=self.base.contact_fk(q);R=R@self.paddle_normal_rotation
        return p,R[:,2],J,W,R

def numeric_validation(cfg,spec):
    numeric={k:v for k,v in asdict(cfg).items() if isinstance(v,(int,float))}
    if not np.isfinite(list(numeric.values())).all():raise ValueError('Nonfinite tuning parameter')
    for n in FIELDS:
        if n not in ('gravity_x','gravity_y','gravity_z') and getattr(cfg,n)<0:raise ValueError('Negative '+n)
    if cfg.mass<=0 or cfg.radius<=0 or cfg.inertia_ratio<=0 or cfg.gravity_z>=0:raise ValueError('Invalid ball model')
    if not 0<=cfg.contact_edge_margin<cfg.paddle_radius or not 0<=cfg.contact_center_grip_offset<cfg.paddle_radius:raise ValueError('Invalid contact preference/margin')
    if cfg.objective_scale<=0 or cfg.proximal<=0 or cfg.rho<=0 or cfg.trust<=0:raise ValueError('Invalid solver setting')
    for n in ('sqp_iters','osqp_iters','horizon'):
        if getattr(cfg,n)<1 or int(getattr(cfg,n))!=getattr(cfg,n):raise ValueError('Invalid '+n)
    if cfg.sqp_iters>4 or cfg.control_dt<=0 or cfg.sim_dt<=0:raise ValueError('Invalid iteration count/timing')

def write_runtime(directory, scalars, arrays, structural):
    schema=hashlib.sha256(json.dumps(structural,sort_keys=True).encode()).hexdigest()
    # Header defaults are useful for native checks. The session wrapper overrides
    # them once at process startup, before any worker or robot callback exists.
    entries=list(scalars.items())+[(n+'['+str(i)+']',float(v)) for n,vs in arrays.items() for i,v in enumerate(vs)]
    layout=[n for n,v in entries]
    schema=hashlib.sha256((schema+json.dumps(layout)).encode()).hexdigest()
    text=schema+'\n'+'\n'.join(n+' '+repr(float(v)) for n,v in entries)+'\n'
    (directory/'runtime_values.txt').write_text(text)
    lines=['inline std::string loaded_runtime_values_path;', 'inline const bool runtime_values_loaded=[](){',
      'const char* path=std::getenv("JUGGLING_RUNTIME_VALUES"); if(!path)return true;',
      'loaded_runtime_values_path=path; std::ifstream in(path); std::string token; if(!(in>>token)||token!="'+schema+'")throw std::runtime_error("Runtime tuning ABI mismatch");',
      'auto read=[&](const char* name,double& value){std::string key,text;if(!(in>>key>>text)||key!=name)throw std::runtime_error("Invalid runtime tuning layout");size_t used=0;value=std::stod(text,&used);if(used!=text.size()||std::isnan(value))throw std::runtime_error("Invalid runtime tuning value");};']
    lines += ['read('+json.dumps(n)+','+n+');' for n in scalars]
    lines += ['for(size_t i=0;i<'+n+'.size();++i)read(('+json.dumps(n+'[')+'+std::to_string(i)+"]").c_str(),'+n+'[i]);' for n in arrays]
    lines+=['if(in>>token)throw std::runtime_error("Extra runtime tuning data");return true;}();']
    return lines,schema
