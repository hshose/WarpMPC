"""Check edge geometry and native derivatives for the optional impact-centering cost."""
import argparse,json
from pathlib import Path
import numpy as np
from arm.model import Panda
from arm.spin_formulation import Config,SCALE,OFFSET
from optimization.pruning.formulation import create_factors
from cpp.python_controller import NativeController

def main():
 p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('recording',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 meta=json.loads((a.factors/'metadata.json').read_text());cfg=Config(**meta['config']);spec=meta['spec'];robot=Panda(assembly=spec['assembly'],**spec['geometry'])
 home=json.loads((a.recording/'fixture_home.json').read_text());q=np.array(home['q_home']);face,normal,_,_,R=robot.contact_fk(q);face=np.array(face).ravel()-robot.origin;normal=np.array(normal).ravel();tangent=np.array(R)[:,0]
 _,stage,_,_,_,_,groups=create_factors(cfg,robot);radius=cfg.paddle_radius-cfg.contact_edge_margin;edge=[]
 for factor in [.9,1.,1.1]:
  y=OFFSET.copy();y[:3]=face+cfg.radius*normal+factor*radius*tangent;y[6:13]=q;y[30]=8.;y[31]=0.;y[32]=cfg.h
  z=(y-OFFSET)/SCALE;r,c=stage(z,z,np.zeros(30));edge.append(float(c[groups['contact'][0]+2]))
  if cfg.contact_center_weight:
   expected=cfg.contact_center_weight*factor*radius*tangent/cfg.paddle_radius
   assert np.max(abs(np.array(r).ravel()[-3:]-expected))<1e-10
 assert edge[0]<0 and abs(edge[1])<1e-10 and edge[2]>0,edge
 row=json.loads(next((a.recording/'mpc.jsonl').open()));x=np.array(row['state']);tau=np.array(row['executed_trajectory'][0][14:21])
 c=NativeController(library=a.factors.parent/'build/controller/libjuggling_controller.so',factors=a.factors);c.set_reference(home['face_offset_m'],home['q_home']);c.initialize(x,tau)
 _,_,metrics=c.step(x,tau,0.);z,_=c.warmstart();param=np.r_[x,tau];H,g,A,con,cost=c.linearize(z,param)
 rng=np.random.default_rng(853);errors=[]
 for _ in range(3):
  d=rng.normal(size=len(z));d/=np.linalg.norm(d);eps=1e-7
  _,_,_,cp,fp=c.linearize(z+eps*d,param);_,_,_,cm,fm=c.linearize(z-eps*d,param)
  errors.append(dict(cost=abs((fp-fm)/(2*eps)-g@d),constraints=float(np.max(abs((cp-cm)/(2*eps)-A[:len(con)]@d)))))
 assert max(e['cost'] for e in errors)<1e-6,errors
 assert max(e['constraints'] for e in errors)<1e-5,errors
 assert abs(cost-metrics[5])<1e-10,(cost,metrics[5])
 c.close();result=dict(passed=True,robot_motion=False,physical_radius_m=cfg.paddle_radius,mpc_contact_radius_m=radius,edge_constraints=edge,derivative_errors=errors,merit_error=abs(cost-metrics[5]));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
