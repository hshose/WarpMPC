"""Verify native projected posture cost, derivatives, and fixture reference."""
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from arm.model import Panda,HOME
from cpp.python_controller import NativeController

def main():
 p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('--baseline',type=Path,default=Path('build/variants/fixture39_gain50a65/factors'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 meta=json.loads((a.factors/'metadata.json').read_text());spec=meta['spec'];robot=Panda(assembly=spec['assembly'],**spec['geometry']);gains=spec['nullspace_regularization'];errors={};rng=np.random.default_rng(83)
 for label,qref,offset in [('home',HOME,np.zeros(3)),('fixture',np.array([.07,-.06,.08,-1.86,.005,1.80,.94]),np.array([.23,.087,-.05]))]:
  base=NativeController(library=a.baseline.parent/'controller/libjuggling_controller.so',factors=a.baseline);native=NativeController(library=a.factors.parent/'controller/libjuggling_controller.so',factors=a.factors)
  for c in [base,native]:c.set_reference(offset,qref)
  x=np.r_[offset+np.array([0,0,.8]),np.zeros(3),qref,np.zeros(10)];tau=np.asarray(robot.bias(qref,np.zeros(7))).ravel()
  base.initialize(x,tau);native.initialize(x,tau);z=base.warmstart()[0];z=z+rng.normal(0,.005,z.shape);param=np.r_[x,tau]
  H0,g0,A0,c0,f0=base.linearize(z,param);H,g,A,c,f=native.linearize(z,param)
  fk=robot.contact_fk(qref);n=np.array(fk[1]).ravel();cross=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]]);J=np.vstack([np.array(fk[2]),-cross@np.array(fk[3])]);N=np.eye(7)-np.linalg.pinv(J,rcond=1e-8)@J
  expected_H=sp.lil_matrix(H.shape);expected_g=np.zeros_like(g);expected_f=0.
  for k,h in enumerate(meta['stage_steps']):
   for start,scale,gain in [(6,1.,gains['position_weight']),(13,2.,gains['velocity_weight'])]:
    sl=slice(k*34+start,k*34+start+7);delta=scale*z[sl]-(qref-HOME if start==6 else 0.);w=meta['config']['objective_scale']*h/.03*gain**2;Q=N.T@N
    expected_H[sl,sl]=w*scale**2*Q;expected_g[sl]=w*scale*Q@delta;expected_f+=.5*w*np.linalg.norm(N@delta)**2
  dH=H-H0-sp.triu(expected_H.tocsc(),format='csc');dA=A-A0
  errors[label]=dict(H=float(np.max(abs(dH.data),initial=0)),gradient=float(np.max(abs(g-g0-expected_g))),cost=float(abs(f-f0-expected_f)),constraints=float(np.max(abs(c-c0))),A=float(np.max(abs(dA.data),initial=0)),projector_task_residual=float(np.linalg.norm(J@N)),projector_rank=int(round(np.trace(N))))
  assert errors[label]['projector_rank']==2
  assert max(v for k,v in errors[label].items() if k!='projector_rank')<1e-9,errors
  # Ensure the nonlinear merit value agrees with assembled cost at returned solution.
  _,_,stats=native.step(x,tau,0.);_,_,_,_,linear_cost=native.linearize(native.warmstart()[0],param)
  errors[label]['merit_cost']=float(abs(stats[5]-linear_cost));assert errors[label]['merit_cost']<1e-9
  base.close();native.close()
 a.output.write_text(json.dumps(dict(passed=True,errors=errors),indent=2)+'\n');print(a.output.read_text())
if __name__=='__main__':main()
