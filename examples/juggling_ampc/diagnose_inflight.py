"""Separate factor, QP, and SQP discrepancies on an in-flight fixture."""
import bootstrap
import argparse,json
import numpy as np
import scipy.sparse as sp
import osqp
from pathlib import Path
from factors import Factors
from plan_parity import metrics
from cpp.python_controller import NativeController

def main(a):
 f=Factors(a.directory/'factors');d=np.load(a.fixture);i=a.world;p=d['p'][i].astype(float);z=d['z'][i].astype(float);y=d['dual'][i].astype(float)
 native=NativeController(a.directory/'controller/libjuggling_controller.so',a.directory/'factors',compiled=True)
 native.set_reference(p[30:33],p[33:]);native.set_warmstart(z,y);expected=native.step(p[:23],p[23:30],0.)[0];native.close()
 gpu=None
 if a.gpu:
  from mpc import MPC
  gpu=MPC(a.directory/'gpu_factors')
 arr=f.arrays;lo=np.array(arr['all_lower']);hi=np.array(arr['all_upper']);cl=np.array(arr['constraint_lower']);cu=np.array(arr['constraint_upper']);scale=np.array(arr['scale']);offset=np.array(arr['offset'])
 unlimited=np.isin(np.arange(f.n)%34,np.r_[13:20,23:30]);solver=None;steps=[]
 def violation(c,z):return np.maximum(cl-c,0).sum()+np.maximum(c-cu,0).sum()+np.maximum(lo-z,0).sum()+np.maximum(z-hi,0).sum()
 for k in range(f.cfg['sqp_iters']):
  h,g,mat,c,cost=f.numpy(z,p);full=h+h.T-sp.diags(h.diagonal());q=g-full@z;az=(mat@z)[:f.m];center=np.clip(z,lo,hi)
  lb=np.where(unlimited,lo,np.maximum(lo,center-f.cfg['trust']));ub=np.where(unlimited,hi,np.minimum(hi,center+f.cfg['trust']));measured=(p[:30]-offset[:30])/scale[:30];lb[:30]=np.minimum(lb[:30],measured);ub[:30]=np.maximum(ub[:30],measured)
  l=np.r_[cl-c+az,lb];u=np.r_[cu-c+az,ub]
  if solver is None:
   solver=osqp.OSQP();solver.setup(P=h,A=mat,q=q,l=l,u=u,rho=f.cfg['rho'],max_iter=f.cfg['osqp_iters'],check_termination=0,adaptive_rho=False,polishing=False,verbose=False,eps_abs=1e-4,eps_rel=1e-4)
  else:solver.update(q=q,l=l,u=u);solver.update(Px=h.data,Ax=mat.data)
  solver.warm_start(x=z,y=y);out=solver.solve()
  errors={}
  if gpu is not None:
   inp=[np.array(v[None],np.float32) for v in (z,p,f.projector(p[30:]).ravel(order='F'))]
   gh,gg,ga,gc,gv=map(np.asarray,gpu.assemble(*inp))
   errors['factors']={name:float(np.max(abs(actual[0]-exact))) for name,actual,exact in zip(('h','g','a','c','cost'),(gh,gg,ga,gc,gv),(h.data,g,mat.data,c,cost))}
   exactargs=[np.array(v[None],np.float32) for v in (h.data,mat.data,q,l,u,z,y)]
   gx,gy=map(np.asarray,gpu.qp.solve(*exactargs));errors['qp32_scaled_error']=float(np.max(abs(gx[0]-out.x)))
   gpuplan=np.asarray(gpu.solve(*[np.array(v[None],np.float32) for v in (z,y,p,f.projector(p[30:]).ravel(order='F'))])[2]) if k==0 else None
   if k==0:errors['current_gpu_native']=metrics(f,gpuplan,expected[None])
  m0=cost+1000*f.cfg['objective_scale']*violation(c,z);alphas=[]
  for alpha in (1.,.5,.25,.125,.0625):
   trial=z+alpha*(out.x-z);_,_,_,ct,co=f.numpy(trial,p);mt=co+1000*f.cfg['objective_scale']*violation(ct,trial);alphas.append([alpha,float(mt-m0)])
   if mt<=m0+1e-8:z=trial;y=out.y;break
  steps.append(dict(alphas=alphas,errors=errors))
 plan=z.reshape(1,-1,34)*scale+offset
 report=dict(python_native=metrics(f,plan,expected[None]),recorded_native=metrics(f,d['plan'][i:i+1],expected[None]),steps=steps)
 print(json.dumps(report,indent=2));a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('fixture',type=Path);p.add_argument('--gpu',action='store_true');p.add_argument('--world',type=int,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
