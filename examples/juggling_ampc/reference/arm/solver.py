"""One-step sparse RTI for the full Panda problem, with physical rollout warmstarts."""
from pathlib import Path
from time import perf_counter
import hashlib
import os
import subprocess
import shutil
import casadi as ca
import numpy as np
import scipy.sparse as sp
import osqp
import pinocchio as pin
from .formulation import NX,NZ,NP,SCALE,OFFSET,build_problem,bounds
from .simulator import Simulator,SimConfig

class Solver:
    def __init__(self,cfg,robot,formulation=None):
        assert cfg.sqp_iters==1 and cfg.osqp_iters in (25,50)
        if formulation is None:
            from . import formulation
        self.formulation=formulation
        NX,NZ,NP=formulation.NX,formulation.NZ,formulation.NP
        build_problem,bounds=formulation.build_problem,formulation.bounds
        self.cfg=cfg;self.robot=robot;self.problem=build_problem(cfg,robot)
        self.data=self.problem['data'];self.value=self.problem['value']
        if cfg.codegen:
            key=hashlib.sha256(self.data.serialize().encode()).hexdigest()[:16]
            compiler=shutil.which('clang') or shutil.which('cc')
            optimization=cfg.codegen_optimization
            assert optimization in ('O0','O1')
            directory=Path('build')/('arm_'+Path(compiler).name+'_'+optimization+'_'+key);directory.mkdir(parents=True,exist_ok=True)
            lib=directory/'arm_problem.so'
            if not lib.exists():
                cwd=Path.cwd()
                try:
                    os.chdir(directory);cg=ca.CodeGenerator('arm_problem.c',{'with_header':True})
                    for fun in (self.data,self.value,self.problem['stage'],self.problem['first'],self.problem['last']):cg.add(fun)
                    cg.generate()
                    subprocess.run([compiler,'-'+optimization,'-fPIC','-shared','arm_problem.c','-o','arm_problem.so','-lm'],check=True)
                finally:os.chdir(cwd)
            self.data=ca.external('arm_sqp_data',str(lib));self.value=ca.external('arm_nlp_value',str(lib))
        self.lb,self.ub=bounds(cfg,robot);self.lower=np.tile(self.lb,cfg.horizon+1);self.upper=np.tile(self.ub,cfg.horizon+1)
        self.n=NZ*(cfg.horizon+1);self.z=None;self.dual=None;self.qp=None
        self.nominal=Simulator(robot,SimConfig(restitution=cfg.restitution,drag=0,max_interval=.003))

    def initialize(self,x,tau):
        f=self.formulation;NX,NZ,SCALE,OFFSET=f.NX,f.NZ,f.SCALE,f.OFFSET
        cfg=self.cfg;Y=np.zeros((cfg.horizon+1,NZ));Y[:,NX+9]=cfg.h;Y[:,NX+8]=.5
        state=x.copy()
        for k in range(cfg.horizon):
            Y[k,:NX]=state;Y[k,NX:NX+7]=np.asarray(self.robot.rnea(state[6:13],state[13:20],np.zeros(7))).ravel()
            state,contacts=self.nominal.step(state,tau,tau,cfg.h,acc=np.zeros(7))
            if contacts:Y[k,NX+7]=contacts[0]['impulse_Ns']/cfg.mass;Y[k,NX+8]=contacts[0]['time']/cfg.h
        Y[-1,:NX]=state;Y[-1,NX:NX+7]=np.asarray(self.robot.rnea(state[6:13],state[13:20],np.zeros(7))).ravel()
        Y[:,NX+10]=np.arange(cfg.horizon+1)*cfg.h;Y[0,NX:NX+7]=tau
        self.z=((Y-OFFSET)/SCALE).ravel()

    def shift(self,dt,x,tau):
        cfg=self.cfg;N=cfg.horizon;old=self.z.reshape(-1,NZ)*SCALE+OFFSET
        times=np.arange(N+1)*cfg.h;source=np.minimum(np.searchsorted(times,times+dt,side='right')-1,N)
        if self.dual is not None:
            nf=self.problem['ncon_first'];nc=self.problem['ncon_stage'];mc=len(self.problem['cl'])
            previous=self.dual;self.dual=np.zeros_like(previous)
            for k,i in enumerate(source[:-1]):
                if i<N:self.dual[nf+k*nc:nf+(k+1)*nc]=previous[nf+i*nc:nf+(i+1)*nc]
            for k,i in enumerate(source):
                if i<N:self.dual[mc+k*NZ:mc+(k+1)*NZ]=previous[mc+i*NZ:mc+(i+1)*NZ]
            self.dual[mc:mc+NP]=0
            for k in range(N):
                l,r=self.problem['groups']['contact'];self.dual[nf+k*nc+l:nf+k*nc+r]=0
        # Shift the previous joint accelerations, then regenerate physical ball
        # contacts. Inverse dynamics supplies torques for this primal seed only.
        accelerations=np.diff(old[:,13:20],axis=0)/cfg.h
        Y=old.copy();state=x.copy();aa=[]
        for k in range(N):
            start=k*cfg.h+dt;end=start+cfg.h;cursor=start;acc=np.zeros(7)
            while cursor<end-1e-12:
                i=min(int((cursor+1e-12)/cfg.h),N-1);stop=min(end,(i+1)*cfg.h if i<N-1 else end)
                acc+=(stop-cursor)*accelerations[i];cursor=stop
            acc/=cfg.h;aa.append(acc)
            Y[k,:NX]=state;Y[k,27]=0;Y[k,28]=.5
            state,contacts=self.nominal.step(state,tau,tau,cfg.h,acc=acc)
            if contacts:
                Y[k,27]=contacts[0]['impulse_Ns']/cfg.mass;Y[k,28]=contacts[0]['time']/cfg.h
        Y[-1,:NX]=state
        for k in range(N+1):
            a=aa[0] if k==0 else aa[-1] if k==N else .5*(aa[k-1]+aa[k])
            Y[k,20:27]=np.asarray(self.robot.rnea(Y[k,6:13],Y[k,13:20],a)).ravel()
        Y[0,20:27]=tau;Y[:,29]=cfg.h;Y[:,30]=times
        self.z=((Y-OFFSET)/SCALE).ravel()

    def violation(self,c):
        return np.maximum(self.problem['cl']-c,0)+np.maximum(c-self.problem['ch'],0)

    def bound_violation(self,z):
        return np.maximum(self.lower-z,0)+np.maximum(z-self.upper,0)

    def solve(self,x,tau):
        f=self.formulation;NX,NZ,NP,SCALE,OFFSET=f.NX,f.NZ,f.NP,f.SCALE,f.OFFSET
        start=perf_counter();cfg=self.cfg;p=np.r_[x,tau]
        if self.z is None:self.initialize(x,tau)
        if not np.all(np.isfinite(self.z)) or np.max(self.bound_violation(self.z))>100:
            raise RuntimeError('Warmstart diverged outside the research model range')
        t=perf_counter();H,grad,A,c,cost=self.data(self.z,p)
        H=H.sparse();A=A.sparse();grad=np.asarray(grad).ravel();c=np.asarray(c).ravel()
        if not all(np.all(np.isfinite(v)) for v in (H.data,A.data,grad,c)):
            raise RuntimeError('Non-finite SQP data; no QP update or plant command applied')
        q=grad-(H+H.T-sp.diags(H.diagonal()))@self.z
        center=np.clip(self.z,self.lower,self.upper)
        low=np.maximum(self.lower,center-cfg.trust);high=np.minimum(self.upper,center+cfg.trust)
        indices=np.concatenate([np.r_[np.arange(k*NZ+13,k*NZ+20),np.arange(k*NZ+NX,k*NZ+NX+7)] for k in range(cfg.horizon+1)])
        low[indices]=self.lower[indices];high[indices]=self.upper[indices]
        low[:NP]=np.minimum(low[:NP],(p-OFFSET[:NP])/SCALE[:NP]);high[:NP]=np.maximum(high[:NP],(p-OFFSET[:NP])/SCALE[:NP])
        AA=sp.vstack([A,sp.eye(self.n,format='csc')],format='csc')
        l=np.r_[self.problem['cl']-c+A@self.z,low];u=np.r_[self.problem['ch']-c+A@self.z,high]
        data_ms=1000*(perf_counter()-t);t=perf_counter()
        if self.qp is None:
            self.qp=osqp.OSQP();self.qp.setup(P=H,q=q,A=AA,l=l,u=u,verbose=False,max_iter=cfg.osqp_iters,
                check_termination=0,polishing=False,warm_starting=True,adaptive_rho=False,rho=cfg.rho,eps_abs=1e-4,eps_rel=1e-4)
        else:self.qp.update(Px=H.data,Ax=AA.data,q=q,l=l,u=u)
        if self.dual is None:self.dual=np.zeros(AA.shape[0])
        warm_nonzeros=int(np.count_nonzero(self.dual));self.qp.warm_start(x=self.z,y=self.dual)
        result=self.qp.solve(raise_error=False);qp_ms=1000*(perf_counter()-t);alpha=0.
        merit_weight=1000*cfg.objective_scale
        merit0=float(cost)+merit_weight*(sum(self.violation(c))+sum(self.bound_violation(self.z)))
        if result.x is not None and np.all(np.isfinite(result.x)) and max(abs(result.x))<1e6:
            dz=result.x-self.z
            for value in [1.,.5,.25,.125,.0625]:
                trial=self.z+value*dz;obj,con=self.value(trial,p);con=np.asarray(con).ravel()
                merit=float(obj)+merit_weight*(sum(self.violation(con))+sum(self.bound_violation(trial)))
                if merit<=merit0+1e-8:self.z=trial;self.dual=result.y.copy();alpha=value;break
        obj,con=self.value(self.z,p);con=np.asarray(con).ravel();Y=self.z.reshape(-1,NZ)*SCALE+OFFSET
        info=dict(total_ms=1000*(perf_counter()-start),data_ms=data_ms,qp_ms=qp_ms,cost=float(obj),
            violation=float(max(self.violation(con))),bound_violation=float(max(self.bound_violation(self.z))),
            warm_dual_nonzeros=warm_nonzeros,iterations=[dict(iter=result.info.iter,status=result.info.status,
            alpha=alpha,prim_res=result.info.prim_res,dual_res=result.info.dual_res)])
        return Y,info
