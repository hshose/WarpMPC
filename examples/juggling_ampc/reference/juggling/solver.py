"""Sparse Gauss–Newton RTI with absolute-iterate QPs and explicit warmstarts."""
from pathlib import Path
from time import perf_counter
import subprocess
import os
import casadi as ca
import numpy as np
import scipy.sparse as sp
import osqp
from .formulation import NX,NZ,SCALE,build_problem,variable_bounds


def csc(dm):
    return dm.sparse()


def npnormal(a):
    r,p=a
    return np.array([np.sin(p)*np.cos(r),-np.sin(r),np.cos(p)*np.cos(r)])


class RTI:
    def __init__(self,cfg):
        if not 1 <= cfg.sqp_iters <= 10 or not 25 <= cfg.osqp_iters <= 100:
            raise ValueError('Requested RTI/QP iteration budget violated')
        self.cfg=cfg
        self.problem=build_problem(cfg)
        self.data=self.problem['data']; self.value=self.problem['value']
        if cfg.codegen:
            # Each local factor is separately available; export the assembled sparse
            # evaluation too. Cache by graph identity, never by a stale configuration.
            import hashlib
            key=hashlib.sha256(self.data.serialize().encode()).hexdigest()[:16]
            directory=Path('build')/key
            directory.mkdir(parents=True,exist_ok=True)
            lib=directory/'sqp_data.so'
            if not lib.exists():
                cwd=Path.cwd()
                try:
                    os.chdir(directory)
                    self.data.generate('sqp_data.c',{'with_header':True})
                    subprocess.run(['cc','-O3','-fPIC','-shared','sqp_data.c','-o','sqp_data.so','-lm'],check=True)
                finally: os.chdir(cwd)
            self.data=ca.external('sqp_data',str(lib))
        self.lb,self.ub=variable_bounds(cfg)
        self.n=NZ*(cfg.horizon+1)
        self.full_lb=np.tile(self.lb,cfg.horizon+1)
        self.full_ub=np.tile(self.ub,cfg.horizon+1)
        self.z=None; self.dual=None; self.qp=None
        self.last_info={}

    def initialize(self,x):
        """Passive ballistic seed; no planned strike schedule is imposed on solves."""
        cfg=self.cfg
        Y=np.zeros((cfg.horizon+1,NZ)); Y[:,23]=cfg.h_nom;Y[:,22]=.5
        Y[0,:NX]=x
        Y[:,24]=np.arange(cfg.horizon+1)*cfg.h_nom
        for k in range(cfg.horizon):
            row=Y[k]; h=row[23]; b=row[:3]; v=row[3:6]
            g=np.array([0.,0.,-9.81])
            # Startup is vertical with stationary level paddle. Subsequent seeds
            # come only from shifting the previous optimized trajectory.
            gap=b[2]-row[8]-cfg.radius
            root=(v[2]+np.sqrt(max(0.,v[2]**2+2*9.81*gap)))/9.81
            impulse=0.; tau=h/2
            if 0<=root<h and v[2]-9.81*root<0:
                tau=root;impulse=-(1+cfg.restitution)*(v[2]-9.81*tau)
            row[21]=impulse;row[22]=tau/h
            Y[k+1,:3]=b+h*v+.5*h*h*g+(h-tau)*impulse*np.array([0,0,1])
            Y[k+1,3:6]=v+h*g+impulse*np.array([0,0,1])
            Y[k+1,6:NX]=row[6:NX]
        self.z=(Y/SCALE).reshape(-1)
        return Y

    def shift(self,dt,x):
        """Sample old continuous arcs and remap impulses onto the shifted time mesh.

        QP variables are absolute scaled trajectories, not SQP increments. Warmstart
        x with the shifted trajectory, and map interval/bound multipliers by old
        interval identity; reset measurement and newly appended multipliers.
        """
        Y=self.z.reshape(-1,NZ)*SCALE
        N=self.cfg.horizon
        times=np.r_[0,np.cumsum(Y[:N,23])]
        newtimes=np.r_[0,np.cumsum(Y[:N,23])]+dt
        Yn=Y.copy(); Yn[:,21]=0; Yn[:,22]=.5
        Yn[:,24]=newtimes-dt
        src=np.minimum(np.searchsorted(times,newtimes,side='right')-1,N)
        g=np.array([0.,0.,-9.81])
        for k,t in enumerate(newtimes):
            i=src[k]; row=Y[i]; s=t-times[i]
            u=row[16:21]; tau=row[22]*row[23]
            pose=row[6:11]+s*row[11:16]+.5*s*s*u
            Yn[k,:3]=row[:3]+s*row[3:6]+.5*s*s*g
            Yn[k,3:6]=row[3:6]+s*g
            if i<N and s+1e-10>=tau:
                pc=row[6:11]+tau*row[11:16]+.5*tau*tau*u
                impulse=row[21]*npnormal(pc[3:5])
                Yn[k,:3]+=(s-tau)*impulse;Yn[k,3:6]+=impulse
            Yn[k,6:11]=pose;Yn[k,11:16]=row[11:16]+s*u
            Yn[k,16:21]=u
        for i in range(N):
            te=times[i]+Y[i,22]*Y[i,23]
            # Arc sampling returns the post-impact state at an exact event.
            # Put boundary impulses at the END of the preceding interval.
            k=np.searchsorted(newtimes,te+1e-12,side='left')-1
            if np.any(np.abs(newtimes-te)<1e-10):
                k=int(np.argmin(np.abs(newtimes-te)))-1
            if 0<=k<N and Y[i,21]>1e-8:
                # At most one contact per interval; detect unsupported remeshing.
                if Yn[k,21]>1e-8:
                    self.dual=None
                Yn[k,21]+=Y[i,21]
                Yn[k,22]=np.clip((te-newtimes[k])/Yn[k,23],0,1)
        Yn[0,:NX]=x
        self.z=(Yn/SCALE).reshape(-1)
        if self.dual is not None:
            nc=self.problem['ncon_stage']; nf=self.problem['ncon_first']; mc=nf+N*nc+self.problem['ncon_last']
            old=self.dual.copy(); d=np.zeros_like(old)
            for k in range(N):
                i=src[k]
                if i<N:
                    d[nf+k*nc:nf+(k+1)*nc]=old[nf+i*nc:nf+(i+1)*nc]
            for k in range(N+1):
                i=src[k]
                if i<N:
                    d[mc+k*NZ:mc+(k+1)*NZ]=old[mc+i*NZ:mc+(i+1)*NZ]
            d[mc:mc+NX]=0
            self.dual=d

    def violation(self,c):
        return np.maximum(self.problem['cl']-c,0)+np.maximum(c-self.problem['ch'],0)

    def bound_violation(self,z):
        return np.maximum(self.full_lb-z,0)+np.maximum(z-self.full_ub,0)

    def solve(self,x):
        start=perf_counter()
        if self.z is None:self.initialize(x)
        data_time=qp_time=0.; history=[]
        warm_dual_nonzeros=0 if self.dual is None else int(np.count_nonzero(self.dual))
        cl,ch=self.problem['cl'],self.problem['ch']
        for iteration in range(self.cfg.sqp_iters):
            t=perf_counter(); H,grad,A,con,cost=self.data(self.z,x)
            H=csc(H); A=csc(A); grad=np.asarray(grad).ravel(); con=np.asarray(con).ravel()
            symH=H+H.T-sp.diags(H.diagonal())
            q=grad-symH@self.z
            trust_center=np.clip(self.z,self.full_lb,self.full_ub)
            lower=np.maximum(self.full_lb,trust_center-self.cfg.trust)
            upper=np.minimum(self.full_ub,trust_center+self.cfg.trust)
            # Trust boxes on the linear acceleration/velocity subproblem can
            # make deceleration infeasible. Keep their physical bounds only.
            linear_indices=np.concatenate([np.arange(k*NZ+11,k*NZ+21)
                                           for k in range(self.cfg.horizon+1)])
            lower[linear_indices]=self.full_lb[linear_indices]
            upper[linear_indices]=self.full_ub[linear_indices]
            # A measured out-of-bounds state cannot be changed retrospectively.
            # First-node equality fixes it; subsequent nodes must recover bounds.
            lower[:NX]=np.minimum(lower[:NX],x/SCALE[:NX])
            upper[:NX]=np.maximum(upper[:NX],x/SCALE[:NX])
            AA=sp.vstack([A,sp.eye(self.n,format='csc')],format='csc')
            l=np.r_[cl-con+A@self.z,lower]; u=np.r_[ch-con+A@self.z,upper]
            data_time+=perf_counter()-t
            t=perf_counter()
            if self.qp is None:
                self.qp=osqp.OSQP()
                self.qp.setup(P=H,q=q,A=AA,l=l,u=u,verbose=False,
                    max_iter=self.cfg.osqp_iters,check_termination=0,polishing=False,
                    warm_starting=True,adaptive_rho=False,rho=self.cfg.rho,
                    eps_abs=1e-4,eps_rel=1e-4)
            else:self.qp.update(Px=H.data,Ax=AA.data,q=q,l=l,u=u)
            if self.dual is None:self.dual=np.zeros(AA.shape[0])
            self.qp.warm_start(x=self.z,y=self.dual)
            result=self.qp.solve(raise_error=False)
            qp_time+=perf_counter()-t
            if result.x is None or not np.all(np.isfinite(result.x)) or np.max(np.abs(result.x))>1e6:
                history.append({'status':result.info.status,'rejected':True});break
            step=result.x-self.z
            merit_weight=1000*self.cfg.objective_scale
            merit0=float(cost)+merit_weight*(np.sum(self.violation(con))+np.sum(self.bound_violation(self.z)))
            accepted=False
            for alpha in [1.,.5,.25,.125,.0625]:
                trial=self.z+alpha*step
                obj,ct=self.value(trial,x);ct=np.asarray(ct).ravel()
                merit=float(obj)+merit_weight*(np.sum(self.violation(ct))+np.sum(self.bound_violation(trial)))
                if merit<=merit0+1e-8:
                    self.z=trial;self.dual=result.y.copy();accepted=True;break
            if not accepted:
                alpha=0.
                # Rejected QP multipliers do not describe the retained trajectory.
            history.append(dict(status=result.info.status,iter=result.info.iter,
                        prim_res=result.info.prim_res,dual_res=result.info.dual_res,
                        alpha=alpha,cost=float(obj),violation=float(np.max(self.violation(ct)))))
        obj,ct=self.value(self.z,x)
        self.last_info=dict(total_ms=(perf_counter()-start)*1e3,data_ms=data_time*1e3,
                qp_ms=qp_time*1e3,cost=float(obj),violation=float(np.max(self.violation(np.asarray(ct).ravel()))),
                iterations=history,warm_dual_nonzeros=warm_dual_nonzeros)
        Y=self.z.reshape(-1,NZ)*SCALE
        dd=np.array([np.array(self.problem['diagnostics'](row/SCALE)).ravel() for row in Y[:-1]])
        active=Y[:-1,21]>.5
        self.last_info.update(bound_violation=float(np.max(self.bound_violation(self.z))),
            contact_gap_active_m=float(np.max(np.abs(dd[active,0]))) if np.any(active) else 0.,
            restitution_active_m_s=float(np.max(np.abs(dd[active,1]))) if np.any(active) else 0.,
            small_impulses_m_s=float(np.sum(np.maximum(Y[:-1,21][~active],0))),
            complementarity_gap_scaled=float(np.max(np.abs(Y[:-1,21]/8*dd[:,0]/.05))),
            complementarity_restitution_scaled=float(np.max(np.abs(Y[:-1,21]/8*dd[:,1]/8))),
            horizon_s=float(np.sum(Y[:-1,23])))
        self.last_info['total_ms']=(perf_counter()-start)*1000
        return Y[0,16:21].copy(),Y.copy(),self.last_info
