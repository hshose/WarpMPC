"""Nonuniform-grid physical warmstart with timestamp-aware shifting."""
import numpy as np
from arm.solver import Solver
from arm.spin_simulator import SpinSimulator
from arm.simulator import SimConfig
from . import formulation
from .formulation import NX,NZ,NP,SCALE,OFFSET,grid

class MeshSolver(Solver):
    def __init__(self,cfg,robot):
        super().__init__(cfg,robot,formulation=formulation)
        self.steps,self.times,self.growth=grid(cfg)
        self.nominal=SpinSimulator(robot,SimConfig(gravity_x=cfg.gravity_x,gravity_y=cfg.gravity_y,gravity_z=cfg.gravity_z,restitution=cfg.restitution,drag=cfg.drag,ball_mass=cfg.mass,radius=cfg.radius,paddle_radius=cfg.paddle_radius,max_interval=.003),
                                   friction=cfg.friction,inertia_ratio=cfg.inertia_ratio)
        for k,h in enumerate(np.r_[self.steps,self.steps[-1]]):
            self.lower[k*NZ+32]=self.upper[k*NZ+32]=h/SCALE[32]

    def initialize(self,x,tau):
        cfg=self.cfg;Y=np.zeros((cfg.horizon+1,NZ));Y[:,32]=np.r_[self.steps,self.steps[-1]];Y[:,31]=.5
        state=x.copy()
        for k,h in enumerate(self.steps):
            Y[k,:NX]=state;Y[k,23:30]=np.asarray(self.robot.rnea(state[6:13],state[13:20],np.zeros(7))).ravel()
            try:state,contacts=self.nominal.step(state,tau,tau,h,acc=np.zeros(7))
            except RuntimeError as error:
                if str(error) != 'Unsupported resting contact':raise
                # Initial seed only: weak rebounds settle within the horizon.
                p,n,J,*_=self.robot.numeric_fk(state[6:13]);p=p-self.robot.origin
                state[:3]+=n*(cfg.radius-n@(state[:3]-p));state[3:6]=J@state[13:20]
                contacts=[dict(impulse_Ns=-cfg.mass*cfg.gravity_z*h,time=.5*h)]
            if contacts:Y[k,30]=contacts[0]['impulse_Ns']/cfg.mass;Y[k,31]=contacts[0]['time']/h
        Y[-1,:NX]=state;Y[-1,23:30]=np.asarray(self.robot.rnea(state[6:13],state[13:20],np.zeros(7))).ravel()
        Y[:,33]=self.times;Y[0,23:30]=tau
        self.z=((Y-OFFSET)/SCALE).ravel()

    def shift(self,dt,x,tau):
        cfg=self.cfg;N=cfg.horizon;old=self.z.reshape(-1,NZ)*SCALE+OFFSET
        new=self.times+dt;src=np.minimum(np.searchsorted(self.times,new,side='right')-1,N)
        Y=old.copy();Y[:,30]=0.;Y[:,31]=.5
        for k,t in enumerate(new):
            i=src[k];row=old[i];s=t-self.times[i];h=self.steps[min(i,N-1)]
            acc=(old[min(i+1,N),13:20]-row[13:20])/h
            Y[k,:6]=self.nominal.free(row[:NX],row[23:30],row[23:30],s,np.zeros(3),acc=acc)[:6]
            Y[k,6:13]=row[6:13]+s*row[13:20]+.5*s*s*acc;Y[k,13:20]=row[13:20]+s*acc
            Y[k,20:23]=row[20:23]
            if i<N:
                event=row[31]*h
                if s>=event-1e-12:
                    end=self.nominal.free(row[:NX],row[23:30],row[23:30],h,np.zeros(3),acc=acc)
                    impulse=(old[i+1,3:6]-end[3:6])*np.exp(cfg.drag*(h-event))
                    hit=self.nominal.free(row[:NX],row[23:30],row[23:30],event,np.zeros(3),acc=acc);hit[3:6]+=impulse
                    Y[k,:6]=self.nominal.free(hit,row[23:30],row[23:30],s-event,np.zeros(3),acc=acc)[:6];Y[k,20:23]=old[i+1,20:23]
            for j in range(7):Y[k,23+j]=np.interp(t,self.times,old[:,23+j])
        strongest_impulse=np.zeros(N)
        for i,h in enumerate(self.steps):
            event=self.times[i]+old[i,31]*h;k=np.searchsorted(new,event+1e-12,side='left')-1
            if 0<=k<N:
                impulse=old[i,30];Y[k,30]+=impulse
                if abs(impulse)>strongest_impulse[k]:
                    strongest_impulse[k]=abs(impulse)
                    Y[k,31]=np.clip((event-new[k])/self.steps[k],0,1)
        Y[0,:NX]=x;Y[0,23:30]=tau;Y[:,32]=np.r_[self.steps,self.steps[-1]];Y[:,33]=self.times
        if self.dual is not None:
            nf=self.problem['ncon_first'];nc=self.problem['ncon_stage'];mc=len(self.problem['cl'])
            previous=self.dual;self.dual=np.zeros_like(previous)
            for k,i in enumerate(src[:-1]):
                if i<N:self.dual[nf+k*nc:nf+(k+1)*nc]=previous[nf+i*nc:nf+(i+1)*nc]
            for k,i in enumerate(src):
                if i<N:self.dual[mc+k*NZ:mc+(k+1)*NZ]=previous[mc+i*NZ:mc+(i+1)*NZ]
            self.dual[mc:mc+NP]=0.
            for k in range(N):
                l,r=self.problem['groups']['contact'];self.dual[nf+k*nc+l:nf+k*nc+r]=0.
        self.z=((Y-OFFSET)/SCALE).ravel()
        self.warmstart_fallback=None;self.nominal.cfg.max_interval=.01
        ball=x[:6].copy();spin=x[20:23].copy()
        for k,h in enumerate(self.steps):
            Y[k,:6]=ball;Y[k,20:23]=spin;Y[k,30]=0.;Y[k,31]=.5
            acc=(Y[k+1,13:20]-Y[k,13:20])/h
            state=Y[k,:NX].copy()
            try:
                out,contacts=self.nominal.step(state,Y[k,23:30],Y[k+1,23:30],h,acc=acc)
                if getattr(self,'guard_ball_rollout',False) and not (-.15<=out[2]<=1.5):
                    raise RuntimeError('Ball rollout left optimizer workspace')
            except RuntimeError as error:
                if str(error) not in ['Unsupported resting contact','Ball rollout left optimizer workspace']:raise
                self.warmstart_fallback=dict(reason=str(error),stage=k);return
            ball=out[:6];spin=out[20:23]
            if contacts:Y[k,30]=contacts[0]['impulse_Ns']/cfg.mass;Y[k,31]=contacts[0]['time']/h
        Y[-1,:6]=ball;Y[-1,20:23]=spin
        self.z=((Y-OFFSET)/SCALE).ravel()
