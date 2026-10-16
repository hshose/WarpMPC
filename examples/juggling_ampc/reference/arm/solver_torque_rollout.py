"""Alternative warmstart: shift torque curves and integrate the full nominal arm."""
import numpy as np
from .solver import Solver
from .formulation import NZ,NX,NP,SCALE,OFFSET

class TorqueRolloutSolver(Solver):
    def shift(self,dt,x,tau):
        cfg=self.cfg;N=cfg.horizon;old=self.z.reshape(-1,NZ)*SCALE+OFFSET
        times=np.arange(N+1)*cfg.h;src=np.minimum(np.searchsorted(times,times+dt,side='right')-1,N)
        if self.dual is not None:
            nf=self.problem['ncon_first'];nc=self.problem['ncon_stage'];mc=len(self.problem['cl'])
            previous=self.dual;self.dual=np.zeros_like(previous)
            for k,i in enumerate(src[:-1]):
                if i<N:self.dual[nf+k*nc:nf+(k+1)*nc]=previous[nf+i*nc:nf+(i+1)*nc]
            for k,i in enumerate(src):
                if i<N:self.dual[mc+k*NZ:mc+(k+1)*NZ]=previous[mc+i*NZ:mc+(i+1)*NZ]
            self.dual[mc:mc+NP]=0
            for k in range(N):
                l,r=self.problem['groups']['contact'];self.dual[nf+k*nc+l:nf+k*nc+r]=0
        Y=old.copy()
        for j in range(7):Y[:,20+j]=np.interp(times+dt,times,old[:,20+j])
        Y[0,20:27]=tau;Y[:,29]=cfg.h;Y[:,30]=times
        state=x.copy();self.nominal.cfg.max_interval=.01
        for k in range(N):
            Y[k,:NX]=state;Y[k,27]=0;Y[k,28]=.5
            state,contacts=self.nominal.step(state,Y[k,20:27],Y[k+1,20:27],cfg.h)
            if contacts:
                Y[k,27]=contacts[0]['impulse_Ns']/cfg.mass;Y[k,28]=contacts[0]['time']/cfg.h
        Y[-1,:NX]=state
        self.z=((Y-OFFSET)/SCALE).ravel()
