"""Shift the existing optimized arcs directly, retaining bounded joint predictions."""
import numpy as np
from .solver import Solver
from .formulation import NX,NZ,NP,SCALE,OFFSET

class StateShiftSolver(Solver):
    def shift(self,dt,x,tau):
        ff=self.formulation;NX,NZ,NP,SCALE,OFFSET=ff.NX,ff.NZ,ff.NP,ff.SCALE,ff.OFFSET
        cfg=self.cfg;N=cfg.horizon;old=self.z.reshape(-1,NZ)*SCALE+OFFSET
        times=np.arange(N+1)*cfg.h;new=times+dt;src=np.minimum(np.searchsorted(times,new,side='right')-1,N)
        Y=old.copy();Y[:,NX+7]=0;Y[:,NX+8]=.5
        for k,t in enumerate(new):
            i=src[k];row=old[i];s=t-times[i]
            acc=(old[min(i+1,N),13:20]-row[13:20])/cfg.h
            Y[k,:3]=row[:3]+s*row[3:6]+.5*s*s*np.array([0,0,-9.81]);Y[k,3:6]=row[3:6]+s*np.array([0,0,-9.81])
            Y[k,6:13]=row[6:13]+s*row[13:20]+.5*s*s*acc;Y[k,13:20]=row[13:20]+s*acc
            if NX>20:Y[k,20:NX]=row[20:NX]
            if i<N:
                event=row[NX+8]*cfg.h
                qc=row[6:13]+event*row[13:20]+.5*event*event*acc
                normal=np.asarray(self.robot.fk(qc)[1]).ravel()
                if s>=event-1e-12:
                    impulse=row[NX+7]*normal
                    if NX>20:
                        impulse=old[i+1,3:6]-row[3:6]-cfg.h*np.array([0,0,-9.81])
                        Y[k,20:NX]=old[i+1,20:NX]
                    Y[k,:3]+=(s-event)*impulse;Y[k,3:6]+=impulse
            for j in range(7):Y[k,NX+j]=np.interp(t,times,old[:,NX+j])
        strongest_impulse=np.zeros(N)
        for i in range(N):
            event=times[i]+old[i,NX+8]*cfg.h;k=np.searchsorted(new,event+1e-12,side='left')-1
            if 0<=k<N:
                impulse=old[i,NX+7];Y[k,NX+7]+=impulse
                if abs(impulse)>strongest_impulse[k]:
                    strongest_impulse[k]=abs(impulse)
                    Y[k,NX+8]=np.clip((event-new[k])/cfg.h,0,1)
        Y[0,:NX]=x;Y[0,NX:NX+7]=tau;Y[:,NX+9]=cfg.h;Y[:,NX+10]=times
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
        self.z=((Y-OFFSET)/SCALE).ravel()
