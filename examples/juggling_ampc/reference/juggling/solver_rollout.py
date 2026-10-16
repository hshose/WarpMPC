"""Alternative warmstart: shift controls, then strictly roll out their physics.

This removes cross-knot shooting defects and lets predicted collisions migrate
between fixed time cells. It does not supply a contact schedule or new controller.
"""
import numpy as np
from .formulation import NX,NZ,SCALE
from .solver import RTI
from .simulator import SimConfig,step


class RolloutRTI(RTI):
    def shift(self,dt,x):
        old=self.z.reshape(-1,NZ)*SCALE
        times=np.r_[0.,np.cumsum(old[:-1,23])]
        # Remap multipliers independently from relaxed-impulse interpolation.
        # Tiny predicted impulses can share a new cell; the strict primal rollout
        # below replaces them, so that must not erase all dynamic/box warmstarts.
        Y=old.copy()
        if self.dual is not None:
            N=self.cfg.horizon;nf=self.problem['ncon_first'];nc=self.problem['ncon_stage']
            mc=nf+N*nc+self.problem['ncon_last']
            shifted=np.zeros_like(self.dual)
            source=np.minimum(np.searchsorted(times,times+dt,side='right')-1,N)
            for k in range(N):
                i=source[k]
                if i<N:shifted[nf+k*nc:nf+(k+1)*nc]=self.dual[nf+i*nc:nf+(i+1)*nc]
            for k in range(N+1):
                i=source[k]
                if i<N:shifted[mc+k*NZ:mc+(k+1)*NZ]=self.dual[mc+i*NZ:mc+(i+1)*NZ]
                shifted[mc+k*NZ+24]=0.
            shifted[mc:mc+NX]=0.
            self.dual=shifted
        cfg=SimConfig(ball_mass=self.cfg.mass,ball_radius=self.cfg.radius,
                      # Seed real face impacts, including those outside the
                      # optimizer's conservative edge margin. Treating that
                      # margin as a missing physical surface erases contact
                      # derivatives exactly when the next solve must recenter it.
                      paddle_radius=self.cfg.paddle_radius,
                      restitution=self.cfg.restitution)
        state=x.copy();t=dt;Y[0,24]=0
        for k in range(self.cfg.horizon):
            h=float(np.clip(Y[k,23],self.cfg.h_min,self.cfg.h_max))
            if not self.cfg.variable_time:h=self.cfg.h_nom
            Y[k,23]=h
            accum=np.zeros(5);start=t
            while start<t+h-1e-12:
                i=min(np.searchsorted(times,start+1e-12,side='right')-1,self.cfg.horizon)
                end=min(t+h,times[i+1] if i<self.cfg.horizon else t+h)
                accum+=(end-start)*old[i,16:21];start=end
            u=accum/h
            Y[k,:NX]=state;Y[k,16:21]=u;Y[k,21]=0.;Y[k,22]=.5
            nextstate,contacts=step(state,u,h,cfg)
            if len(contacts)>1:
                raise RuntimeError('Rollout contains multiple impacts in one MPC interval')
            if contacts:
                Y[k,21]=contacts[0]['impulse_Ns']/self.cfg.mass
                Y[k,22]=contacts[0]['time']/h
            Y[k+1,:NX]=nextstate;Y[k+1,24]=Y[k,24]+h
            state=nextstate;t+=h
        self.z=(Y/SCALE).reshape(-1)
        if self.dual is not None:
            # Inactive/active contact multipliers cannot be reused when a contact
            # changes cells. Dynamics and box duals remain warmstarted.
            nf=self.problem['ncon_first'];nc=self.problem['ncon_stage']
            for k in range(self.cfg.horizon):
                self.dual[nf+k*nc+NX+1:nf+(k+1)*nc]=0.
