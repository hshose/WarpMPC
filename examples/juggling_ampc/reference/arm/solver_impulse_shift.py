"""Roll out the measured ball with impulses shifted from the optimized plan.

This preserves contact seeds when a strict auxiliary collision rollout would
miss a future strike and fall below the workspace. The actual plant is never
advanced here; complementarity/gap/support remain part of the same SQP problem.
"""
import numpy as np
from .solver_state_shift import StateShiftSolver
from .formulation import NZ,SCALE,OFFSET


class ImpulseShiftSolver(StateShiftSolver):
    def shift(self,dt,x,tau):
        super().shift(dt,x,tau)
        cfg=self.cfg;Y=self.z.reshape(-1,NZ)*SCALE+OFFSET;ball=x[:6].copy();g=np.array([0.,0.,-9.81])
        for k in range(cfg.horizon):
            Y[k,:6]=ball;j=np.clip(Y[k,27],0.,12.);f=np.clip(Y[k,28],0.,1.);tc=f*cfg.h
            Y[k,27]=j;Y[k,28]=f
            acc=(Y[k+1,13:20]-Y[k,13:20])/cfg.h
            qc=Y[k,6:13]+tc*Y[k,13:20]+.5*tc*tc*acc
            normal=np.asarray(self.robot.fk(qc)[1]).ravel()
            ball=np.r_[ball[:3]+cfg.h*ball[3:6]+.5*cfg.h**2*g+(cfg.h-tc)*j*normal,
                       ball[3:6]+cfg.h*g+j*normal]
        Y[-1,:6]=ball;self.z=((Y-OFFSET)/SCALE).ravel()
