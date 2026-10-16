"""Preserve shifted joint/torque arcs, regenerate only physical ball contacts."""
import numpy as np
from .solver_state_shift import StateShiftSolver
from .formulation import NX,NZ,SCALE,OFFSET

class BallRolloutSolver(StateShiftSolver):
    def shift(self,dt,x,tau):
        ff=self.formulation;NX,NZ,SCALE,OFFSET=ff.NX,ff.NZ,ff.SCALE,ff.OFFSET
        super().shift(dt,x,tau)
        self.warmstart_fallback=None
        cfg=self.cfg;Y=self.z.reshape(-1,NZ)*SCALE+OFFSET
        self.nominal.cfg.max_interval=.01
        ball=x[:6].copy();spin=x[20:NX].copy()
        for k in range(cfg.horizon):
            Y[k,:6]=ball;Y[k,20:NX]=spin;Y[k,NX+7]=0.;Y[k,NX+8]=.5
            acc=(Y[k+1,13:20]-Y[k,13:20])/cfg.h
            state=Y[k,:NX].copy()
            try:
                out,contacts=self.nominal.step(state,Y[k,NX:NX+7],Y[k+1,NX:NX+7],cfg.h,acc=acc)
                if getattr(self,'guard_ball_rollout',False) and not (-.15<=out[2]<=1.5):
                    raise RuntimeError('Ball rollout left optimizer workspace')
            except RuntimeError as error:
                if str(error) not in ['Unsupported resting contact','Ball rollout left optimizer workspace']:raise
                # A heuristic seed rollout may encounter Zeno-like tiny bounces
                # in a future interval. Keep the already shifted primal/dual
                # solution; this fallback never changes the physical plant.
                self.warmstart_fallback=dict(reason=str(error),stage=k)
                return
            ball=out[:6];spin=out[20:NX]
            if contacts:
                Y[k,NX+7]=contacts[0]['impulse_Ns']/cfg.mass
                Y[k,NX+8]=contacts[0]['time']/cfg.h
        Y[-1,:6]=ball;Y[-1,20:NX]=spin
        self.z=((Y-OFFSET)/SCALE).ravel()
