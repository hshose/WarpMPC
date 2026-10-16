"""Timestamp-aware mesh warmstarts with variable retained constraint rows."""
import numpy as np
from arm.solver import Solver
from arm.spin_simulator import SpinSimulator
from arm.simulator import SimConfig
from optimization.mesh.solver import MeshSolver
from optimization.mesh.formulation import grid
from . import combined_formulation as formulation
from .formulation import NZ,NP,SCALE


class CombinedSolver(MeshSolver):
    def __init__(self,cfg,robot):
        Solver.__init__(self,cfg,robot,formulation=formulation)
        self.steps,self.times,self.growth=grid(cfg)
        self.nominal=SpinSimulator(robot,SimConfig(restitution=cfg.restitution,drag=cfg.drag,ball_mass=cfg.mass,radius=cfg.radius,paddle_radius=cfg.paddle_radius,max_interval=.003),
            friction=cfg.friction,inertia_ratio=cfg.inertia_ratio)
        for k,h in enumerate(np.r_[self.steps,self.steps[-1]]):
            self.lower[k*NZ+32]=self.upper[k*NZ+32]=h/SCALE[32]

    def shift(self,dt,x,tau):
        previous=self.dual;self.dual=None
        super().shift(dt,x,tau)
        if previous is None:return
        p=self.problem;N=self.cfg.horizon
        src=np.minimum(np.searchsorted(self.times,self.times+dt,side='right')-1,N)
        mc=len(p['cl']);offsets=p['stage_offsets'];rows=p['kept_rows']
        self.dual=np.zeros_like(previous);contact=set(range(*p['groups']['contact']))
        for k,i in enumerate(src[:-1]):
            if i>=N:continue
            source={int(row):offsets[i]+j for j,row in enumerate(rows[i])}
            for j,row in enumerate(rows[k]):
                if row not in contact and row in source:self.dual[offsets[k]+j]=previous[source[row]]
        for k,i in enumerate(src):
            if i<N:self.dual[mc+k*NZ:mc+(k+1)*NZ]=previous[mc+i*NZ:mc+(i+1)*NZ]
        self.dual[mc:mc+NP]=0
