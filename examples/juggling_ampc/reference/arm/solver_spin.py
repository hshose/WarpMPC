"""One SQP / 50 OSQP with a frictional, spinning-ball physical warmstart."""
from . import spin_formulation
from .solver_ball_rollout import BallRolloutSolver
from .spin_simulator import SpinSimulator
from .simulator import SimConfig


class SpinSolver(BallRolloutSolver):
    def __init__(self,cfg,robot):
        super().__init__(cfg,robot,formulation=spin_formulation)
        self.nominal=SpinSimulator(robot,SimConfig(gravity_x=cfg.gravity_x,gravity_y=cfg.gravity_y,gravity_z=cfg.gravity_z,restitution=cfg.restitution,drag=cfg.drag,ball_mass=cfg.mass,radius=cfg.radius,paddle_radius=cfg.paddle_radius,max_interval=.003),
                                   friction=cfg.friction,inertia_ratio=cfg.inertia_ratio)
