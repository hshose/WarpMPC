"""Batched live acceleration-ramp reference and joint velocity callback.

The physical actuator is separate: measured lag/dead time belongs to the plant.
MPC wall time never advances the simulation clock.
"""
import jax
import jax.numpy as jnp
import numpy as np
import casadi as ca
from warpmpc.numpysadi import convert


class TorqueDynamics:
    """Policy rollouts only need seven inverse-dynamics torque outputs."""
    def __init__(self, factors):
        rnea=ca.Function.load(str(factors.directory/'panda_rnea.casadi'))
        q=ca.MX.sym('q',7);dq=ca.MX.sym('dq',7);a=ca.MX.sym('a',7)
        self.torque=jax.vmap(convert(ca.Function('policy_torque',[q,dq,a],[rnea(q,dq,a)]).expand(),backend='warp'))


class Execution:
    def __init__(self, factors, warmstart, backend='warp', observer_geometry=True):
        self.factors = factors
        self.warmstart = warmstart
        arr = factors.arrays
        self.acc_limit = jnp.asarray(arr['acceleration_limit'], dtype=jnp.float32)
        self.jerk_limit = jnp.asarray(arr['jerk_limit'], dtype=jnp.float32)
        self.velocity_limit = jnp.asarray(arr['velocity_limit'], dtype=jnp.float32)
        self.h = float(factors.meta['stage_steps'][0])
        if factors.cfg.get('control_dt', .02) > self.h:
            raise ValueError('This synchronous execution needs control_dt <= first stage')
        settings = factors.meta['spec'].get('execution', {})
        self.kp = settings.get('position_gain', 100.)
        self.kd = settings.get('velocity_gain', 20.)
        self.jerk_fraction = settings.get('jerk_fraction', .2)
        if not observer_geometry:
            return
        directory = factors.directory
        fk = ca.Function.load(str(directory/'panda_contact_fk.casadi'))
        mass = ca.Function.load(str(directory/'panda_mass.casadi'))
        q = ca.MX.sym('q', 7); dq = ca.MX.sym('dq', 7)
        p,n,J,W,_ = fk(q,ca.DM(arr['factor_parameters']))
        spatial = ca.vertcat(J,W)
        inertia = spatial@ca.solve(mass(q),spatial.T,'symbolicqr')
        # Observer ArmSample positions use the controller's relative frame.
        geometry = ca.vertcat(p-ca.DM(arr['origin']),n,J@dq,W@dq,ca.vec(inertia.T))
        self.geometry = jax.jit(jax.vmap(convert(ca.Function('observer_geometry',[q,dq],[geometry]).expand(), backend=backend)))

    def project(self, plan, q, dq, initial_acceleration):
        batch = plan.shape[0]
        knots = plan[:,:4]
        acc = self.warmstart.acceleration(knots[:,:,6:13].reshape(-1,7),knots[:,:,13:20].reshape(-1,7),knots[:,:,23:30].reshape(-1,7)).reshape(batch,4,7)
        raw = jnp.max(jnp.abs(acc[:,1:3])/self.acc_limit,axis=(1,2))
        acc = jnp.clip(acc,-.95*self.acc_limit,.95*self.acc_limit).at[:,0].set(initial_acceleration)
        # For synchronous 20 ms updates, only the first 30 ms ramp is reached.
        return (q,dq,acc[:,0],(acc[:,1]-acc[:,0])/self.h), raw

    def action(self, plan):
        """Seven accelerations at the first future knot, after native projection."""
        a=self.warmstart.acceleration(plan[:,1,6:13],plan[:,1,13:20],plan[:,1,23:30]).reshape(-1,7)
        return jnp.clip(a,-.95*self.acc_limit,.95*self.acc_limit)

    def applied_action(self, knot):
        """Convert only the applied knot's q/dq/torque; evaluate dynamics once."""
        a = self.warmstart.acceleration(knot[:, :7], knot[:, 7:14], knot[:, 14:21]).reshape(-1, 7)
        raw = jnp.max(jnp.abs(a) / self.acc_limit, axis=1)
        return jnp.clip(a, -.95*self.acc_limit, .95*self.acc_limit), raw

    def project_action(self, action, q, dq, initial_acceleration):
        raw=jnp.max(jnp.abs(action)/self.acc_limit,axis=1)
        a=jnp.clip(action,-.95*self.acc_limit,.95*self.acc_limit)
        return (q,dq,initial_acceleration,(a-initial_acceleration)/self.h),raw

    @staticmethod
    def reference(ramp, age):
        q,v,a,j = ramp
        return q+v*age+.5*a*age**2+j*age**3/6,v+a*age+.5*j*age**2,a+j*age

    def callback(self, reference, q, dq, commanded_velocity, commanded_acceleration, dt=.001):
        qr,vr,ar = reference
        desired = ar+self.kp*(qr-q)+self.kd*(vr-dq)
        change = self.jerk_fraction*self.jerk_limit*dt
        acceleration = jnp.clip(jnp.clip(desired,-.95*self.acc_limit,.95*self.acc_limit),commanded_acceleration-change,commanded_acceleration+change)
        velocity = commanded_velocity+dt*acceleration
        violation = jnp.any(jnp.abs(velocity)>self.velocity_limit,axis=1)
        return velocity,acceleration,violation
