"""Position-only Vicon surrogate and observer adapter (no truth velocity/spin)."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from estimation.ball import BallObserver,ObserverConfig


@dataclass
class ViconConfig:
    rate_hz: float = 200.
    latency_s: float = .005
    position_std_m: float = .001  # standard deviation of EACH world coordinate
    seed: int = 7
    dropout_probability: float = 0.


class PositionFeedback:
    def __init__(self,robot,position,q,dq,sensor=None,observer=None):
        self.sensor=sensor or ViconConfig();c=self.sensor
        if c.rate_hz<=0 or c.latency_s<0 or c.position_std_m<=0 or not 0<=c.dropout_probability<1:
            raise ValueError('Invalid Vicon settings.')
        self.observer=BallObserver(robot,observer or ObserverConfig(position_std=c.position_std_m))
        if c.latency_s>self.observer.cfg.max_measurement_age:raise ValueError('Latency exceeds supported observation age.')
        self.rng=np.random.default_rng(c.seed);self.queue=deque();self.samples=[];self.frame=0
        self.previous_t=0.;self.previous_position=np.array(position).copy()
        # A held ball was visible before release. Its most recent available
        # pre-release position is valid at t=0 because it was stationary then.
        held=np.asarray(position)+self.rng.normal(0,c.position_std_m,3)
        self.initial_measurement=dict(capture_time=-c.latency_s,arrival_time=0.,position=held.tolist(),
                                      assumption='Ball held stationary before release; zero-mean velocity and spin priors.')
        self.observer.record_arm(0.,q,dq);self.observer.initialize(held,0.)
        self._capture(0.,position)
        self._deliver(0.)

    def _capture(self,t,position):
        c=self.sensor;z=np.asarray(position)+self.rng.normal(0,c.position_std_m,3)
        dropped=self.rng.random()<c.dropout_probability
        self.samples.append(dict(capture_time=t,arrival_time=t+c.latency_s,position=z.tolist(),dropped=dropped))
        if not dropped:self.queue.append((t+c.latency_s,t,z))
        self.frame+=1

    def _deliver(self,t):
        while self.queue and self.queue[0][0]<=t+1e-10:
            _,capture,z=self.queue.popleft();self.observer.update_position(capture,z,t)

    def step(self,t,position,q,dq):
        self.observer.record_arm(t,q,dq)
        while self.frame/self.sensor.rate_hz<=t+1e-10:
            capture=self.frame/self.sensor.rate_hz
            f=(capture-self.previous_t)/(t-self.previous_t)
            at=self.previous_position+f*(np.asarray(position)-self.previous_position)
            self._capture(capture,at)
        self._deliver(t);self.previous_t=t;self.previous_position=np.asarray(position).copy()

    def state(self,t,q,dq):return self.observer.mpc_state(t,q,dq)
