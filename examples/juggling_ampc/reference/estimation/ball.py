"""Small hybrid EKF with delayed position updates and measured-arm prediction.

State: [position, velocity, angular velocity, disturbance acceleration], all in
world axes, with position relative to the nominal racket face. No plant state,
contact flag, force, commanded wind, or ball velocity is accepted by this API.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from arm.spin_contact import impact


@dataclass
class ObserverConfig:
    position_std: float = .001
    drag: float = .34
    gravity_x:float=0.
    gravity_y:float=0.
    gravity_z:float=-9.81
    acceleration_time_constant: float = 1.
    acceleration_noise: float = .3  # m/s^2 / sqrt(s), unknown smooth wind
    velocity_noise: float = .015  # m/s / sqrt(s), unmodeled flight acceleration
    spin_noise: float = .1  # rad/s / sqrt(s)
    initial_velocity_std: float = .3
    initial_spin_std: float = 20.
    initial_acceleration_std: float = .2
    restitution: float = .54
    friction: float = .3
    ball_mass: float = .0027
    radius: float = .02
    paddle_radius: float = .075
    inertia_ratio: float = 2/3
    contact_delay: float = .001  # effective impulse time after first touch
    impact_time_std: float = .001  # uncertainty of effective compliant-contact timing
    impact_velocity_std: float = .15
    impact_tangent_std: float = .04
    impact_spin_std: float = 1.
    contact_measurement_std: float = .004
    contact_measurement_window: float = .004  # seconds after predicted impact
    impact_refractory: float = .03
    innovation_gate: float = 25.  # squared Mahalanobis distance, 3 position DOFs
    max_measurement_age: float = .1
    history_seconds: float = .25
    impact_independent_tangent_std: float = 0.  # additive velocity kick; no spin reaction


@dataclass
class ArmSample:
    t: float
    q: np.ndarray
    dq: np.ndarray
    p: np.ndarray
    n: np.ndarray


def skew(v):
    return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])


class BallObserver:
    """Keep the posterior at capture time; predict a copy to the present.

    Measurements must arrive in increasing capture-time order. Late/duplicate,
    invalid and gated samples are reported rather than applied to the wrong time.
    Arm history contains only measurements that have already arrived. initialize()
    uses an explicit held/released-from-rest prior, not a measured velocity/spin.
    """
    def __init__(self, robot, config=None):
        self.robot=robot;self.cfg=config or ObserverConfig()
        c=self.cfg
        if min(c.position_std,c.acceleration_time_constant,c.history_seconds,c.contact_measurement_window)<=0:
            raise ValueError('Noise and observer timing parameters must be positive.')
        if not np.isfinite(c.impact_time_std) or c.impact_time_std<0:
            raise ValueError('Impact timing uncertainty must be finite and nonnegative.')
        if not np.isfinite(c.impact_independent_tangent_std) or c.impact_independent_tangent_std<0:
            raise ValueError('Independent tangential impact uncertainty must be finite and nonnegative.')
        if c.max_measurement_age>=c.history_seconds:
            raise ValueError('Arm history must exceed maximum measurement age.')
        self.history=deque();self.initialized=False;self.t=0.;self.last_impact=-np.inf
        self.last_capture=-np.inf;self.last_accepted=-np.inf
        self.events=[];self.updates=[]

    def record_arm(self,t,q,dq):
        q=np.asarray(q,float).copy();dq=np.asarray(dq,float).copy()
        if q.shape!=(7,) or dq.shape!=(7,) or not np.all(np.isfinite(np.r_[t,q,dq])):
            raise ValueError('Finite timestamp and seven joint positions/velocities required.')
        if self.history and t<=self.history[-1].t:
            raise ValueError('Arm timestamps must increase.')
        p,n,_,_=self.robot.numeric_fk(q)
        self.history.append(ArmSample(float(t),q,dq,p-self.robot.origin,n))
        # Keep a bracketing sample. Never discard history needed by the posterior.
        cutoff=t-self.cfg.history_seconds
        if self.initialized and self.t<cutoff:
            self.x,self.P,self.last_impact,events=self._advance(cutoff)
            self.events.extend(events);self.t=cutoff
        while len(self.history)>2 and self.history[1].t<cutoff:self.history.popleft()

    def initialize(self,position,t=0.):
        """Position of a held ball at release; zero-mean uncertain v/spin priors."""
        p=np.asarray(position,float)
        if p.shape!=(3,) or not np.all(np.isfinite(np.r_[p,t])):raise ValueError('Finite position and timestamp required.')
        if not self.history or abs(self.history[-1].t-t)>1e-9:raise ValueError('Record initial arm state first.')
        c=self.cfg;self.x=np.r_[p,np.zeros(9)]
        self.P=np.diag(np.repeat([c.position_std**2,c.initial_velocity_std**2,
                                 c.initial_spin_std**2,c.initial_acceleration_std**2],3))
        self.t=float(t);self.initialized=True;self.last_accepted=float(t)

    @lru_cache(maxsize=256)
    def _flight(self,dt):
        c=self.cfg;A=np.zeros((12,12));A[:3,3:6]=np.eye(3)
        A[3:6,3:6]=-c.drag*np.eye(3);A[3:6,9:12]=np.eye(3)
        A[9:12,9:12]=-np.eye(3)/c.acceleration_time_constant
        L=np.diag(np.repeat([0.,c.velocity_noise**2,c.spin_noise**2,c.acceleration_noise**2],3))
        V=expm(np.block([[A,L],[np.zeros_like(A),-A.T]])*dt)
        F=V[:12,:12];Q=V[:12,12:]@F.T
        affine=np.zeros((13,13));affine[:12,:12]=A;affine[3:6,12]=[c.gravity_x,c.gravity_y,c.gravity_z]
        b=expm(affine*dt)[:12,12]
        return F,b,.5*(Q+Q.T)

    def _free(self,x,dt):
        # Closed form for the mean, including negative intervals used to
        # differentiate the event-time map. This avoids an expm per derivative.
        c=self.cfg;d=c.drag;k=1/c.acceleration_time_constant
        ed=np.exp(-d*dt);ek=np.exp(-k*dt)
        vd=-np.expm1(-d*dt)/d if d else dt
        pd=(dt-vd)/d if d else .5*dt*dt
        va=(ek-ed)/(d-k) if abs(d-k)>1e-9 else dt*ed
        pa=((1-ek)/k-vd)/(d-k) if abs(d-k)>1e-9 else (vd-dt*ed)/d
        y=x.copy();g=np.array([c.gravity_x,c.gravity_y,c.gravity_z])
        y[:3]+=vd*x[3:6]+pd*g+pa*x[9:12]
        y[3:6]=ed*x[3:6]+vd*g+va*x[9:12];y[9:12]=ek*x[9:12]
        return y

    @staticmethod
    def _arm(a,b,t):
        f=(t-a.t)/(b.t-a.t)
        q=a.q+f*(b.q-a.q);dq=a.dq+f*(b.dq-a.dq)
        p=a.p+f*(b.p-a.p);n=a.n+f*(b.n-a.n);n=n/np.linalg.norm(n)
        return q,dq,p,n

    def _geometry(self,x,a,b,t):
        _,_,p,n=self._arm(a,b,t);rel=x[:3]-p
        speed=n@(x[3:6]-(b.p-a.p)/(b.t-a.t))
        return float(n@rel-self.cfg.radius),np.linalg.norm(rel-n*(n@rel)),float(speed)

    def _hit_map(self,x,dt,a,b,t,hit):
        """Event-time map with a moving local plane and finite-arm impulse.

        Differentiating this map includes the change in impact time with position;
        differentiating just the velocity reset would miss that uncertainty.
        """
        c=self.cfg
        # Refine the nominal event time for this perturbed state.
        for _ in range(2):
            at=self._free(x,hit);gap,_,speed=self._geometry(at,a,b,t+hit)
            residual=gap-max(-speed,0.)*(-c.contact_delay)
            hit-=residual/min(speed,-.05)
        at=self._free(x,hit);q,dq,_,_=self._arm(a,b,t+hit)
        p,n,Jp,W,R=self.robot.contact_fk(q)
        p=np.asarray(p).ravel()-self.robot.origin;n=np.asarray(n).ravel()
        rel=at[:3]-p;tangent=rel-n*(n@rel)
        J=np.asarray(Jp)-skew(tangent)@np.asarray(W)
        relative=at[3:6]-np.cross(at[6:9],c.radius*n)-J@dq
        u,domega,_,_=impact(n,np.asarray(R)[:,:2],J,np.asarray(self.robot.mass(q)),relative,
                            c.ball_mass,c.radius,c.restitution,c.friction,c.inertia_ratio)
        at[3:6]+=u;at[6:9]+=domega
        return self._free(at,dt-hit),hit,n

    def _step(self,x,P,a,b,t,end,last):
        dt=end-t;F,offset,Q=self._flight(round(dt,12));y=F@x+offset
        ga,ra,va=self._geometry(x,a,b,t);gb,rb,vb=self._geometry(y,a,b,end)
        c=self.cfg
        # Effective plane accounts for the ~2 ms compliant Newton impact. This
        # is an estimator approximation, not a change to native contact physics.
        ea=ga+max(-va,0.)*c.contact_delay;eb=gb+max(-vb,0.)*c.contact_delay
        if t-last>c.impact_refractory and eb<=0 and max(ra,rb)<c.paddle_radius and vb<-.2 and ga>-.02:
            hit=dt*np.clip(ea/max(ea-eb,1e-12),0,1)
            y,hit,n=self._hit_map(x,dt,a,b,t,hit)
            D=np.empty((12,12))
            for j in range(12):
                eps=2e-5 if 6<=j<9 else 2e-6;dx=np.zeros(12);dx[j]=eps
                yp,_,_=self._hit_map(x+dx,dt,a,b,t,hit)
                ym,_,_=self._hit_map(x-dx,dt,a,b,t,hit)
                D[:,j]=(yp-ym)/(2*eps)
            tangent=np.eye(3)-np.outer(n,n)
            impulse_cov=c.impact_velocity_std**2*np.outer(n,n)+c.impact_tangent_std**2*tangent
            G=np.zeros((12,3));G[3:6]=np.eye(3);G[6:9]=-skew(n)/(c.inertia_ratio*c.radius)
            P=D@P@D.T+Q+G@impulse_cov@G.T
            if c.impact_independent_tangent_std>0:
                # Inject a velocity-only event kick, then propagate its
                # position/velocity covariance through the remaining flight.
                # Spin and acceleration receive no independent kick.
                post_F,_,_=self._flight(round(max(0.,dt-hit),12))
                injection=post_F[:,3:6]
                P+=c.impact_independent_tangent_std**2*injection@tangent@injection.T
            # Contact-time uncertainty shifts the rebound position by the
            # velocity jump times timing error. Velocity-only impulse noise
            # otherwise leaves the immediate post-contact position overconfident.
            timing=np.zeros(12)
            timing[:3]=y[3:6]-(F@x+offset)[3:6]
            P+=c.impact_time_std**2*np.outer(timing,timing)
            P[6:9,6:9]+=c.impact_spin_std**2*tangent
            return y,.5*(P+P.T),t+hit
        return y,F@P@F.T+Q,last

    def _advance(self,end):
        if not self.initialized:raise RuntimeError('Initialize with a held-ball position first.')
        if not np.isfinite(end) or end<self.t-1e-10 or end>self.history[-1].t+1e-10:raise ValueError('Prediction requires causal arm history.')
        x=self.x.copy();P=self.P.copy();t=self.t;last=self.last_impact;events=[]
        history=list(self.history)
        if t<history[0].t-1e-10:raise ValueError('Measurement gap exceeds retained arm history.')
        for a,b in zip(history[:-1],history[1:]):
            if b.t<=t+1e-10:continue
            if t>=end-1e-10:break
            stop=min(b.t,end);old=last
            x,P,last=self._step(x,P,a,b,t,stop,last)
            if last!=old:events.append(last)
            t=stop
        return x,P,last,events

    def update_position(self,capture_time,position,arrival_time):
        z=np.asarray(position,float);c=self.cfg
        info=dict(capture_time=float(capture_time),arrival_time=float(arrival_time),accepted=False)
        if not np.all(np.isfinite([capture_time,arrival_time])):info['reason']='invalid timestamp'
        elif z.shape!=(3,) or not np.all(np.isfinite(z)):info['reason']='invalid position'
        elif capture_time>arrival_time+1e-10:info['reason']='future timestamp'
        elif arrival_time>self.history[-1].t+1e-10:info['reason']='missing arm history'
        elif capture_time<=self.last_capture+1e-10 or capture_time<self.t-1e-10:info['reason']='old or duplicate timestamp'
        elif arrival_time-capture_time>c.max_measurement_age+1e-10:info['reason']='measurement too old'
        else:
            self.x,self.P,self.last_impact,events=self._advance(capture_time)
            self.events.extend(events);self.t=float(capture_time);self.last_capture=float(capture_time)
            residual=z-self.x[:3]
            near_impact=bool(abs(capture_time-self.last_impact)<c.contact_measurement_window)
            R=np.eye(3)*(c.contact_measurement_std if near_impact else c.position_std)**2
            S=self.P[:3,:3]+R;nis=float(residual@np.linalg.solve(S,residual))
            info.update(innovation=residual.tolist(),nis=nis,near_impact=near_impact)
            if nis<=c.innovation_gate:
                K=np.linalg.solve(S,self.P[:3,:]).T;self.x+=K@residual
                A=np.eye(12);A[:,:3]-=K
                self.P=A@self.P@A.T+K@R@K.T  # Joseph form
                self.P=.5*(self.P+self.P.T)
                self.last_accepted=float(capture_time);info.update(accepted=True,reason='accepted')
            else:info['reason']='innovation gate'
        self.updates.append(info)
        return info

    def estimate(self,t):
        x,P,last,_=self._advance(t)
        return x,P,dict(last_impact=last,measurement_age=t-self.last_accepted,
                       stale=t-self.last_accepted>self.cfg.max_measurement_age)

    def mpc_state(self,t,q,dq):
        x,P,info=self.estimate(t)
        return np.r_[x[:6],q,dq,x[6:9]],P,info

    def contact_pending(self,t):
        x,_,info=self.estimate(t)
        if len(self.history)<2:return False
        gap,radial,speed=self._geometry(x,self.history[-2],self.history[-1],t)
        return bool(radial<self.cfg.paddle_radius and
                    (0<=t-info['last_impact']<.004 or (-.01<gap<.008 and speed<-.2)))
