"""Stationary fixture alignment and position-only release detection."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from arm.model import HOME
from .raw_ball import MAX_BALL_GAP


def predicted_landing(state,drag,radius=.02,gravity=(0.,0.,-9.81)):
    """Nominal strike-plane crossing using the fitted linear-drag flight law."""
    x=np.asarray(state,dtype=float);g=np.asarray(gravity,dtype=float)
    if g.shape!=(3,) or not np.isfinite(g).all() or g[2]>=0:raise ValueError("Invalid ball gravity")
    if not np.isfinite(x[:6]).all() or not np.isfinite(drag) or drag<0:
        raise ValueError('Invalid landing prediction state/model')
    if x[2]<=radius:return x[:2].copy()
    if drag<1e-8:
        t=(x[5]+np.sqrt(x[5]**2-2*g[2]*(x[2]-radius)))/(-g[2])
        return x[:2]+t*x[3:5]+.5*t*t*g[:2]
    def flight(t):
        gain=-np.expm1(-drag*t)/drag
        return gain,x[2]+gain*x[5]+g[2]*(t-gain)/drag
    lo=0.;hi=1.
    while flight(hi)[1]>radius:
        hi*=2
        if hi>64:raise ValueError('No finite near-term landing')
    for _ in range(40):
        mid=.5*(lo+hi)
        if flight(mid)[1]>radius:lo=mid
        else:hi=mid
    t=.5*(lo+hi);gain=flight(t)[0]
    return x[:2]+gain*x[3:5]+(t-gain)/drag*g[:2]


def fixture_home(robot, ball_base, preparation_only=False, strike_plane_offset=0.):
    """Align below the fixture with a bounded vertical strike-plane offset."""
    ball = np.asarray(ball_base, float)
    if ball.shape != (3,) or not np.isfinite(ball).all(): raise ValueError('Invalid ball position')
    if not np.isfinite(strike_plane_offset) or abs(strike_plane_offset)>.05:
        raise ValueError('Strike-plane adjustment must stay within 5 cm of nominal')
    target = robot.origin.copy(); target[:2] = ball[:2];target[2]+=strike_plane_offset
    if np.linalg.norm(target[:2]-robot.origin[:2]) > .32:
        raise ValueError('Ball outside reachable startup region (32 cm from nominal HOME)')
    gap = float(ball[2]-target[2]-.02)
    supported = .40 <= gap <= 1.4
    if not supported and not (preparation_only and .40 <= gap <= 3.0):
        raise ValueError('Ball height outside startup range (0.40–1.40 m above face, ball bottom)')
    # Level the corrected physical face, retaining the original heading.
    nominal_R = np.asarray(robot.fk(HOME)[4])@robot.paddle_normal_rotation.T
    def fun(q):
        p, n, _, _, rot = robot.fk(q)
        return np.r_[(np.asarray(p).ravel()-target)/.001,
                     (np.asarray(rot)-nominal_R).ravel()/.01, .1*(q-HOME)]
    lo = np.maximum(robot.q_min, HOME-1.0); hi = np.minimum(robot.q_max, HOME+1.0)
    fit = least_squares(fun, HOME, bounds=(lo, hi), max_nfev=100)
    p, n, *_ = robot.numeric_fk(fit.x)
    if not fit.success or np.linalg.norm(p-target) > .001 or np.linalg.norm(n-[0, 0, 1]) > .01:
        raise ValueError('Fixture HOME cannot be reached within the local joint envelope')
    # Check the whole joint interpolation's paddle envelope, not only endpoint.
    for a in np.linspace(0, 1, 101):
        p, n, *_ = robot.numeric_fk((1-a)*HOME+a*fit.x)
        if (np.linalg.norm(p[:2]-robot.origin[:2]) > .35 or abs(p[2]-robot.origin[2]) > .15
                or n[2] < np.cos(.35)):
            raise ValueError('Ball is unreachable within the bounded homing path')
    return dict(q_home=fit.x.tolist(), ball_fixture_base_m=ball.tolist(),
                free_fall_gap_m=gap, release_height_supported=supported,
                face_target_base_m=target.tolist(), face_offset_m=(target-robot.origin).tolist(),
                position_error_m=float(np.linalg.norm(np.asarray(robot.fk(fit.x)[0]).ravel()-target)))


@dataclass
class ReleaseConfig:
    stable_seconds: float = .6
    max_gap: float = MAX_BALL_GAP
    stationary_radius: float = .004
    drop: float = .008
    downward_speed: float = .15
    confirm_frames: int = 3
    max_lateral: float = .025


class ReleaseGate:
    """Never apply gravity to a held ball. Fit a short measured fall on release.

    Repeated frames, occlusion, timing gaps, fixture relocation and side pushes
    cannot arm/release the controller. A gate is single-use; faults are latched.
    """
    def __init__(self, config=None, gravity=(0.,0.,-9.81)):
        self.gravity=np.asarray(gravity,dtype=float)
        if self.gravity.shape!=(3,) or not np.isfinite(self.gravity).all() or self.gravity[2]>=0:raise ValueError("Invalid ball gravity")
        self.cfg = config or ReleaseConfig(); self.history = deque()
        self.last_frame = None; self.last_time = None; self.anchor = None
        self.armed = False; self.released = None; self.fault = None; self.count = 0

    def fail(self, reason):
        self.fault = self.fault or reason
        raise RuntimeError(self.fault)

    def missing(self, now):
        if self.last_time is not None and now-self.last_time > self.cfg.max_gap:
            self.fail('Ball tracking lost; release is inhibited')

    def arm(self):
        if self.fault or self.anchor is None: raise RuntimeError('No stable fixture position')
        self.armed = True

    def update(self, t, position, frame):
        if self.fault: raise RuntimeError(self.fault)
        p = np.asarray(position, float)
        if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(t):
            return self.fail('Invalid ball observation')
        if self.last_frame is not None and (frame <= self.last_frame or t <= self.last_time):
            return None
        if self.last_time is not None and t-self.last_time > self.cfg.max_gap:
            return self.fail('Ball frame gap; release is inhibited')
        self.last_frame, self.last_time = frame, t
        self.history.append((t, p.copy()))
        while self.history[0][0] < t-max(.8, self.cfg.stable_seconds+.1): self.history.popleft()
        if self.anchor is None:
            recent = [(s, q) for s, q in self.history if s >= t-self.cfg.stable_seconds-.02]
            points = np.array([q for s, q in recent]); center = np.median(points, axis=0)
            if (recent[-1][0]-recent[0][0] >= self.cfg.stable_seconds
                    and len(recent) >= 30 and np.max(np.linalg.norm(points-center, axis=1)) <= self.cfg.stationary_radius):
                self.anchor = center
            return None
        drift = p-self.anchor
        if self.released is not None: return self.released
        if not self.armed:
            if np.linalg.norm(drift) > .006: return self.fail('Ball moved/released before robot was ready')
            return None
        if np.linalg.norm(drift[:2]) > self.cfg.max_lateral or drift[2] > .008:
            return self.fail('Fixture moved or ball pushed outside the release corridor')
        recent = [(s, q) for s, q in self.history if s >= t-.06]
        times = np.array([s-t for s, q in recent]); points = np.array([q for s, q in recent])
        velocity = np.linalg.lstsq(np.c_[np.ones(len(times)), times], points, rcond=None)[0][1]
        falling = drift[2] < -self.cfg.drop and velocity[2] < -self.cfg.downward_speed
        self.count = self.count+1 if falling else 0
        if self.count >= self.cfg.confirm_frames:
            # Estimate release instant from anchor and observed fall, then fit
            # p,v at current capture time under known gravity using falling data.
            fallen = [(s, q) for s, q in self.history if q[2] < self.anchor[2]-.003 and s >= t-.10]
            tt = np.array([s-t for s, q in fallen]); pp = np.array([q for s, q in fallen])
            corrected = pp-.5*tt[:,None]**2*self.gravity
            pv = np.linalg.lstsq(np.c_[np.ones(len(tt)), tt], corrected, rcond=None)[0]
            residual = pp-(pv[0]+tt[:, None]*pv[1]+.5*tt[:, None]**2*self.gravity)
            if len(fallen) < 3 or np.max(np.linalg.norm(residual, axis=1)) > .004:
                return self.fail('Release observations do not fit a free fall')
            if not -.9 < pv[1, 2] < -.15 or np.linalg.norm(pv[1, :2]) > .25:
                return self.fail('Release velocity outside the supported local drop')
            self.released = dict(capture_time=t, position=pv[0].tolist(), velocity=pv[1].tolist(),
                                 observed_drop_m=float(-drift[2]), frame=frame)
            return self.released
        return None
