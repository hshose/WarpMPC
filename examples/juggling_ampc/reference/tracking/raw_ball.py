"""Select a stationary unlabeled point above the paddle, then track continuity.

SDK IDs and array indices are diagnostic only. Selection uses several seconds
of reconstructed positions, measured CAD face location and temporal coverage.
No physical marker diameter is available from this 3D API.
"""
from collections import deque, Counter
import numpy as np
from .streams import capture_time, paddle_pose

MAX_BALL_GAP = .08  # 100 Hz: bridge several frames, never relock after losing identity.


class RawBall:
    def __init__(self, paddle_to_face, seconds=3., lateral=.35, height=(.35, 3.0), extra_latency=0., max_gap=MAX_BALL_GAP):
        self.paddle_to_face = np.asarray(paddle_to_face)
        self.seconds = seconds; self.lateral = lateral; self.height = height; self.extra_latency = extra_latency
        self.frames = deque(); self.tracks = []; self.last_frame = None; self.last_time = None
        self.selection = None; self.position = None; self.seen_time = None
        self.allow_motion = False; self.fault = None
        self.max_gap = max_gap; self.position_std = .001
        self.x = None; self.P = None; self.filter_time = None
        self.acceleration = np.zeros(3)
        self.status = 'acquiring'; self.rejections = Counter(); self.startup_resets = 0
        self.last_association = None

    def fail(self, message):
        self.fault = message; self.status = 'lost'
        raise RuntimeError(message)

    def _predict(self, t):
        dt = t-self.filter_time
        if dt < -1e-9: raise ValueError('Cannot predict raw ball backwards')
        dt = max(dt, 0.)
        F = np.eye(6); F[:3, 3:] = np.eye(3)*dt
        # Continuous white acceleration, used only for point association.
        # During MPC, the contact-aware BallObserver supplies the prediction.
        Q = 2.**2*np.kron(np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]), np.eye(3))
        x = F@self.x+np.r_[.5*dt*dt*self.acceleration, dt*self.acceleration]
        return x, F@self.P@F.T+Q

    def estimate(self, t):
        if self.selection is None: return None
        x, P = self._predict(t)
        age = t-self.seen_time
        return x, P, dict(measurement_age=age, stale=bool(self.fault or age > self.max_gap),
                         status=self.status)

    @staticmethod
    def points(frame):
        out = []
        for point in frame.get('unlabeled_markers') or []:
            p = np.asarray(point['position_m'], float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError('Invalid unlabeled marker position')
            out.append((point.get('id'), p))
        return out

    def update(self, frame, prediction=None):
        """Return a real accepted observation, or None (never a synthetic sample).

        Optional prediction is (six-state mean, covariance, measurement std),
        in Vicon axes at this frame's capture time, from the main ball EKF.
        """
        if self.fault: raise RuntimeError(self.fault)
        # DataStream can deliver empty/buffered frames during subscription.
        # They cannot initialize identity or the acquisition timestamp window.
        if self.selection is None and paddle_pose(frame) is None: return None
        t = capture_time(frame, self.extra_latency); number = frame['frame']
        if not np.isfinite(t): self.fail('Invalid raw marker timestamp')
        if self.last_frame is not None and (number <= self.last_frame or t <= self.last_time):
            if self.selection is not None:
                self.fail('Raw marker frame/time restarted or duplicated')
            # No identity has been selected and no robot has been started.
            # Restart the full stationary window after a startup timing jump.
            self.frames.clear(); self.tracks.clear(); self.startup_resets += 1
        self.last_frame = number; self.last_time = t
        points = self.points(frame)
        if self.selection is not None:
            dt = t-self.seen_time
            if dt > self.max_gap+1e-9: self.fail('Selected raw ball tracking lost; reacquisition requires a new session')
            x, P = self._predict(t); std = self.position_std
            if prediction is not None:
                x, P, std = prediction
                x = np.asarray(x, float); P = np.asarray(P, float)
                if (x.shape != (6,) or P.shape != (6,6) or not np.isfinite(x).all()
                        or not np.isfinite(P).all() or not np.isfinite(std) or std <= 0):
                    raise ValueError('Invalid ball association prediction')
            R = np.eye(3)*std**2; S = P[:3, :3]+R
            near = []
            for identity, p in points:
                residual = p-x[:3]
                # Covariance alone must not open an arbitrarily large gate.
                if np.linalg.norm(residual) > .020: continue
                if not self.allow_motion and np.linalg.norm(p-self.selection['position_vicon_m']) > .030: continue
                nis = float(residual@np.linalg.solve(S, residual))
                if nis <= 25.: near.append((identity, p, residual, nis))
            self.last_association = dict(capture_time=t, candidate_count=len(near),
                prediction_vicon_m=x[:3].tolist(), accepted=False)
            if len(near) != 1:
                reason = 'ambiguous' if near else 'missing_or_outlier'
                self.status = 'predicted'; self.rejections[reason] += 1
                self.last_association['reason'] = reason
                return None
            identity, p, residual, nis = near[0]
            K = np.linalg.solve(S, P[:3, :]).T
            self.x = x+K@residual
            A = np.eye(6); A[:, :3] -= K
            self.P = A@P@A.T+K@R@K.T; self.P = .5*(self.P+self.P.T)
            self.filter_time = t; self.status = 'measured'
            self.last_association.update(accepted=True, reason='accepted', nis=nis,
                residual_m=residual.tolist(), marker_id=identity)
            self.position = p.copy(); self.seen_time = t
            return p.copy()
        P = paddle_pose(frame)
        if P is None: return None
        face = (P@self.paddle_to_face)[:3, 3]
        self.frames.append(t)
        while self.frames and self.frames[0] < t-self.seconds-.025: self.frames.popleft()
        for tr in self.tracks:
            while tr['observations'] and tr['observations'][0][0] < t-self.seconds-.025:
                tr['observations'].popleft()
        self.tracks = [tr for tr in self.tracks if tr['observations']]
        used = set()
        for identity, p in points:
            d = p-face
            if np.linalg.norm(d[:2]) > self.lateral or not self.height[0] <= d[2] <= self.height[1]: continue
            nearest = [(np.linalg.norm(p-tr['center']), k) for k, tr in enumerate(self.tracks)]
            nearest.sort()
            if nearest and nearest[0][0] < .012:
                k = nearest[0][1]
                if k in used: self.fail('Multiple raw points occupy one stationary candidate')
                tr = self.tracks[k]; tr['center'] = .9*tr['center']+.1*p
            else:
                k = len(self.tracks); tr = dict(center=p.copy(), observations=deque()); self.tracks.append(tr)
            used.add(k); tr['observations'].append((t, p.copy(), identity))
        if len(self.frames) < 30 or t-self.frames[0] < self.seconds: return None
        candidates = []
        for tr in self.tracks:
            obs = list(tr['observations']); xyz = np.array([o[1] for o in obs]); center = np.median(xyz, axis=0)
            radius = np.linalg.norm(xyz-center, axis=1); span = obs[-1][0]-obs[0][0]
            coverage = len(obs)/len(self.frames); d = center-face
            if (coverage >= .95 and span >= self.seconds-.03 and t-obs[-1][0] <= .02
                    and np.percentile(radius, 95) <= .004 and np.max(radius) <= .008):
                candidates.append(dict(position_vicon_m=center.tolist(), coverage=coverage, span_s=span,
                    scatter_p95_m=float(np.percentile(radius,95)), lateral_from_face_m=float(np.linalg.norm(d[:2])),
                    height_above_face_m=float(d[2]), samples=len(obs), id_counts=dict(Counter(str(o[2]) for o in obs)),
                    last_position=xyz[-1], last_time=obs[-1][0]))
        # First require a persistent, stationary point; among these choose the
        # nearest in XY. Tiny coverage differences must not prefer distant
        # ceiling clutter over the stable marker directly above the paddle.
        candidates.sort(key=lambda c: (c['lateral_from_face_m'], -c['coverage'],
                                       c['scatter_p95_m'], c['position_vicon_m']))
        if not candidates: return None
        best = candidates[0]; self.position = best.pop('last_position'); self.seen_time = best.pop('last_time')
        self.selection = dict(best, diameter_m=None, diameter_available=False,
            eligible_candidates=len(candidates), method='nearest_xy_among_stable_95pct_3s')
        self.x = np.r_[self.position, np.zeros(3)]
        self.P = np.diag([self.position_std**2]*3+[.03**2]*3)
        self.filter_time = self.seen_time; self.status = 'measured'
        return self.position.copy()
