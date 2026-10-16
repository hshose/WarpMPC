"""Continuous ball flight and frictionless impacts on an accelerated disk.

State units are metres, metres/second, radians, and radians/second.  The paddle
coordinates are world x/y/z followed by roll/pitch (R_y(pitch) R_x(roll)).
Inputs are the corresponding five generalized accelerations.  This simulator
contains no controller and resolves contact independently of MPC stage knots.
"""

from dataclasses import dataclass

import numpy as np
from .specs import BALL_MASS_KG, BALL_RADIUS_M, PADDLE_RADIUS_M


@dataclass(frozen=True)
class SimConfig:
    """Physical parameters and continuous event-search tolerances (SI units)."""

    gravity: float = 9.81
    ball_radius: float = BALL_RADIUS_M
    ball_mass: float = BALL_MASS_KG
    paddle_radius: float = PADDLE_RADIUS_M
    restitution: float = 0.88
    drag: float = 0.0  # Linear flight drag rate [1/s], acceleration = -drag * v.
    max_event_interval: float = 0.002  # Gap sampling interval [s].
    max_angle_interval: float = 0.01  # Maximum sampled angular sweep [rad].
    event_time_tolerance: float = 1e-11  # Root bracket width [s].
    separation_tolerance: float = 1e-10  # Numerical post-impact separation [m].
    impact_speed_tolerance: float = 1e-9  # Closing relative normal speed [m/s].
    max_impacts_per_step: int = 16

    def __post_init__(self):
        positive = (
            self.gravity, self.ball_radius, self.ball_mass, self.paddle_radius,
            self.max_event_interval, self.max_angle_interval,
            self.event_time_tolerance, self.separation_tolerance,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Physical sizes and event tolerances must be finite and positive")
        if not np.isfinite(self.drag) or self.drag < 0:
            raise ValueError("drag must be a nonnegative rate [1/s]")
        if not 0 <= self.restitution <= 1:
            raise ValueError("restitution must be between zero and one")
        if self.max_impacts_per_step < 1:
            raise ValueError("max_impacts_per_step must be positive")


def surface_normal(pose):
    """Unit normal of a paddle pose (..., 5), pointing out of its upper face."""
    pose = np.asarray(pose)
    roll, pitch = pose[..., 3], pose[..., 4]
    return np.stack(
        (np.sin(pitch) * np.cos(roll), -np.sin(roll), np.cos(pitch) * np.cos(roll)),
        axis=-1,
    )


def surface_normal_rate(pose, velocity):
    """Time derivative of the normal, including both generalized angle rates."""
    roll, pitch = pose[3:5]
    roll_rate, pitch_rate = velocity[3:5]
    return roll_rate * np.array(
        [-np.sin(pitch) * np.sin(roll), -np.cos(roll), -np.cos(pitch) * np.sin(roll)]
    ) + pitch_rate * np.array(
        [np.cos(pitch) * np.cos(roll), 0.0, -np.sin(pitch) * np.cos(roll)]
    )


def _free_flight(x, u, time, cfg, wind):
    """Exact free flight for constant acceleration and optional linear drag."""
    t = np.asarray(time)[..., None]
    out = np.empty(t.shape[:-1] + (16,))
    acceleration = np.asarray(wind, dtype=float) + np.array([0.0, 0.0, -cfg.gravity])
    if cfg.drag < 1e-8:
        out[..., :3] = x[:3] + t * x[3:6] + 0.5 * t * t * acceleration
        out[..., 3:6] = x[3:6] + t * acceleration
    else:
        decay = np.exp(-cfg.drag * t)
        phi1 = -np.expm1(-cfg.drag * t) / cfg.drag
        phi2 = (t - phi1) / cfg.drag
        out[..., :3] = x[:3] + phi1 * x[3:6] + phi2 * acceleration
        out[..., 3:6] = decay * x[3:6] + phi1 * acceleration
    out[..., 6:11] = x[6:11] + t * x[11:16] + 0.5 * t * t * u
    out[..., 11:16] = x[11:16] + t * u
    return out


def _gap(x, cfg):
    return np.sum(surface_normal(x[..., 6:11]) * (x[..., :3] - x[..., 6:9]), axis=-1) - cfg.ball_radius


def _contact_geometry(x):
    normal = surface_normal(x[6:11])
    offset = x[:3] - x[6:9]
    tangent = offset - np.dot(normal, offset) * normal
    normal_rate = surface_normal_rate(x[6:11], x[11:16])
    # The normal velocity of the material surface at this offset is
    # n . paddle_translation_velocity - n_dot . offset.
    relative_normal_speed = np.dot(normal, x[3:6] - x[11:14]) + np.dot(normal_rate, offset)
    return normal, np.linalg.norm(tangent), relative_normal_speed


def _first_contact(x, u, duration, cfg, wind):
    """Bracket a closing plane crossing, rejecting intersections off the disk."""
    angle_rate_bound = np.linalg.norm(x[14:16]) + duration * np.linalg.norm(u[3:5])
    count = max(
        1,
        int(np.ceil(duration / cfg.max_event_interval)),
        int(np.ceil(duration * angle_rate_bound / cfg.max_angle_interval)),
    )
    times = np.linspace(0.0, duration, count + 1)
    states = _free_flight(x, u, times, cfg, wind)
    gaps = _gap(states, cfg)
    candidates = np.flatnonzero((gaps[:-1] >= 0.0) & (gaps[1:] <= 0.0))
    # A ball initialized on the face while closing impacts immediately. A ball
    # already below the plane is allowed to fall; we do not teleport it back.
    if abs(gaps[0]) <= cfg.separation_tolerance:
        normal, radius, speed = _contact_geometry(x)
        if radius <= cfg.paddle_radius and speed < -cfg.impact_speed_tolerance:
            return 0.0, x.copy(), normal, radius, speed
    for index in candidates:
        lo, hi = times[index:index + 2]
        while hi - lo > cfg.event_time_tolerance:
            mid = 0.5 * (lo + hi)
            if _gap(_free_flight(x, u, mid, cfg, wind), cfg) > 0:
                lo = mid
            else:
                hi = mid
        time = 0.5 * (lo + hi)
        state = _free_flight(x, u, time, cfg, wind)
        normal, radius, speed = _contact_geometry(state)
        if radius <= cfg.paddle_radius and speed < -cfg.impact_speed_tolerance:
            return time, state, normal, radius, speed
    return None


def step(x, u, dt, cfg=None, wind=None):
    """Advance a 16-state system and return ``(state, contact_records)``.

    ``u`` is a five-vector [m/s², m/s², m/s², rad/s², rad/s²]. ``wind``
    is an external ball acceleration [m/s²], constant over the step. The
    impulse is frictionless against an infinitely actuated disk; it changes
    ball velocity but not the prescribed paddle trajectory. Each record's
    ``time`` is seconds after the beginning of this call, ``position`` is the
    surface contact point [m], and normal speeds are relative to the moving
    surface [m/s]. The incoming speed is negative and outgoing speed positive.
    ``impulse_Ns`` is the scalar normal impulse on the ball [N s].

    The disk has zero thickness and only its upper face produces impacts.
    Rim and underside collisions and persistent resting contact are omitted.
    Flight below a missed paddle remains unconstrained.
    """
    cfg = SimConfig() if cfg is None else cfg
    current = np.asarray(x, dtype=float).copy()
    u = np.asarray(u, dtype=float)
    wind = np.zeros(3) if wind is None else np.asarray(wind, dtype=float)
    if current.shape != (16,) or u.shape != (5,) or wind.shape != (3,):
        raise ValueError("Expected x.shape=(16,), u.shape=(5,), wind.shape=(3,)")
    if not all(np.all(np.isfinite(v)) for v in (current, u, wind)):
        raise ValueError("State, control, and wind must be finite")
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    contacts = []
    elapsed = 0.0
    while elapsed < dt:
        remaining = dt - elapsed
        event = _first_contact(current, u, remaining, cfg, wind)
        if event is None:
            return _free_flight(current, u, remaining, cfg, wind), contacts
        if len(contacts) >= cfg.max_impacts_per_step:
            raise RuntimeError("Impact count exceeded; this flight/impact model does not support resting contact")
        event_time, current, normal, radius, incoming = event
        elapsed += event_time
        impulse = -(1.0 + cfg.restitution) * cfg.ball_mass * incoming
        contact_position = current[:3] - cfg.ball_radius * normal
        current[3:6] += (impulse / cfg.ball_mass) * normal
        # Remove only root-solver roundoff, then leave a microscopic positive
        # gap so the departing ball does not generate a duplicate t=0 event.
        current[:3] += (cfg.separation_tolerance - _gap(current, cfg)) * normal
        _, _, outgoing = _contact_geometry(current)
        contacts.append({
            "time": float(elapsed),
            "position": contact_position.tolist(),
            "normal": normal.tolist(),
            "incoming_normal_speed": float(incoming),
            "outgoing_normal_speed": float(outgoing),
            "impulse_Ns": float(impulse),
            "radial_distance": float(radius),
            "ball_velocity_after": current[3:6].tolist(),
        })
    return current, contacts
