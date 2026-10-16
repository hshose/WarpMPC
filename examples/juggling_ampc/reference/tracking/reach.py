"""Landing reach in robot-base axes, matching the native supervisor."""
import numpy as np


def within_landing_reach(hit, fixture, config):
    def inside(point, front, side, rear):
        x, y = point
        return (x/(front if x >= 0 else rear))**2 + (y/side)**2 <= 1.
    hit = np.asarray(hit)
    return (inside(hit, config.get('landing_forward', .35), config.get('landing_side', .35), .35)
            and (config.get('fixture_landing_forward', .12) <= 0
                 or inside(hit-np.asarray(fixture), config.get('fixture_landing_forward', .12),
                           config.get('fixture_landing_side', .12), .12)))


def nominal_return_fraction(time, first_impact, duration):
    if duration <= 0 or not np.isfinite(first_impact): return 0.
    u = np.clip((time-first_impact)/duration, 0., 1.)
    return float(u*u*u*(10.+u*(-15.+6.*u)))
