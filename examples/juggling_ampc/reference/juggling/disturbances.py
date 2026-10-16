"""Reproducible external ball acceleration profiles (metres/second squared)."""
import numpy as np


def wind_acceleration(t, profile):
    if profile == 'none':
        return np.zeros(3)
    if profile == 'pulse':
        return np.array([.5,-.3,0.]) if 1.2 <= t < 1.4 else np.zeros(3)
    if profile == 'smooth':
        # A gently rotating breeze with two smaller components. The analytic
        # signal has no start/stop window and smoothly changes magnitude and
        # direction throughout the trial. The controller is not told its value.
        return np.array([.12*np.cos(.8*t)+.04*np.sin(1.7*t),
                         .10*np.sin(.8*t)+.03*np.cos(1.3*t),0.])
    raise ValueError(f'Unknown wind profile: {profile}')


SMOOTH_WIND_DESCRIPTION = {
    'kind': 'unmeasured external ball acceleration, evaluated at each step midpoint',
    'units': 'm/s^2',
    'ax': '0.12*cos(0.8*t) + 0.04*sin(1.7*t)',
    'ay': '0.10*sin(0.8*t) + 0.03*cos(1.3*t)',
    'az': '0',
    'note': 'Smooth persistent breeze; acceleration disturbance, not an airspeed/aerodynamic model',
}
