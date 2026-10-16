"""Explicit uncertainty groups, including compatibility with legacy reports."""
import math


def settings(multiplier=1.5,arm_multiplier=None):
    arm=multiplier if arm_multiplier is None else arm_multiplier
    if not all(math.isfinite(v) and v>0 for v in (multiplier,arm)):
        raise ValueError('Uncertainty multipliers must be finite and positive')
    return dict(ball_flight=float(multiplier),ball_contact=float(multiplier),sensing=float(multiplier),
                plan_delivery=float(multiplier),arm_following=float(arm))


def from_report(report):
    result=report.get('uncertainty_multipliers')
    return result if result is not None else settings(report['noise_disturbance_multiplier'])


def require_matching(a,b):
    if from_report(a)!=from_report(b):
        raise ValueError('Uncertainty group multipliers differ')
