"""Optional numeric limits for isolated prediction experiments.

The generated Cartesian paddle-velocity expression is J(q)dq/1.2. Changing
its bounds reuses the exact factors and leaves joint/torque limits intact.
"""
import numpy as np


def apply(spec, groups, arrays, constraint_ids):
    if 'prediction_limits' not in spec:
        return None
    options = spec['prediction_limits']
    definitions = {'paddle_velocity_m_s': ('paddle_velocity', np.full(3, 1.2)),
                   'torque_rate_nm_s': ('torque_rate', np.asarray(arrays['torque_rate_limit']))}
    if not isinstance(options, dict) or not options or set(options) - set(definitions):
        raise ValueError('Unknown or empty numeric prediction limits')
    ids = np.asarray(constraint_ids)
    lower = np.asarray(arrays['constraint_lower'], dtype=float).copy()
    upper = np.asarray(arrays['constraint_upper'], dtype=float).copy()
    annotation = {}
    for name, value in options.items():
        group, normalization = definitions[name]
        limit = np.asarray(value, dtype=float)
        if limit.shape != normalization.shape or not np.isfinite(limit).all() or np.any(limit <= 0):
            raise ValueError('Prediction limit needs positive finite values: ' + name)
        begin, end = groups[group]
        if end - begin != len(limit):
            raise ValueError('Unexpected prediction-limit factor layout: ' + group)
        rows = np.flatnonzero((ids >= begin) & (ids < end))
        if not len(rows):
            raise ValueError('No retained prediction-limit constraints: ' + group)
        annotation[name] = dict(value=limit.tolist(), rows=rows.tolist(),
            factor_normalization=normalization.tolist(),
            original_lower=lower[rows].tolist(), original_upper=upper[rows].tolist())
        scaled = (limit / normalization)[ids[rows] - begin]
        lower[rows], upper[rows] = -scaled, scaled
    arrays['constraint_lower'], arrays['constraint_upper'] = lower, upper
    return dict(limits=annotation,
                scope='Existing retained prediction rows only; execution guards and physical model unchanged')
