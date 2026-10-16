"""Optional mixed-fidelity prediction using existing native constraint bounds.

Only legacy stages can become kinematic: their position and contact equations
already use the endpoint joint velocities. Full-ramp stages instead require
the dynamics row to integrate velocity, so freeing that row is invalid there.
This relaxation does not certify actuator feasibility outside the physical
prefix. The annotation retains the original bounds for offline assessment.
"""
import numpy as np


def mask(spec, groups, arrays, stage_offsets, constraint_ids, stage_steps, stage_times):
    """Modify numeric constraint bounds and return a JSON-ready annotation.

    The absent option is an exact no-op. Enable with
    ``prediction_fidelity={"kinematic_tail_start_s": 0.30}``.
    Dimensions, factor outputs, row IDs and runtime parameter layout stay fixed.
    """
    if 'prediction_fidelity' not in spec:
        return None
    option = spec['prediction_fidelity']
    if not isinstance(option, dict) or set(option) != {'kinematic_tail_start_s'}:
        raise ValueError('Prediction fidelity requires only kinematic_tail_start_s')
    start = float(option['kinematic_tail_start_s'])
    if not np.isfinite(start) or start < .18:
        raise ValueError('Kinematic tail must start at or after 0.18 seconds')
    model = spec.get('arm_interval_model', 'legacy')
    if model not in ('legacy', 'ramp_prefix'):
        raise ValueError('Kinematic tail requires legacy factors, not full-ramp factors')

    steps = np.asarray(stage_steps, dtype=float)
    times = np.asarray(stage_times, dtype=float)
    n = len(steps)
    if (steps.shape != (n,) or times.shape != (n + 1,)
            or not np.isfinite(steps).all() or not np.isfinite(times).all()
            or np.any(steps <= 0)
            or not np.allclose(times, np.r_[0., np.cumsum(steps)], rtol=1e-10, atol=1e-12)):
        raise ValueError('Invalid prediction fidelity stage mesh')
    first = int(np.searchsorted(times[:-1], start - 1e-12, side='left'))
    if not 5 <= first < n:
        raise ValueError('Kinematic tail must preserve five stages and affect at least one stage')
    if model == 'ramp_prefix':
        prefix = float(spec.get('prefix', n))
        if not np.isfinite(prefix) or prefix % 1 or not 0 <= prefix <= n or first < prefix:
            raise ValueError('Kinematic tail cannot mask any ramp-prefix stage')

    def integer_vector(values, name):
        result = np.asarray(values)
        if result.ndim != 1 or not np.isfinite(result).all() or np.any(result != np.floor(result)):
            raise ValueError('Invalid prediction fidelity ' + name)
        return result.astype(int)

    offsets = integer_vector(stage_offsets, 'stage offsets')
    ids = integer_vector(constraint_ids, 'constraint IDs')
    lower = np.asarray(arrays['constraint_lower'], dtype=float)
    upper = np.asarray(arrays['constraint_upper'], dtype=float)
    if (lower.shape != ids.shape or upper.shape != ids.shape
            or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper)
            or offsets.shape != (n + 1,) or offsets[0] < 0
            or offsets[-1] > len(ids) or np.any(np.diff(offsets) <= 0)):
        raise ValueError('Invalid prediction fidelity constraint layout/bounds')
    ranges = {}
    for name in ('position', 'dynamics', 'endpoint_acceleration', 'jerk'):
        if name == 'jerk' and name not in groups:
            continue
        if name not in groups:
            raise ValueError('Missing prediction fidelity constraint group: ' + name)
        interval = integer_vector(groups[name], name)
        if interval.shape != (2,) or interval[0] < 0 or interval[1] <= interval[0]:
            raise ValueError('Invalid prediction fidelity constraint group: ' + name)
        ranges[name] = tuple(interval)
    joints = ranges['position'][1] - ranges['position'][0]
    if (ranges['dynamics'][1] - ranges['dynamics'][0] != joints
            or ranges['endpoint_acceleration'][1] - ranges['endpoint_acceleration'][0] not in (joints, 2 * joints)):
        raise ValueError('Inconsistent prediction fidelity joint constraint dimensions')

    rows, stages, row_groups, boundary = [], [], [], []
    relaxed = ('dynamics', 'endpoint_acceleration', 'jerk')
    for k in range(first, n):
        for row in range(offsets[k], offsets[k + 1]):
            for name in relaxed:
                if name not in ranges:
                    continue
                begin, end = ranges[name]
                if not begin <= ids[row] < end:
                    continue
                # The boundary torque still closes the preceding physical
                # interval. Retain its left endpoint acceleration bound.
                if name == 'endpoint_acceleration' and k == first and ids[row] < begin + joints:
                    boundary.append(row)
                    break
                rows.append(row)
                stages.append(k)
                row_groups.append(name)
                break
    if not rows:
        raise ValueError('Prediction fidelity selected no existing constraint rows')
    annotation = dict(
        mode='kinematic_tail', arm_interval_model=model,
        requested_start_s=start, effective_start_s=float(times[first]),
        first_stage=first, protected_initial_stages=5,
        relaxed_groups=list(relaxed),
        disabled_rows=rows, disabled_stages=stages, disabled_groups=row_groups,
        disabled_constraint_ids=ids[rows].tolist(),
        original_lower=lower[rows].tolist(), original_upper=upper[rows].tolist(),
        preserved_boundary_endpoint_rows=boundary,
        limitation='Tail torque consistency and torque-derived acceleration/jerk feasibility are not enforced; retained torque-rate rows constrain auxiliary tail torques.',
    )
    # Commit only after all validation; absent/invalid options never partially
    # mutate the caller's bounds. Other arrays and appended native caps survive.
    lower = lower.copy()
    upper = upper.copy()
    lower[rows] = -np.inf
    upper[rows] = np.inf
    arrays['constraint_lower'] = lower
    arrays['constraint_upper'] = upper
    return annotation
