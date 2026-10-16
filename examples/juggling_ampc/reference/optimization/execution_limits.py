"""Opt-in execution envelopes for isolated simulation studies.

Only existing exported numeric arrays change. MPC bound arrays, symbolic factors
and all hardware/default profiles remain unchanged. Runtime helpers, including
the optional terminal-velocity regularizer, also read these arrays. The
plant's controller layer receives these arrays when it loads candidate factors.
If a separate plant factor profile is supplied, it must carry the same execution
policy; the instantaneous-output study uses candidate factors for both.
"""
import numpy as np


def apply(spec, arrays):
    """Apply ``simulation_execution_limits`` and annotate changed denominators.

    Values use SI units and joint order 1..7. Bounds are at most 90% of the
    verified Panda nominal limits (installed libfranka 0.9.2), not FR3 limits.
    An absent option is an exact no-op, including array identities and keys.
    """
    if 'simulation_execution_limits' not in spec:
        return None
    options = spec['simulation_execution_limits']
    definitions = {
        'velocity_rad_s': ('velocity_limit', np.array([2.175]*4+[2.61]*3)),
        'acceleration_rad_s2': ('acceleration_limit', np.array([15.,7.5,10.,12.5,15.,20.,20.])),
        'jerk_rad_s3': ('jerk_limit', np.array([7500.,3750.,5000.,6250.,7500.,10000.,10000.])),
    }
    if not isinstance(options, dict) or not options or set(options)-set(definitions):
        raise ValueError('Unknown or empty simulation execution limits')
    replacements, changes = {}, {}
    for name, value in options.items():
        array, nominal = definitions[name]
        limit = np.asarray(value, dtype=float)
        if (limit.shape != (7,) or not np.isfinite(limit).all()
                or np.any(limit <= 0) or np.any(limit > .9*nominal+1e-12)):
            raise ValueError('Simulation execution limit must be seven positive finite values at most 90% of Panda: '+name)
        original = np.asarray(arrays[array], dtype=float)
        if original.shape != (7,) or not np.isfinite(original).all() or np.any(original <= 0):
            raise ValueError('Invalid existing execution array: '+array)
        replacements[array] = limit.copy()
        changes[name] = dict(array=array, original=original.tolist(), value=limit.tolist(),
                             panda_nominal=nominal.tolist(), maximum_fraction_of_panda=.9)
    # Validate all requested fields before mutation and never modify the shared
    # research arrays imported from arm.model in place.
    arrays.update(replacements)
    return dict(simulation_only=True, hardware_deployment_validated=False,
                limits=changes,
                diagnostic_denominators={name:np.asarray(arrays[name]).tolist()
                    for name in ('velocity_limit','acceleration_limit','jerk_limit')},
                scope='Existing execution/projector arrays only; symbolic MPC factors and constraint-bound arrays unchanged',
                auxiliary_cost_note='An enabled native terminal-velocity regularizer also uses velocity_limit; it is disabled in the execution-limit candidates',
                plant_invariants='Mass, inertia, geometry, contact, lag, deadtime, sensing and noise unchanged',
                propagation='Applied by NativePlant loading candidate factors; a separately supplied plant factor profile must carry the same execution policy',
                diagnostic_note='Velocity/acceleration ratios use these new arrays; measured filtered torque-rate guard stays 800 Nm/s')
