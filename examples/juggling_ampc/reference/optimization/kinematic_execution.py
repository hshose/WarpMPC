"""Offline adapter for acceleration-valued prediction tails.

Only a copy of the first four reference nodes is translated. The optimizer's
plan and warmstart retain their documented torque/acceleration slot meanings.
The ordinary reference projection and physical execution limits are unchanged.
"""
import numpy as np


def acceleration_view(robot, metadata, torque_plan):
    """Import a full torque plan into the selected offline tail control units."""
    hybrid = metadata.get('spec', {}).get('kinematic_tail')
    if not hybrid:
        return torque_plan
    converted = torque_plan.copy()
    for k in range(int(hybrid['dynamic_prefix']) + 1, len(converted)):
        node = torque_plan[k]
        if hybrid.get('endpoint_accelerations', False):
            converted[k, 23:30] = np.linalg.solve(np.asarray(robot.mass(node[6:13])),
                node[23:30]-np.asarray(robot.bias(node[6:13], node[13:20])).ravel())
        else:
            converted[k, 23:30] = ((node[13:20]-torque_plan[k-1, 13:20])
                                   / metadata['stage_steps'][k-1])
    return converted


def torque_view(robot, metadata, plan):
    """Return the legacy torque view consumed by the offline ramp projector."""
    hybrid = metadata.get('spec', {}).get('kinematic_tail')
    if not hybrid or int(hybrid['dynamic_prefix']) >= 3:
        return plan
    converted = plan.copy()
    for k in range(int(hybrid['dynamic_prefix']) + 1, min(4, len(converted))):
        node = converted[k]
        node[23:30] = np.asarray(robot.rnea(node[6:13], node[13:20], node[23:30])).ravel()
    return converted
