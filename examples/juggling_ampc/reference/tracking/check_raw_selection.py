"""Brief offline startup-selection regression; no controller or hardware."""
import json
import numpy as np
from .raw_ball import RawBall


def main():
    near = np.array([.03, .01, 1.2]); far = np.array([.18, .01, 1.2])
    for scenario in range(5):
        raw = RawBall(np.eye(4))
        for k in range(301):
            points = [(999, far), (1, [.5, 0, 1.2]), (2, [0, 0, .1])]
            if (scenario != 3 and (scenario != 1 or k % 25 != 12)
                    and (scenario != 2 or k % 4 != 2)):
                p = near.copy()
                if scenario == 4: p[2] += .0045 if k % 2 else -.0045
                points.append((k % 7, p))
            if k % 2: points.reverse()
            raw.update(dict(frame=k, receive_time=k*.01+.005, latency_s=.005,
                paddle=dict(position_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1]),
                unlabeled_markers=[dict(id=i, position_m=list(p)) for i, p in points]))
        assert raw.selection is not None, scenario
        expected = near if scenario < 2 else far
        assert np.allclose(raw.selection['position_vicon_m'], expected, atol=1e-10), (scenario, raw.selection)
        if scenario < 2: assert raw.selection['eligible_candidates'] == 2
    print(json.dumps(dict(passed=True, scenarios=5, robot_connected=False)))


if __name__ == '__main__':
    main()
