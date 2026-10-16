"""Small symbolic checks for the optional tail; no compilation or plant rollout."""

import argparse
import json
from pathlib import Path

import casadi as ca
import numpy as np

from arm.model import Panda, HOME, ACCEL_LIMIT
from optimization import runtime_parameters as rt
from optimization.pruning.formulation import Config, SCALE, OFFSET
from optimization.kinematic_tail import (
    create_kinematic_tail_factors, acceleration_slot_bounds, integration_metadata)
from optimization.contact_prediction import create_edge_correction
from optimization.contact_bias_prediction import create_contact_bias_correction
from optimization.simplified_dynamics import create_contact_plane_correction


def check(spec):
    cfg = Config(**spec['config'])
    base = Panda(assembly=spec.get('assembly'),
                 **dict(spec.get('geometry', {}), paddle_normal_rotvec_deg=[0, 0, 0]))
    tuning = ca.MX.sym('tuning', rt.SIZE)
    symbolic = rt.symbolic(cfg, tuning)
    robot = rt.Robot(base, tuning)
    values = rt.values(cfg, spec.get('geometry', {}))

    class Probe:
        def __init__(self, boundary):
            self.boundary, self.calls = boundary, []

        def __getattr__(self, name):
            if name in ('mass', 'bias', 'rnea'):
                self.calls.append(name)
                if name != 'rnea' or not self.boundary:
                    raise AssertionError('Forbidden rigid-body call: ' + name)
            return getattr(robot, name)

    def support(sparsity):
        return set(zip(*sparsity.get_triplet()))

    checks = {}
    rng = np.random.default_rng(4705)
    for boundary in (False, True):
        probe = Probe(boundary)
        _, stage, _, _, _, _, groups = create_kinematic_tail_factors(
            symbolic, probe, runtime=tuning, boundary=boundary)
        z, p = ca.MX.sym('z', 68), ca.MX.sym('p', 30)
        r, c = stage(z[:34], z[34:], p, tuning)
        jr, jc = ca.jacobian(r, z), ca.jacobian(c, z)
        data = ca.Function('tail_check_data', [z, p, tuning], [r, c, jr, jc])
        errors = []
        for sample in range(3):
            h = .03 + .02 * sample
            y = np.zeros((2, 34))
            y[0, 6:13] = HOME + rng.uniform(-.03, .03, 7)
            y[0, 13:20] = rng.uniform(-.06, .06, 7)
            acceleration = rng.uniform(-.08, .08, 7) * ACCEL_LIMIT
            y[1, 6:13] = y[0, 6:13] + h*y[0, 13:20] + .5*h*h*acceleration
            y[1, 13:20] = y[0, 13:20] + h*acceleration
            y[:, :3], y[:, 3:6], y[:, 20:23] = [.19, .02, .6], [.06, -.025, -1.2], [.3, -.4, 1.2]
            y[0, 23:30] = (np.asarray(base.rnea(y[0, 6:13], y[0, 13:20], acceleration)).ravel()
                           if boundary else acceleration + .02)
            y[1, 23:30] = acceleration
            y[:, 30], y[:, 31], y[:, 32], y[1, 33] = 3., .35, h, h
            point = ((y - OFFSET) / SCALE).ravel()
            _, cv, rvj, cvj = map(np.asarray, data(point, np.zeros(30), values))
            for group in ['position', 'dynamics', 'clock'] + (['torque_rate'] if boundary else []):
                assert np.max(abs(cv.ravel()[slice(*groups[group])])) < 1e-12, group
            direction = rng.normal(size=68)
            direction /= np.linalg.norm(direction)
            eps = 1e-6
            plus = data(point + eps*direction, np.zeros(30), values)
            minus = data(point - eps*direction, np.zeros(30), values)
            errors.append(max(np.max(abs((np.asarray(plus[0]) - np.asarray(minus[0]))[:, 0]/(2*eps) - rvj@direction)),
                              np.max(abs((np.asarray(plus[1]) - np.asarray(minus[1]))[:, 0]/(2*eps) - cvj@direction))))
        assert max(errors) < 1e-4, errors
        assert probe.calls == (['rnea'] if boundary else []), probe.calls

        # Native additive helpers must not require missing sparse factor slots.
        r_support, c_support = support(jr.sparsity()), support(jc.sparsity())
        edge = create_edge_correction(symbolic, robot, tuning)[0]
        bias = create_contact_bias_correction(symbolic, robot, tuning)[0]
        plane = create_contact_plane_correction(symbolic, robot, tuning,
                                                neglect_ball_reaction=True)[0]
        assert all((52, col) in r_support for _, col in support(edge.sparsity_out(1)))
        assert support(bias.sparsity_out(1)) <= c_support
        rmap, cmap = [51, 52], [*range(9), *range(16, 23), 24, 25]
        assert all((rmap[row], col) in r_support for row, col in support(plane.sparsity_out(1)))
        assert all((cmap[row], col) in c_support for row, col in support(plane.sparsity_out(3)))
        _, reduced, _, _, _, _, _ = create_kinematic_tail_factors(
            symbolic, Probe(boundary), spec.get('drop', ()), True,
            runtime=tuning, boundary=boundary)
        checks['boundary' if boundary else 'pure'] = dict(
            residuals=int(r.numel()), constraints=int(c.numel()),
            reduced_constraints=reduced.size1_out(1), rigid_body_requests=probe.calls,
            max_directional_fd_error=float(max(errors)), helper_support_subset=True)
    lower, upper = acceleration_slot_bounds(cfg)
    np.testing.assert_allclose(upper*SCALE[23:30], cfg.actuator_margin*ACCEL_LIMIT)
    np.testing.assert_allclose(lower*SCALE[23:30], -cfg.actuator_margin*ACCEL_LIMIT)
    return dict(checks=checks, metadata=integration_metadata(), compiled=False,
                controller_solves=0, physical_plant_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'experiments/simplification_20261007/baseline_spec.json')
    args = parser.parse_args()
    print(json.dumps(check(json.loads(args.spec.read_text())), indent=2))
