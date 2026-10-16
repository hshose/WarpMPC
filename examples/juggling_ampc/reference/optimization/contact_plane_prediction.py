"""Exact optional contact-plane displacement for legacy arm prediction.

The old stage factors remain unchanged. Add these corrections before computing
costs, Gauss--Newton terms, constraints or restitution caps. A displacement along
the corrected paddle normal changes contact position and its velocity Jacobian;
the latter changes friction, ball/spin dynamics and finite-arm recoil slightly.
Nominal pose costs, workspace constraints, robot origin and the physical plant
are unaffected. Existing edge-restitution and private-bias helpers compose with
this helper without modification.
"""

import casadi as ca

from arm.ball_flight import flight
from optimization.pruning.formulation import NZ, NP, SCALE, OFFSET
from optimization.runtime_parameters import SIZE


RESIDUAL_ROWS = (51, 52)
CONSTRAINT_ROWS = (*range(9), *range(16, 23), 24, 25)


def create_contact_plane_correction(cfg, robot, runtime, plane_offset=None):
    """Return data/value functions using ``[z68,p30,tuning35,plane_offset]``.

    Data outputs are ``[dr2,dJr2x68,dc18,dJc18x68]``; value outputs are
    ``[dr2,dc18]``. ``dr`` already includes objective/contact-weight scaling;
    ``dc`` already uses the original constraint normalization. Row destinations
    are RESIDUAL_ROWS and CONSTRAINT_ROWS in the *full* original stage layout.
    Map constraint IDs when assembling pruned stage types.

    ``cfg`` and ``robot`` must use the supplied symbolic runtime vector. Native
    callers should bypass the helper at zero offset to preserve baseline floating
    point execution exactly. The runtime rotation must remain a proper rotation,
    as required by the existing factor ABI.
    """
    if runtime.shape != (SIZE, 1):
        raise ValueError('Contact plane requires the unchanged runtime vector')
    if plane_offset is None:
        plane_offset = ca.MX.sym('plane_offset')
    if plane_offset.shape != (1, 1):
        raise ValueError('Contact plane offset must be a scalar input symbol')

    z = ca.MX.sym('z', 2 * NZ)
    p = ca.MX.sym('p', NP)
    y = z[:NZ] * ca.DM(SCALE) + ca.DM(OFFSET)
    yn = z[NZ:] * ca.DM(SCALE) + ca.DM(OFFSET)
    q, dq, qn, dqn = y[6:13], y[13:20], yn[6:13], yn[13:20]
    omega, j, fraction, h = y[20:23], y[30], y[31], y[32]
    contact_time = fraction * h
    acceleration = (dqn-dq)/h
    contact_q = q+contact_time*dq+.5*contact_time**2*acceleration
    contact_dq = dq+contact_time*acceleration
    position, normal, jacobian, angular_jacobian, rotation = robot.contact_fk(contact_q)
    ball_position, ball_velocity = flight(
        y[:3], y[3:6], contact_time, cfg.drag,
        [cfg.gravity_x, cfg.gravity_y, cfg.gravity_z])
    relative_position = ball_position-position+ca.DM(robot.origin)
    tangent = relative_position-normal*ca.dot(normal, relative_position)
    J0 = jacobian-ca.skew(tangent) @ angular_jacobian
    Jdelta = J0-ca.skew(plane_offset*normal) @ angular_jacobian
    # A displacement along n leaves the tangent and normal surface speed
    # unchanged. Its tangential velocity and effective mass still change.
    mass = robot.mass(contact_q)
    responses = ca.solve(mass, ca.horzcat(J0.T, Jdelta.T))
    ball_inverse_inertia = ca.MX.eye(3)+(ca.MX.eye(3)-normal @ normal.T)/cfg.inertia_ratio
    tangents = rotation[:, :2]
    impulse_maps = []
    for J, response in ((J0, responses[:, :3]), (Jdelta, responses[:, 3:])):
        K = ball_inverse_inertia+cfg.mass*J @ response
        relative_velocity = ball_velocity-ca.cross(omega, cfg.radius*normal)-J @ contact_dq
        desired = -ca.solve(tangents.T @ K @ tangents,
                            tangents.T @ (relative_velocity+K @ normal*j))
        capacity = cfg.friction*j
        tangential_impulse = desired*capacity/ca.sqrt(capacity**2+ca.dot(desired, desired)+1e-10)
        impulse = j*normal+tangents @ tangential_impulse
        impulse_maps.append((impulse, K))
    u0, K0 = impulse_maps[0]
    udelta, Kdelta = impulse_maps[1]
    delta_impulse = udelta-u0
    delta_wrench = Jdelta.T @ udelta-J0.T @ u0
    delta_position, delta_velocity = flight(
        ca.MX.zeros(3), delta_impulse, h-contact_time, cfg.drag, [0., 0., 0.])
    delta_ball = -ca.vertcat(delta_position, delta_velocity)/ca.DM(SCALE[:6])
    delta_spin = ca.cross(normal, delta_impulse)/(cfg.inertia_ratio*cfg.radius*20.)
    delta_dynamics = cfg.mass/2.*ca.solve(robot.mass(.5*(q+qn)), delta_wrench)
    delta_constraints = ca.vertcat(delta_ball, delta_spin, delta_dynamics,
                                   -plane_offset/.5, -plane_offset/.5)
    # (1+e)*vn cancels exactly, including the independently applied edge law.
    delta_restitution = ca.dot(normal, Kdelta @ udelta-K0 @ u0)
    delta_residuals = ca.sqrt(cfg.objective_scale)*cfg.comp_weight*ca.vertcat(
        -(j/8.)*plane_offset/.05, (j/64.)*delta_restitution)
    inputs = [z, p, runtime, plane_offset]
    names = ['z', 'p', 'tuning', 'plane_offset']
    data = ca.Function('contact_plane_prediction_data', inputs,
                       [delta_residuals, ca.jacobian(delta_residuals, z),
                        delta_constraints, ca.jacobian(delta_constraints, z)],
                       names, ['residual_correction', 'residual_jacobian',
                               'constraint_correction', 'constraint_jacobian'])
    value = ca.Function('contact_plane_prediction_value', inputs,
                        [delta_residuals, delta_constraints], names,
                        ['residual_correction', 'constraint_correction'])
    return data, value


def _selfcheck():
    """Symbolic/NumPy/FD checks only; no native build or MPC/plant rollout."""
    import json
    from pathlib import Path
    import numpy as np
    from arm.model import Panda, HOME
    from optimization.pruning.formulation import Config, create_factors
    from optimization import runtime_parameters as rt
    from optimization.contact_prediction import create_edge_correction

    spec = json.loads((Path(__file__).resolve().parents[1]/'config/cpp_ball_spec.json').read_text())
    geometry = spec.get('geometry', {})
    cfg = Config(mass=.0039, restitution=.5568564815669839, drag=.2257156027819272,
                 comp_weight=400., objective_scale=.0001, contact_center_weight=16.)
    base = Panda(assembly=spec.get('assembly'), **dict(geometry, paddle_normal_rotvec_deg=[0., 0., 0.]))
    actual_robot = Panda(assembly=spec.get('assembly'), **geometry)
    tuning = ca.MX.sym('tuning', rt.SIZE)
    symbolic_cfg, symbolic_robot = rt.symbolic(cfg, tuning), rt.Robot(base, tuning)
    values = rt.values(cfg, geometry)
    data, value = create_contact_plane_correction(symbolic_cfg, symbolic_robot, tuning)
    _, edge_value = create_edge_correction(symbolic_cfg, symbolic_robot, tuning)
    state = ca.MX.sym('state', 2*NZ)
    parameter = ca.MX.sym('parameter', NP)
    support = {}
    for label, dropped, dedup in [('full', (), False),
                                  ('reduced', cfg.pruning_groups, True),
                                  ('tail', cfg.pruning_groups, False)]:
        _, stage, _, _, _, _, groups = create_factors(
            symbolic_cfg, symbolic_robot, dropped, dedup, '_'+label, runtime=tuning)
        residuals, constraints = stage(state[:NZ], state[NZ:], parameter, tuning)
        jr = ca.jacobian(residuals, state).sparsity()
        jc = ca.jacobian(constraints, state).sparsity()
        original_r = set(zip(*jr.get_triplet()))
        original_c = set(zip(*jc.get_triplet()))
        full_groups = {'ball': (0, 6), 'spin': (6, 9), 'dynamics': (16, 23), 'contact': (24, 28)}
        row_map = {full_groups[name][0]+i: bounds[0]+i
                   for name, bounds in groups.items() if name in full_groups
                   for i in range(bounds[1]-bounds[0])}
        added_r = {(RESIDUAL_ROWS[r], c) for r, c in zip(*data.sparsity_out(1).get_triplet())}
        added_c = {(row_map[CONSTRAINT_ROWS[r]], c)
                   for r, c in zip(*data.sparsity_out(3).get_triplet())}
        assert not added_r-original_r, (label, 'residual sparsity', added_r-original_r)
        assert not added_c-original_c, (label, 'constraint sparsity', added_c-original_c)
        support[label] = dict(residual_nonzeros=len(added_r), constraint_nonzeros=len(added_c),
                              contained_in_original=True)

    def skew(v):
        x, y, z = v
        return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

    gravity = np.array([cfg.gravity_x, cfg.gravity_y, cfg.gravity_z])
    def numeric_flight(b, v, dt):
        decay = np.exp(-cfg.drag*dt)
        L = -np.expm1(-cfg.drag*dt)/cfg.drag
        return b+L*v+(dt-L)/cfg.drag*gravity, decay*v+L*gravity

    def independent_map(flat, delta, edge_loss=0.):
        y = flat[:NZ]*SCALE+OFFSET
        yn = flat[NZ:]*SCALE+OFFSET
        q, dq, qn, dqn = y[6:13], y[13:20], yn[6:13], yn[13:20]
        j, h, tc = y[30], y[32], y[31]*y[32]
        acceleration = (dqn-dq)/h
        qc, vc = q+tc*dq+.5*tc*tc*acceleration, dq+tc*acceleration
        pc, n, J, W, R = [np.asarray(x) for x in actual_robot.contact_fk(qc)]
        pc, n = pc.ravel(), n.ravel()
        # Independently shift the actual point and differentiate its offset;
        # do not assume tangent/normal-speed invariance in this reference.
        pc = pc-actual_robot.origin+delta*n
        J = J-skew(delta*n) @ W
        bc, bv = numeric_flight(y[:3], y[3:6], tc)
        relative_position = bc-pc
        gap = n @ relative_position-cfg.radius
        tangent = relative_position-n*(n @ relative_position)
        J = J-skew(tangent) @ W
        relative = bv-np.cross(y[20:23], cfg.radius*n)-J @ vc
        M = np.asarray(actual_robot.mass(qc))
        K = np.eye(3)+(np.eye(3)-np.outer(n, n))/cfg.inertia_ratio+cfg.mass*J @ np.linalg.solve(M, J.T)
        T = R[:, :2]
        desired = -np.linalg.solve(T.T @ K @ T, T.T @ (relative+K @ n*j))
        capacity = cfg.friction*j
        u = j*n+T @ (desired*capacity/np.sqrt(capacity**2+desired @ desired+1e-10))
        next_b, next_v = numeric_flight(bc, bv+u, h-tc)
        next_spin = y[20:23]-np.cross(n, u)/(cfg.inertia_ratio*cfg.radius)
        qm, vm = .5*(q+qn), .5*(dq+dqn)
        torque = np.asarray(actual_robot.rnea(qm, vm, acceleration)).ravel()+cfg.mass/h*J.T @ u
        dynamics = h/2.*np.linalg.solve(np.asarray(actual_robot.mass(qm)), torque-.5*(y[23:30]+yn[23:30]))
        endp, endn, *_ = [np.asarray(x) for x in actual_robot.contact_fk(qn)]
        endp = endp.ravel()-actual_robot.origin+delta*endn.ravel()
        endgap = endn.ravel() @ (yn[:3]-endp)-cfg.radius
        c = np.r_[(yn[:3]-next_b)/SCALE[:3], (yn[3:6]-next_v)/SCALE[3:6],
                  (yn[20:23]-next_spin)/20., dynamics, gap/.5, endgap/.5]
        effective_restitution = np.clip(cfg.restitution-edge_loss*(np.linalg.norm(tangent)/.075)**4, 0., 1.5)
        rest = n @ (K @ u)+(1+effective_restitution)*(n @ relative)
        residual = np.sqrt(cfg.objective_scale)*cfg.comp_weight*np.array([j/8*gap/.05, j/64*rest])
        return np.r_[residual, c]

    rng = np.random.default_rng(20261007)
    maxima = dict(independent_value=0., edge_composition=0., finite_difference=0., zero_offset=0.,
                  zero_impulse_dynamic_correction=0.)
    cases = 12
    for i in range(cases):
        y, yn = OFFSET.copy(), OFFSET.copy()
        y[6:13] = HOME+rng.normal(0., .05, 7)
        y[13:20] = rng.normal(0., .2, 7)
        y[:3] = rng.normal(0., .03, 3)
        y[3:6] = np.r_[rng.normal(0., .3, 2), -3.]
        y[20:23] = rng.normal(0., 8., 3)
        y[23:30] = np.asarray(actual_robot.bias(y[6:13], y[13:20])).ravel()
        y[30:34] = [0. if i % 4 == 0 else 5.+rng.random()*3., [.05, .5, .95][i % 3], .03 if i < 6 else .09, .2]
        yn[6:13] = y[6:13]+y[32]*y[13:20]
        yn[13:20] = y[13:20]+rng.normal(0., .02, 7)
        yn[:6] = y[:6]+rng.normal(0., .02, 6)
        yn[20:23] = y[20:23]+rng.normal(0., 1., 3)
        yn[23:30] = y[23:30]+rng.normal(0., .1, 7)
        yn[30:34] = [0., .5, y[32], y[33]+y[32]]
        flat = np.r_[(y-OFFSET)/SCALE, (yn-OFFSET)/SCALE]
        delta = [-.0036831441747697535, .002, 0.][i % 3]
        args = [flat, np.zeros(NP), values, delta]
        dr, jr, dc, jc = [np.asarray(x) for x in data(*args)]
        result = np.r_[dr.ravel(), dc.ravel()]
        jacobian = np.vstack([jr, jc])
        expected = independent_map(flat, delta)-independent_map(flat, 0.)
        maxima['independent_value'] = max(maxima['independent_value'], float(np.max(abs(result-expected))))
        composed = independent_map(flat, 0.)+result
        composed[1] += float(edge_value(flat, np.zeros(NP), values, .17840861146757123))
        expected_edge = independent_map(flat, delta, .17840861146757123)
        maxima['edge_composition'] = max(maxima['edge_composition'], float(np.max(abs(composed-expected_edge))))
        eps = 2e-6
        fd = np.zeros_like(jacobian)
        for k in range(2*NZ):
            plus, minus = flat.copy(), flat.copy()
            plus[k] += eps
            minus[k] -= eps
            a = np.concatenate([np.asarray(x).ravel() for x in value(plus, np.zeros(NP), values, delta)])
            b = np.concatenate([np.asarray(x).ravel() for x in value(minus, np.zeros(NP), values, delta)])
            fd[:, k] = (a-b)/(2*eps)
        maxima['finite_difference'] = max(maxima['finite_difference'], float(np.max(abs(jacobian-fd))))
        if delta == 0.:
            maxima['zero_offset'] = max(maxima['zero_offset'], float(np.max(abs(result))))
        if y[30] == 0.:
            maxima['zero_impulse_dynamic_correction'] = max(
                maxima['zero_impulse_dynamic_correction'], float(np.max(abs(result[:18]))))
    assert maxima['independent_value'] < 1e-10, maxima
    assert maxima['edge_composition'] < 1e-10, maxima
    assert maxima['finite_difference'] < 2e-7, maxima
    assert maxima['zero_offset'] < 1e-14, maxima
    assert maxima['zero_impulse_dynamic_correction'] < 1e-14, maxima
    return dict(cases=cases, maxima=maxima, sparsity=support,
                residual_rows=list(RESIDUAL_ROWS), constraint_rows=list(CONSTRAINT_ROWS))


if __name__ == '__main__':
    import json
    print(json.dumps(_selfcheck(), indent=2))
