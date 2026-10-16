"""Focused algebra/geometry/warmstart checks, without closed-loop experiments.

Run with ``.venv/bin/python -m juggling.check_formulation``. These checks do
not establish closed-loop success; they isolate expensive-to-debug model and
transcription errors. Cross-knot warmstart defects are reported, not asserted
zero, because multiple shooting intentionally permits an infeasible seed.
"""

import json

import numpy as np

from .formulation import Config, NX, NZ, SCALE, create_factors
from .simulator import SimConfig, step, surface_normal
from .solver import RTI, csc


G = np.array([0.0, 0.0, -9.81])


def advance(row, duration):
    """Evaluate the model's one-impulse arc in physical coordinates."""
    out = row[:NX].copy()
    tau = row[22] * row[23]
    pose_contact = row[6:11] + tau * row[11:16] + 0.5 * tau**2 * row[16:21]
    out[:3] += duration * row[3:6] + 0.5 * duration**2 * G
    out[3:6] += duration * G
    if duration >= tau:
        jump = row[21] * surface_normal(pose_contact)
        out[:3] += (duration - tau) * jump
        out[3:6] += jump
    out[6:11] += duration * row[11:16] + 0.5 * duration**2 * row[16:21]
    out[11:16] += duration * row[16:21]
    return out


def add_clock_if_present(Y):
    # The current layout leaves the optional adjacent-node clock after h.
    if NZ > 24:
        Y[:, 24] = np.r_[0.0, np.cumsum(Y[:-1, 23])]


def dynamic_defects(solver, Y):
    values = []
    for k in range(solver.cfg.horizon):
        _, c = solver.problem['stage'](Y[k] / SCALE, Y[k + 1] / SCALE, Y[0, :NX])
        values.append(np.asarray(c).ravel()[:NX] * SCALE[:NX])
    return np.asarray(values)


def derivatives_and_locality():
    solver = RTI(Config(horizon=4, sqp_iters=1))
    x = np.zeros(NX)
    x[2] = 0.67
    solver.initialize(x)
    z = solver.z.copy()
    H, gradient, A, constraints, _ = solver.data(z, x)
    nfirst = solver.problem['first'].size1_out(0)
    nstage = solver.problem['ncon_stage']
    rows, cols = A.sparsity().get_triplet()
    invalid = 0
    for row, col in zip(rows, cols):
        if row < nfirst:
            allowed = {0}
        elif row < nfirst + solver.cfg.horizon * nstage:
            k = (row - nfirst) // nstage
            allowed = {k, k + 1}
        else:
            allowed = {solver.cfg.horizon}
        invalid += col // NZ not in allowed
    hr, hc = H.sparsity().get_triplet()
    bandwidth = max(abs(r // NZ - c // NZ) for r, c in zip(hr, hc))
    assert invalid == 0 and bandwidth <= 1, 'Non-adjacent graph dependency'
    direction = np.random.default_rng(17).normal(size=z.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    op, cp = solver.value(z + epsilon * direction, x)
    om, cm = solver.value(z - epsilon * direction, x)
    dc_fd = (np.asarray(cp).ravel() - np.asarray(cm).ravel()) / (2 * epsilon)
    dc_ad = csc(A) @ direction
    jac_error = float(np.max(np.abs(dc_fd - dc_ad)))
    cost_error = abs(float((op - om) / (2 * epsilon)) - np.asarray(gradient).ravel() @ direction)
    H2, _, A2, _, _ = solver.data(z + 0.01 * direction, x)
    assert jac_error < 1e-6 and cost_error < 1e-5, 'Directional derivative disagreement'
    assert H.sparsity() == H2.sparsity() and A.sparsity() == A2.sparsity()
    return dict(jacobian_error=jac_error, objective_gradient_error=cost_error,
                invalid_stage_nonzeros=int(invalid), hessian_stage_bandwidth=bandwidth)


def rotating_impact():
    cfg = Config(horizon=1)
    _, stage, _, diagnostics, _, _ = create_factors(cfg)
    row = np.zeros(NZ)
    row[23], row[22] = 0.03, 0.4
    row[6:11] = [0.02, -0.01, 0.005, 0.18, -0.22]
    row[11:16] = [0.1, -0.2, 0.05, 0.7, -0.4]
    row[16:21] = [0.2, 0.1, 0.3, -0.2, 0.15]
    row[3:6] = [0.03, -0.02, -1.2]
    tau = row[22] * row[23]
    pc = row[6:11] + tau * row[11:16] + 0.5 * tau**2 * row[16:21]
    n = surface_normal(pc)
    offset = np.array([0.025, 0.01, 0.0])
    offset -= n * np.dot(n, offset)
    contact_center = pc[:3] + cfg.radius * n + offset
    row[:3] = contact_center - tau * row[3:6] - 0.5 * tau**2 * G
    nu = float(diagnostics(row / SCALE)[3])
    row[21] = -(1 + cfg.restitution) * nu
    following = row.copy()
    following[:NX] = advance(row, row[23])
    if NZ > 24:
        following[24] = row[23]
    _, constraints = stage(row / SCALE, following / SCALE, row[:NX])
    defect = float(np.max(np.abs(np.asarray(constraints).ravel()[:NX])))
    actual, contacts = step(row[:NX], row[16:21], row[23], SimConfig())
    difference = float(np.max(np.abs(actual - following[:NX])))
    assert len(contacts) == 1, 'Expected one finite-disk contact'
    assert abs(contacts[0]['time'] - tau) < 1e-8
    assert defect < 1e-12 and difference < 1e-8, 'Rotating impulse disagreement'
    return dict(stage_defect=defect, simulator_state_error=difference,
                contact_time_error=abs(contacts[0]['time'] - tau))


def boundary_shift(event_time):
    cfg = Config(horizon=3, h_nom=0.03, sqp_iters=1)
    solver = RTI(cfg)
    Y = np.zeros((cfg.horizon + 1, NZ))
    Y[:, 23], Y[:, 22] = cfg.h_nom, 0.5
    Y[0, 2] = cfg.radius + event_time + 0.5 * 9.81 * event_time**2
    Y[0, 5] = -1.0
    event_interval = int(event_time // cfg.h_nom)
    Y[event_interval, 21] = (1 + cfg.restitution) * (1 + 9.81 * event_time)
    Y[event_interval, 22] = (event_time - event_interval * cfg.h_nom) / cfg.h_nom
    for k in range(cfg.horizon):
        Y[k + 1, :NX] = advance(Y[k], cfg.h_nom)
    add_clock_if_present(Y)
    x = advance(Y[0], cfg.h_min)
    solver.z = (Y / SCALE).ravel()
    solver.shift(cfg.h_min, x)
    shifted = solver.z.reshape(-1, NZ) * SCALE
    defects = dynamic_defects(solver, shifted)
    error = float(np.max(np.abs(defects)))
    assert error < 1e-8, f'Boundary shift duplicates/loses an impulse: {error:g}'
    return dict(event_time=event_time, max_physical_dynamics_defect=error)


def cross_knot_shift_defect():
    cfg = Config(horizon=3, h_nom=0.03, sqp_iters=1)
    solver = RTI(cfg)
    Y = np.zeros((cfg.horizon + 1, NZ))
    Y[:, 23], Y[:, 22] = cfg.h_nom, 0.5
    Y[:, 16] = [10.0, -10.0, 10.0, 10.0]
    Y[0, 2] = 0.8
    for k in range(cfg.horizon):
        Y[k + 1, :NX] = advance(Y[k], cfg.h_nom)
    add_clock_if_present(Y)
    x = advance(Y[0], cfg.h_min)
    solver.z = (Y / SCALE).ravel()
    solver.shift(cfg.h_min, x)
    shifted = solver.z.reshape(-1, NZ) * SCALE
    defects = dynamic_defects(solver, shifted)
    return dict(first_paddle_position_defect_m=float(defects[0, 6]),
                first_paddle_velocity_defect_m_s=float(defects[0, 11]),
                interpretation='Expected infeasible seed when a new interval crosses an old control knot')


def clock_and_dual_offsets():
    cfg = Config(horizon=6, h_nom=0.03, variable_time=True, sqp_iters=1)
    solver = RTI(cfg)
    x = np.zeros(NX)
    x[2] = 0.8
    Y = solver.initialize(x)
    Y[:-1, 23] = [0.01, 0.02, 0.05, 0.03, 0.04, 0.05]
    Y[:, 21] = 0.0
    for k in range(cfg.horizon):
        Y[k + 1, :NX] = advance(Y[k], Y[k, 23])
    add_clock_if_present(Y)
    solver.z = (Y / SCALE).ravel()
    nf = solver.problem['ncon_first']
    ns = solver.problem['ncon_stage']
    nl = solver.problem['ncon_last']
    nc = nf + cfg.horizon * ns + nl
    solver.dual = np.arange(nc + solver.n, dtype=float) + 1.0
    solver.shift(cfg.h_min, advance(Y[0], cfg.h_min))
    shifted = solver.z.reshape(-1, NZ) * SCALE
    error = float(np.max(np.abs(np.diff(shifted[:, 24]) - shifted[:-1, 23])))
    terminal = nf + cfg.horizon * ns
    assert error < 1e-12 and shifted[0, 24] == 0.0
    assert np.all(solver.dual[:nf] == 0.0)
    assert np.all(solver.dual[terminal:terminal + nl] == 0.0)
    assert np.all(solver.dual[nc:nc + NX] == 0.0)
    _, constraints = solver.value(solver.z, shifted[0, :NX])
    constraints = np.asarray(constraints).ravel()
    assert solver.problem['cl'][terminal] <= constraints[terminal] <= solver.problem['ch'][terminal]
    return dict(clock_continuity_error=error, first_clock_s=float(shifted[0, 24]),
                horizon_s=float(shifted[-1, 24]),
                minimum_horizon_s=float(solver.problem['cl'][terminal]),
                first_terminal_and_measurement_bound_duals_reset=True)


def main():
    checks = {
        'derivatives_and_locality': derivatives_and_locality,
        'rotating_impact': rotating_impact,
        'initial_boundary_shift': lambda: boundary_shift(0.01),
        'interior_boundary_shift': lambda: boundary_shift(0.04),
        'just_before_boundary_shift': lambda: boundary_shift(0.04 - 5e-11),
        'just_after_boundary_shift': lambda: boundary_shift(0.04 + 5e-11),
        'cross_knot_shift_diagnostic': cross_knot_shift_defect,
        'clock_and_dual_offsets': clock_and_dual_offsets,
    }
    results = {}
    failed = []
    for name, check in checks.items():
        try:
            results[name] = check()
        except Exception as error:
            failed.append(name)
            results[name] = {'error': f'{type(error).__name__}: {error}'}
    print(json.dumps({'checks': results, 'failed': failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
