"""Optional acceleration-controlled prediction tail; never a physical plant.

The optional ``endpoint_accelerations`` design uses endpoint control values
instead of the original incoming-interval accelerations documented below. It
averages adjacent endpoint accelerations in the velocity equation, maps the
boundary torque through exact endpoint forward dynamics, and uses the original
proxy torque-rate inequality at that boundary. Costs retain their torque units.
This is the full frozen-affine midpoint formulation after a variable change;
the fixed arm approximation and the actual nonlinear boundary still differ.

Integration contract (the exporter/native caller must opt in explicitly):

* For a dynamic prefix of P intervals, retain legacy factors at k=0..P-1.
  Nodes0..P keep torque in slots23:30. At stageP select ``boundary=True``.
  NodesP+1..N store acceleration in those slots, using the existing global
  SCALE/OFFSET. Each a[k+1] belongs to the interval ending at node k+1.
* The boundary's seven torque_rate rows are an exact conversion equality,
  (RNEA(q[P],dq[P],a[P+1])-tau[P])/TORQUE_LIMIT=0. This is the sole rigid-body
  evaluation added at the boundary. All later factors use only FK/kinematics.
  Prefix endpoint acceleration constraints still apply to tau[P].
* Pure tail torque_rate rows instead bound (a[k+1]-a[k])/(h[k]*JERK_LIMIT).
  No tail torque or torque-rate limit is asserted. Bounds on acceleration
  slots MUST replace torque boxes; ``acceleration_slot_bounds`` supplies them.
  Existing global variable scaling is deliberately retained; native packing
  remains z*scale+offset, and decoded tail slots have units rad/s^2.
* Contact position/velocity use exactly the legacy endpoint-derived constant
  acceleration. Thus edge, private-bias, radial scaling, and first-event
  centering helpers retain their kinematics and row destinations. Tail K omits
  finite-arm effective mass, and there is no arm recoil. The legacy finite-arm
  plane correction is NOT compatible: bypass its tail mapping at zero offset,
  or select a no-reaction plane helper for those stages.
* Native initialization/regeneration must seed interval accelerations from
  (dq[k+1]-dq[k])/h[k], not call inverse dynamics on tail control slots. At the
  boundary reconcile a[P+1] with torque[P], or record the resulting seed defect.
  Warm shifts must transfer decoded physical controls by semantic type: convert
  a shifted tail interval entering the prefix to torque through RNEA, and a
  dynamic interval entering the tail to its q/dq-derived acceleration. Never
  interpolate Newton-metres and rad/s^2. Rebuild the boundary relation after
  arm feedback and shifts. Tail contact rollout must use the no-reaction K.
* Shift duals only between equal row meanings; clear boundary-conversion duals
  and any dynamics/endpoint/rate dual whose source crosses the type boundary.
  In particular a torque_rate equality multiplier is not a jerk-bound dual.
  Variable-bound duals must likewise be cleared when torque/acceleration meaning
  changes. First-state/torque pinning remains unchanged.
* Returned full plans retain acceleration tail slots and need metadata marking
  the switch. The executed first four nodes stay torque-valued when P>=3, so
  existing reference projection remains unchanged. For diagnostics requiring a
  torque view, convert a COPY, never mutate the optimizer's acceleration plan.
* For P<3, execution must convert acceleration-valued entries in a COPY of the
  first four nodes to RNEA torques before the ordinary torque-reference
  projector. The optimizer plan itself retains its original control meanings.
* Tail endpoint elasticity and arm-shift repair have different semantics and
  must remain disabled until adapted. The baseline extra_fraction=0 and
  warmstart_arm_shift=0 satisfy this. Plant factors/library stay baseline.

The 57 residual slots and original full constraint IDs are preserved. Residuals
23:30 become a small normalized acceleration penalty; 30:37 become a normalized
acceleration-change penalty, zero at the boundary where no previous acceleration
variable exists. This does not pretend that an acceleration is a torque.
"""

import casadi as ca
import numpy as np

from arm.ball_flight import flight
from arm.model import HOME, ACCEL_LIMIT, TORQUE_LIMIT, TORQUE_RATE_LIMIT, JERK_LIMIT
from optimization.pruning.formulation import NX, NZ, NP, SCALE, OFFSET


def acceleration_slot_bounds(cfg):
    """Normalized boxes to apply at nodes P+1..N, replacing torque boxes."""
    extent = float(cfg.actuator_margin) * np.asarray(ACCEL_LIMIT, dtype=float)
    if not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError('Kinematic-tail acceleration margin must be positive')
    return ((-extent - OFFSET[23:30]) / SCALE[23:30],
            (extent - OFFSET[23:30]) / SCALE[23:30])


def actuation_proxy_coefficients(robot, posture):
    """Fixed first-order RNEA model, evaluated once while exporting factors.

    tau_hat(q,a) = M(q_ref) a + g(q_ref) + dg/dq(q_ref) (q-q_ref).
    It matches RNEA and its q/a derivatives at (q_ref, dq=0, a=0).
    Coriolis and configuration-dependent inertia remain approximations. This
    is a prediction proxy only, never an update to the physical arm model.
    """
    reference = np.asarray(posture, dtype=float)
    physical_robot = getattr(robot, 'base', robot)
    if (reference.shape != (7,) or not np.isfinite(reference).all()
            or np.any(reference < physical_robot.q_min)
            or np.any(reference > physical_robot.q_max)):
        raise ValueError('Tail actuation proxy needs seven finite in-bounds joint angles')
    mass = np.asarray(robot.mass(reference), dtype=float)
    gravity = np.asarray(robot.bias(reference, np.zeros(7)), dtype=float).ravel()
    q = ca.MX.sym('q_proxy_reference', 7)
    derivative = ca.Function('tail_gravity_derivative', [q],
                            [ca.jacobian(robot.bias(q, ca.DM.zeros(7)), q)])
    stiffness = np.asarray(derivative(reference), dtype=float)
    if (not np.isfinite(np.r_[mass.ravel(), gravity, stiffness.ravel()]).all()
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=0)
            or np.linalg.eigvalsh(mass)[0] <= 0):
        raise ValueError('Invalid fixed RNEA actuation proxy')
    return dict(posture=reference, mass=mass, gravity=gravity,
                gravity_jacobian=stiffness)


def integration_metadata(prefix=5, *, actuation_proxy_posture=None,
                         endpoint_accelerations=False):
    """Machine-readable additions for an explicitly selected hybrid profile."""
    if isinstance(prefix, bool) or int(prefix) != prefix or prefix < 1:
        raise ValueError('Need at least one fully dynamic prefix interval')
    prefix = int(prefix)
    result = dict(dynamic_intervals=prefix, boundary_stage=prefix,
                first_acceleration_node=prefix + 1,
                control_columns=[23, 30],
                control_variable_scale=SCALE[23:30].tolist(),
                tail_control_units='rad/s^2',
                tail_control_convention='node k+1 stores constant acceleration for interval k',
                boundary_rate_rows='exact RNEA(q_k,dq_k,a_{k+1})=tau_k conversion',
                tail_rate_rows='(a_{k+1}-a_k)/(h_k*JERK_LIMIT)',
                contact_model='moving paddle with infinite arm effective mass and no arm recoil',
                physical_plant_changed=False)
    if actuation_proxy_posture is not None:
        posture = np.asarray(actuation_proxy_posture, dtype=float)
        if posture.shape != (7,) or not np.isfinite(posture).all():
            raise ValueError('Tail actuation proxy posture must contain seven finite angles')
        result.update(actuation_proxy_posture=posture.tolist(),
            actuation_proxy='M_ref*a + g_ref + dg_dq_ref*(q-q_ref); fixed zero-velocity RNEA linearization',
            tail_effort_cost='Original torque/TORQUE_LIMIT and delta_torque/[10,10,10,10,2,2,2] costs on the affine proxy',
            tail_rate_rows='delta_proxy_torque/(h*TORQUE_RATE_LIMIT), original actuator_margin*rate_margin bounds',
            tail_endpoint_rows='Affine proxy torque/TORQUE_LIMIT boxes; acceleration slots remain explicitly boxed',
            boundary_endpoint_rows='Unchanged acceleration bounds; exact RNEA conversion retained',
            boundary_rate_cost='proxy_torque(q_next,a_next)-actual_tau_boundary; original rate-cost units',
            execution_conversion_required=prefix < 3)
    if endpoint_accelerations:
        if actuation_proxy_posture is None:
            raise ValueError('Endpoint acceleration tail requires the affine actuation proxy')
        result.update(endpoint_accelerations=True,
            tail_control_convention='node k stores endpoint acceleration; interval velocity uses the endpoint average',
            boundary_rate_rows='delta_proxy_torque/(h*TORQUE_RATE_LIMIT), original actuator_margin*rate_margin bounds',
            boundary_acceleration='Exact forward dynamics of the last torque-valued node',
            boundary_endpoint_rows='Exact left endpoint acceleration and right acceleration variable',
            tail_dynamics='dq_next-dq = h*(a_left+a_right)/2; legacy constant-average contact interpolation')
    return result


def create_kinematic_tail_factors(cfg, robot, drop=(), left_endpoint_only=False,
                                  suffix='', runtime=None, *, boundary=False,
                                  actuation_proxy_posture=None,
                                  endpoint_accelerations=False):
    """Drop-in factor tuple for one boundary or purely kinematic tail stage.

    API/row grouping matches pruning.formulation.create_factors. ``boundary``
    is structural, never inferred from stage data or added as a runtime scalar.
    The caller uses the ordinary first/last factors from the dynamic builder;
    compatible first/last functions are returned for the standard tuple ABI.
    The optional fixed-posture actuation proxy matches the full formulation's
    torque/rate units and restores approximate tail torque/rate feasibility.
    It evaluates only constant matrices during export: no online rigid-body
    dynamics or matrix inverse is added to pure-tail factors. Omitting it
    preserves the original tail graph and costs exactly.
    """
    if not isinstance(boundary, bool):
        raise ValueError('Boundary selection must be boolean')
    if not isinstance(endpoint_accelerations, bool):
        raise ValueError('Endpoint acceleration selection must be boolean')
    if endpoint_accelerations and actuation_proxy_posture is None:
        raise ValueError('Endpoint acceleration tail requires the affine actuation proxy')
    if 'torque_rate' in drop:
        raise ValueError('The conversion/acceleration-change rows cannot be dropped')
    if runtime is None:
        if not 0 <= cfg.contact_edge_margin < cfg.paddle_radius:
            raise ValueError('Contact margin must be smaller than paddle radius')
        if (not np.isfinite(cfg.contact_center_grip_offset)
                or not 0 <= cfg.contact_center_grip_offset < cfg.paddle_radius):
            raise ValueError('Preferred contact offset must lie inside the paddle')
        if (not np.isfinite([cfg.gravity_x, cfg.gravity_y, cfg.gravity_z]).all()
                or cfg.gravity_z >= 0):
            raise ValueError('Ball gravity must be finite and downward')

    z = ca.MX.sym('z', NZ)
    zn = ca.MX.sym('zn', NZ)
    p = ca.MX.sym('p', NP)
    y = z * ca.DM(SCALE) + ca.DM(OFFSET)
    yn = zn * ca.DM(SCALE) + ca.DM(OFFSET)
    b, v, q, dq = y[:3], y[3:6], y[6:13], y[13:20]
    omega, left_control, j, fraction, h = y[20:23], y[23:30], y[30], y[31], y[32]
    qn, dqn, acceleration = yn[6:13], yn[13:20], yn[23:30]
    mean_velocity = .5 * (dq + dqn)
    interpolation_acceleration = (dqn - dq) / h
    contact_time = fraction * h
    contact_q = q + contact_time * dq + .5 * contact_time**2 * interpolation_acceleration
    contact_dq = dq + contact_time * interpolation_acceleration
    pc, normal, Jc, Wc, Rc = robot.contact_fk(contact_q)
    pc = pc - ca.DM(robot.origin)
    pos, pose_normal, J, N, _ = robot.fk(q)
    pos = pos - ca.DM(robot.origin)
    posn, nn, _, _, _ = robot.fk(qn)
    posn = posn - ca.DM(robot.origin)
    gravity = [cfg.gravity_x, cfg.gravity_y, cfg.gravity_z]
    bc, contact_ball_velocity = flight(b, v, contact_time, cfg.drag, gravity)
    relative_position = bc - pc
    gap = ca.dot(normal, relative_position) - cfg.radius
    tangent = relative_position - normal * ca.dot(normal, relative_position)
    Jcontact = Jc - ca.skew(tangent) @ Wc
    relative_velocity = (contact_ball_velocity - ca.cross(omega, cfg.radius * normal)
                         - Jcontact @ contact_dq)
    normal_velocity = ca.dot(normal, relative_velocity)
    tangents = Rc[:, :2]
    K = ca.MX.eye(3) + (ca.MX.eye(3) - normal @ normal.T) / cfg.inertia_ratio
    desired_tangent = -ca.solve(tangents.T @ K @ tangents,
                               tangents.T @ (relative_velocity + K @ normal * j))
    capacity = cfg.friction * j
    jt = desired_tangent * capacity / ca.sqrt(
        capacity**2 + ca.dot(desired_tangent, desired_tangent) + 1e-10)
    impulse = j * normal + tangents @ jt
    restitution = ca.dot(normal, K @ impulse) + (1 + cfg.restitution) * normal_velocity
    bnext, vnext = flight(bc, contact_ball_velocity + impulse, h - contact_time,
                         cfg.drag, gravity)
    omeganext = omega - ca.cross(normal, impulse) / (cfg.inertia_ratio * cfg.radius)
    endgap = ca.dot(nn, yn[:3] - posn) - cfg.radius
    contactgap = (j / 8) * gap / .05
    contactrest = (j / 8) * restitution / 8

    if boundary and not endpoint_accelerations:
        # Original incoming-interval design: one inverse-dynamics conversion.
        rate_rows = (robot.rnea(q, dq, acceleration) - left_control) / ca.DM(TORQUE_LIMIT)
        rate_lower = rate_upper = 0.
        previous_acceleration = acceleration
        acceleration_change = ca.MX.zeros(7)
    else:
        previous_acceleration = left_control
        acceleration_change = acceleration - previous_acceleration
        rate_rows = acceleration_change / h / ca.DM(JERK_LIMIT)
        rate_lower = -cfg.actuator_margin * cfg.jerk_margin
        rate_upper = cfg.actuator_margin * cfg.jerk_margin

    if endpoint_accelerations:
        # Unlike the original incoming-interval control, each acceleration is
        # now the endpoint counterpart of a torque variable. The frozen affine
        # arm's midpoint force balance is then exactly the trapezoidal velocity
        # equation. At the type boundary retain the actual endpoint dynamics.
        previous_acceleration = (ca.solve(robot.mass(q), left_control-robot.bias(q, dq))
                                 if boundary else left_control)
        acceleration_change = acceleration - previous_acceleration
    interval_acceleration = (.5*(previous_acceleration+acceleration)
                             if endpoint_accelerations else acceleration)

    proxy = None
    if actuation_proxy_posture is not None:
        proxy = actuation_proxy_coefficients(robot, actuation_proxy_posture)
        def proxy_torque(position, interval_acceleration):
            return (ca.DM(proxy['mass']) @ interval_acceleration
                    + ca.DM(proxy['gravity'])
                    + ca.DM(proxy['gravity_jacobian'])
                    @ (position - ca.DM(proxy['posture'])))
        left_torque = (left_control if boundary else
                       proxy_torque(q, previous_acceleration))
        right_torque = proxy_torque(qn, acceleration)
        torque_change = right_torque - left_torque
        if not boundary or endpoint_accelerations:
            rate_rows = torque_change / h / ca.DM(TORQUE_RATE_LIMIT)
            rate_lower = -cfg.actuator_margin * cfg.rate_margin
            rate_upper = cfg.actuator_margin * cfg.rate_margin

    constraints, lo, hi, groups = [], [], [], {}

    def add(name, expression, lower, upper):
        if name in drop:
            return
        start, count = len(lo), int(expression.numel())
        constraints.append(expression)
        lo.extend(np.broadcast_to(lower, (count,)))
        hi.extend(np.broadcast_to(upper, (count,)))
        groups[name] = (start, start + count)

    add('ball', ca.vertcat(yn[:3] - bnext, yn[3:6] - vnext) / ca.DM(SCALE[:6]), 0, 0)
    add('spin', (yn[20:23] - omeganext) / 20., 0, 0)
    add('position', qn - q - h * mean_velocity, 0, 0)
    add('dynamics', (dqn - dq - h * interval_acceleration) / 2., 0, 0)
    add('clock', yn[33] - y[33] - h, 0, 0)
    add('contact', ca.vertcat(gap / .5, endgap / .5,
        (j / 8) * (ca.dot(tangent, tangent) / (cfg.paddle_radius - cfg.contact_edge_margin)**2 - 1),
        (j / 8) * normal_velocity / 4),
        [0, 0, -np.inf, -np.inf], [np.inf, np.inf, 0, 0])
    add('workspace', pos / ca.DM([.4, .4, .18]),
        [-1, -cfg.workspace_side / .4, -1], [cfg.workspace_forward / .4, cfg.workspace_side / .4, 1])
    add('tilt', pose_normal[:2] / np.sin(.4), -1, 1)
    add('upward', pose_normal[2], np.cos(.4), np.inf)
    add('paddle_velocity', J @ dq / 1.2, -1, 1)
    add('acceleration', interpolation_acceleration / ca.DM(ACCEL_LIMIT),
        -cfg.actuator_margin, cfg.actuator_margin)
    endpoint = (previous_acceleration / ca.DM(ACCEL_LIMIT) if left_endpoint_only else
                ca.vertcat(previous_acceleration, acceleration) / ca.DM(np.tile(ACCEL_LIMIT, 2)))
    if proxy is not None and not boundary:
        # Acceleration variables already have physical acceleration boxes.
        # Reuse these redundant rows for the original torque-box envelope.
        endpoint = (left_torque / ca.DM(TORQUE_LIMIT) if left_endpoint_only else
                    ca.vertcat(left_torque, right_torque) / ca.DM(np.tile(TORQUE_LIMIT, 2)))
    add('endpoint_acceleration', endpoint, -cfg.actuator_margin, cfg.actuator_margin)
    add('torque_rate', rate_rows, rate_lower, rate_upper)
    add('jerk', acceleration_change / h / ca.DM(JERK_LIMIT),
        -cfg.actuator_margin * cfg.jerk_margin, cfg.actuator_margin * cfg.jerk_margin)

    energy = b[2] - cfg.radius + v[2]**2 / (-2 * cfg.gravity_z)
    effort_residual = cfg.torque_weight * acceleration / ca.DM(ACCEL_LIMIT)
    rate_residual = cfg.rate_weight * acceleration_change / ca.DM(ACCEL_LIMIT)
    if proxy is not None:
        effort_residual = cfg.torque_weight * left_torque / ca.DM(TORQUE_LIMIT)
        rate_residual = cfg.rate_weight * torque_change / ca.DM([10.] * 4 + [2.] * 3)
    residual = ca.vertcat(
        cfg.energy_weight * (energy - cfg.apex), cfg.lateral_weight * b[:2] / .2,
        2 * cfg.ball_velocity_weight * v[:2],
        cfg.pose_weight * pos * ca.vertcat(cfg.paddle_lateral_weight / .3,
                                          cfg.paddle_lateral_weight / .3, 1 / .15),
        .3 * pose_normal[:2] / .3,
        cfg.velocity_weight * (J @ dq), .5 * cfg.velocity_weight * (N @ dq),
        cfg.accel_weight * interpolation_acceleration / ca.DM(ACCEL_LIMIT),
        effort_residual, rate_residual,
        cfg.posture_weight * (q - ca.DM(HOME)), .3 * dq,
        cfg.comp_weight * contactgap, cfg.comp_weight * contactrest, .001 * j / 8)
    if runtime is not None or cfg.contact_center_weight:
        center_error = tangent
        if runtime is not None or cfg.contact_center_grip_offset:
            grip = robot.paddle_normal_rotation.T @ ca.DM([-1., 0., 0.])
            grip = ca.vertcat(grip[0], grip[1], 0)
            grip = grip / ca.norm_2(grip)
            center_error = tangent - cfg.contact_center_grip_offset * (Rc @ grip)
        residual = ca.vertcat(residual, cfg.contact_center_weight * (j / 8)
                             * center_error / cfg.paddle_radius)
    terminal = ca.vertcat(2 * cfg.energy_weight * (energy - cfg.apex),
                         2 * cfg.lateral_weight * b[:2] / .2,
                         4 * cfg.ball_velocity_weight * v[:2])
    extra = [] if runtime is None else [runtime]
    first = ca.Function('arm_first', [z, p] + extra, [ca.vertcat(
        (y[:NX] - p[:NX]) / ca.DM(SCALE[:NX]),
        (left_control - p[NX:]) / ca.DM(SCALE[23:30]), y[33])])
    stage = ca.Function('arm_stage' + suffix, [z, zn, p] + extra,
                        [residual, ca.vertcat(*constraints)])
    last = ca.Function('arm_last', [z, p] + extra, [terminal, y[33]])
    diagnostic = ca.Function('arm_contact' + suffix, [z, zn] + extra,
                             [gap, restitution, ca.norm_2(tangent), normal_velocity])
    return first, stage, last, diagnostic, np.array(lo), np.array(hi), groups
