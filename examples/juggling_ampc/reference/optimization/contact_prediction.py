"""Optional edge-restitution residual correction for legacy arm prediction.

This separate factor leaves the existing stage graphs unchanged. Its output
already includes objective/contact-weight scaling; add it to residual row 52
and its Jacobian before forming costs, Gauss--Newton terms, or restitution
constraints. It does not change the arm trajectory, impulse, or physical plant.
"""
import casadi as ca

from arm.ball_flight import flight
from optimization.pruning.formulation import NZ, NP, SCALE, OFFSET
from optimization.runtime_parameters import SIZE


def create_edge_correction(cfg, robot, runtime, edge_loss=None):
    """Return data/value functions with inputs ``[z68, p30, tuning35, edge]``.

    ``cfg`` and ``robot`` use the supplied symbolic runtime vector, as in the
    parametric native exporter. ``edge_loss`` is a separate scalar symbol (or
    one is created here), keeping the existing tuning ABI unchanged. The data
    function returns the scalar correction and its 1-by-68 Jacobian; the value
    function returns only the correction. Both use the unmodified legacy arm
    contact kinematics and the shared nominal ball-flight function.

    The quartic radius is fixed at 75 mm, matching the fitted contact law;
    it is independent of the optimizer's disk radius or edge safety margin.
    Clamp boundaries inherit CasADi's piecewise derivative convention.
    """
    if runtime.shape != (SIZE, 1):
        raise ValueError('Edge prediction requires the unchanged runtime vector')
    if edge_loss is None:
        edge_loss = ca.MX.sym('edge_loss')
    if edge_loss.shape != (1, 1):
        raise ValueError('Edge loss must be a scalar input symbol')

    z = ca.MX.sym('z', 2 * NZ)
    p = ca.MX.sym('p', NP)
    y = z[:NZ] * ca.DM(SCALE) + ca.DM(OFFSET)
    yn = z[NZ:] * ca.DM(SCALE) + ca.DM(OFFSET)
    b, v, q, dq = y[:3], y[3:6], y[6:13], y[13:20]
    omega, j, fraction, h = y[20:23], y[30], y[31], y[32]
    acceleration = (yn[13:20] - dq) / h
    contact_time = fraction * h
    contact_q = q + contact_time * dq + .5 * contact_time**2 * acceleration
    contact_dq = dq + contact_time * acceleration

    position, normal, jacobian, angular_jacobian, _ = robot.contact_fk(contact_q)
    position = position - ca.DM(robot.origin)
    ball_position, ball_velocity = flight(
        b, v, contact_time, cfg.drag,
        [cfg.gravity_x, cfg.gravity_y, cfg.gravity_z])
    relative_position = ball_position - position
    tangent = relative_position - normal * ca.dot(normal, relative_position)
    contact_jacobian = jacobian - ca.skew(tangent) @ angular_jacobian
    relative_velocity = (ball_velocity - ca.cross(omega, cfg.radius * normal)
                         - contact_jacobian @ contact_dq)
    normal_velocity = ca.dot(normal, relative_velocity)

    radial_quartic = (ca.dot(tangent, tangent) / .075**2)**2
    effective_restitution = ca.fmin(
        1.5, ca.fmax(0., cfg.restitution - edge_loss * radial_quartic))
    correction = (ca.sqrt(cfg.objective_scale) * cfg.comp_weight * (j / 8)
                  * ((effective_restitution - cfg.restitution) * normal_velocity / 8))
    inputs = [z, p, runtime, edge_loss]
    names = ['z', 'p', 'tuning', 'edge_loss']
    data = ca.Function('edge_prediction_data', inputs,
                       [correction, ca.jacobian(correction, z)],
                       names, ['correction', 'jacobian'])
    value = ca.Function('edge_prediction_value', inputs, [correction],
                        names, ['correction'])
    return data, value
