"""Controller-only learned contact velocity bias for legacy arm prediction.

The correction is a phenomenological ball-velocity kick in the nominal paddle
frame, per j/8. It has no arm/spin reaction and does not modify the plant. The
outputs add to normalized ball dynamics constraints, not objective residuals.
"""
import casadi as ca

from arm.ball_flight import flight
from optimization.pruning.formulation import NZ, NP, SCALE, OFFSET
from optimization.runtime_parameters import SIZE


def create_contact_bias_correction(cfg, robot, runtime, bias=None):
    """Return data/value with inputs ``[z68, p30, tuning35, bias3]``.

    ``cfg`` and ``robot`` must use the supplied symbolic runtime vector.
    Data outputs are correction6 and its 6-by-68 Jacobian; value returns only
    correction6. A positive local bias adds to the predicted rebound velocity,
    hence subtracts from the endpoint-minus-prediction ball constraint.
    """
    if runtime.shape != (SIZE, 1):
        raise ValueError('Contact bias requires the unchanged runtime vector')
    if bias is None:
        bias = ca.MX.sym('contact_bias', 3)
    if bias.shape != (3, 1):
        raise ValueError('Contact bias must be a three-vector input symbol')
    z = ca.MX.sym('z', 2 * NZ)
    p = ca.MX.sym('p', NP)
    y = z[:NZ] * ca.DM(SCALE) + ca.DM(OFFSET)
    yn = z[NZ:] * ca.DM(SCALE) + ca.DM(OFFSET)
    q, dq = y[6:13], y[13:20]
    j, fraction, h = y[30], y[31], y[32]
    contact_time = fraction * h
    acceleration = (yn[13:20] - dq) / h
    contact_q = q + contact_time * dq + .5 * contact_time**2 * acceleration
    rotation = robot.contact_fk(contact_q)[4]
    kick = (j / 8) * (rotation @ bias)
    delta_position, delta_velocity = flight(
        ca.MX.zeros(3), kick, h - contact_time, cfg.drag, [0., 0., 0.])
    correction = -ca.vertcat(delta_position, delta_velocity) / ca.DM(SCALE[:6])
    inputs = [z, p, runtime, bias]
    names = ['z', 'p', 'tuning', 'bias_per_j8_local']
    data = ca.Function('contact_bias_prediction_data', inputs,
                       [correction, ca.jacobian(correction, z)],
                       names, ['correction', 'jacobian'])
    value = ca.Function('contact_bias_prediction_value', inputs, [correction],
                        names, ['correction'])
    return data, value
