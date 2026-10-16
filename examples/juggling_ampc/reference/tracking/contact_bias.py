"""Causal, optional adaptation of an effective contact-velocity residual.

This module is deliberately independent of simulator truth and disturbances.  Its
inputs are associated, delivered marker positions, measured joint states and an
observer's impact timestamp/incoming spin estimate.  Nothing is enabled merely by
importing it.  ``bias_per_j8_local`` is a velocity correction in the corrected
paddle frame: the controller adds ``R @ bias * (j / 8)`` after nominal impact,
where ``j`` is normal impulse divided by ball mass, in m/s.

Run the small analytic check with ``python -m tracking.contact_bias``.  It fits
synthetic piecewise quadratic observations, not a closed-loop simulation.
"""

from collections import deque
from collections.abc import Mapping
import json

import numpy as np
from scipy.optimize import minimize_scalar

from arm.spin_contact import impact


def _config(cfg, name, default=None):
    value = cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)
    if value is None:
        raise ValueError(f"Contact bias estimator needs config parameter {name}")
    return float(value)


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


class _Rejected(Exception):
    pass


class ContactBiasEstimator:
    """Small residual estimator; invalid contacts are skipped, never extrapolated.

    ``record_position`` uses capture/arrival timestamps in the arm/observer clock;
    positions are relative to ``robot.origin``.  ``record_arm`` takes *measured*
    q/dq.  ``observe`` takes the delivered 23-state observer estimate and its
    ``last_impact`` timestamp; call it during normal healthy supervisor updates.
    It returns a new diagnostic dictionary, or None.  All dictionaries also
    remain in ``records`` for JSON logging, including rejected contacts.

    The learned quantity combines repeatable model errors, including uncertain
    spin. It is not an identification of separate restitution, normal or friction
    parameters. Fits estimate acceleration independently of configured gravity.
    ``polynomial_degree=3`` optionally fits a constant acceleration derivative;
    it reduces flight-curvature bias but increases extrapolated velocity variance.
    The default degree 2 retains the original quadratic fit.
    Only already observed neighboring impacts can be used for isolation checks.
    """

    def __init__(self, robot, cfg, edge_loss=0., *, gain=.15, max_bias=.20,
                 pre_window=(-.16, -.02), post_window=(.025, .16), min_samples=8,
                 min_span=.06, max_position_gap=.04, fit_noise_floor=.001,
                 max_fit_rms=.003, max_continuity_gap=.004,
                 impact_refinement=.008, max_contact_radius=.055,
                 max_plane_gap=.010, min_impulse=2., min_contact_gap=.32,
                 max_arm_gap=.02, max_spin_age=.10, max_sample_norm=.65,
                 edge_reference_radius=.075, plane_offset=0.,
                 impact_event_revision_window=.02, polynomial_degree=2):
        self.robot = robot
        self.ball_mass = _config(cfg, 'mass', _config(cfg, 'ball_mass', .0039))
        self.radius = _config(cfg, 'radius')
        self.restitution = _config(cfg, 'restitution')
        self.friction = _config(cfg, 'friction')
        self.inertia_ratio = _config(cfg, 'inertia_ratio', 2/3)
        self.edge_loss = float(edge_loss)
        self.edge_reference_radius = float(edge_reference_radius)
        self.plane_offset = float(plane_offset)
        self.gain, self.max_bias = float(gain), float(max_bias)
        self.pre_window, self.post_window = tuple(pre_window), tuple(post_window)
        self.min_samples, self.min_span = int(min_samples), float(min_span)
        self.max_position_gap = float(max_position_gap)
        self.fit_noise_floor, self.max_fit_rms = float(fit_noise_floor), float(max_fit_rms)
        self.max_continuity_gap = float(max_continuity_gap)
        self.impact_refinement = float(impact_refinement)
        self.max_contact_radius = min(float(max_contact_radius), _config(cfg, 'paddle_radius'))
        self.max_plane_gap, self.min_impulse = float(max_plane_gap), float(min_impulse)
        self.min_contact_gap = float(min_contact_gap)
        self.impact_event_revision_window = float(impact_event_revision_window)
        if polynomial_degree not in (2, 3):
            raise ValueError('Contact bias polynomial degree must be 2 or 3')
        self.polynomial_degree = int(polynomial_degree)
        self.max_arm_gap, self.max_spin_age = float(max_arm_gap), float(max_spin_age)
        self.max_sample_norm = float(max_sample_norm)
        positive = [self.ball_mass, self.radius, self.inertia_ratio, self.edge_reference_radius,
                    self.max_bias, self.min_span, self.max_position_gap, self.fit_noise_floor,
                    self.max_fit_rms, self.max_continuity_gap, self.max_contact_radius,
                    self.max_plane_gap, self.min_impulse, self.min_contact_gap,
                    self.max_arm_gap, self.max_spin_age, self.max_sample_norm,
                    self.impact_event_revision_window]
        if not all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError('Contact bias estimator requires positive finite bounds')
        if not 0 < self.gain <= 1 or self.max_bias > .30 or self.min_samples < 8:
            raise ValueError('Invalid contact bias gain, bound or sample count')
        if self.impact_event_revision_window >= self.min_contact_gap:
            raise ValueError('Impact revision window must be shorter than contact isolation gap')
        if (len(self.pre_window) != 2 or len(self.post_window) != 2
                or not self.pre_window[0] < self.pre_window[1] < 0
                or not 0 < self.post_window[0] < self.post_window[1]
                or not np.isfinite(self.pre_window + self.post_window).all()
                or not 0 <= self.impact_refinement <= .008
                or not np.isfinite([self.plane_offset, self.edge_loss, self.restitution,
                                    self.friction]).all()
                or self.edge_loss < 0 or self.friction < 0):
            raise ValueError('Invalid contact bias fit windows or contact parameters')
        self._positions = deque(maxlen=4096)
        self._arms = deque(maxlen=8192)
        self._spins = deque(maxlen=2048)
        self._events = deque(maxlen=256)
        self._pending = deque()
        self._latest_event = -np.inf
        self._last_observe = -np.inf
        self._bias = np.zeros(3)
        self._covariance = np.zeros((3, 3))
        self._residual_covariance = np.zeros((3, 3))
        self.accepted_samples = 0
        self.records = []

    @property
    def bias_per_j8_local(self):
        return self._bias.copy()

    @property
    def covariance(self):
        return self._covariance.copy()

    def record_position(self, capture, p_local, arrival):
        p = np.asarray(p_local, dtype=float).reshape(-1)
        if p.size == 3 and np.isfinite(p).all() and np.isfinite([capture, arrival]).all():
            self._positions.append((float(capture), p.copy(), float(arrival)))

    def record_arm(self, t, q, dq):
        q, dq = np.asarray(q, float).reshape(-1), np.asarray(dq, float).reshape(-1)
        if q.size == dq.size == 7 and np.isfinite(t) and np.isfinite(q).all() and np.isfinite(dq).all():
            self._arms.append((float(t), q.copy(), dq.copy()))

    def observe(self, t, estimated_state23, last_impact):
        """Consume delivered evidence only; a fit is available after its post-window."""
        t = float(t)
        state = np.asarray(estimated_state23, float).reshape(-1)
        if not np.isfinite(t) or t < self._last_observe:
            return None
        self._last_observe = t
        if state.size >= 23 and np.isfinite(state[20:23]).all():
            self._spins.append((t, state[20:23].copy()))
        if last_impact is not None and np.isfinite(last_impact):
            event = float(last_impact)
            if event <= t:
                if abs(event-self._latest_event) <= self.impact_event_revision_window:
                    # Observer replay can revise a provisional impact time in
                    # either direction. Update that contact, never fit it twice.
                    if self._pending and self._pending[-1] == self._latest_event:
                        self._pending[-1] = event
                    self._events[-1] = event
                    self._latest_event = event
                elif event > self._latest_event:
                    self._latest_event = event
                    self._events.append(event)
                    self._pending.append(event)
        result = None
        while self._pending and t > self._pending[0] + self.post_window[1]:
            event = self._pending.popleft()
            record = dict(available_time_s=t, observer_impact_time_s=event,
                          accepted=False, bias_before=self._bias.copy())
            if self.polynomial_degree == 3:
                record['polynomial_degree'] = 3
            try:
                self._fit_contact(event, t, record)
            except _Rejected as exc:
                record['reason'] = str(exc)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                record['reason'] = 'invalid_nominal_calculation: ' + str(exc)
            record.update(bias_per_j8_local=self._bias.copy(), covariance=self._covariance.copy(),
                          residual_covariance=self._residual_covariance.copy(),
                          accepted_samples=self.accepted_samples)
            result = _json_safe(record)
            self.records.append(result)
        return result

    def _window_fit(self, event, now, window):
        # Filter arrival explicitly, even if a caller preloads future records.
        selected = {capture: (p, arrival) for capture, p, arrival in self._positions
                    if arrival <= now and capture <= now
                    and event + window[0] - 1e-10 <= capture <= event + window[1] + 1e-10}
        times = np.array(sorted(selected))
        if len(times) < self.min_samples:
            raise _Rejected('insufficient_delivered_positions')
        if times[-1] - times[0] < self.min_span:
            raise _Rejected('position_window_too_short')
        if np.max(np.diff(times)) > self.max_position_gap:
            raise _Rejected('position_window_gap')
        values = np.array([selected[t][0] for t in times])
        u = (times - event) / .1
        design = np.column_stack([np.ones(len(times)), u, .5*u*u])
        if self.polynomial_degree == 3:
            design = np.column_stack([design, u*u*u/6.])
        weights = np.ones(len(times))
        for _ in range(4):
            coefficients = np.linalg.lstsq(design*np.sqrt(weights[:, None]),
                                            values*np.sqrt(weights[:, None]), rcond=None)[0]
            residual = values - design @ coefficients
            norms = np.linalg.norm(residual, axis=1)
            robust_scale = max(self.fit_noise_floor,
                               1.4826*np.median(np.abs(norms-np.median(norms))))
            weights = np.minimum(1., 2.5*robust_scale / np.maximum(norms, 1e-15))
        rms = float(np.sqrt(np.mean(np.sum(residual*residual, axis=1))))
        if rms > self.max_fit_rms:
            raise _Rejected('position_fit_residual')
        normal = design.T @ (weights[:, None]*design)
        if np.linalg.cond(normal) > 1e8:
            raise _Rejected('ill_conditioned_position_fit')
        variance = np.maximum(self.fit_noise_floor**2,
                              np.sum(weights[:, None]*residual**2, axis=0)
                              / max(1., weights.sum()-(self.polynomial_degree+1)))
        return dict(coefficients=coefficients, inverse_normal=np.linalg.inv(normal), variance=variance,
                    count=len(times), rms=rms, span_s=float(times[-1]-times[0]),
                    latest_arrival_s=max(selected[t][1] for t in times))

    @staticmethod
    def _evaluate_fit(fit, delta):
        u = delta/.1
        coefficients = fit['coefficients']
        if len(coefficients) == 4:
            position = np.array([1., u, .5*u*u, u*u*u/6.]) @ coefficients
            derivative = np.array([0., 10., 10.*u, 5.*u*u])
        else:
            position = np.array([1., u, .5*u*u]) @ coefficients
            derivative = np.array([0., 10., 10.*u])
        velocity = derivative @ coefficients
        covariance = np.diag(fit['variance'] * (derivative @ fit['inverse_normal'] @ derivative))
        return position, velocity, covariance

    def _arm_at(self, when, now):
        samples = {t: (q, dq) for t, q, dq in self._arms if t <= now}
        times = np.array(sorted(samples))
        k = np.searchsorted(times, when)
        if k < len(times) and abs(times[k]-when) < 1e-10:
            return samples[times[k]]
        if k == 0 or k == len(times) or times[k]-times[k-1] > self.max_arm_gap:
            raise _Rejected('missing_bracketing_measured_arm')
        fraction = (when-times[k-1]) / (times[k]-times[k-1])
        return tuple((1-fraction)*a + fraction*b for a, b in zip(samples[times[k-1]], samples[times[k]]))

    def _fit_contact(self, event, now, record):
        if any(1e-6 < abs(other-event) < self.min_contact_gap for other in self._events):
            raise _Rejected('nonisolated_observed_contact')
        pre = self._window_fit(event, now, self.pre_window)
        post = self._window_fit(event, now, self.post_window)
        for name, fit in [('pre', pre), ('post', post)]:
            record[name+'_fit'] = {key: fit[key] for key in ('count', 'rms', 'span_s', 'latest_arrival_s')}
        def discrepancy(delta):
            difference = self._evaluate_fit(pre, delta)[0] - self._evaluate_fit(post, delta)[0]
            return float(difference @ difference)
        delta = 0.
        if self.impact_refinement > 0:
            refined = minimize_scalar(discrepancy, bounds=(-self.impact_refinement, self.impact_refinement),
                                      method='bounded', options={'xatol': 1e-10})
            if refined.success:
                delta = float(refined.x)
            if abs(delta) >= self.impact_refinement - 1e-5:
                raise _Rejected('impact_time_refinement_at_bound')
        pminus, vminus, covminus = self._evaluate_fit(pre, delta)
        pplus, vplus, covplus = self._evaluate_fit(post, delta)
        gap = float(np.linalg.norm(pplus-pminus))
        record.update(refined_impact_time_s=event+delta, impact_time_correction_s=delta,
                      continuity_gap_m=gap, velocity_pre=vminus, velocity_post=vplus)
        if gap > self.max_continuity_gap:
            raise _Rejected('impact_position_discontinuity')
        q, dq = self._arm_at(event+delta, now)
        spins = [(t, spin) for t, spin in self._spins if t <= event + min(-.02, delta-1e-6)]
        if not spins:
            raise _Rejected('missing_incoming_spin_estimate')
        spin_time, spin = max(spins, key=lambda pair: pair[0])
        if event+delta-spin_time > self.max_spin_age:
            raise _Rejected('stale_incoming_spin_estimate')
        p, n, Jp, W, R = [np.asarray(value, float) for value in self.robot.contact_fk(q)]
        p, n = p.reshape(3), n.reshape(3)
        origin = np.asarray(self.robot.origin, float).reshape(3)
        p = p-origin+self.plane_offset*n
        Jp = Jp-_skew(self.plane_offset*n) @ W
        relative_position = .5*(pminus+pplus)-p
        tangent = relative_position-n*(n @ relative_position)
        radial = float(np.linalg.norm(tangent))
        plane_gap = float(n @ relative_position-self.radius)
        record.update(contact_radius_m=radial, plane_gap_m=plane_gap, spin_pre=spin,
                      spin_time_s=spin_time, q=q, dq=dq)
        if radial > self.max_contact_radius:
            raise _Rejected('edge_contact')
        if abs(plane_gap) > self.max_plane_gap:
            raise _Rejected('contact_plane_mismatch')
        J = Jp-_skew(tangent) @ W
        relative_velocity = vminus-np.cross(spin, self.radius*n)-J @ dq
        if float(n @ relative_velocity) >= -.2:
            raise _Rejected('not_descending_contact')
        restitution = float(np.clip(self.restitution-self.edge_loss*(radial/self.edge_reference_radius)**4, 0., 1.5))
        mass = np.asarray(self.robot.mass(q), float)
        kick, _, _, impulse = impact(n, R[:, :2], J, mass, relative_velocity,
                                    ball_mass=self.ball_mass, radius=self.radius,
                                    restitution=restitution, friction=self.friction,
                                    inertia_ratio=self.inertia_ratio)
        if not np.isfinite(impulse) or impulse < self.min_impulse:
            raise _Rejected('weak_nominal_contact')
        nominal = vminus+kick
        sample = R.T @ (vplus-nominal)*(8./impulse)
        # Approximate position-fit covariance only: it omits the nonlinear
        # nominal-map derivative, timing, spin and arm uncertainty. It is a
        # diagnostic, not a calibrated confidence bound. Learned scatter below
        # also includes unknown spin, arm interpolation and model errors.
        covariance = R.T @ (covplus+covminus) @ R*(8./impulse)**2
        record.update(nominal_post_velocity=nominal, nominal_j_m_s=impulse,
                      radial_restitution=restitution, bias_sample_per_j8_local=sample,
                      sample_fit_covariance=covariance)
        if not np.isfinite(sample).all() or np.linalg.norm(sample) > self.max_sample_norm:
            raise _Rejected('implausible_contact_residual')
        innovation = sample-self._bias
        proposal = self._bias+self.gain*innovation
        length = np.linalg.norm(proposal)
        saturated = bool(length > self.max_bias)
        if saturated:
            proposal *= self.max_bias/length
        self._bias = proposal
        self._covariance = (1-self.gain)**2*self._covariance+self.gain**2*covariance
        self._residual_covariance = ((1-self.gain)*self._residual_covariance
                                     +self.gain*np.outer(innovation, innovation))
        self.accepted_samples += 1
        record.update(accepted=True, reason='accepted', bias_clamped=saturated)


def _analytic_selftest():
    """Check sign, convergence, observer timing refinement and delivery causality."""
    class FixedPaddle:
        origin = np.array([.4, -.1, .3])
        def contact_fk(self, q):
            return self.origin.copy(), np.array([0., 0., 1.]), np.zeros((3, 7)), np.zeros((3, 7)), np.eye(3)
        def mass(self, q):
            return np.eye(7)
    robot = FixedPaddle()
    cfg = dict(mass=.0039, radius=.02, restitution=.556, friction=.3,
               inertia_ratio=2/3, paddle_radius=.07)
    incoming = np.array([0., 0., -3.])
    nominal_kick, _, _, j = impact(np.array([0., 0., 1.]), np.eye(3)[:, :2],
                                   np.zeros((3, 7)), np.eye(7), incoming,
                                   ball_mass=cfg['mass'], restitution=cfg['restitution'])
    acceleration = np.array([.03, -.02, -9.7])
    state = np.zeros(23)
    def feed(estimator, event, bias, delayed=False, revise=False, jerk=None):
        for dt in np.arange(-.18, .181, .002):
            estimator.record_arm(event+dt, np.zeros(7), np.zeros(7))
        estimator.observe(event-.04, state, np.nan)
        for dt in np.arange(-.17, .171, .005):
            velocity = incoming if dt < 0 else incoming+nominal_kick+(j/8)*bias
            p = np.array([0., 0., cfg['radius']])+velocity*dt+.5*acceleration*dt*dt
            if jerk is not None:
                p += np.asarray(jerk)*(dt*dt*dt/6.)
            arrival = event+dt+.01 if not delayed or dt < 0 else event+1.
            estimator.record_position(event+dt, p, arrival)
        # The event is reported 3 ms late; continuity should refine it back.
        estimator.observe(event+.003, state, event+.003)
        reported = event+.003
        if revise:
            estimator.observe(event+.01, state, reported+.0002)
            estimator.observe(event+.02, state, reported-.0001)
            reported -= .0001
        before = estimator.bias_per_j8_local
        assert estimator.observe(event+.15, state, reported) is None
        assert np.array_equal(before, estimator.bias_per_j8_local)
        return estimator.observe(event+.19, state, reported)
    nominal = ContactBiasEstimator(robot, cfg)
    result = feed(nominal, 2., np.zeros(3))
    assert result['accepted'], result
    assert np.linalg.norm(nominal.bias_per_j8_local) < 1e-8
    assert abs(result['impact_time_correction_s']+.003) < 1e-8
    desired = np.array([.08, -.04, -.05])
    adaptive = ContactBiasEstimator(robot, cfg)
    for index in range(40):
        result = feed(adaptive, 2.+index, desired)
        assert result['accepted'], result
    assert np.linalg.norm(adaptive.bias_per_j8_local-desired) < .001
    withheld = ContactBiasEstimator(robot, cfg)
    result = feed(withheld, 2., desired, delayed=True)
    assert not result['accepted'] and result['reason'] == 'insufficient_delivered_positions'
    assert np.array_equal(withheld.bias_per_j8_local, np.zeros(3))
    revised = ContactBiasEstimator(robot, cfg)
    result = feed(revised, 2., np.zeros(3), revise=True)
    assert result['accepted'] and revised.accepted_samples == len(revised.records) == 1
    assert abs(result['observer_impact_time_s']-2.0029) < 1e-12
    assert np.linalg.norm(revised.bias_per_j8_local) < 1e-8
    # Revisions after processing also must not create a second fit.
    assert revised.observe(2.25, state, 2.0031) is None
    assert revised.observe(2.26, state, 2.0028) is None
    assert revised.accepted_samples == len(revised.records) == 1
    # Distinct close contacts still fail isolation rather than being deduplicated.
    close = ContactBiasEstimator(robot, cfg)
    close.observe(2., state, 2.)
    close.observe(2.1, state, 2.1)
    close.observe(2.18, state, 2.1)
    close.observe(2.28, state, 2.1)
    assert len(close.records) == 2 and close.accepted_samples == 0
    assert all(record['reason'] == 'nonisolated_observed_contact' for record in close.records)
    # A known nonzero acceleration derivative is represented exactly by cubic
    # fitting; retain the same delay, revision and impact-time refinement logic.
    cubic = ContactBiasEstimator(robot, cfg, polynomial_degree=3)
    jerk = np.array([.1, -.2, 3.5])
    result = feed(cubic, 2., np.zeros(3), revise=True, jerk=jerk)
    assert result['accepted'] and result['polynomial_degree'] == 3, result
    cubic_nominal_norm = float(np.linalg.norm(cubic.bias_per_j8_local))
    assert cubic_nominal_norm < 1e-8
    result = feed(cubic, 3., desired, jerk=jerk)
    assert result['accepted'], result
    assert np.linalg.norm(np.array(result['bias_sample_per_j8_local'])-desired) < 1e-8
    assert np.linalg.norm(cubic.bias_per_j8_local-cubic.gain*desired) < 1e-8
    result = feed(cubic, 4., desired, delayed=True, jerk=jerk)
    assert not result['accepted'] and result['reason'] == 'insufficient_delivered_positions'
    assert cubic.accepted_samples == 2
    json.dumps(adaptive.records+withheld.records+revised.records+close.records+cubic.records,
               allow_nan=False)
    return dict(nominal_bias_norm=float(np.linalg.norm(nominal.bias_per_j8_local)),
                target_bias=desired.tolist(), estimated_bias=adaptive.bias_per_j8_local.tolist(),
                accepted=adaptive.accepted_samples, future_delivery_rejected=True,
                nominal_time_refinement_s=nominal.records[0]['impact_time_correction_s'],
                revised_event_fits=len(revised.records), distinct_close_contacts_rejected=True,
                cubic_nominal_bias_norm=cubic_nominal_norm,
                cubic_known_jerk=jerk.tolist(), cubic_known_bias_sample=cubic.records[1]['bias_sample_per_j8_local'])


if __name__ == '__main__':
    print(json.dumps(_analytic_selftest(), indent=2, allow_nan=False))
