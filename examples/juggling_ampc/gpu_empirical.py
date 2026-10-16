"""Measured plant draws and inexpensive noisy truth observations on the GPU.

The plant uncertainty follows ``EmpiricalBatch``: fixed run effects, new flight
and rebound draws at each contact, and interpolation of measured acceleration
profiles. All evolving state and outputs are preallocated Warp arrays.

There is deliberately no observer here. At each MPC update, ball position gets
an iid measured Vicon residual. Paired clock/delivery rows retain their measured
block ordering and perturb the true ball state by a small time shift. The mean
delivery delay is treated as compensated, as in a predictor; only its measured
variation and capture timestamp error remain. Position and velocity are shifted
with the current physical flight acceleration. This is a cheap approximation
of sensor-induced state error, not a fitted observer-error distribution. Joint
positions, joint velocities, and spin remain the exact simulation state.
"""

import hashlib

import numpy as np
import warp as wp


@wp.func
def _sample_index(seed: wp.uint64, salt: wp.uint64, world: int,
                  counter: wp.uint64, size: int):
    # Same independent world/channel counter hash as EmpiricalBatch.indices.
    value = (seed ^ salt
             ^ (wp.uint64(world + 1) * wp.uint64(0x9E3779B97F4A7C15))
             ^ ((counter + wp.uint64(1)) * wp.uint64(0xD1B54A32D192ED03)))
    value = (value ^ (value >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    value = value ^ (value >> wp.uint64(31))
    return int(value % wp.uint64(size))


@wp.kernel
def _update_flight(
    time: float, counts: wp.array(dtype=int), previous: wp.array(dtype=int),
    first_contact: wp.array(dtype=float),
    run_acceleration: wp.array2d(dtype=float), run_impact: wp.array2d(dtype=float),
    flight_rows: wp.array2d(dtype=float), impact_rows: wp.array2d(dtype=float),
    profiles: wp.array3d(dtype=float), profile_lengths: wp.array(dtype=int),
    flight_mean: wp.vec3, impact_mean: wp.vec3, expansion: float, profile_dt: float,
    seed: wp.uint64, salts: wp.array(dtype=wp.uint64),
    counters: wp.array2d(dtype=wp.uint64),
    flight_base: wp.array2d(dtype=float), profile_id: wp.array(dtype=int),
    flight_start: wp.array(dtype=float),
    acceleration: wp.array2d(dtype=float), rebound: wp.array2d(dtype=float),
):
    world = wp.tid()
    if counts[world] != previous[world]:
        if previous[world] == 0:
            first_contact[world] = time
        previous[world] = counts[world]
        fi = _sample_index(seed, salts[0], world, counters[world, 0], flight_rows.shape[0])
        ii = _sample_index(seed, salts[1], world, counters[world, 1], impact_rows.shape[0])
        pi = _sample_index(seed, salts[2], world, counters[world, 2], profiles.shape[0])
        for channel in range(3):
            counters[world, channel] += wp.uint64(1)
            flight_base[world, channel] = (run_acceleration[world, channel]
                + flight_mean[channel] + expansion * flight_rows[fi, channel])
            rebound[world, channel] = (run_impact[world, channel]
                + impact_mean[channel] + expansion * impact_rows[ii, channel])
        profile_id[world] = pi
        flight_start[world] = time

    profile = profile_id[world]
    end = profile_lengths[profile] - 1
    phase = wp.clamp((time - flight_start[world]) / profile_dt, 0.0, float(end))
    lower = int(phase)
    upper = wp.min(lower + 1, end)
    fraction = phase - float(lower)
    for channel in range(3):
        residual = ((1.0 - fraction) * profiles[profile, lower, channel]
                    + fraction * profiles[profile, upper, channel])
        acceleration[world, channel] = flight_base[world, channel] + expansion * residual


@wp.kernel
def _truth_observation(
    truth: wp.array2d(dtype=float), acceleration: wp.array2d(dtype=float),
    position_rows: wp.array2d(dtype=float), clock_rows: wp.array2d(dtype=float),
    clock_index: wp.array(dtype=int), clock_remaining: wp.array(dtype=int),
    camera_frames: int, block_frames: int, mean_delay: float,
    multiplier: float, gravity: wp.vec3, drag: float, quadratic_drag: float,
    seed: wp.uint64, salts: wp.array(dtype=wp.uint64),
    counters: wp.array2d(dtype=wp.uint64), observed: wp.array2d(dtype=float),
):
    world = wp.tid()
    # Advance the measured frame sequence at its original camera cadence,
    # even though only the final frame's residual is needed by the MPC.
    clock = int(0)
    for frame in range(camera_frames):
        if clock_remaining[world] == 0:
            clock_index[world] = _sample_index(
                seed, salts[4], world, counters[world, 4], clock_rows.shape[0])
            counters[world, 4] += wp.uint64(1)
            clock_remaining[world] = block_frames
        clock = clock_index[world] % clock_rows.shape[0]
        clock_index[world] += 1
        clock_remaining[world] -= 1

    position = _sample_index(seed, salts[3], world, counters[world, 3], position_rows.shape[0])
    counters[world, 3] += wp.uint64(1)
    # A positive capture timestamp error makes the predicted state too young.
    # Extra delivery delay has the same sign after nominal-delay compensation.
    time_shift = -multiplier * (clock_rows[clock, 0] + clock_rows[clock, 1] - mean_delay)
    velocity = wp.vec3(truth[world, 3], truth[world, 4], truth[world, 5])
    physical_acceleration = (gravity
        + wp.vec3(acceleration[world, 0], acceleration[world, 1], acceleration[world, 2])
        - drag * velocity - quadratic_drag * wp.length(velocity) * velocity)
    for channel in range(23):
        observed[world, channel] = truth[world, channel]
    for channel in range(3):
        observed[world, channel] += (multiplier * position_rows[position, channel]
            + velocity[channel] * time_shift
            + 0.5 * physical_acceleration[channel] * time_shift * time_shift)
        observed[world, channel + 3] += physical_acceleration[channel] * time_shift


class GPUEmpirical:
    """GPU continuation of an initialized :class:`EmpiricalBatch`.

    Call ``update(t, plant.counts)`` before each physics step and call
    ``truth_observation(plant.observe(), t)`` only at MPC boundaries. The latter
    returns the persistent ``observed_state`` buffer, suitable for zero-copy
    ``wp.to_jax`` use. Neither method copies to the host or allocates arrays.
    """

    def __init__(self, draws, device="cuda:0"):
        self.batch = draws.batch
        self.device = device
        self.seed = wp.uint64(draws.seed)
        self.multiplier = float(draws.multiplier)
        self.expansion = float(draws.expansion)
        spec = draws.spec
        plant = draws.model["plant"]

        def array(values, dtype=float):
            numpy_dtype = np.float32 if dtype is float else np.int32 if dtype is int else np.uint64
            return wp.array(np.asarray(values, dtype=numpy_dtype), dtype=dtype, device=device)

        self.run_acceleration = array(draws.run_acceleration)
        self.run_impact = array(draws.run_impact)
        self.flight_base = array(draws.flight_base)
        self.rebound = array(draws.impact)
        self.flight_start = array(draws.flight_start)
        self.profile_id = array(draws.profile_id, int)
        self.profiles = array(draws.profiles)
        self.profile_lengths = array(draws.lengths, int)
        self.flight_rows = array(spec["flight_acceleration_samples_m_s2"])
        self.impact_rows = array(spec["impact_velocity_samples_local_m_s"])
        self.position_rows = array(spec["noise_position_samples_m"])
        self.clock_rows = array(spec["capture_error_receive_delay_samples_s"])
        self.flight_mean = wp.vec3(*spec.get("flight_acceleration_mean_m_s2", [0., 0., 0.]))
        self.impact_mean = wp.vec3(*spec.get("impact_velocity_mean_local_m_s", [0., 0., 0.]))
        self.profile_dt = float(spec["within_flight_profile_dt_s"])
        self.block_frames = max(1, int(spec.get("bootstrap_block_frames", 1)))
        self.mean_delay = float(np.mean(np.asarray(spec["capture_error_receive_delay_samples_s"])[:, 1]))
        self.camera_hz = float(draws.model.get("vicon_rate_hz", 100.))
        self.last_camera_frame = -1
        self.gravity = wp.vec3(*plant["gravity_m_s2"])
        self.drag = float(plant["drag_s_inv"])
        self.quadratic_drag = float(plant["quadratic_drag_m_inv"])

        keys = ("flight_acceleration_samples_m_s2", "impact_velocity_samples_local_m_s",
                "profile_id", "noise_position_samples_m", "capture_error_receive_delay_samples_s")
        self.salts = array([int.from_bytes(hashlib.sha256(k.encode()).digest()[:8], "little")
                            for k in keys], wp.uint64)
        self.counters = array(np.stack([draws.counters.get(k, np.zeros(self.batch, np.uint64))
                                         for k in keys], axis=1), wp.uint64)
        self.counts = wp.zeros(self.batch, dtype=int, device=device)
        self.first_contact_time = array(np.full(self.batch, np.inf))
        self.acceleration = array(draws.acceleration(0.))
        self.observed_state = wp.empty((self.batch, 23), dtype=float, device=device)
        self.clock_index = wp.zeros(self.batch, dtype=int, device=device)
        self.clock_remaining = wp.zeros(self.batch, dtype=int, device=device)

    def update(self, time, counts):
        wp.launch(_update_flight, self.batch, inputs=[
            float(time), counts, self.counts, self.first_contact_time,
            self.run_acceleration, self.run_impact, self.flight_rows, self.impact_rows,
            self.profiles, self.profile_lengths, self.flight_mean, self.impact_mean,
            self.expansion, self.profile_dt, self.seed, self.salts, self.counters,
            self.flight_base, self.profile_id, self.flight_start,
            self.acceleration, self.rebound], device=self.device)
        return self.acceleration, self.rebound

    def truth_observation(self, truth, time):
        frame = int(round(float(time) * self.camera_hz))
        frames = max(1, frame - self.last_camera_frame)
        self.last_camera_frame = frame
        wp.launch(_truth_observation, self.batch, inputs=[
            truth, self.acceleration, self.position_rows, self.clock_rows,
            self.clock_index, self.clock_remaining, frames, self.block_frames,
            self.mean_delay, self.multiplier, self.gravity, self.drag, self.quadratic_drag,
            self.seed, self.salts, self.counters, self.observed_state], device=self.device)
        return self.observed_state
