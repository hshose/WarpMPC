"""Batched draws from the measured October 6 uncertainty, with explicit inflation.

Physical parameter means are unchanged. Ball/contact/sensing uncertainty and
servo lag/dead-time deviations have independently selected multipliers.
Entire acceleration profiles and paired clock/delivery rows retain correlation.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from uncertainty_settings import settings


class EmpiricalBatch:
    def __init__(self, path, batch, seed=2026, multiplier=1.5, arm_multiplier=None):
        if batch < 1 or multiplier <= 0:
            raise ValueError('Positive batch size and uncertainty multiplier required')
        self.model = json.loads(Path(path).read_text())
        self.spec = self.model['uncertainty']
        self.batch = batch
        self.multipliers = settings(multiplier,arm_multiplier)
        self.arm_multiplier = self.multipliers["arm_following"]
        self.multiplier = multiplier
        self.expansion = multiplier * self.spec.get('residual_expansion_factor', 1.)
        self.seed = int(seed)
        self.counters = {}
        self.blocks = {}
        self.run_acceleration = self.residual('run_acceleration_samples_m_s2', batch)
        self.run_impact = self.residual('run_impact_velocity_samples_local_m_s', batch)
        self.flight_base = np.zeros((batch, 3))
        self.impact = np.zeros((batch, 3))
        self.flight_start = np.zeros(batch)
        profiles = self.spec['within_flight_acceleration_profiles_m_s2']
        self.lengths = np.array([len(p) for p in profiles])
        self.profiles = np.zeros((len(profiles), max(self.lengths), 3))
        for i, profile in enumerate(profiles):
            self.profiles[i, :len(profile)] = profile
            self.profiles[i, len(profile):] = profile[-1]
        self.profile_id = np.zeros(batch, dtype=int)
        self.last_arrival = np.full(batch, -np.inf)
        self.new_flight(np.arange(batch), 0., initial=True)

    def indices(self,key,size,worlds):
        """Independent counter stream per world/channel, unaffected by other hits."""
        worlds=np.asarray(worlds,dtype=int)
        counter=self.counters.setdefault(key,np.zeros(self.batch,dtype=np.uint64))
        salt=np.uint64(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
        value=(np.uint64(self.seed)^salt^((worlds.astype(np.uint64)+1)*np.uint64(0x9E3779B97F4A7C15))
               ^((counter[worlds]+1)*np.uint64(0xD1B54A32D192ED03)))
        value=(value^(value>>30))*np.uint64(0xBF58476D1CE4E5B9)
        value=(value^(value>>27))*np.uint64(0x94D049BB133111EB)
        value=value^(value>>31);counter[worlds]+=1
        return (value%np.uint64(size)).astype(int)

    def row(self, key, count=None, blocked=False, worlds=None):
        count = self.batch if count is None else count
        rows = np.asarray(self.spec[key], dtype=float)
        if not blocked:
            return rows[self.indices(key,len(rows),np.arange(count) if worlds is None else worlds)]
        # Independent block cursors per world, preserving measured frame order.
        if count != self.batch:
            raise ValueError('Blocked draws require the full batch')
        index, left = self.blocks.get(key, (np.zeros(count, int), np.zeros(count, int)))
        renew = left == 0
        index[renew] = self.indices(key,len(rows),np.flatnonzero(renew))
        left[renew] = int(self.spec.get('bootstrap_block_frames', 1))
        result = rows[index % len(rows)]
        self.blocks[key] = (index + 1, left - 1)
        return result

    def residual(self, key, count, worlds=None):
        mean = np.asarray(self.spec.get(key.replace('_samples', '_mean'), [0., 0., 0.]))
        return mean + self.expansion * self.row(key, count, worlds=worlds)

    def new_flight(self, indices, time, initial=False):
        indices = np.asarray(indices, dtype=int)
        key = ('initial_flight_acceleration_samples_m_s2' if initial else 'flight_acceleration_samples_m_s2')
        self.flight_base[indices] = self.run_acceleration[indices] + self.residual(key, len(indices),indices)
        self.impact[indices] = self.run_impact[indices] + self.residual('impact_velocity_samples_local_m_s', len(indices),indices)
        self.profile_id[indices] = self.indices('profile_id',len(self.profiles),indices)
        self.flight_start[indices] = time

    def acceleration(self, time):
        index = np.maximum(0., (time - self.flight_start) / self.spec['within_flight_profile_dt_s'])
        index = np.minimum(index, self.lengths[self.profile_id] - 1)
        lower = index.astype(int)
        upper = np.minimum(lower + 1, self.lengths[self.profile_id] - 1)
        fraction = (index - lower)[:, None]
        residual = ((1 - fraction) * self.profiles[self.profile_id, lower]
                    + fraction * self.profiles[self.profile_id, upper])
        return self.flight_base + self.expansion * residual

    def measurement(self, time, position):
        error, delay = (self.multiplier * self.row('capture_error_receive_delay_samples_s', blocked=True)).T
        if np.any(delay < 0):
            raise ValueError('Invalid negative empirical delivery delay')
        arrival = np.maximum(time + delay, self.last_arrival + 1e-9)
        self.last_arrival = arrival
        # White position residuals MUST be iid, unlike clock/delivery block rows.
        point = np.asarray(position) + self.multiplier * self.row('noise_position_samples_m')
        return time + error, arrival, point

    def plan_delay(self):
        return self.multiplier * self.row('plan_delay_samples_s', blocked=True)

    def parameter_draws(self):
        plant = self.model['plant']
        samples = self.spec['plant_parameter_samples']
        rows = self.indices('plant_parameters',len(samples),np.arange(self.batch))
        result = {}
        for key in ('servo_lag_s', 'servo_dead_time_s', 'center_offset_local_m'):
            mean = np.asarray(plant[key])
            values = np.asarray([samples[i].get(key, plant[key]) for i in rows])
            factor=self.arm_multiplier if key in ("servo_lag_s","servo_dead_time_s") else self.multiplier
            result[key] = mean + factor * (values - mean)
        if np.any(result['servo_lag_s'] <= 0) or np.any(result['servo_dead_time_s'] < 0):
            raise ValueError('Inflated servo samples leave the physical domain')
        return result
