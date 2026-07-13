#!/usr/bin/env python3
"""Batched TuRBO-style tuning of a pure-quadratic identified-parameter cartpole MPC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from benchmarks.jax_cache import configure_jax_compilation_cache

    configure_jax_compilation_cache()
except Exception:
    pass

import jax
import jax.numpy as jnp

from warpmpc.jax_ampc import sample_scaled_gaussian_noise
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import MPAXSettings, FilterLineSearchSettings, build_sparse_mpc_plan, compile_sparse_mpc_sqp

from benchmarks.problems.cartpole_physical_quadratic import (
    CARTPOLE_COST_PARAMETER_NAMES,
    CARTPOLE_COST_PARAM_LOWER,
    CARTPOLE_COST_PARAM_UPPER,
    CARTPOLE_M_TIP,
    CARTPOLE_NOMINAL_COST_PARAMS,
    CARTPOLE_NX,
    CARTPOLE_NU,
    CARTPOLE_NZ,
    CARTPOLE_N_COST_PARAMS,
    CARTPOLE_N_STEPS,
    CARTPOLE_RAIL_LIMIT,
    CARTPOLE_U_MAX,
    CARTPOLE_U_MIN,
    cartpole_cost_function_description,
    cartpole_warm_start_description,
    cost_params_to_log,
    format_cost_parameters,
    make_cartpole_initialization_kernel,
    make_cartpole_sqp_problem,
    update_cartpole_params_initial_state,
)


STATE_PLOT_LABELS = ("x [m]", "v [m/s]", "theta [rad]", "omega [rad/s]")
STATE_PLOT_YLIMS = (
    (-0.5, 0.5),
    (-6.0, 6.0),
    (-2.0 * np.pi, 2.0 * np.pi),
    (-18.0, 18.0),
)
DEFAULT_PROCESS_NOISE_SCALE = "0.0005,0.01,0.001,0.02"
DEFAULT_INPUT_NOISE_SCALE = "0.05"
RAIL_RATE_PENALTY = 40.0
RAIL_MAX_PENALTY = 40.0
RAIL_MEAN_SQUARED_PENALTY = 150.0
SUCCESS_ANGLE_BOUND = float(np.deg2rad(10.0))
SYSID_ENV_PARAMETER_NAMES = ("M_CART", "M_ROD", "AB", "AC", "B_EQ", "B_P", "L_ROD")
SYSID_ENV_PARAMETER_TABLE = (
    ("M_CART", 0.5, 0.4472965, 0.03673086, 0.4105657, 0.4840274, 0.4105671, 0.4840251),
    ("M_ROD", 0.1, 0.01840051, 0.004307272, 0.01409324, 0.02270778, 0.01409341, 0.02270763),
    ("AB", -4.0, -1.959361, 0.1772289, -2.13659, -1.782132, -2.136574, -1.782149),
    ("AC", 1.5, 0.5374264, 0.1123448, 0.4250816, 0.6497712, 0.425084, 0.6497689),
    ("B_EQ", 4.0, 1.959361, 0.1772289, 1.782132, 2.13659, 1.782149, 2.136574),
    ("B_P", 0.01, 0.00172243, 0.0005311022, 0.001191328, 0.002253532, 0.001191324, 0.002253513),
    ("L_ROD", 0.7, 0.6696171, 0.04486594, 0.6247511, 0.714483, 0.6247518, 0.7144823),
)
SYSID_ENV_PARAMETER_NOMINAL = np.asarray([row[2] for row in SYSID_ENV_PARAMETER_TABLE], dtype=np.float64)
SYSID_ENV_PARAMETER_SEG_MIN = np.asarray([row[6] for row in SYSID_ENV_PARAMETER_TABLE], dtype=np.float64)
SYSID_ENV_PARAMETER_SEG_MAX = np.asarray([row[7] for row in SYSID_ENV_PARAMETER_TABLE], dtype=np.float64)
SYSID_INPUT_DISTURBANCE_BOUND = 0.478184104


class CartpoleBatchOutput(NamedTuple):
    states: object
    commanded_actions: object
    applied_actions: object
    rollout_returns: object
    rollout_success: object
    rollout_rail_violation: object
    experiment_returns: object
    experiment_success_rates: object
    experiment_rail_violation_rates: object
    step_lengths: object
    constraint_violations: object
    line_search_accepted: object
    prim_res: object
    dual_res: object
    sqp_finite: object


@dataclass
class TurboBatchOptimizer:
    """Small self-contained TuRBO-style optimizer for q-way batches."""

    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray
    batch_size: int
    seed: int
    candidate_pool: int
    max_gp_points: int
    length: float = 0.8
    length_min: float = 0.01
    length_max: float = 1.0
    success_tolerance: int = 3
    failure_tolerance: int = 4
    best_value: float = -np.inf
    success_counter: int = 0
    failure_counter: int = 0

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        self.nominal = np.asarray(self.nominal, dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)
        self.history_x: list[np.ndarray] = []
        self.history_y: list[np.ndarray] = []
        self._ask_count = 0

    def ask(self) -> np.ndarray:
        if not self.history_y:
            points = self._initial_batch()
        else:
            points = self._turbo_batch()
        self._ask_count += 1
        return np.clip(points, self.lower, self.upper)

    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.shape != (y.shape[0], self.lower.size):
            raise ValueError(f"bad x/y shapes: {x.shape}, {y.shape}")
        previous_best = self.best_value
        batch_best = float(np.max(y))
        self.history_x.append(x.copy())
        self.history_y.append(y.copy())
        if batch_best > self.best_value:
            self.best_value = batch_best
        if np.isfinite(previous_best):
            if batch_best > previous_best + 1.0e-6 * max(1.0, abs(previous_best)):
                self.success_counter += 1
                self.failure_counter = 0
            else:
                self.failure_counter += 1
                self.success_counter = 0
            if self.success_counter >= self.success_tolerance:
                self.length = min(2.0 * self.length, self.length_max)
                self.success_counter = 0
            if self.failure_counter >= self.failure_tolerance:
                self.length = max(0.5 * self.length, self.length_min)
                self.failure_counter = 0

    @property
    def history_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.history_y:
            return (
                np.empty((0, self.lower.size), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        return np.vstack(self.history_x), np.concatenate(self.history_y)

    def best(self) -> tuple[np.ndarray, float]:
        x, y = self.history_arrays
        if y.size == 0:
            return self.nominal.copy(), float("nan")
        index = int(np.argmax(y))
        return x[index].copy(), float(y[index])

    def _initial_batch(self) -> np.ndarray:
        points = self._sobol(self.batch_size, self.lower.size)
        points = self.lower[None, :] + points * (self.upper - self.lower)[None, :]
        points[0] = self.nominal
        return points

    def _turbo_batch(self) -> np.ndarray:
        x_hist, y_hist = self.history_arrays
        best_index = int(np.argmax(y_hist))
        center = x_hist[best_index]
        center_unit = self._normalize(center)
        train_x, train_y = self._training_subset(x_hist, y_hist, center)

        pool = max(self.candidate_pool, 4 * self.batch_size)
        unit = self._sobol(pool, self.lower.size)
        trust = center_unit[None, :] + (unit - 0.5) * self.length
        perturb_prob = min(1.0, 20.0 / self.lower.size)
        mask = self.rng.random((pool, self.lower.size)) < perturb_prob
        empty = ~np.any(mask, axis=1)
        if np.any(empty):
            mask[empty, self.rng.integers(0, self.lower.size, size=int(np.sum(empty)))] = True
        candidates_unit = np.where(mask, trust, center_unit[None, :])
        candidates_unit = np.clip(candidates_unit, 0.0, 1.0)
        candidates = self._unnormalize(candidates_unit)

        scores = self._thompson_scores(train_x, train_y, candidates)
        chosen = np.argpartition(scores, -self.batch_size)[-self.batch_size :]
        chosen = chosen[np.argsort(scores[chosen])[::-1]]
        return candidates[chosen]

    def _training_subset(
        self,
        x_hist: np.ndarray,
        y_hist: np.ndarray,
        center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_hist.size <= self.max_gp_points:
            return x_hist, y_hist
        unit = self._normalize(x_hist)
        center_unit = self._normalize(center)
        distance = np.linalg.norm(unit - center_unit[None, :], axis=1)
        local_count = min(self.max_gp_points // 2, y_hist.size)
        elite_count = self.max_gp_points - local_count
        local = np.argpartition(distance, local_count - 1)[:local_count]
        elite = np.argpartition(y_hist, -elite_count)[-elite_count:]
        index = np.unique(np.concatenate([local, elite]))
        if index.size > self.max_gp_points:
            index = index[np.argsort(y_hist[index])[-self.max_gp_points :]]
        return x_hist[index], y_hist[index]

    def _thompson_scores(self, train_x: np.ndarray, train_y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        try:
            from scipy.linalg import cho_factor, cho_solve
        except Exception:
            return self.rng.standard_normal(candidates.shape[0])

        x_train = self._normalize(train_x)
        x_cand = self._normalize(candidates)
        y_mean = float(np.mean(train_y))
        y_std = float(np.std(train_y))
        y_std = y_std if y_std > 1.0e-12 else 1.0
        y_scaled = (train_y - y_mean) / y_std
        lengthscale = self._median_lengthscale(x_train)
        k_xx = self._rbf_kernel(x_train, x_train, lengthscale)
        k_xx.flat[:: k_xx.shape[0] + 1] += 1.0e-5
        try:
            factor = cho_factor(k_xx, lower=True, check_finite=False)
            alpha = cho_solve(factor, y_scaled, check_finite=False)
            k_xs = self._rbf_kernel(x_train, x_cand, lengthscale)
            mean = k_xs.T @ alpha
            solve = cho_solve(factor, k_xs, check_finite=False)
            var = np.maximum(1.0 - np.sum(k_xs * solve, axis=0), 1.0e-9)
            return mean + np.sqrt(var) * self.rng.standard_normal(candidates.shape[0])
        except Exception:
            return self.rng.standard_normal(candidates.shape[0])

    def _median_lengthscale(self, x_unit: np.ndarray) -> float:
        if x_unit.shape[0] < 2:
            return 0.2
        sample = x_unit
        if sample.shape[0] > 256:
            index = self.rng.choice(sample.shape[0], size=256, replace=False)
            sample = sample[index]
        diff = sample[:, None, :] - sample[None, :, :]
        distance = np.sqrt(np.sum(diff * diff, axis=2))
        distance = distance[np.triu_indices(distance.shape[0], k=1)]
        distance = distance[distance > 1.0e-12]
        if distance.size == 0:
            return 0.2
        return float(np.clip(np.median(distance), 0.05, 1.0))

    @staticmethod
    def _rbf_kernel(xa: np.ndarray, xb: np.ndarray, lengthscale: float) -> np.ndarray:
        diff = xa[:, None, :] - xb[None, :, :]
        return np.exp(-0.5 * np.sum(diff * diff, axis=2) / (lengthscale**2))

    def _sobol(self, n: int, dim: int) -> np.ndarray:
        try:
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(0, 2**31 - 1)))
            power = int(math.ceil(math.log2(max(1, n))))
            return sampler.random_base2(power)[:n]
        except Exception:
            return self.rng.random((n, dim))

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.lower) / (self.upper - self.lower)

    def _unnormalize(self, x_unit: np.ndarray) -> np.ndarray:
        return self.lower[None, :] + np.asarray(x_unit, dtype=np.float64) * (self.upper - self.lower)[None, :]


def _parse_scale(text: str, dim: int, *, name: str) -> np.ndarray:
    values = np.asarray([float(item) for item in text.split(",") if item.strip()], dtype=np.float64)
    if values.size == 1:
        values = np.full((dim,), float(values[0]), dtype=np.float64)
    if values.shape != (dim,):
        raise ValueError(f"{name} needs {dim} comma-separated values or one scalar, got {text!r}")
    return values


def _sample_initial_states(args: argparse.Namespace, batch_size: int, seed: int, dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = np.zeros((batch_size, CARTPOLE_NX), dtype=dtype)
    angle_spread = np.deg2rad(args.initial_angle_spread_deg)
    x0[:, 0] = rng.uniform(-args.initial_position_range, args.initial_position_range, size=batch_size)
    x0[:, 1] = rng.normal(0.0, args.initial_velocity_std, size=batch_size)
    downward = rng.choice(np.asarray([-np.pi, np.pi]), size=batch_size)
    x0[:, 2] = downward + rng.uniform(-angle_spread, angle_spread, size=batch_size)
    x0[:, 3] = rng.normal(0.0, args.initial_omega_std, size=batch_size)
    return x0.astype(dtype, copy=False)


def _sample_env_parameters(batch_size: int, seed: int, dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.uniform(
        SYSID_ENV_PARAMETER_SEG_MIN[None, :],
        SYSID_ENV_PARAMETER_SEG_MAX[None, :],
        size=(batch_size, len(SYSID_ENV_PARAMETER_NAMES)),
    )
    return values.astype(dtype, copy=False)


def _nominal_env_parameters(batch_size: int, dtype: np.dtype) -> np.ndarray:
    nominal = SYSID_ENV_PARAMETER_NOMINAL.astype(dtype, copy=False)
    return np.broadcast_to(nominal[None, :], (batch_size, nominal.size)).copy()


def _format_sysid_env_parameter_table() -> str:
    lines = [
        "name prior nominal radius lower upper seg_min seg_max",
    ]
    for row in SYSID_ENV_PARAMETER_TABLE:
        name, prior, nominal, radius, lower, upper, seg_min, seg_max = row
        lines.append(
            f"{name}: prior={prior:.8g}, nominal={nominal:.8g}, radius={radius:.8g}, "
            f"lower={lower:.8g}, upper={upper:.8g}, seg_min={seg_min:.8g}, seg_max={seg_max:.8g}"
        )
    return "\n".join(lines)


def _make_sqp(args: argparse.Namespace, dtype: np.dtype):
    settings = OSQPSettings(
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        max_iter=args.osqp_max_iter,
        scaling=0,
        adaptive_rho=False,
        rho_is_vec=False,
        check_termination=0,
        warm_starting=True,
        polishing=False,
    )
    line_search_settings = FilterLineSearchSettings(line_search_step_min=args.line_search_step_min)
    problem = make_cartpole_sqp_problem(
        args.horizon_steps,
        rail_constraint=args.enable_rail_constraint,
    )
    plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
    sqp = compile_sparse_mpc_sqp(
        problem,
        plan,
        dtype=dtype,
        qp_solver=getattr(args, "qp_solver", "jax_osqp"),
        osqp_settings=settings,
        mpax_settings=MPAXSettings(
            eps_abs=getattr(args, "mpax_eps_abs", 1e-3),
            eps_rel=getattr(args, "mpax_eps_rel", 1e-3),
            iteration_limit=getattr(args, "mpax_iteration_limit", None) or args.osqp_max_iter,
            termination_evaluation_frequency=getattr(
                args,
                "mpax_termination_evaluation_frequency",
                100,
            ),
            l_inf_ruiz_iterations=getattr(args, "mpax_l_inf_ruiz_iterations", 10),
            pock_chambolle_alpha=getattr(args, "mpax_pock_chambolle_alpha", 1.0),
            regularization=getattr(args, "mpax_regularization", 0.0),
            unroll=getattr(args, "mpax_unroll", False),
        ),
        transpose_work=getattr(args, "transpose_work", True),
        segmented=getattr(args, "segmented", True),
        segment_budget=args.segment_budget,
        segment_strategy=args.segment_strategy,
        level_scheduled_solve=getattr(args, "level_scheduled_solve", False),
        level_scheduled_solve_threshold=getattr(args, "level_scheduled_solve_threshold", 1),
        qdldl_backend=getattr(args, "qdldl_backend", "jax"),
        qdldl_factor_backend=getattr(args, "qdldl_factor_backend", None),
        qdldl_solve_backend=getattr(args, "qdldl_solve_backend", None),
        line_search_settings=line_search_settings,
        group_repeated_stages=args.group_repeated_stages,
    )
    return plan, sqp


def _make_rollout(args: argparse.Namespace, sqp, dtype: np.dtype):
    jdtype = jnp.dtype(dtype)
    process_scale = jnp.asarray(
        _parse_scale(args.process_noise_scale, CARTPOLE_NX, name="process noise scale"),
        dtype=jdtype,
    )
    input_scale = jnp.asarray(
        _parse_scale(args.input_noise_scale, CARTPOLE_NU, name="input noise scale"),
        dtype=jdtype,
    )
    control_dt = jnp.asarray(args.control_dt, dtype=jdtype)
    input_disturbance_bound = jnp.asarray(args.input_disturbance_bound, dtype=jdtype)
    substeps = int(args.integrator_substeps)
    rollout_steps = int(args.rollout_steps)
    horizon_steps = int(args.horizon_steps)
    sqp_iterations = int(args.sqp_iterations)
    num_experiments = int(args.experiments_per_episode)
    rollouts_per_experiment = int(args.rollouts_per_experiment)

    def env_cartpole_jax_dynamics(x, u, env_params):
        m_cart = env_params[:, 0]
        m_rod = env_params[:, 1]
        ab = env_params[:, 2]
        ac = env_params[:, 3]
        b_eq = env_params[:, 4]
        b_p = env_params[:, 5]
        l_rod = env_params[:, 6]
        m_tip = jnp.asarray(CARTPOLE_M_TIP, dtype=jdtype)
        m_pole = m_rod + m_tip
        center_of_mass = (m_rod * (0.5 * l_rod) + m_tip * l_rod) / m_pole
        pole_inertia_pivot = m_rod * l_rod**2 / jnp.asarray(3.0, dtype=jdtype) + m_tip * l_rod**2
        pole_inertia_com = pole_inertia_pivot - m_pole * center_of_mass**2

        v = x[:, 1]
        theta = x[:, 2]
        omega = x[:, 3]
        force = (ab - b_eq) * v + ac * u[:, 0]

        h1 = m_cart + m_pole
        h2 = m_pole * center_of_mass
        h4 = m_pole * center_of_mass**2 + pole_inertia_com
        h7 = m_pole * center_of_mass * jnp.asarray(9.81, dtype=jdtype)
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        denominator = h2**2 * cos_theta**2 - h1 * h4
        vdot = (
            h2 * h4 * omega**2 * sin_theta
            - h2 * h7 * cos_theta * sin_theta
            + h4 * force
            - h2 * cos_theta * b_p * omega
        ) / (-denominator)
        omegadot = (
            h2**2 * omega**2 * cos_theta * sin_theta
            - h1 * h7 * sin_theta
            + h2 * cos_theta * force
            + h1 * b_p * omega
        ) / denominator
        return jnp.stack((v, vdot, omega, omegadot), axis=1)

    def env_cartpole_jax_euler_step(x, u, env_params, dt):
        return x + dt * env_cartpole_jax_dynamics(x, u, env_params)

    def noisy_dynamics_step(key, x, u, env_params, noise_scale):
        key_u, key_disturbance, key_x = jax.random.split(key, 3)
        scale = jnp.asarray(noise_scale, dtype=jdtype)
        input_noise = scale * sample_scaled_gaussian_noise(key_u, x.shape[0], input_scale, dtype=jdtype)
        process_noise = scale * sample_scaled_gaussian_noise(key_x, x.shape[0], process_scale, dtype=jdtype)
        sysid_input_disturbance = jax.random.uniform(
            key_disturbance,
            shape=(x.shape[0],),
            minval=-input_disturbance_bound,
            maxval=input_disturbance_bound,
            dtype=jdtype,
        )
        input_disturbance = jnp.zeros_like(u).at[:, 0].set(sysid_input_disturbance)
        u_applied = jnp.clip(
            u + input_noise + input_disturbance,
            jnp.asarray(CARTPOLE_U_MIN, dtype=jdtype),
            jnp.asarray(CARTPOLE_U_MAX, dtype=jdtype),
        )
        step_dt = control_dt / jnp.asarray(substeps, dtype=jdtype)

        def body(_, x_cur):
            return env_cartpole_jax_euler_step(x_cur, u_applied, env_params, step_dt)

        x_next = jax.lax.fori_loop(0, substeps, body, x)
        return x_next + process_noise, u_applied

    def reward_fn(states, actions):
        theta = states[:, :, 2]
        angle = theta
        x_pos = states[:, :, 0]
        v = states[:, :, 1]
        omega = states[:, :, 3]
        tail_start = max(1, rollout_steps // 2)
        tail_angle = angle[tail_start:]
        tail_x = x_pos[tail_start:]
        tail_v = v[tail_start:]
        tail_omega = omega[tail_start:]
        upright_score = jnp.exp(-0.5 * (angle / 0.45) ** 2)
        upright_tail = jnp.exp(-0.5 * (tail_angle / 0.35) ** 2)
        center_tail = jnp.exp(-0.5 * (tail_x / 0.15) ** 2)
        swing_up_time = jnp.abs(angle) < 0.35
        swing_up = jnp.any(swing_up_time, axis=0)
        within_success_angle = jnp.abs(angle) <= jnp.asarray(SUCCESS_ANGLE_BOUND, dtype=jdtype)
        stay_up_from_time = jnp.flip(
            jnp.cumprod(jnp.flip(within_success_angle.astype(jdtype), axis=0), axis=0),
            axis=0,
        )
        success = jnp.any(stay_up_from_time[:-1] > jnp.asarray(0.5, dtype=jdtype), axis=0)
        near_upright_fraction = jnp.mean((jnp.abs(angle) < 0.55).astype(jdtype), axis=0)
        max_height_score = jnp.max(0.5 * (1.0 + jnp.cos(angle)), axis=0)
        balance_time = (
            (jnp.abs(tail_angle) < 0.25)
            & (jnp.abs(tail_omega) < 1.0)
        )
        balance_fraction = jnp.mean(balance_time.astype(jdtype), axis=0)
        best_upright_score = jnp.max(upright_score, axis=0)
        rail_violation = jnp.maximum(jnp.abs(x_pos) - jnp.asarray(CARTPOLE_RAIL_LIMIT, dtype=jdtype), 0.0)
        rail_violation_rate = jnp.mean((rail_violation > 0.0).astype(jdtype), axis=0)
        rail_penalty = (
            jnp.asarray(RAIL_RATE_PENALTY, dtype=jdtype) * rail_violation_rate
            + jnp.asarray(RAIL_MAX_PENALTY, dtype=jdtype) * jnp.max(rail_violation, axis=0)
            + jnp.asarray(RAIL_MEAN_SQUARED_PENALTY, dtype=jdtype) * jnp.mean(rail_violation**2, axis=0)
        )
        tail_velocity_penalty = 0.10 * jnp.mean(tail_v**2, axis=0) + 0.01 * jnp.mean(tail_omega**2, axis=0)
        action_penalty = 0.002 * jnp.mean(actions[:, :, 0] ** 2, axis=0)
        reward = (
            250.0 * swing_up.astype(jdtype)
            + 80.0 * max_height_score
            + 60.0 * best_upright_score
            + 50.0 * near_upright_fraction
            + 45.0 * balance_fraction
            + 25.0 * jnp.mean(upright_tail * center_tail, axis=0)
            - rail_penalty
            - tail_velocity_penalty
            - action_penalty
        )
        finite = jnp.all(jnp.isfinite(states), axis=(0, 2)) & jnp.all(jnp.isfinite(actions), axis=(0, 2))
        return jnp.where(finite, reward, -1.0e6), success & finite, rail_violation_rate

    @jax.jit
    def rollout(z, params, x, env_params, key, noise_scale):
        def step(carry, _):
            z_cur, params_cur, x_cur, key_cur, solver_state_cur = carry
            params_cur = update_cartpole_params_initial_state(params_cur, x_cur)

            def sqp_body(iter_carry, _):
                z_iter, solver_state_iter = iter_carry
                result_iter, solver_state_next = sqp.step(
                    z_iter,
                    params_cur,
                    state=solver_state_iter,
                )
                return (result_iter.z_next, solver_state_next), result_iter

            (z_next, solver_state_next), sqp_results = jax.lax.scan(
                sqp_body,
                (z_cur, solver_state_cur),
                xs=None,
                length=sqp_iterations,
            )
            result = jax.tree_util.tree_map(lambda leaf: leaf[-1], sqp_results)
            stages = result.z_next.reshape((x_cur.shape[0], horizon_steps + 1, CARTPOLE_NZ))
            commanded_u = stages[:, 0, CARTPOLE_NX : CARTPOLE_NX + CARTPOLE_NU]
            key_next, noise_key = jax.random.split(key_cur)
            x_next, applied_u = noisy_dynamics_step(noise_key, x_cur, commanded_u, env_params, noise_scale)
            params_next = update_cartpole_params_initial_state(params_cur, x_next)
            finite = result.is_finite & jnp.all(jnp.isfinite(x_cur), axis=1) & jnp.all(jnp.isfinite(commanded_u), axis=1)
            output = (
                x_cur,
                commanded_u,
                applied_u,
                result.line_search.step_length,
                result.line_search.constraint_violation,
                result.line_search.accepted,
                result.solve.prim_res,
                result.solve.dual_res,
                finite,
            )
            return (z_next, params_next, x_next, key_next, solver_state_next), output

        solver_state = sqp.init_state(z.shape[0])
        final_carry, outputs = jax.lax.scan(
            step,
            (z, params, x, key, solver_state),
            xs=None,
            length=rollout_steps,
        )
        _, _, x_final, _, _ = final_carry
        (
            states_before,
            commanded_actions,
            applied_actions,
            step_lengths,
            constraint_violations,
            line_search_accepted,
            prim_res,
            dual_res,
            sqp_finite,
        ) = outputs
        states = jnp.concatenate([states_before, x_final[None, :, :]], axis=0)
        rollout_returns, rollout_success, rollout_rail_violation = reward_fn(states, applied_actions)
        exp_shape = (num_experiments, rollouts_per_experiment)
        experiment_returns = jnp.mean(rollout_returns.reshape(exp_shape), axis=1)
        experiment_success_rates = jnp.mean(rollout_success.reshape(exp_shape).astype(jdtype), axis=1)
        experiment_returns = experiment_returns + 300.0 * experiment_success_rates
        experiment_rail_violation_rates = jnp.mean(rollout_rail_violation.reshape(exp_shape), axis=1)
        return CartpoleBatchOutput(
            states=states,
            commanded_actions=commanded_actions,
            applied_actions=applied_actions,
            rollout_returns=rollout_returns,
            rollout_success=rollout_success,
            rollout_rail_violation=rollout_rail_violation,
            experiment_returns=experiment_returns,
            experiment_success_rates=experiment_success_rates,
            experiment_rail_violation_rates=experiment_rail_violation_rates,
            step_lengths=step_lengths,
            constraint_violations=constraint_violations,
            line_search_accepted=line_search_accepted,
            prim_res=prim_res,
            dual_res=dual_res,
            sqp_finite=sqp_finite,
        )

    return rollout


def _run_batch(
    args: argparse.Namespace,
    initialize_fn,
    rollout_fn,
    cost_params_by_experiment: np.ndarray,
    *,
    seed: int,
    dtype: np.dtype,
) -> tuple[CartpoleBatchOutput, float]:
    batch_size = args.experiments_per_episode * args.rollouts_per_experiment
    cost_params_by_experiment = np.asarray(cost_params_by_experiment, dtype=dtype)
    if cost_params_by_experiment.shape != (args.experiments_per_episode, CARTPOLE_N_COST_PARAMS):
        raise ValueError(f"bad experiment cost shape {cost_params_by_experiment.shape}")
    cost_params = np.repeat(cost_params_by_experiment, args.rollouts_per_experiment, axis=0)
    x0 = _sample_initial_states(args, batch_size, seed, dtype)
    if args.simulation_parameter_mode == "nominal":
        env_params = _nominal_env_parameters(batch_size, dtype)
    elif args.simulation_parameter_mode == "randomized":
        env_params = _sample_env_parameters(batch_size, seed + 4_202_011, dtype)
    else:
        raise ValueError(f"unsupported simulation parameter mode: {args.simulation_parameter_mode!r}")
    key = jax.random.PRNGKey(seed + 1_000_003)
    start = time.perf_counter()
    z0, params0 = initialize_fn(jnp.asarray(x0), jnp.asarray(cost_params))
    output = rollout_fn(
        z0,
        params0,
        jnp.asarray(x0),
        jnp.asarray(env_params),
        key,
        jnp.asarray(args.noise_scale, dtype=jnp.dtype(dtype)),
    )
    jax.block_until_ready(output.experiment_returns)
    elapsed = time.perf_counter() - start
    return jax.device_get(output), elapsed


def _return_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10.0)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _plot_experiment_state_distribution(
    path: pathlib.Path,
    time_grid: np.ndarray,
    states: np.ndarray,
    rollout_returns: np.ndarray,
    cost_params: np.ndarray,
    *,
    title: str,
    success_rate: float,
    rail_violation_rate: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.asarray(states, dtype=np.float64)
    plot_states = np.array(states, dtype=np.float64, copy=True)
    rollout_returns = np.asarray(rollout_returns, dtype=np.float64).reshape(-1)
    fig = plt.figure(figsize=(13.5, 7.8))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.95))
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    text_ax = fig.add_subplot(grid[:, 2])
    for idx, (ax, label, ylim) in enumerate(zip(axes, STATE_PLOT_LABELS, STATE_PLOT_YLIMS, strict=True)):
        values = plot_states[:, :, idx]
        finite = np.isfinite(values)
        values = np.where(finite, values, np.nan)
        ax.plot(time_grid, values, color="black", alpha=0.05, linewidth=0.45)
        median = np.nanmedian(values, axis=1)
        p10 = np.nanpercentile(values, 10.0, axis=1)
        p90 = np.nanpercentile(values, 90.0, axis=1)
        vmin = np.nanmin(values, axis=1)
        vmax = np.nanmax(values, axis=1)
        ax.fill_between(time_grid, vmin, vmax, color="tab:blue", alpha=0.10, linewidth=0.0)
        ax.fill_between(time_grid, p10, p90, color="tab:blue", alpha=0.24, linewidth=0.0)
        ax.plot(time_grid, median, color="tab:blue", linewidth=1.7)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        if idx == 0:
            ax.axhline(CARTPOLE_RAIL_LIMIT, color="tab:red", linewidth=0.8, alpha=0.5)
            ax.axhline(-CARTPOLE_RAIL_LIMIT, color="tab:red", linewidth=0.8, alpha=0.5)
        if idx == 2:
            ax.axhline(np.pi, color="tab:gray", linewidth=0.7, alpha=0.35)
            ax.axhline(-np.pi, color="tab:gray", linewidth=0.7, alpha=0.35)
        ax.set_ylim(*ylim)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    for ax in axes[2:]:
        ax.set_xlabel("time [s]")

    stats = _return_summary(rollout_returns)
    cost_text = format_cost_parameters(cost_params)
    text = (
        f"{title}\n\n"
        f"rollouts: {states.shape[1]:,}\n"
        f"return mean: {stats['mean']:.3g}\n"
        f"return p10/p90: {stats['p10']:.3g} / {stats['p90']:.3g}\n"
        f"return min/max: {stats['min']:.3g} / {stats['max']:.3g}\n"
        f"success rate: {success_rate:.2%}\n"
        f"rail violation time: {rail_violation_rate:.2%}\n\n"
        "cost parameters:\n"
        f"{cost_text}"
    )
    text_ax.axis("off")
    text_ax.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=6.1,
        linespacing=1.05,
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_return_stats(path: pathlib.Path, episode_rows: list[dict[str, float]]) -> None:
    if not episode_rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray([row["episode"] for row in episode_rows], dtype=np.int32)
    mean = np.asarray([row["mean"] for row in episode_rows], dtype=np.float64)
    median = np.asarray([row["median"] for row in episode_rows], dtype=np.float64)
    best = np.asarray([row["max"] for row in episode_rows], dtype=np.float64)
    p10 = np.asarray([row["p10"] for row in episode_rows], dtype=np.float64)
    p90 = np.asarray([row["p90"] for row in episode_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.fill_between(episodes, p10, p90, color="tab:blue", alpha=0.18, label="p10-p90")
    ax.plot(episodes, mean, marker="o", label="mean")
    ax.plot(episodes, median, marker="o", label="median")
    ax.plot(episodes, best, marker="o", label="best")
    ax.set_xlabel("episode")
    ax.set_ylabel("experiment return")
    ax.set_title("Cartpole cost tuning return statistics")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_return_histogram(path: pathlib.Path, returns: np.ndarray, *, baseline: float | None) -> None:
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    if returns.size == 0:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(returns[np.isfinite(returns)], bins=40, color="tab:blue", alpha=0.82)
    if baseline is not None and np.isfinite(baseline):
        ax.axvline(baseline, color="tab:orange", linewidth=2.0, label="nominal baseline mean")
        ax.legend()
    ax.set_xlabel("experiment return")
    ax.set_ylabel("count")
    ax.set_title("Distribution of all tuned experiments")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_experiment_csv(path: pathlib.Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _save_history_npz(path: pathlib.Path, optimizer: TurboBatchOptimizer) -> None:
    x, y = optimizer.history_arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        log_params=x,
        cost_params=np.exp(x) if x.size else np.empty_like(x),
        returns=y,
        names=np.asarray(CARTPOLE_COST_PARAMETER_NAMES),
    )


def _plot_episode_experiments(
    args: argparse.Namespace,
    output: CartpoleBatchOutput,
    cost_params_by_experiment: np.ndarray,
    time_grid: np.ndarray,
    episode_dir: pathlib.Path,
    *,
    episode: int,
    best_index: int,
) -> dict[int, pathlib.Path]:
    states = np.asarray(output.states)
    returns = np.asarray(output.rollout_returns)
    paths: dict[int, pathlib.Path] = {}
    if args.max_experiment_plots < 0:
        plot_indices = list(range(args.experiments_per_episode))
    else:
        plot_indices = list(range(min(args.max_experiment_plots, args.experiments_per_episode)))
        if best_index not in plot_indices:
            plot_indices.append(best_index)
    for experiment in plot_indices:
        start = experiment * args.rollouts_per_experiment
        stop = start + args.rollouts_per_experiment
        path = episode_dir / "plots" / f"experiment_{experiment:03d}_state_distribution.png"
        title = f"episode {episode:03d}, experiment {experiment:03d}"
        _plot_experiment_state_distribution(
            path,
            time_grid,
            states[:, start:stop, :],
            returns[start:stop],
            cost_params_by_experiment[experiment],
            title=title,
            success_rate=float(output.experiment_success_rates[experiment]),
            rail_violation_rate=float(output.experiment_rail_violation_rates[experiment]),
        )
        paths[experiment] = path
    return paths


def run(args: argparse.Namespace) -> dict:
    dtype = np.dtype(args.dtype)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)
    if args.simulation_parameter_mode == "nominal":
        simulation_parameter_text = "MPC and simulation both use the nominal row below.\n"
        plant_params_label = "nominal"
    else:
        simulation_parameter_text = (
            "MPC uses the nominal row below; simulation samples one fixed parameter vector per rollout "
            "from seg_min..seg_max.\n"
        )
        plant_params_label = "randomized_per_rollout"
    if args.input_disturbance_bound == 0.0:
        input_disturbance_text = "Simulation input disturbance: disabled.\n"
        input_disturbance_label = "disabled"
    else:
        input_disturbance_text = (
            "Simulation input disturbance: every applied first-channel action gets an independent "
            f"uniform additive disturbance in [{-args.input_disturbance_bound:g}, "
            f"{args.input_disturbance_bound:g}] V before clipping.\n"
        )
        input_disturbance_label = f"uniform +/-{args.input_disturbance_bound:g} V"
    (args.output_dir / "cost_function.txt").write_text(
        cartpole_cost_function_description()
        + "\n\n"
        + cartpole_warm_start_description()
        + "\n\n"
        + f"SQP rail constraint enabled: {args.enable_rail_constraint}\n"
        + (
            "BO rail penalty per rollout: "
            f"{RAIL_RATE_PENALTY:g}*mean(1[|x|>0.35]) "
            f"+ {RAIL_MAX_PENALTY:g}*max((|x|-0.35)_+) "
            f"+ {RAIL_MEAN_SQUARED_PENALTY:g}*mean((|x|-0.35)_+^2).\n"
        )
        + "\nSimulation physical parameters:\n"
        + simulation_parameter_text
        + f"Simulation also includes a fixed {CARTPOLE_M_TIP:g} kg point mass at the pole tip.\n"
        + _format_sysid_env_parameter_table()
        + "\n"
        + input_disturbance_text
        + "\n\nNominal parameters:\n"
        + format_cost_parameters(CARTPOLE_NOMINAL_COST_PARAMS)
        + "\n"
    )

    print(cartpole_cost_function_description(), flush=True)
    print("\n" + cartpole_warm_start_description(), flush=True)
    print(
        "\nRail setup:",
        f"hard_sqp_constraint={args.enable_rail_constraint}",
        f"limit=+/-{CARTPOLE_RAIL_LIMIT:g} m",
        flush=True,
    )
    print("\nSimulation physical parameters:", f"mode={args.simulation_parameter_mode}", flush=True)
    print(f"Fixed pole-tip point mass: {CARTPOLE_M_TIP:g} kg", flush=True)
    print(_format_sysid_env_parameter_table(), flush=True)
    print("Simulation input disturbance:", input_disturbance_label, flush=True)
    print("\nNominal cost parameters:", flush=True)
    print(format_cost_parameters(CARTPOLE_NOMINAL_COST_PARAMS), flush=True)

    print("building cartpole SQP solver...", flush=True)
    plan, sqp = _make_sqp(args, dtype)
    initialize_fn = make_cartpole_initialization_kernel(
        n_steps=args.horizon_steps,
        dt_start=args.dt_start,
        dt_growth=args.dt_growth,
        dtype=dtype,
    )
    rollout_fn = _make_rollout(args, sqp, dtype)
    batch_size = args.experiments_per_episode * args.rollouts_per_experiment
    print(
        "Cartpole tuning setup:",
        f"dtype={dtype}",
        f"episodes={args.episodes}",
        f"experiments_per_episode={args.experiments_per_episode}",
        f"rollouts_per_experiment={args.rollouts_per_experiment}",
        f"batch={batch_size:,}",
        f"horizon_steps={args.horizon_steps}",
        f"rollout_steps={args.rollout_steps}",
        f"sqp_iterations={args.sqp_iterations}",
        f"osqp_max_iter={args.osqp_max_iter}",
        f"rail_constraint={args.enable_rail_constraint}",
        f"qp_solver={args.qp_solver}",
        f"qdldl_backend={args.qdldl_backend}",
        f"qdldl_factor_backend={args.qdldl_factor_backend}",
        f"qdldl_solve_backend={args.qdldl_solve_backend}",
        f"level_scheduled_solve={args.level_scheduled_solve}",
        f"level_scheduled_solve_threshold={args.level_scheduled_solve_threshold}",
        f"plant_params={plant_params_label}",
        f"input_disturbance_bound={args.input_disturbance_bound:g}",
        f"n={plan.n_variables}",
        f"m={plan.n_constraints}",
        f"nnz_P={plan.p_pattern.nnz}",
        f"nnz_A={plan.a_pattern.nnz}",
        flush=True,
    )

    time_grid = np.arange(args.rollout_steps + 1, dtype=np.float64) * args.control_dt
    nominal_by_experiment = np.broadcast_to(
        CARTPOLE_NOMINAL_COST_PARAMS[None, :],
        (args.experiments_per_episode, CARTPOLE_N_COST_PARAMS),
    ).copy()

    print("running nominal baseline...", flush=True)
    baseline_output, baseline_elapsed = _run_batch(
        args,
        initialize_fn,
        rollout_fn,
        nominal_by_experiment,
        seed=args.seed + 17,
        dtype=dtype,
    )
    baseline_return_mean = float(np.mean(np.asarray(baseline_output.experiment_returns)))
    baseline_dir = args.output_dir / "baseline"
    _plot_experiment_state_distribution(
        baseline_dir / "baseline_state_distribution.png",
        time_grid,
        np.asarray(baseline_output.states),
        np.asarray(baseline_output.rollout_returns),
        CARTPOLE_NOMINAL_COST_PARAMS,
        title="nominal baseline",
        success_rate=float(np.mean(np.asarray(baseline_output.rollout_success))),
        rail_violation_rate=float(np.mean(np.asarray(baseline_output.rollout_rail_violation))),
    )
    baseline_summary = {
        "elapsed_s": baseline_elapsed,
        "return_mean": baseline_return_mean,
        "experiment_returns": np.asarray(baseline_output.experiment_returns, dtype=np.float64).tolist(),
        "success_rate": float(np.mean(np.asarray(baseline_output.rollout_success))),
        "rail_violation_rate": float(np.mean(np.asarray(baseline_output.rollout_rail_violation))),
    }
    _write_json(baseline_dir / "baseline_summary.json", baseline_summary)
    print(
        f"baseline elapsed={baseline_elapsed:.3f}s "
        f"mean_return={baseline_return_mean:.3g} "
        f"success={baseline_summary['success_rate']:.2%} "
        f"rail_violation={baseline_summary['rail_violation_rate']:.2%}",
        flush=True,
    )

    optimizer = TurboBatchOptimizer(
        lower=np.log(CARTPOLE_COST_PARAM_LOWER),
        upper=np.log(CARTPOLE_COST_PARAM_UPPER),
        nominal=cost_params_to_log(CARTPOLE_NOMINAL_COST_PARAMS),
        batch_size=args.experiments_per_episode,
        seed=args.seed,
        candidate_pool=args.turbo_candidate_pool,
        max_gp_points=args.turbo_max_gp_points,
        length=args.turbo_initial_length,
        length_min=args.turbo_min_length,
    )

    episode_rows: list[dict[str, float]] = []
    experiment_rows: list[dict[str, float | int]] = []
    all_experiment_returns: list[np.ndarray] = []
    best_plot_path: pathlib.Path | None = None
    best_payload: dict | None = None

    for episode in range(args.episodes):
        log_params = optimizer.ask()
        cost_params = np.exp(log_params)
        print(f"\nepisode {episode:03d}: evaluating {args.experiments_per_episode} experiments...", flush=True)
        output, elapsed = _run_batch(
            args,
            initialize_fn,
            rollout_fn,
            cost_params,
            seed=args.seed + 10_000 * (episode + 1),
            dtype=dtype,
        )
        experiment_returns = np.asarray(output.experiment_returns, dtype=np.float64)
        optimizer.tell(log_params, experiment_returns)
        all_experiment_returns.append(experiment_returns.copy())
        stats = _return_summary(experiment_returns)
        stats["episode"] = float(episode)
        episode_rows.append(stats)
        episode_dir = args.output_dir / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        best_index = int(np.argmax(experiment_returns))
        plot_paths = _plot_episode_experiments(
            args,
            output,
            cost_params,
            time_grid,
            episode_dir,
            episode=episode,
            best_index=best_index,
        )
        for experiment in range(args.experiments_per_episode):
            start = experiment * args.rollouts_per_experiment
            stop = start + args.rollouts_per_experiment
            experiment_rows.append(
                {
                    "episode": episode,
                    "experiment": experiment,
                    "return": float(experiment_returns[experiment]),
                    "success_rate": float(output.experiment_success_rates[experiment]),
                    "rail_violation_rate": float(output.experiment_rail_violation_rates[experiment]),
                    "elapsed_s": elapsed,
                }
            )
        _, current_best_return = optimizer.best()
        if best_index in plot_paths and float(experiment_returns[best_index]) >= current_best_return - 1.0e-8:
            best_plot_path = plot_paths[best_index]
            shutil.copyfile(best_plot_path, args.output_dir / "plots" / "best_so_far_state_distribution.png")
        if float(experiment_returns[best_index]) >= current_best_return - 1.0e-8:
            best_payload = {
                "episode": episode,
                "experiment": best_index,
                "return": float(experiment_returns[best_index]),
                "cost_params": {
                    name: float(value)
                    for name, value in zip(CARTPOLE_COST_PARAMETER_NAMES, cost_params[best_index], strict=True)
                },
            }
        _write_json(
            episode_dir / "episode_summary.json",
            {
                "episode": episode,
                "elapsed_s": elapsed,
                "return_stats": stats,
                "best_experiment": best_index,
                "best_return": float(experiment_returns[best_index]),
                "best_cost_params": {
                    name: float(value)
                    for name, value in zip(CARTPOLE_COST_PARAMETER_NAMES, cost_params[best_index], strict=True)
                },
                "rail_violation_mean": float(np.mean(np.asarray(output.experiment_rail_violation_rates))),
                "best_rail_violation_rate": float(output.experiment_rail_violation_rates[best_index]),
                "turbo_length": optimizer.length,
            },
        )
        _write_experiment_csv(args.output_dir / "experiment_returns.csv", experiment_rows)
        _plot_return_stats(args.output_dir / "plots" / "return_statistics.png", episode_rows)
        _plot_return_histogram(
            args.output_dir / "plots" / "all_experiment_returns_histogram.png",
            np.concatenate(all_experiment_returns),
            baseline=baseline_return_mean,
        )
        _save_history_npz(args.output_dir / "tuning_history.npz", optimizer)
        if best_payload is not None:
            _write_json(args.output_dir / "best_so_far_cost_params.json", best_payload)
        print(
            f"episode {episode:03d}: elapsed={elapsed:.3f}s "
            f"return mean={stats['mean']:.3g} median={stats['median']:.3g} "
            f"best={stats['max']:.3g} min={stats['min']:.3g} "
            f"success_mean={float(np.mean(np.asarray(output.experiment_success_rates))):.2%} "
            f"rail_violation_mean={float(np.mean(np.asarray(output.experiment_rail_violation_rates))):.2%} "
            f"turbo_length={optimizer.length:.3g}",
            flush=True,
        )
        print("best cost params this episode:", flush=True)
        print(format_cost_parameters(cost_params[best_index], max_lines=8), flush=True)
        del output

    best_log, best_return = optimizer.best()
    summary = {
        "output_dir": str(args.output_dir),
        "episodes": args.episodes,
        "experiments_per_episode": args.experiments_per_episode,
        "rollouts_per_experiment": args.rollouts_per_experiment,
        "batch_size": batch_size,
        "horizon_steps": args.horizon_steps,
        "dt_start": args.dt_start,
        "dt_growth": args.dt_growth,
        "sim_time": args.sim_time,
        "control_dt": args.control_dt,
        "rollout_steps": args.rollout_steps,
        "integrator_substeps": args.integrator_substeps,
        "sqp_iterations": args.sqp_iterations,
        "qp_solver": args.qp_solver,
        "osqp_max_iter": args.osqp_max_iter,
        "rho": args.rho,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "qdldl_backend": args.qdldl_backend,
        "qdldl_factor_backend": args.qdldl_factor_backend,
        "qdldl_solve_backend": args.qdldl_solve_backend,
        "transpose_work": bool(args.transpose_work),
        "segmented": bool(args.segmented),
        "segment_budget": args.segment_budget,
        "segment_strategy": args.segment_strategy,
        "level_scheduled_solve": bool(args.level_scheduled_solve),
        "level_scheduled_solve_threshold": args.level_scheduled_solve_threshold,
        "line_search_step_min": args.line_search_step_min,
        "enable_rail_constraint": args.enable_rail_constraint,
        "rail_limit": CARTPOLE_RAIL_LIMIT,
        "noise_scale": args.noise_scale,
        "process_noise_scale": args.process_noise_scale,
        "input_noise_scale": args.input_noise_scale,
        "bo_rail_penalty": {
            "rate": RAIL_RATE_PENALTY,
            "max": RAIL_MAX_PENALTY,
            "mean_squared": RAIL_MEAN_SQUARED_PENALTY,
        },
        "mpc_physical_parameters": {
            name: float(value)
            for name, value in zip(SYSID_ENV_PARAMETER_NAMES, SYSID_ENV_PARAMETER_NOMINAL, strict=True)
        },
        "simulation_parameter_mode": args.simulation_parameter_mode,
        "simulation_physical_parameter_randomization": [
            {
                "name": str(row[0]),
                "prior": float(row[1]),
                "nominal": float(row[2]),
                "radius": float(row[3]),
                "lower": float(row[4]),
                "upper": float(row[5]),
                "segment_min": float(row[6]),
                "segment_max": float(row[7]),
            }
            for row in SYSID_ENV_PARAMETER_TABLE
        ],
        "simulation_input_disturbance": {
            "distribution": "none" if args.input_disturbance_bound == 0.0 else "uniform",
            "channel": 0,
            "lower": -args.input_disturbance_bound,
            "upper": args.input_disturbance_bound,
            "applied": "disabled" if args.input_disturbance_bound == 0.0 else "before voltage clipping, independent of noise_scale",
        },
        "baseline": baseline_summary,
        "best_return": best_return,
        "best_cost_params": {
            name: float(value)
            for name, value in zip(CARTPOLE_COST_PARAMETER_NAMES, np.exp(best_log), strict=True)
        },
        "best_plot_path": str(best_plot_path) if best_plot_path is not None else None,
        "episode_return_stats": episode_rows,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/cartpole_physical_quadratic_tuning"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--experiments-per-episode", type=int, default=100)
    parser.add_argument("--rollouts-per-experiment", type=int, default=1000)
    parser.add_argument("--horizon-steps", type=int, default=CARTPOLE_N_STEPS)
    parser.add_argument("--dt-start", type=float, default=100e-3)
    parser.add_argument("--dt-growth", type=float, default=1.0)
    parser.add_argument("--sim-time", type=float, default=10.0)
    parser.add_argument("--control-dt", type=float, default=100e-3)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--integrator-substeps", type=int, default=1)
    parser.add_argument("--sqp-iterations", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--osqp-max-iter", type=int, default=50)
    parser.add_argument("--rho", type=float, default=0.02)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.4)
    parser.add_argument("--qp-solver", choices=("jax_osqp", "mpax"), default="jax_osqp")
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="jax")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default=None)
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-budget", type=int, default=384)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action="store_true")
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=1)
    parser.add_argument("--line-search-step-min", type=float, default=0.01)
    parser.add_argument(
        "--enable-rail-constraint",
        action="store_true",
        help=f"Add hard SQP rail bounds |x| <= {CARTPOLE_RAIL_LIMIT:g} to the quadratic MPC.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--process-noise-scale", default=DEFAULT_PROCESS_NOISE_SCALE)
    parser.add_argument("--input-noise-scale", default=DEFAULT_INPUT_NOISE_SCALE)
    parser.add_argument("--simulation-parameter-mode", choices=("randomized", "nominal"), default="randomized")
    parser.add_argument("--input-disturbance-bound", type=float, default=SYSID_INPUT_DISTURBANCE_BOUND)
    parser.add_argument("--initial-position-range", type=float, default=0.30)
    parser.add_argument("--initial-angle-spread-deg", type=float, default=20.0)
    parser.add_argument("--initial-velocity-std", type=float, default=0.15)
    parser.add_argument("--initial-omega-std", type=float, default=0.5)
    parser.add_argument("--turbo-candidate-pool", type=int, default=4096)
    parser.add_argument("--turbo-max-gp-points", type=int, default=300)
    parser.add_argument("--turbo-initial-length", type=float, default=0.8)
    parser.add_argument("--turbo-min-length", type=float, default=0.01)
    parser.add_argument(
        "--max-experiment-plots",
        type=int,
        default=-1,
        help="Plot all experiments when negative; useful to limit local smoke-test output.",
    )
    parser.add_argument(
        "--no-group-repeated-stages",
        action="store_false",
        dest="group_repeated_stages",
    )
    parser.set_defaults(group_repeated_stages=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.qdldl_factor_backend is None:
        args.qdldl_factor_backend = args.qdldl_backend
    if args.qdldl_solve_backend is None:
        args.qdldl_solve_backend = args.qdldl_backend
    if args.rollout_steps is None:
        args.rollout_steps = int(math.ceil(args.sim_time / args.control_dt))
    if args.experiments_per_episode <= 0 or args.rollouts_per_experiment <= 0:
        raise ValueError("experiments and rollouts per experiment must be positive")
    if args.integrator_substeps <= 0:
        raise ValueError("integrator substeps must be positive")
    if args.sqp_iterations <= 0:
        raise ValueError("sqp iterations must be positive")
    if args.input_disturbance_bound < 0.0:
        raise ValueError("input disturbance bound must be non-negative")
    summary = run(args)
    print(f"\nWrote results to {summary['output_dir']}", flush=True)
    print(f"Best return: {summary['best_return']:.6g}", flush=True)
    print("Best cost parameters:", flush=True)
    print(format_cost_parameters(np.asarray(list(summary["best_cost_params"].values()))), flush=True)


if __name__ == "__main__":
    main()
