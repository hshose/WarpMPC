"""Pure MPPI/CEM optimizers and model adapters for the nonlinear benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Settings:
    method: str = "mppi"
    samples: int = 256
    iterations: int = 4
    initial_iterations: int = 0
    warm_sigma: float = 1.0
    knots: int = 16
    sigma: float = 1.0
    temperature: float = 0.1
    elite_fraction: float = 0.05
    smoothing: float = 0.1
    min_std: float = 0.05
    penalty: float = 1000.0
    noise_decay: float = 0.0
    noise_correlation: float = 0.0
    control_scale: tuple = ()
    feedback: bool = False
    shift: bool = True

    def __post_init__(self):
        if self.feedback:
            raise ValueError("Only direct-action MPPI/CEM is supported; prestabilizing feedback is forbidden")
        object.__setattr__(self, "control_scale", tuple(self.control_scale))
        if not math.isfinite(self.noise_decay) or self.noise_decay < 0:
            raise ValueError("noise_decay must be finite and nonnegative")
        if not 0 <= self.noise_correlation < 1:
            raise ValueError("noise_correlation must be in [0, 1) so every control direction remains sampled")
        if any(not math.isfinite(v) or v <= 0 for v in self.control_scale):
            raise ValueError("control_scale entries must be finite and positive")
        if self.method not in {"mppi", "cem"}:
            raise ValueError("method must be mppi or cem")
        if min(self.samples, self.iterations, self.knots) < 1 or self.samples < 2:
            raise ValueError("positive iterations/knots and at least two samples required")
        if self.initial_iterations < 0 or not math.isfinite(self.warm_sigma) or self.warm_sigma <= 0:
            raise ValueError("initial_iterations must be nonnegative and warm_sigma positive")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.sigma, self.temperature, self.min_std, self.penalty)):
            raise ValueError("sigma, temperature, min_std and penalty must be finite and positive")
        if not 0 < self.elite_fraction <= 1 or not 0 <= self.smoothing < 1:
            raise ValueError("invalid elite fraction or smoothing")


def make_optimizer(score, settings: Settings):
    """score(x[B,D], candidates[B,S,K,U]) -> costs[B,S].

    MPPI includes the Gaussian proposal/prior likelihood correction in latent
    control space. Bounds are applied by the rollout, not by changing the noise
    in that correction. CEM fits elite mean and diagonal variance. Invalid
    candidates receive zero weight, and an all-invalid population keeps its mean.
    """
    cfg = settings
    elites = max(2, min(cfg.samples, math.ceil(cfg.samples * cfg.elite_fraction)))

    def optimize(x, mean, keys, iterations=None, noise_scale=1.):
        time_scale = jnp.exp(-cfg.noise_decay * jnp.linspace(0., 1., mean.shape[1]))
        initial_std = jnp.ones_like(mean) * cfg.sigma * noise_scale * time_scale[None, :, None]
        if cfg.noise_correlation:
            rho = cfg.noise_correlation
            indices = jnp.arange(mean.shape[1])
            powers = indices[:, None] - indices[None, :]
            correlation_cholesky = jnp.where(powers >= 0, rho**jnp.maximum(powers, 0), 0.)
            correlation_cholesky *= jnp.where(indices == 0, 1., jnp.sqrt(1.-rho*rho))[None, :]

        def whiten(values):
            if not cfg.noise_correlation:
                return values
            return jnp.concatenate((values[..., :1, :],
                (values[..., 1:, :] - cfg.noise_correlation*values[..., :-1, :])
                / jnp.sqrt(1.-cfg.noise_correlation**2)), axis=-2)

        def iteration(i, carry):
            center, std = carry
            iter_keys = jax.vmap(lambda k: jax.random.fold_in(k, i))(keys)
            noise = jax.vmap(lambda k: jax.random.normal(
                k, (cfg.samples, *mean.shape[1:]), dtype=mean.dtype))(iter_keys)
            if cfg.noise_correlation:
                noise = jnp.einsum("lk,bsku->bslu", correlation_cholesky, noise,
                                   precision=jax.lax.Precision.HIGHEST)
            # Preserve the incumbent as a candidate; no oracle/warm-start solver.
            noise = noise.at[:, 0].set(0)
            delta = noise * std[:, None]
            candidates = center[:, None] + delta
            costs = score(x, candidates)
            valid = jnp.isfinite(costs)
            any_valid = jnp.any(valid, axis=1)
            if cfg.method == "mppi":
                correction = cfg.temperature * jnp.sum(
                    (whiten(center/std)[:, None] * whiten(delta/std[:, None]))
                    if cfg.noise_correlation else center[:, None] * delta / (std[:, None] ** 2),
                    axis=(2, 3))
                costs = jnp.where(valid, costs + correction, jnp.inf)
                best = jnp.min(costs, axis=1, keepdims=True)
                logits = jnp.where(valid, -(costs - best) / cfg.temperature, -jnp.inf)
                weights = jax.nn.softmax(jnp.where(any_valid[:, None], logits, 0), axis=1)
                updated = center + jnp.sum(weights[:, :, None, None] * delta, axis=1)
                next_std = std
            else:
                _, indices = jax.lax.top_k(-jnp.where(valid, costs, jnp.inf), elites)
                selected = jnp.take_along_axis(candidates, indices[:, :, None, None], axis=1)
                selected_valid = jnp.take_along_axis(valid, indices, axis=1)
                weights = selected_valid / jnp.maximum(jnp.sum(selected_valid, axis=1, keepdims=True), 1)
                elite_mean = jnp.sum(selected * weights[:, :, None, None], axis=1)
                variance = jnp.sum((selected - elite_mean[:, None]) ** 2 * weights[:, :, None, None], axis=1)
                updated = cfg.smoothing * center + (1 - cfg.smoothing) * elite_mean
                next_std = jnp.sqrt(cfg.smoothing * std**2 + (1 - cfg.smoothing) * variance)
                next_std = jnp.maximum(next_std, cfg.min_std * initial_std)
            return (jnp.where(any_valid[:, None, None], updated, center), next_std)

        budget = cfg.iterations if iterations is None else iterations
        return jax.lax.fori_loop(0, budget, iteration, (mean, initial_std))[0]

    return optimize


BATCH_SIZES = (512, 2048, 10000, 20000, 50000, 100000, 200000, 300000)
CHUNK_SIZES = {"cartpole": 1024, "crazyflie": 128}
# Match the current run_cartpole_quadratic_mpc.sh / run_crazyflie_sqp_rollout.sh.
SIM_STEPS = {"cartpole": 20, "crazyflie": 100}
REFERENCE_SQP_ITERATIONS = {"cartpole": 5, "crazyflie": 1}


def default_settings(system: str, method: str) -> Settings:
    if method not in ("mppi", "cem"):
        raise ValueError(f"Unknown sampling method: {method}")
    if system == "cartpole":
        return Settings(method=method, samples=1024, iterations=4, knots=100,
                        sigma=3., temperature=.01 if method == "mppi" else 1.,
                        elite_fraction=.02, min_std=.02,
                        noise_correlation=.95 if method == "mppi" else .8)
    if system == "crazyflie":
        return Settings(method=method, samples=256, iterations=8, knots=40,
                        warm_sigma=.25 if method == "mppi" else .5,
                        temperature=1e-6 if method == "mppi" else .1,
                        elite_fraction=.01, min_std=.01 if method == "mppi" else .02,
                        noise_decay=2.3, control_scale=(.03, .03, .01, .1))
    raise ValueError(f"No paper sampling baseline for {system}; humanoid failed quality checks")


@dataclass
class Model:
    name: str
    nx: int
    nu: int
    horizon: int
    steps: int
    dt: float
    schedule: np.ndarray
    q: np.ndarray
    r: np.ndarray
    terminal_q: np.ndarray
    scale: np.ndarray
    dynamics: object
    project: object
    violation: object
    sample_initial: object


def make_model(name):
    if name == "cartpole":
        from benchmarks.problems.cartpole_physical_quadratic import (
            CARTPOLE_U_MAX, CARTPOLE_RAIL_LIMIT,
            cartpole_dt_schedule, cartpole_jax_euler_step, cartpole_tube_schedule)
        from benchmarks.nonlinear_mpc.evaluate_cartpole_quadratic_mpc import TUNED_COST_PARAMS
        from examples.cartpole_tuning.tune_cartpole_physical_quadratic_mpc import _sample_initial_states
        params = TUNED_COST_PARAMS
        schedule = cartpole_dt_schedule(100, dt_start=.1, dt_growth=1.)
        tube = cartpole_tube_schedule(params, schedule)[0]
        bounds = jnp.asarray(CARTPOLE_U_MAX - params[11] * tube)
        rails = jnp.asarray(CARTPOLE_RAIL_LIMIT - params[12] * tube)
        initial_args = SimpleNamespace(initial_position_range=.30, initial_angle_spread_deg=20.,
                                       initial_velocity_std=.15, initial_omega_std=.5)
        return Model(name, 4, 1, 100, 20, .1, schedule, params[:4], params[4:5],
                     schedule[-1] * params[5:9], np.array([3.]), cartpole_jax_euler_step,
                     lambda u, t: jnp.clip(u, -bounds[t], bounds[t]),
                     lambda x, t: jnp.maximum(jnp.abs(x[..., 0]) - rails[t], 0.) ** 2,
                     lambda b, seed: _sample_initial_states(initial_args, b, seed, np.dtype("float32")))
    if name == "crazyflie":
        from benchmarks.problems.crazyflie_sqp import (
            CRAZYFLIE_Q, CRAZYFLIE_R, CRAZYFLIE_ARM, CRAZYFLIE_THRUST_TO_TORQUE,
            CRAZYFLIE_ACTION_SCALING, CRAZYFLIE_M, CRAZYFLIE_G, CRAZYFLIE_THRUST_MAX,
            crazyflie_dt_schedule, crazyflie_jax_euler_step)
        from benchmarks.nonlinear_mpc.evaluate_crazyflie_sqp import _sample_initial_states
        a, k = CRAZYFLIE_ARM, CRAZYFLIE_THRUST_TO_TORQUE
        mixer = .25 * np.array([[-1/a,-1/a,-1/k,1],[-1/a,1/a,1/k,1],
                               [1/a,1/a,-1/k,1],[1/a,-1/a,1/k,1]])
        mix = jnp.asarray(mixer, dtype=jnp.float32)
        unmix = jnp.asarray(np.linalg.inv(mixer), dtype=jnp.float32)
        scale = jnp.asarray(CRAZYFLIE_ACTION_SCALING, dtype=jnp.float32)
        hover = jnp.asarray([0,0,0,CRAZYFLIE_M*CRAZYFLIE_G], dtype=jnp.float32)
        def project(u, t):
            thrusts = (u * scale + hover) @ mix.T
            return (jnp.clip(thrusts, 0, CRAZYFLIE_THRUST_MAX) @ unmix.T - hover) / scale
        return Model(name, 12, 4, 40, 100, .01, crazyflie_dt_schedule(40),
                     np.diag(CRAZYFLIE_Q), np.diag(CRAZYFLIE_R), np.diag(CRAZYFLIE_Q),
                     np.array([.01, .01, .01, .1]), crazyflie_jax_euler_step, project,
                     lambda x, t: jnp.zeros(x.shape[:-1], dtype=x.dtype),
                     lambda b, seed: _sample_initial_states(b, seed, np.dtype("float32")))
    raise ValueError(name)


def require_full_controls(system, cfg):
    """Reduced control knots are historical diagnostics, not final baselines."""
    expected = {"cartpole": 100, "crazyflie": 40}[system]
    if cfg.knots != expected:
        raise ValueError(f"{system} requires {expected} independent control stages; got {cfg.knots} knots")


def make_rollout_functions(model, cfg):
    if cfg.control_scale and len(cfg.control_scale) != model.nu:
        raise ValueError("control_scale must have one entry per control dimension")
    times = np.concatenate(([0.], np.cumsum(model.schedule)))[:-1]
    # Full-horizon sampling puts a separate control at every shooting stage,
    # including the early, shorter intervals of the Crazyflie grid.
    knot_times = np.interp(np.linspace(0, model.horizon-1, cfg.knots),
                           np.arange(model.horizon), times)
    # Linear interpolation in physical time, including the nonuniform Crazyflie grid.
    def matrix_at(target):
        return np.stack([np.interp(target, knot_times, np.eye(cfg.knots)[i])
                         for i in range(cfg.knots)], axis=-1).astype(np.float32)
    interpolation = jnp.asarray(matrix_at(times))
    shift_matrix = jnp.asarray(matrix_at(knot_times + model.dt))
    scale, q, r, qf = map(lambda a: jnp.asarray(a, dtype=jnp.float32),
                         (cfg.control_scale or model.scale, model.q, model.r, model.terminal_q))
    schedule = jnp.asarray(model.schedule, dtype=jnp.float32)

    def action(controls, t):
        return model.project(controls * scale, t)

    def score(x0, candidates):
        batch, samples = candidates.shape[:2]
        x = jnp.broadcast_to(x0[:, None], (batch, samples, model.nx)).reshape(-1, model.nx)
        latent = candidates.reshape(batch * samples, cfg.knots, model.nu)
        def stage(carry, t):
            x, cost = carry
            controls = jnp.einsum("k,bku->bu", interpolation[t], latent)
            u = action(controls, t)
            c = .5 * schedule[t] * (jnp.sum(x*x*q, axis=-1) + jnp.sum(u*u*r, axis=-1))
            c += cfg.penalty * model.violation(x, t)
            x_next = model.dynamics(x, u, schedule[t])
            return (x_next, cost + c), None
        (xn, cost), _ = jax.lax.scan(stage, (x, jnp.zeros(x.shape[0], x.dtype)),
                                   jnp.arange(model.horizon))
        cost += .5 * jnp.sum(xn*xn*qf, axis=-1) + cfg.penalty * model.violation(xn, model.horizon)
        return jnp.where(jnp.isfinite(cost), cost, jnp.inf).reshape(batch, samples)

    def execute(x, mean):
        u = action(mean[:, 0], 0)
        xn = model.dynamics(x, u, model.dt)
        next_mean = jnp.einsum("lk,bku->blu", shift_matrix, mean) if cfg.shift else mean
        return xn, next_mean, u

    return score, execute
