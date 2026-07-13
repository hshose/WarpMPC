#!/usr/bin/env python3
"""Headless batched Humanoid Robot Software simulation with sparse SQP MPC."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import casadi as cs
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency.
    raise ModuleNotFoundError("headless humanoid simulation requires casadi") from exc

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.jax_cache import configure_jax_compilation_cache
from benchmarks.problems.humanoid_mpc import (
    HUMANOID_CASADI_FUNCTION_DIR,
    HUMANOID_HIGH_LEVEL_TIMESTEP,
    HUMANOID_LOW_LEVEL_TIMESTEP,
    HUMANOID_MPC_PERIOD_S,
    HUMANOID_MPC_PERIOD_STEPS,
    HUMANOID_MPC_STANDING_REFERENCE_S,
    HUMANOID_MODEL_NAME,
    HUMANOID_NDQ,
    HUMANOID_NQ,
    HUMANOID_NTAU,
    HUMANOID_PARAMETERS_PATH,
    HUMANOID_SIM_TIMESTEP,
    HUMANOID_TORQUE_FEEDFORWARD_LOOKAHEAD_S,
    ACTUATOR_STATIC_TAU_MAX_NM,
    gains_for_contact,
    humanoid_initial_guess_and_params,
    humanoid_jax_trajectory_from_solution,
    humanoid_make_references,
    humanoid_walking_gait_schedule,
    load_humanoid_mpc_parameters,
    make_humanoid_sqp_problem,
    standing_gait_schedule,
    update_humanoid_params_mpc,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import FilterLineSearchSettings, build_sparse_mpc_plan, compile_sparse_mpc_sqp
from warpmpc.numpysadi import convert


configure_jax_compilation_cache()

GROUND_HEIGHT = 0.0
GROUND_MU = 1.25
GROUND_RESTITUTION = 0.0
DEFAULT_CONTACT_ITERATIONS = 20
SOFT_STOP_LOWER = np.array(
    [
        -0.75,
        -1.1,
        -3.14,
        -3.14,
        -3.14,
        -1.0,
        -1.0,
        -3.14,
        -3.14,
        -3.14,
        -1.9,
        -1.57,
        -1.17,
        -3.14,
        -1.9,
        -0.1,
        -1.17,
        -3.14,
    ],
    dtype=np.float64,
)
SOFT_STOP_UPPER = np.array(
    [
        1.0,
        1.0,
        3.14,
        3.14,
        3.14,
        0.75,
        1.1,
        3.14,
        3.14,
        3.14,
        1.9,
        0.1,
        1.17,
        3.14,
        1.9,
        1.57,
        1.17,
        3.14,
    ],
    dtype=np.float64,
)
SOFT_STOP_KP = 100.0
SOFT_STOP_KD = 1.0
INITIAL_BASE_QUAT_WXYZ = np.array(
    [0.9995500337489875, 0.0, 0.02999550020249566, 0.0],
    dtype=np.float64,
)


ArrayFn = Callable[..., Any]


def _single_output(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"expected one output, got {len(value)}")
        return value[0]
    return value


def _outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, dict):
        return tuple(value[name] for name in value)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _dense_sx_function(function: cs.Function, function_name: str) -> cs.Function:
    inputs = [function.sx_in(i) for i in range(function.n_in())]
    outputs = [cs.densify(output) for output in function.call(inputs)]
    input_names = [function.name_in(i) or f"arg{i}" for i in range(function.n_in())]
    output_names = [function.name_out(i) or f"out{i}" for i in range(function.n_out())]
    return cs.Function(function_name, inputs, outputs, input_names, output_names)


def _convert_casadi_file(path: Path, function_name: str) -> ArrayFn:
    function = cs.Function.load(str(path))
    return convert(
        _dense_sx_function(function, f"{function_name}_dense"),
        function_name=function_name,
        backend="jax",
    )


def _load_optional_jax_function(dirs: list[Path | None], filename: str, function_name: str) -> ArrayFn | None:
    for directory in dirs:
        if directory is None:
            continue
        path = directory / filename
        if not path.exists():
            continue
        return _convert_casadi_file(path, function_name)
    return None


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2.0, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def make_grid_xy(batch_size: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(batch_size)))
    rows = int(np.ceil(batch_size / cols))
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    xy = np.stack([ii.reshape(-1), jj.reshape(-1)], axis=1)[:batch_size].astype(np.float64)
    xy[:, 0] -= (rows - 1) * 0.5
    xy[:, 1] -= (cols - 1) * 0.5
    return xy * float(spacing)


def make_initial_q(batch_size: int, grid_xy: np.ndarray, qhome: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(qhome, (batch_size, HUMANOID_NQ)).copy()
    q[:, :2] = grid_xy
    q[:, 2] = 0.61
    q[:, 3:6] = quat_to_rpy(INITIAL_BASE_QUAT_WXYZ)
    return q


def _jax_rpy_to_rotmat(rpy: jax.Array, dtype: jnp.dtype) -> jax.Array:
    roll, pitch, yaw = rpy
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=dtype,
    )


def _jax_binv_rpy(rpy: jax.Array, dtype: jnp.dtype) -> jax.Array:
    _roll, pitch, yaw = rpy
    cp = jnp.cos(pitch)
    cp_abs = jnp.maximum(jnp.abs(cp), 1.0e-8)
    cp = jnp.copysign(cp_abs, cp)
    return jnp.asarray(
        [
            [jnp.cos(yaw) / cp, jnp.sin(yaw) / cp, 0.0],
            [-jnp.sin(yaw), jnp.cos(yaw), 0.0],
            [jnp.cos(yaw) * jnp.tan(pitch), jnp.sin(yaw) * jnp.tan(pitch), 1.0],
        ],
        dtype=dtype,
    )


def _jax_integrate_q(q: jax.Array, dq: jax.Array, dt: float, dtype: jnp.dtype) -> jax.Array:
    rot = _jax_rpy_to_rotmat(q[3:6], dtype)
    q_next = q
    q_next = q_next.at[:3].set(q[:3] + rot @ dq[3:6] * dt)
    q_next = q_next.at[3:6].set(q[3:6] + _jax_binv_rpy(q[3:6], dtype) @ (rot @ dq[:3]) * dt)
    q_next = q_next.at[6:].set(q[6:] + dq[6:] * dt)
    q_next = q_next.at[3:6].set(jnp.mod(q_next[3:6] + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
    return q_next


class JaxRobotSoftwareDynamics:
    def __init__(
        self,
        casadi_dir: Path,
        model_name: str,
        dtype: jnp.dtype,
        *,
        actuator_casadi_dir: Path | None = None,
        use_friction_model: bool = True,
        use_voltage_limits: bool = True,
    ) -> None:
        self.dtype = dtype
        self.mass_bias = _convert_casadi_file(
            casadi_dir / f"{model_name}_mass_matrix_and_bias_force_vector.casadi",
            "jax_mass_matrix_and_bias_force_vector",
        )
        self.contact_pos_jac = _convert_casadi_file(
            casadi_dir / f"{model_name}_contact_points_pos_jac.casadi",
            "jax_contact_points_pos_jac",
        )
        actuator_dirs = [p for p in (actuator_casadi_dir, casadi_dir) if p is not None]
        self.low_level_cmd = _load_optional_jax_function(
            actuator_dirs,
            "humanoid_low_level_cmd.casadi",
            "jax_low_level_cmd",
        )
        self.friction_model = (
            _load_optional_jax_function(actuator_dirs, "friction_model.casadi", "jax_friction_model")
            if use_friction_model
            else None
        )
        self.voltage_limits = (
            _load_optional_jax_function(
                actuator_dirs,
                "actuator_voltage_saturation_limits.casadi",
                "jax_actuator_voltage_saturation_limits",
            )
            if use_voltage_limits
            else None
        )
        self.saturation_limits = _load_optional_jax_function(
            actuator_dirs,
            "actuator_saturation_limits.casadi",
            "jax_actuator_saturation_limits",
        )
        self._tau_static_max = jnp.asarray(ACTUATOR_STATIC_TAU_MAX_NM, dtype=dtype)
        if self.saturation_limits is not None:
            saturation_outputs = _outputs(self.saturation_limits())
            if len(saturation_outputs) >= 1:
                tau_max = np.asarray(jax.device_get(saturation_outputs[0]), dtype=np.float64).reshape(-1)
                if tau_max.size == HUMANOID_NTAU:
                    self._tau_static_max = jnp.asarray(np.abs(tau_max), dtype=dtype)

    def mass_bias_terms(self, q: jax.Array, dq: jax.Array) -> tuple[jax.Array, jax.Array]:
        h_mat, bias = self.mass_bias(q, dq)
        return jnp.asarray(h_mat, dtype=self.dtype), jnp.ravel(jnp.asarray(bias, dtype=self.dtype))

    def contacts(self, q: jax.Array) -> tuple[jax.Array, jax.Array]:
        outputs = self.contact_pos_jac(q)
        positions = jnp.stack(
            [jnp.ravel(jnp.asarray(output, dtype=self.dtype)) for output in outputs[:4]],
            axis=0,
        )
        jacobians = jnp.stack(
            [jnp.asarray(output, dtype=self.dtype).reshape(3, HUMANOID_NDQ) for output in outputs[4:]],
            axis=0,
        )
        return positions, jacobians

    def low_level_torque(
        self,
        q_joint: jax.Array,
        dq_joint: jax.Array,
        tau_ff: jax.Array,
        q_des: jax.Array,
        dq_des: jax.Array,
        kp: jax.Array,
        kd: jax.Array,
    ) -> jax.Array:
        if self.low_level_cmd is None:
            tau = tau_ff + kp * (q_des - q_joint) + kd * (dq_des - dq_joint)
        else:
            tau = jnp.ravel(
                jnp.asarray(
                    _single_output(self.low_level_cmd(q_joint, dq_joint, tau_ff, q_des, dq_des, kp, kd)),
                    dtype=self.dtype,
                )
            )
        return self.apply_actuator_effects(tau, q_joint, dq_joint)

    def apply_actuator_effects(self, tau: jax.Array, q_joint: jax.Array, dq_joint: jax.Array) -> jax.Array:
        tau_min = -self._tau_static_max
        tau_max = self._tau_static_max
        if self.voltage_limits is not None:
            lower, upper = self.voltage_limits(dq_joint, jnp.asarray([60.0], dtype=self.dtype))
            tau_min = jnp.maximum(tau_min, jnp.ravel(jnp.asarray(lower, dtype=self.dtype)))
            tau_max = jnp.minimum(tau_max, jnp.ravel(jnp.asarray(upper, dtype=self.dtype)))
        tau_limited = jnp.clip(tau, tau_min, tau_max)
        if self.friction_model is None:
            return tau_limited
        return jnp.ravel(
            jnp.asarray(
                _single_output(self.friction_model(tau_limited, q_joint, dq_joint)),
                dtype=self.dtype,
            )
        )


class BatchedRobotSoftwarePlant:
    def __init__(
        self,
        dynamics: JaxRobotSoftwareDynamics,
        *,
        batch_size: int,
        dt: float,
        dtype: jnp.dtype,
        contact_iterations: int = DEFAULT_CONTACT_ITERATIONS,
    ) -> None:
        self.dynamics = dynamics
        self.batch_size = int(batch_size)
        self.dt = float(dt)
        self.dtype = dtype
        self.contact_iterations = int(contact_iterations)
        self._soft_lower = jnp.asarray(SOFT_STOP_LOWER, dtype=dtype)
        self._soft_upper = jnp.asarray(SOFT_STOP_UPPER, dtype=dtype)
        self._step_jit = jax.jit(jax.vmap(self._step_one, in_axes=(0, 0, 0, 0, 0, 0, 0)))
        self.q = jnp.zeros((batch_size, HUMANOID_NQ), dtype=dtype)
        self.dq = jnp.zeros((batch_size, HUMANOID_NDQ), dtype=dtype)

    def reset(self, q: np.ndarray) -> None:
        self.q = jnp.asarray(q, dtype=self.dtype)
        self.dq = jnp.zeros((self.batch_size, HUMANOID_NDQ), dtype=self.dtype)

    def set_state(self, q: jax.Array, dq: jax.Array) -> None:
        self.q = q
        self.dq = dq

    def step(
        self,
        q_des: jax.Array,
        dq_des: jax.Array,
        tau_ff: jax.Array,
        kp: jax.Array,
        kd: jax.Array,
    ) -> None:
        self.q, self.dq = self._step_jit(self.q, self.dq, q_des, dq_des, tau_ff, kp, kd)

    def _apply_soft_stops(
        self,
        q: jax.Array,
        q_des: jax.Array,
        dq_des: jax.Array,
        tau_ff: jax.Array,
        kp: jax.Array,
        kd: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        q_joint = q[6:]
        lower_active = q_joint < self._soft_lower
        upper_active = q_joint > self._soft_upper
        active = jnp.logical_or(lower_active, upper_active)
        q_des = jnp.where(lower_active, self._soft_lower, q_des)
        q_des = jnp.where(upper_active, self._soft_upper, q_des)
        dq_des = jnp.where(active, 0.0, dq_des)
        tau_ff = jnp.where(active, 0.0, tau_ff)
        kp = jnp.where(active, SOFT_STOP_KP, kp)
        kd = jnp.where(active, SOFT_STOP_KD, kd)
        return q_des, dq_des, tau_ff, kp, kd

    def _step_one(
        self,
        q: jax.Array,
        dq: jax.Array,
        q_des: jax.Array,
        dq_des: jax.Array,
        tau_ff: jax.Array,
        kp: jax.Array,
        kd: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        q_des, dq_des, tau_ff, kp, kd = self._apply_soft_stops(q, q_des, dq_des, tau_ff, kp, kd)
        h_mat, bias = self.dynamics.mass_bias_terms(q, dq)
        contact_positions, contact_jacobians = self.dynamics.contacts(q)
        tau_joint = self.dynamics.low_level_torque(q[6:], dq[6:], tau_ff, q_des, dq_des, kp, kd)
        tau_gen = jnp.zeros(HUMANOID_NDQ, dtype=self.dtype).at[6:].set(tau_joint)
        qdd = jnp.linalg.solve(h_mat, tau_gen - bias)
        dq_next = dq + qdd * self.dt
        dq_next = self._resolve_impulse_contacts(dq, dq_next, h_mat, contact_positions, contact_jacobians)
        q_next = _jax_integrate_q(q, dq_next, self.dt, self.dtype)
        return q_next, dq_next

    def _resolve_impulse_contacts(
        self,
        dq_before: jax.Array,
        dq_after: jax.Array,
        h_mat: jax.Array,
        contact_positions: jax.Array,
        contact_jacobians: jax.Array,
    ) -> jax.Array:
        active = contact_positions[:, 2] < GROUND_HEIGHT
        h_inv = jnp.linalg.inv(h_mat)
        qdot = dq_after
        local_impulses = jnp.zeros((4, 3), dtype=self.dtype)
        old_vels = jnp.stack([contact_jacobians[i] @ dq_before for i in range(4)], axis=0)
        desired_z = jnp.where(old_vels[:, 2] < 0.0, -GROUND_RESTITUTION * old_vels[:, 2], old_vels[:, 2])
        desired_z = jnp.where(active, desired_z, 0.0)
        desired_tangent = jnp.zeros(4, dtype=self.dtype)
        min_z = jnp.zeros(4, dtype=self.dtype)
        max_z = jnp.full(4, 1.0e5, dtype=self.dtype)
        lambdas, ainvb = self._contact_impulse_maps(active, h_inv, contact_jacobians)
        for _outer in range(self.contact_iterations):
            qdot, local_impulses = self._update_qdot_one_direction(
                2,
                contact_jacobians,
                lambdas[:, 2],
                ainvb[:, 2, :],
                desired_z,
                min_z,
                max_z,
                local_impulses,
                qdot,
            )
            max_tangent = GROUND_MU * local_impulses[:, 2]
            min_tangent = -max_tangent
            for axis in (0, 1):
                qdot, local_impulses = self._update_qdot_one_direction(
                    axis,
                    contact_jacobians,
                    lambdas[:, axis],
                    ainvb[:, axis, :],
                    desired_tangent,
                    min_tangent,
                    max_tangent,
                    local_impulses,
                    qdot,
                )
        return qdot

    def _contact_impulse_maps(
        self,
        active: jax.Array,
        h_inv: jax.Array,
        contact_jacobians: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        lambda_rows = []
        ainvb_rows = []
        for i in range(4):
            lambda_axes = []
            ainvb_axes = []
            jac = contact_jacobians[i]
            for axis in range(3):
                direction = jnp.zeros(3, dtype=self.dtype).at[axis].set(1.0)
                impulse_map = h_inv @ jac.T @ direction
                lambda_inv = direction @ (jac @ impulse_map)
                valid = jnp.logical_and(active[i], jnp.abs(lambda_inv) >= 1.0e-12)
                safe_lambda_inv = jnp.where(jnp.abs(lambda_inv) >= 1.0e-12, lambda_inv, 1.0)
                lambda_axes.append(jnp.where(valid, 1.0 / safe_lambda_inv, 0.0))
                ainvb_axes.append(jnp.where(valid, impulse_map, jnp.zeros(HUMANOID_NDQ, dtype=self.dtype)))
            lambda_rows.append(jnp.stack(lambda_axes, axis=0))
            ainvb_rows.append(jnp.stack(ainvb_axes, axis=0))
        return jnp.stack(lambda_rows, axis=0), jnp.stack(ainvb_rows, axis=0)

    def _update_qdot_one_direction(
        self,
        axis: int,
        contact_jacobians: jax.Array,
        lambda_list: jax.Array,
        ainvb_list: jax.Array,
        desired_vel: jax.Array,
        min_impulse: jax.Array,
        max_impulse: jax.Array,
        local_impulses: jax.Array,
        qdot: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        for _ in range(self.contact_iterations):
            for i in range(4):
                valid = lambda_list[i] != 0.0
                cp_vel = contact_jacobians[i][axis, :] @ qdot
                dvel = desired_vel[i] - cp_vel
                dforce = dvel * lambda_list[i]
                pre_force = local_impulses[i, axis]
                next_force = jnp.clip(pre_force + dforce, min_impulse[i], max_impulse[i])
                applied_force = jnp.where(valid, next_force - pre_force, 0.0)
                local_impulses = local_impulses.at[i, axis].set(pre_force + applied_force)
                qdot = qdot + ainvb_list[i] * applied_force
        return qdot, local_impulses


def make_interpolator(n_nodes: int, n_interp: int, interpolation_dt: float):
    offsets = jnp.arange(n_interp, dtype=jnp.int32)

    @jax.jit
    def interpolate(z: jax.Array, dt: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        q, dq, tau, _force = humanoid_jax_trajectory_from_solution(z, n_nodes=n_nodes)
        elements = jnp.rint(dt / interpolation_dt).astype(jnp.int32)
        cumulative = jnp.cumsum(elements)
        seg_idx = jnp.sum(offsets[:, None] >= cumulative[None, :], axis=1)
        seg_idx = jnp.minimum(seg_idx, n_nodes - 2)
        cumulative_prev = jnp.where(seg_idx > 0, cumulative[seg_idx - 1], 0)
        alpha = (offsets - cumulative_prev).astype(z.dtype) * interpolation_dt / dt[seg_idx]
        next_seg = jnp.minimum(seg_idx + 1, n_nodes - 1)
        tau_seg = jnp.minimum(seg_idx, tau.shape[2] - 1)
        next_tau_seg = jnp.minimum(seg_idx + 1, tau.shape[2] - 1)
        q0 = jnp.take(q[:, 6:, :], seg_idx, axis=2)
        q1 = jnp.take(q[:, 6:, :], next_seg, axis=2)
        dq0 = jnp.take(dq[:, 6:, :], seg_idx, axis=2)
        dq1 = jnp.take(dq[:, 6:, :], next_seg, axis=2)
        tau0 = jnp.take(tau, tau_seg, axis=2)
        tau1 = jnp.take(tau, next_tau_seg, axis=2)
        alpha_b = alpha[None, None, :]
        q_cmd = q0 + alpha_b * (q1 - q0)
        dq_cmd = dq0 + alpha_b * (dq1 - dq0)
        tau_alpha = HUMANOID_TORQUE_FEEDFORWARD_LOOKAHEAD_S / dt[tau_seg]
        tau_cmd = tau0 + tau_alpha[None, None, :] * (tau1 - tau0)
        return q_cmd, dq_cmd, tau_cmd

    return interpolate


def _save_rollout(path: Path, arrays: dict[str, Any], *, save_pickle: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl" or save_pickle:
        with path.open("wb") as f:
            pickle.dump(arrays, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        np.savez(path, **arrays)


def _stat_mask(mask: Any | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    mask_array = np.asarray(jax.device_get(mask), dtype=bool)
    if mask_array.shape != shape:
        mask_array = np.broadcast_to(mask_array, shape)
    return mask_array


def _finite_mean(value: Any, mask: Any | None = None) -> float:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    finite = np.isfinite(array)
    mask_array = _stat_mask(mask, array.shape)
    if mask_array is not None:
        finite &= mask_array
    if not np.any(finite):
        return float("nan")
    return float(np.mean(array[finite]))


def _finite_max(value: Any, mask: Any | None = None) -> float:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    finite = np.isfinite(array)
    mask_array = _stat_mask(mask, array.shape)
    if mask_array is not None:
        finite &= mask_array
    if not np.any(finite):
        return float("nan")
    return float(np.max(array[finite]))


def _resample_commands(
    rng: np.random.Generator,
    commands: np.ndarray,
    next_sample_time: np.ndarray,
    walking_time: float,
    *,
    enabled: bool,
) -> None:
    if not enabled or walking_time < 1.0:
        return
    mask = walking_time + 1e-12 >= next_sample_time
    count = int(mask.sum())
    if count == 0:
        return
    commands[mask, 0] = rng.uniform(-1.0, 1.0, size=count)
    commands[mask, 1] = rng.uniform(-0.2, 0.2, size=count)
    commands[mask, 2] = rng.uniform(-1.0, 1.0, size=count)
    commands[mask, 3] = rng.uniform(0.5, 0.7, size=count)
    next_sample_time[mask] = walking_time + rng.uniform(0.7, 3.0, size=count)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dtype = np.dtype(args.dtype)
    jax_dtype = jnp.float64 if dtype == np.dtype("float64") else jnp.float32
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    rng = np.random.default_rng(args.seed)
    loaded_params = load_humanoid_mpc_parameters(args.parameters)
    if args.horizon_nodes is None:
        args.horizon_nodes = loaded_params.n_nodes
    if args.max_iter is None:
        args.max_iter = loaded_params.osqp_max_iter
    if args.rho is None:
        args.rho = loaded_params.osqp_rho
    if args.alpha is None:
        args.alpha = loaded_params.osqp_alpha
    if args.scaling is None:
        args.scaling = loaded_params.osqp_scaling
    if args.line_search_step_min is None:
        args.line_search_step_min = loaded_params.line_search_step_min
    params = replace(loaded_params, n_nodes=args.horizon_nodes)
    grid_xy = make_grid_xy(args.batch_size, args.grid_spacing)
    q_initial = make_initial_q(args.batch_size, grid_xy, params.qhome).astype(dtype)

    dynamics = JaxRobotSoftwareDynamics(
        args.casadi_dir,
        args.model_name,
        jax_dtype,
        actuator_casadi_dir=args.actuator_casadi_dir,
        use_friction_model=not args.no_friction_model,
        use_voltage_limits=not args.no_voltage_limits,
    )
    plant = BatchedRobotSoftwarePlant(
        dynamics,
        batch_size=args.batch_size,
        dt=args.sim_dt,
        dtype=jax_dtype,
        contact_iterations=args.contact_iterations,
    )
    plant.reset(q_initial)

    z0, params0 = humanoid_initial_guess_and_params(
        q_initial,
        np.zeros((args.batch_size, HUMANOID_NDQ), dtype=dtype),
        n_nodes=args.horizon_nodes,
        dtype=dtype,
        parameters=params,
    )
    plan_commands = np.zeros((args.batch_size, 4), dtype=np.float64)
    plan_commands[:, 3] = params.qhome[2]
    if args.mpc_standing_s > 0.0:
        plan_gait = standing_gait_schedule(args.horizon_nodes, params.discretization_times)
    else:
        plan_gait = humanoid_walking_gait_schedule(
            0.0,
            n_nodes=args.horizon_nodes,
            parameters=params,
        )
    plan_qref, plan_dqref = humanoid_make_references(
        params,
        q_initial,
        plan_commands,
        plan_gait.discretization_times,
        command_dt=HUMANOID_HIGH_LEVEL_TIMESTEP,
    )
    plan_params0 = update_humanoid_params_mpc(
        params0,
        q0=q_initial,
        dq0=np.zeros((args.batch_size, HUMANOID_NDQ), dtype=dtype),
        tau0=np.zeros((args.batch_size, HUMANOID_NTAU), dtype=dtype),
        reference_contacts=plan_gait.gait_pattern,
        reference_foot_height=plan_gait.foot_height,
        reference_q=plan_qref,
        reference_dq=plan_dqref,
        discretization_times=plan_gait.discretization_times,
        weights=params,
    )
    settings = OSQPSettings(
        rho=args.rho,
        sigma=args.sigma,
        alpha=args.alpha,
        max_iter=args.max_iter,
        scaling=args.scaling,
        adaptive_rho=False,
        rho_is_vec=True,
        check_termination=0,
        warm_starting=True,
        polishing=False,
    )
    print("building humanoid sparse SQP problem", flush=True)
    problem = make_humanoid_sqp_problem(args.horizon_nodes, model_name=args.model_name)
    plan = build_sparse_mpc_plan(
        problem,
        osqp_settings=settings,
        representative_z=z0,
        representative_params=plan_params0,
    )
    sqp = compile_sparse_mpc_sqp(
        problem,
        plan,
        dtype=dtype,
        osqp_settings=settings,
        transpose_work=True,
        segmented=True,
        segment_budget=args.segment_budget,
        segment_strategy=args.segment_strategy,
        level_scheduled_solve=args.level_scheduled_solve,
        qdldl_backend=args.qdldl_backend,
        qdldl_factor_backend=args.qdldl_factor_backend,
        qdldl_solve_backend=args.qdldl_solve_backend,
        line_search_settings=FilterLineSearchSettings(
            line_search_step_min=args.line_search_step_min,
            line_search_constraint_scale=params.line_search_constraint_scale,
            line_search_g_max=params.line_search_g_max,
            line_search_g_min=params.line_search_g_min,
            line_search_gamma_c=params.line_search_gamma_c,
            line_search_armijo_factor=params.line_search_armijo_factor,
            line_search_step_decay=params.line_search_step_decay,
            line_search_cost_accept_uses_trial_cost=False,
        ),
        group_repeated_stages=args.group_repeated_stages,
    )
    z = jnp.asarray(z0, dtype=jax_dtype)
    params_np = params0
    solver_state = sqp.init_state(args.batch_size)
    interpolate = make_interpolator(args.horizon_nodes, args.mpc_period_steps, args.interpolation_dt)

    total_s = args.stabilize_s + args.mpc_standing_s + args.walking_s
    total_steps = int(round(total_s / args.sim_dt))
    mpc_start_step = int(round(args.stabilize_s / args.sim_dt))
    record_stride = max(1, int(round(args.record_dt / args.sim_dt)))
    record_steps = total_steps // record_stride + 1
    q_history = np.empty((record_steps, args.batch_size, HUMANOID_NQ), dtype=dtype)
    dq_history = np.empty((record_steps, args.batch_size, HUMANOID_NDQ), dtype=dtype)
    command_history = np.empty((record_steps, args.batch_size, 4), dtype=dtype)
    time_history = np.empty(record_steps, dtype=np.float64)

    commands = np.zeros((args.batch_size, 4), dtype=np.float64)
    commands[:, 3] = params.qhome[2]
    next_sample_time = np.ones(args.batch_size, dtype=np.float64)

    pd_q = jnp.asarray(np.broadcast_to(params.qhome[6:], (args.batch_size, HUMANOID_NTAU)), dtype=jax_dtype)
    pd_dq = jnp.zeros((args.batch_size, HUMANOID_NTAU), dtype=jax_dtype)
    pd_tau = jnp.zeros((args.batch_size, HUMANOID_NTAU), dtype=jax_dtype)
    pd_kp = jnp.asarray(
        np.broadcast_to(np.concatenate([np.full(10, 400.0), np.full(8, 50.0)]), (args.batch_size, HUMANOID_NTAU)),
        dtype=jax_dtype,
    )
    pd_kd = jnp.asarray(
        np.broadcast_to(np.concatenate([np.full(10, 3.0), np.full(8, 1.0)]), (args.batch_size, HUMANOID_NTAU)),
        dtype=jax_dtype,
    )
    q_cmd_buf = pd_q[:, :, None].repeat(args.mpc_period_steps, axis=2)
    dq_cmd_buf = pd_dq[:, :, None].repeat(args.mpc_period_steps, axis=2)
    tau_cmd_buf = pd_tau[:, :, None].repeat(args.mpc_period_steps, axis=2)
    last_tau_ff = np.zeros((args.batch_size, HUMANOID_NTAU), dtype=dtype)
    interp_index = 0
    solve_index = 0
    failed = jnp.zeros(args.batch_size, dtype=bool)
    failure_time = np.full(args.batch_size, np.nan, dtype=np.float64)
    failure_reason = np.zeros(args.batch_size, dtype=np.int8)
    freeze_q = plant.q
    freeze_dq = plant.dq

    def record(record_index: int, sim_time: float) -> None:
        q_history[record_index] = np.asarray(jax.device_get(plant.q), dtype=dtype)
        dq_history[record_index] = np.asarray(jax.device_get(plant.dq), dtype=dtype)
        command_history[record_index] = commands.astype(dtype)
        time_history[record_index] = sim_time

    record_index = 0
    aborted_all_diverged = False
    completed_sim_time = 0.0

    def record_once(sim_time: float) -> None:
        nonlocal record_index
        if record_index >= record_steps:
            return
        if record_index > 0 and np.isclose(time_history[record_index - 1], sim_time):
            return
        record(record_index, sim_time)
        record_index += 1

    record_once(0.0)
    start = time.perf_counter()
    for step in range(total_steps):
        sim_time = step * args.sim_dt
        completed_sim_time = sim_time
        prev_q = plant.q
        prev_dq = plant.dq
        if step < mpc_start_step:
            q_des, dq_des, tau_des, kp, kd = pd_q, pd_dq, pd_tau, pd_kp, pd_kd
        else:
            mpc_time = (step - mpc_start_step) * args.sim_dt
            walking_time = mpc_time - args.mpc_standing_s
            _resample_commands(
                rng,
                commands,
                next_sample_time,
                walking_time,
                enabled=args.randomize_references,
            )
            if interp_index == 0:
                q_cpu = np.asarray(jax.device_get(plant.q), dtype=dtype)
                dq_cpu = np.asarray(jax.device_get(plant.dq), dtype=dtype)
                if mpc_time < args.mpc_standing_s:
                    gait = standing_gait_schedule(args.horizon_nodes, params.discretization_times)
                else:
                    gait = humanoid_walking_gait_schedule(
                        walking_time,
                        n_nodes=args.horizon_nodes,
                        parameters=params,
                    )
                qref, dqref = humanoid_make_references(
                    params,
                    q_cpu,
                    commands,
                    gait.discretization_times,
                    command_dt=HUMANOID_HIGH_LEVEL_TIMESTEP,
                )
                params_np = update_humanoid_params_mpc(
                    params_np,
                    q0=q_cpu,
                    dq0=dq_cpu,
                    tau0=last_tau_ff,
                    reference_contacts=gait.gait_pattern,
                    reference_foot_height=gait.foot_height,
                    reference_q=qref,
                    reference_dq=dqref,
                    discretization_times=gait.discretization_times,
                    weights=params,
                )
                result, solver_state = sqp.step(z, jnp.asarray(params_np, dtype=jax_dtype), state=solver_state)
                z = result.z_next
                q_cmd_buf, dq_cmd_buf, tau_cmd_buf = interpolate(
                    z,
                    jnp.asarray(gait.discretization_times, dtype=jax_dtype),
                )
                mpc_cost = result.line_search.cost
                cost_diverged = (~jnp.isfinite(mpc_cost)) | (
                    mpc_cost > args.mpc_cost_divergence_threshold
                )
                new_cost_failed = (~failed) & cost_diverged
                if bool(jax.device_get(jnp.any(new_cost_failed))):
                    new_cost_failed_np = np.asarray(jax.device_get(new_cost_failed))
                    failure_time[new_cost_failed_np] = sim_time
                    failure_reason[new_cost_failed_np] = 2
                freeze_q = jnp.where(new_cost_failed[:, None], prev_q, freeze_q)
                freeze_dq = jnp.where(new_cost_failed[:, None], prev_dq, freeze_dq)
                failed = failed | new_cost_failed
                alive_mask = np.asarray(jax.device_get(~failed), dtype=bool)
                alive_count = int(alive_mask.sum())
                alive_fraction = alive_count / args.batch_size
                if solve_index % args.log_every == 0 or alive_count == 0:
                    step_len, violation, prim, dual, cost = jax.block_until_ready(
                        (
                            result.line_search.step_length,
                            result.line_search.constraint_violation,
                            result.solve.prim_res,
                            result.solve.dual_res,
                            mpc_cost,
                        )
                    )
                    elapsed = time.perf_counter() - start
                    prefix = (
                        f"solve={solve_index:05d} t={mpc_time:7.3f}s "
                        f"elapsed={elapsed:8.2f}s "
                        f"alive={alive_count}/{args.batch_size} ({100.0 * alive_fraction:5.1f}%)"
                    )
                    if alive_count > 0:
                        print(
                            f"{prefix} "
                            f"mean_step={_finite_mean(step_len, alive_mask):.3g} "
                            f"max_viol={_finite_max(violation, alive_mask):.3e} "
                            f"max_cost={_finite_max(cost, alive_mask):.3e} "
                            f"mean_qp=({_finite_mean(prim, alive_mask):.2e}, "
                            f"{_finite_mean(dual, alive_mask):.2e})",
                            flush=True,
                        )
                    else:
                        print(f"{prefix} no_non_diverged_runs", flush=True)
                if alive_count == 0:
                    aborted_all_diverged = True
                    completed_sim_time = sim_time
                    plant.set_state(freeze_q, freeze_dq)
                    record_once(completed_sim_time)
                    print(
                        f"all simulations diverged at t={completed_sim_time:.3f}s; aborting",
                        flush=True,
                    )
                    break
                solve_index += 1
            q_des = q_cmd_buf[:, :, interp_index]
            dq_des = dq_cmd_buf[:, :, interp_index]
            tau_des = tau_cmd_buf[:, :, interp_index]
            gait_now = (
                standing_gait_schedule(1, params.discretization_times)
                if mpc_time < args.mpc_standing_s
                else humanoid_walking_gait_schedule(
                    walking_time,
                    n_nodes=1,
                    parameters=params,
                )
            )
            kp_np, kd_np = gains_for_contact(params, gait_now.gait_pattern[:, 0])
            if params.gains_stance_velocity_setpoint_to_zero:
                if int(gait_now.gait_pattern[0, 0]) == 1:
                    dq_des = dq_des.at[:, :5].set(0.0)
                if int(gait_now.gait_pattern[1, 0]) == 1:
                    dq_des = dq_des.at[:, 5:10].set(0.0)
            kp = jnp.asarray(np.broadcast_to(kp_np, (args.batch_size, HUMANOID_NTAU)), dtype=jax_dtype)
            kd = jnp.asarray(np.broadcast_to(kd_np, (args.batch_size, HUMANOID_NTAU)), dtype=jax_dtype)
            last_tau_ff = np.asarray(jax.device_get(tau_des), dtype=dtype)
            interp_index = (interp_index + 1) % args.mpc_period_steps

        plant.step(q_des, dq_des, tau_des, kp, kd)
        z_height = plant.q[:, 2]
        finite = jnp.all(jnp.isfinite(plant.q), axis=1) & jnp.all(jnp.isfinite(plant.dq), axis=1)
        violated = (z_height < args.min_body_z) | (z_height > args.max_body_z) | (~finite)
        new_failed = (~failed) & violated
        if bool(jax.device_get(jnp.any(new_failed))):
            new_failed_np = np.asarray(jax.device_get(new_failed))
            failure_time[new_failed_np] = sim_time + args.sim_dt
            failure_reason[new_failed_np] = 1
        freeze_q = jnp.where(new_failed[:, None], prev_q, freeze_q)
        freeze_dq = jnp.where(new_failed[:, None], prev_dq, freeze_dq)
        failed = failed | new_failed
        plant.set_state(
            jnp.where(failed[:, None], freeze_q, plant.q),
            jnp.where(failed[:, None], freeze_dq, plant.dq),
        )
        if (step + 1) % record_stride == 0:
            record_once(sim_time + args.sim_dt)
        completed_sim_time = sim_time + args.sim_dt
        if bool(jax.device_get(jnp.all(failed))):
            aborted_all_diverged = True
            record_once(completed_sim_time)
            print(
                f"all simulations diverged at t={completed_sim_time:.3f}s; aborting",
                flush=True,
            )
            break

    q_history = q_history[:record_index]
    dq_history = dq_history[:record_index]
    command_history = command_history[:record_index]
    time_history = time_history[:record_index]
    failed_np = np.asarray(jax.device_get(failed))
    non_diverged_fraction = 1.0 - float(failed_np.sum()) / args.batch_size
    metadata = {
        "batch_size": args.batch_size,
        "model_name": args.model_name,
        "horizon_nodes": args.horizon_nodes,
        "dtype": args.dtype,
        "sim_dt": args.sim_dt,
        "interpolation_dt": args.interpolation_dt,
        "mpc_period_s": args.mpc_period_s,
        "mpc_period_steps": args.mpc_period_steps,
        "record_dt": args.record_dt,
        "stabilize_s": args.stabilize_s,
        "mpc_standing_s": args.mpc_standing_s,
        "walking_s": args.walking_s,
        "randomize_references": args.randomize_references,
        "min_body_z": args.min_body_z,
        "max_body_z": args.max_body_z,
        "mpc_cost_divergence_threshold": args.mpc_cost_divergence_threshold,
        "qdldl_backend": args.qdldl_backend,
        "qdldl_factor_backend": args.qdldl_factor_backend,
        "qdldl_solve_backend": args.qdldl_solve_backend,
        "level_scheduled_solve": bool(args.level_scheduled_solve),
        "segment_budget": args.segment_budget,
        "segment_strategy": args.segment_strategy,
        "group_repeated_stages": bool(args.group_repeated_stages),
        "aborted_all_diverged": aborted_all_diverged,
        "completed_sim_time_s": completed_sim_time,
        "non_diverged_fraction": non_diverged_fraction,
        "failed_count": int(failed_np.sum()),
        "elapsed_s": time.perf_counter() - start,
    }
    arrays = {
        "time": time_history,
        "q": q_history,
        "dq": dq_history,
        "commands": command_history,
        "grid_xy": grid_xy.astype(dtype),
        "failed": failed_np,
        "failure_time": failure_time,
        "failure_reason": failure_reason,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    _save_rollout(args.output, arrays, save_pickle=args.save_pickle)
    print(f"wrote {args.output}", flush=True)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--horizon-nodes",
        type=int,
        default=None,
        help="Override the humanoid horizon node count from the parameter YAML.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--model-name", default=HUMANOID_MODEL_NAME)
    parser.add_argument("--parameters", type=Path, default=HUMANOID_PARAMETERS_PATH)
    parser.add_argument("--casadi-dir", type=Path, default=HUMANOID_CASADI_FUNCTION_DIR)
    parser.add_argument("--actuator-casadi-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/humanoid/headless_humanoid_10k.npz"))
    parser.add_argument("--save-pickle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-spacing", type=float, default=1.4)
    parser.add_argument("--sim-dt", type=float, default=HUMANOID_SIM_TIMESTEP)
    parser.add_argument("--interpolation-dt", type=float, default=HUMANOID_LOW_LEVEL_TIMESTEP)
    parser.add_argument("--mpc-period-s", type=float, default=HUMANOID_MPC_PERIOD_S)
    parser.add_argument("--record-dt", type=float, default=0.01)
    parser.add_argument("--stabilize-s", type=float, default=2.0)
    parser.add_argument("--mpc-standing-s", type=float, default=HUMANOID_MPC_STANDING_REFERENCE_S)
    parser.add_argument("--walking-s", type=float, default=10.0)
    parser.add_argument("--min-body-z", type=float, default=0.6)
    parser.add_argument("--max-body-z", type=float, default=0.67)
    parser.add_argument("--mpc-cost-divergence-threshold", type=float, default=5_000.0)
    parser.add_argument("--randomize-references", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-iterations", type=int, default=DEFAULT_CONTACT_ITERATIONS)
    parser.add_argument("--no-friction-model", action="store_true")
    parser.add_argument("--no-voltage-limits", action="store_true")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--scaling", type=int, default=None)
    parser.add_argument("--segment-budget", type=int, default=96)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--line-search-step-min", type=float, default=None)
    parser.add_argument("--no-group-repeated-stages", action="store_false", dest="group_repeated_stages")
    parser.set_defaults(group_repeated_stages=True)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    args.mpc_period_steps = int(round(args.mpc_period_s / args.sim_dt))
    if not np.isclose(args.mpc_period_steps * args.sim_dt, args.mpc_period_s):
        raise ValueError("--mpc-period-s must be an integer multiple of --sim-dt")
    return args


if __name__ == "__main__":
    run(parse_args())
