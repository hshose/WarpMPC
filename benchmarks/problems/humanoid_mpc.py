"""Humanoid walking MPC in this repo's sparse SQP style.

The formulation mirrors the tuned MPC used by the Robot Software humanoid
debug simulations.  The first two transitions carry joint torque variables and
full rigid-body dynamics.  Later transitions keep only floating-base dynamics,
joint acceleration smoothing, contact forces, and kinematic constraints.  The
terminal node contains only ``q, dq`` and terminal/contact constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import jax.numpy as jnp
import numpy as np

from warpmpc.jax_sqp import CasadiStageFunction, SparseMPCProblem


HUMANOID_NQ = 24
HUMANOID_NDQ = 24
HUMANOID_NDDQ = 18
HUMANOID_NTAU = 18
HUMANOID_NF = 12
HUMANOID_N_FB = 6
HUMANOID_N_LEG = 10
HUMANOID_N_ARM = 8
HUMANOID_N_TORQUE_STAGES = 2
HUMANOID_NZ_TORQUE = HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU + HUMANOID_NF
HUMANOID_NZ_FORCE = HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NF
HUMANOID_NZ_TERMINAL = HUMANOID_NQ + HUMANOID_NDQ
HUMANOID_NZ = HUMANOID_NZ_TORQUE
HUMANOID_N_NODES = 28
HUMANOID_N_STEPS = HUMANOID_N_NODES - 1
HUMANOID_INFTY = 1e30
HUMANOID_MASS = 24.868527
HUMANOID_G = 9.81
HUMANOID_PHI_VEL = 1.0 / 0.6
HUMANOID_MODEL_NAME = "no_rotors"
HUMANOID_CASADI_FUNCTION_DIR = Path(__file__).resolve().parent / "humanoid_casadi_functions"
HUMANOID_PARAMETERS_PATH = Path(__file__).with_name("humanoid_mpc_parameters.yaml")

HUMANOID_SIM_TIMESTEP = 0.0005
HUMANOID_LOW_LEVEL_TIMESTEP = 0.001
HUMANOID_HIGH_LEVEL_TIMESTEP = 0.002
HUMANOID_MPC_PERIOD_S = 0.010
HUMANOID_MPC_PERIOD_STEPS = int(round(HUMANOID_MPC_PERIOD_S / HUMANOID_SIM_TIMESTEP))
HUMANOID_MPC_STANDING_REFERENCE_S = 0.7
HUMANOID_TORQUE_FEEDFORWARD_LOOKAHEAD_S = 0.010
HUMANOID_COLLISION_SPHERE_RADIUS = 0.06

HUMANOID_REFERENCE_VELOCITY_X_LIMIT = 1.0
HUMANOID_REFERENCE_VELOCITY_Y_LIMIT = 0.2
HUMANOID_REFERENCE_YAW_RATE_LIMIT = 1.0
HUMANOID_REFERENCE_BODY_HEIGHT_MIN = 0.60
HUMANOID_REFERENCE_BODY_HEIGHT_MAX = 0.67

ACTUATOR_STATIC_TAU_MIN_NM = np.array(
    [
        -13.998,
        -25.998,
        -72.000,
        -195.996,
        -51.996,
        -13.998,
        -25.998,
        -72.000,
        -195.996,
        -51.996,
        -13.998,
        -13.998,
        -13.998,
        -20.997,
        -13.998,
        -13.998,
        -13.998,
        -20.997,
    ],
    dtype=np.float64,
)
ACTUATOR_STATIC_TAU_MAX_NM = -ACTUATOR_STATIC_TAU_MIN_NM


@dataclass(frozen=True)
class HumanoidMPCParameters:
    qhome: np.ndarray
    cost_q: np.ndarray
    cost_dq: np.ndarray
    cost_ddq: np.ndarray
    cost_f: np.ndarray
    cost_tau: np.ndarray
    gains_leg_swing_p: np.ndarray
    gains_leg_swing_d: np.ndarray
    gains_leg_stance_p: np.ndarray
    gains_leg_stance_d: np.ndarray
    gains_arm_p: np.ndarray
    gains_arm_d: np.ndarray
    gains_stance_velocity_setpoint_to_zero: bool
    n_nodes: int
    dt_start: float
    dt_exponent: float
    gait_cycle_duration: float
    gait_overlap_duration: float
    gait_right_swing_duration: float
    gait_left_swing_duration: float
    foot_lift_height: float
    osqp_alpha: float
    osqp_rho: float
    osqp_max_iter: int
    osqp_scaling: int
    osqp_print_status: bool
    line_search_g_max: float
    line_search_g_min: float
    line_search_gamma_c: float
    line_search_armijo_factor: float
    line_search_step_decay: float
    line_search_step_min: float
    line_search_constraint_scale: float

    @property
    def discretization_times(self) -> np.ndarray:
        return humanoid_dt_schedule(
            self.n_nodes,
            dt_start=self.dt_start,
            dt_exponent=self.dt_exponent,
        )


@dataclass(frozen=True)
class BodyReferenceCommand:
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    yaw_rate: float = 0.0
    body_height: float = 0.61

    @property
    def velocity_xy(self) -> np.ndarray:
        return np.array([self.velocity_x, self.velocity_y], dtype=np.float64)


@dataclass(frozen=True)
class GaitSchedule:
    gait_pattern: np.ndarray
    foot_height: np.ndarray
    discretization_times: np.ndarray


def _array(values: Any, size: int, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if out.size != size:
        raise ValueError(f"{name} must have {size} entries, got {out.size}")
    return out


def _default_parameter_values() -> HumanoidMPCParameters:
    return HumanoidMPCParameters(
        qhome=np.array(
            [
                0,
                0,
                0.61,
                0,
                0,
                0,
                0,
                0,
                -0.724757,
                1.412282,
                -0.68752,
                0,
                0,
                -0.724757,
                1.412282,
                -0.68752,
                0,
                -0.1,
                0,
                0,
                0,
                0.1,
                0,
                0,
            ],
            dtype=np.float64,
        ),
        cost_q=np.array(
            [
                0,
                0,
                8000,
                300,
                600,
                300,
                20,
                20,
                0.05,
                0.05,
                0.05,
                20,
                20,
                0.05,
                0.05,
                0.05,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
            ],
            dtype=np.float64,
        ),
        cost_dq=np.array(
            [
                100,
                100,
                100,
                800,
                800,
                100,
                5,
                5,
                0.12,
                0.12,
                0.12,
                5,
                5,
                0.12,
                0.12,
                0.12,
                0.4,
                0.4,
                0.4,
                0.3,
                0.4,
                0.4,
                0.4,
                0.3,
            ],
            dtype=np.float64,
        ),
        cost_ddq=np.concatenate([0.001 * np.ones(10), 0.002 * np.ones(8)]),
        cost_f=np.tile(0.005 * np.ones(3), 4),
        cost_tau=np.array(
            [
                0.5,
                0.5,
                0.01,
                0.01,
                0.01,
                0.5,
                0.5,
                0.01,
                0.01,
                0.01,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            dtype=np.float64,
        ),
        gains_leg_swing_p=np.array([50, 50, 50, 50, 50], dtype=np.float64),
        gains_leg_swing_d=np.array([2, 2, 2, 2, 2], dtype=np.float64),
        gains_leg_stance_p=np.array([0, 0, 0, 0, 0], dtype=np.float64),
        gains_leg_stance_d=np.array([3, 3, 3, 1.5, 1.5], dtype=np.float64),
        gains_arm_p=np.array([50, 50, 25, 25], dtype=np.float64),
        gains_arm_d=np.array([1, 1, 1, 1], dtype=np.float64),
        gains_stance_velocity_setpoint_to_zero=True,
        n_nodes=HUMANOID_N_NODES,
        dt_start=0.03,
        dt_exponent=1.0,
        gait_cycle_duration=0.7,
        gait_overlap_duration=0.01,
        gait_right_swing_duration=0.34,
        gait_left_swing_duration=0.34,
        foot_lift_height=0.05,
        osqp_alpha=1.4,
        osqp_rho=0.02,
        osqp_max_iter=25,
        osqp_scaling=2,
        osqp_print_status=False,
        line_search_g_max=1e-1,
        line_search_g_min=1e-3,
        line_search_gamma_c=1e-3,
        line_search_armijo_factor=1e-4,
        line_search_step_decay=0.5,
        line_search_step_min=0.03,
        line_search_constraint_scale=0.05,
    )


def load_humanoid_mpc_parameters(path: Path | str | None = None) -> HumanoidMPCParameters:
    """Load humanoid MPC parameters.

    With ``path=None`` this returns the checked-in tuned defaults without adding
    a PyYAML dependency to the benchmark import path.  Passing a path loads the
    same schema from YAML when PyYAML is installed.
    """

    if path is None:
        return _default_parameter_values()
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
        raise ModuleNotFoundError("loading humanoid parameter YAML requires PyYAML") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    default = _default_parameter_values()
    return HumanoidMPCParameters(
        qhome=_array(raw["qhome"], HUMANOID_NQ, "qhome"),
        cost_q=_array(raw["cost_q"], HUMANOID_NQ, "cost_q"),
        cost_dq=_array(raw["cost_dq"], HUMANOID_NDQ, "cost_dq"),
        cost_ddq=_array(raw["cost_ddq"], HUMANOID_NTAU, "cost_ddq"),
        cost_f=_array(raw["cost_F"], HUMANOID_NF, "cost_F"),
        cost_tau=_array(raw["cost_tau"], HUMANOID_NTAU, "cost_tau"),
        gains_leg_swing_p=_array(raw["gains_leg_swing_p"], 5, "gains_leg_swing_p"),
        gains_leg_swing_d=_array(raw["gains_leg_swing_d"], 5, "gains_leg_swing_d"),
        gains_leg_stance_p=_array(raw["gains_leg_stance_p"], 5, "gains_leg_stance_p"),
        gains_leg_stance_d=_array(raw["gains_leg_stance_d"], 5, "gains_leg_stance_d"),
        gains_arm_p=_array(raw["gains_arm_p"], 4, "gains_arm_p"),
        gains_arm_d=_array(raw["gains_arm_d"], 4, "gains_arm_d"),
        gains_stance_velocity_setpoint_to_zero=bool(
            raw["gains_stance_velocity_setpoint_to_zero"]
        ),
        n_nodes=int(raw.get("N", default.n_nodes)),
        dt_start=float(raw.get("dt_start", default.dt_start)),
        dt_exponent=float(raw.get("dt_exponent", default.dt_exponent)),
        gait_cycle_duration=float(raw.get("gait_cycle_duration", default.gait_cycle_duration)),
        gait_overlap_duration=float(raw.get("gait_overlap_duration", default.gait_overlap_duration)),
        gait_right_swing_duration=float(
            raw.get("gait_right_swing_duration", default.gait_right_swing_duration)
        ),
        gait_left_swing_duration=float(
            raw.get("gait_left_swing_duration", default.gait_left_swing_duration)
        ),
        foot_lift_height=float(raw.get("foot_lift_height", default.foot_lift_height)),
        osqp_alpha=float(raw.get("osqp_alpha", default.osqp_alpha)),
        osqp_rho=float(raw.get("osqp_rho", default.osqp_rho)),
        osqp_max_iter=int(raw.get("osqp_max_iter", default.osqp_max_iter)),
        osqp_scaling=int(raw.get("osqp_scaling", default.osqp_scaling)),
        osqp_print_status=bool(raw.get("osqp_print_status", default.osqp_print_status)),
        line_search_g_max=float(raw.get("line_search_g_max", default.line_search_g_max)),
        line_search_g_min=float(raw.get("line_search_g_min", default.line_search_g_min)),
        line_search_gamma_c=float(raw.get("line_search_gamma_c", default.line_search_gamma_c)),
        line_search_armijo_factor=float(
            raw.get("line_search_armijo_factor", default.line_search_armijo_factor)
        ),
        line_search_step_decay=float(
            raw.get("line_search_step_decay", default.line_search_step_decay)
        ),
        line_search_step_min=float(raw.get("line_search_step_min", default.line_search_step_min)),
        line_search_constraint_scale=float(
            raw.get("line_search_constraint_scale", default.line_search_constraint_scale)
        ),
    )


def humanoid_dt_schedule(
    n_nodes: int = HUMANOID_N_NODES,
    *,
    dt_start: float | None = None,
    dt_exponent: float | None = None,
) -> np.ndarray:
    params = _default_parameter_values()
    start = params.dt_start if dt_start is None else float(dt_start)
    exponent = params.dt_exponent if dt_exponent is None else float(dt_exponent)
    return start * exponent ** np.arange(n_nodes, dtype=np.float64)


def humanoid_qhome(parameters: HumanoidMPCParameters | None = None) -> np.ndarray:
    return (parameters or _default_parameter_values()).qhome.copy()


def humanoid_joint_limits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_min = np.zeros(18)
    q_max = np.zeros(18)
    q_min[[0, 5]] = -np.deg2rad([30, 30])
    q_max[[0, 5]] = np.deg2rad([30, 30])
    q_min[[1, 6]] = -np.deg2rad([60, 60])
    q_max[[1, 6]] = np.deg2rad([60, 60])
    q_min[[2, 7]] = -np.deg2rad([120, 120])
    q_max[[2, 7]] = np.deg2rad([60, 60])
    q_min[[3, 8]] = np.deg2rad([10, 10])
    q_max[[3, 8]] = np.deg2rad([180, 180])
    q_min[[4, 9]] = -np.deg2rad([80, 80])
    q_max[[4, 9]] = np.deg2rad([80, 80])
    q_min[[10, 14]] = -np.deg2rad([90, 90])
    q_max[[10, 14]] = np.deg2rad([80, 80])
    q_min[[11, 15]] = -np.deg2rad([90, 2.5])
    q_max[[11, 15]] = np.deg2rad([2.5, 90])
    q_min[[12, 16]] = -np.deg2rad([90, 45])
    q_max[[12, 16]] = np.deg2rad([45, 90])
    q_min[[13, 17]] = -np.deg2rad([130, 130])
    q_max[[13, 17]] = np.deg2rad([0, 0])
    dq_min = np.concatenate([np.full(10, -50.0), np.full(8, -25.0)])
    dq_max = -dq_min
    return q_min, q_max, dq_min, dq_max


def humanoid_stage_var_dims(n_nodes: int = HUMANOID_N_NODES) -> tuple[int, ...]:
    if n_nodes < HUMANOID_N_TORQUE_STAGES + 1:
        raise ValueError("humanoid MPC requires at least three shooting nodes")
    return (
        HUMANOID_NZ_TORQUE,
        HUMANOID_NZ_TORQUE,
        *([HUMANOID_NZ_FORCE] * max(0, n_nodes - 3)),
        HUMANOID_NZ_TERMINAL,
    )


def humanoid_z_offsets(n_nodes: int = HUMANOID_N_NODES) -> np.ndarray:
    dims = humanoid_stage_var_dims(n_nodes)
    out = np.zeros(len(dims) + 1, dtype=np.int32)
    out[1:] = np.cumsum(np.asarray(dims, dtype=np.int32))
    return out


def humanoid_n_nodes_from_z_width(width: int) -> int:
    if (int(width) - 24) % 60 != 0:
        raise ValueError(f"cannot infer humanoid node count from z width {width}")
    n_nodes = (int(width) - 24) // 60
    if n_nodes < HUMANOID_N_TORQUE_STAGES + 1:
        raise ValueError(f"invalid humanoid node count inferred from z width {width}")
    return n_nodes


def gait_schedule_walking(
    phi: float,
    phi_vel: float,
    dt: np.ndarray,
    n_nodes: int,
    *,
    stance_fraction: float = 0.7,
    height: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy phase-based walking gait used by older benchmark tests."""

    dt = np.asarray(dt, dtype=np.float64)
    gait_pattern = np.zeros((4, n_nodes), dtype=np.float64)
    foot_height = np.zeros((4, n_nodes), dtype=np.float64)
    leg_offsets_phi = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float64)
    elapsed = np.concatenate([[0.0], np.cumsum(dt[: max(0, n_nodes - 1)])])
    for i in range(n_nodes):
        current_phase = (phi + elapsed[i] * phi_vel) % 1.0
        per_leg_phase = (current_phase + leg_offsets_phi) % 1.0
        stance = per_leg_phase <= stance_fraction
        gait_pattern[:, i] = stance.astype(np.float64)
        swing_phase = (per_leg_phase - stance_fraction) / (1.0 - stance_fraction)
        swing = ~stance
        foot_height[swing, i] = 4.0 * swing_phase[swing] * (1.0 - swing_phase[swing]) * height
    return gait_pattern, foot_height


class InPlaceWalkingGait:
    _SWITCH_PHASE_DEADBAND = 0.01
    _MIN_SWITCH_DEADBAND_S = 1.0e-3

    def __init__(
        self,
        *,
        cycle_duration: float = 0.7,
        overlap_duration: float = 0.01,
        right_swing_duration: float = 0.34,
        left_swing_duration: float = 0.34,
        swing_height: float = 0.05,
    ) -> None:
        active_duration = (
            2.0 * overlap_duration + right_swing_duration + left_swing_duration
        )
        if active_duration > cycle_duration + 1e-9:
            raise ValueError("gait timings exceed gait_cycle_duration")
        self.cycle_duration = float(cycle_duration)
        self.overlap_duration = float(overlap_duration)
        self.right_swing_duration = float(right_swing_duration)
        self.left_swing_duration = float(left_swing_duration)
        self.swing_height = float(swing_height)

    def _switch_deadband(self) -> float:
        return max(
            self._MIN_SWITCH_DEADBAND_S,
            self._SWITCH_PHASE_DEADBAND * self.cycle_duration,
        )

    def _switch_boundaries(self) -> list[float]:
        active_duration = (
            2.0 * self.overlap_duration
            + self.right_swing_duration
            + self.left_swing_duration
        )
        raw = [
            self.overlap_duration,
            self.overlap_duration + self.right_swing_duration,
            2.0 * self.overlap_duration + self.right_swing_duration,
            active_duration,
        ]
        out: list[float] = []
        for boundary in raw:
            if boundary <= 1e-12 or boundary > self.cycle_duration + 1e-12:
                continue
            boundary = min(boundary, self.cycle_duration)
            if out and abs(boundary - out[-1]) <= 1e-12:
                continue
            out.append(boundary)
        return out

    def _time_to_next_switch(self, t: float) -> float:
        boundaries = self._switch_boundaries()
        if not boundaries:
            return self.cycle_duration
        deadband = self._switch_deadband()
        phase_time = t % self.cycle_duration
        for cycle_offset in (0.0, self.cycle_duration):
            for boundary in boundaries:
                delta = cycle_offset + boundary - phase_time
                if delta > deadband:
                    return delta
        return self.cycle_duration

    def _adaptive_discretization(
        self,
        gait_time: float,
        nominal_dt: np.ndarray,
        n_nodes: int,
    ) -> np.ndarray:
        dt_opt = np.asarray(nominal_dt, dtype=np.float64).reshape(-1)
        if dt_opt.size < n_nodes:
            raise ValueError(f"nominal_dt must contain at least {n_nodes} entries")
        dt_opt = dt_opt[:n_nodes].copy()
        deadband = self._switch_deadband()
        t = float(gait_time)
        for k in range(max(0, n_nodes - 1)):
            time_to_switch = self._time_to_next_switch(t)
            if deadband < time_to_switch < dt_opt[k] - deadband:
                dt_opt[k] = time_to_switch
            t += dt_opt[k]
        return dt_opt

    def _phase(self, t: float) -> tuple[str, float]:
        phase_time = (t + self._switch_deadband()) % self.cycle_duration
        if phase_time < self.overlap_duration:
            return "double", 0.0
        phase_time -= self.overlap_duration
        if phase_time < self.right_swing_duration:
            return "right", phase_time / self.right_swing_duration
        phase_time -= self.right_swing_duration
        if phase_time < self.overlap_duration:
            return "double", 0.0
        phase_time -= self.overlap_duration
        if phase_time < self.left_swing_duration:
            return "left", phase_time / self.left_swing_duration
        return "double", 0.0

    def schedule(self, gait_time: float, dt: np.ndarray, n_nodes: int) -> GaitSchedule:
        dt_opt = self._adaptive_discretization(gait_time, dt, n_nodes)
        sample_times = gait_time + np.concatenate([[0.0], np.cumsum(dt_opt[: n_nodes - 1])])
        gait_pattern = np.ones((4, n_nodes), dtype=np.float64)
        foot_height = np.zeros((4, n_nodes), dtype=np.float64)
        for k, t in enumerate(sample_times):
            mode, swing_phase = self._phase(float(t))
            swing_z = 16.0 * swing_phase**2 * (1.0 - swing_phase) ** 2
            swing_z *= self.swing_height
            if mode == "right":
                gait_pattern[[0, 2], k] = 0.0
                foot_height[[0, 2], k] = swing_z
            elif mode == "left":
                gait_pattern[[1, 3], k] = 0.0
                foot_height[[1, 3], k] = swing_z
        return GaitSchedule(gait_pattern, foot_height, dt_opt)


def standing_gait_schedule(n_nodes: int, dt: np.ndarray) -> GaitSchedule:
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    if dt.size < n_nodes:
        raise ValueError(f"dt must contain at least {n_nodes} entries")
    return GaitSchedule(
        gait_pattern=np.ones((4, n_nodes), dtype=np.float64),
        foot_height=np.zeros((4, n_nodes), dtype=np.float64),
        discretization_times=dt[:n_nodes].copy(),
    )


def humanoid_walking_gait_schedule(
    gait_time: float,
    *,
    n_nodes: int = HUMANOID_N_NODES,
    parameters: HumanoidMPCParameters | None = None,
) -> GaitSchedule:
    params = parameters or _default_parameter_values()
    gait = InPlaceWalkingGait(
        cycle_duration=params.gait_cycle_duration,
        overlap_duration=params.gait_overlap_duration,
        right_swing_duration=params.gait_right_swing_duration,
        left_swing_duration=params.gait_left_swing_duration,
        swing_height=params.foot_lift_height,
    )
    return gait.schedule(float(gait_time), params.discretization_times, n_nodes)


def generate_desired_ground_forces(gait_pattern: np.ndarray) -> np.ndarray:
    weight = HUMANOID_MASS * HUMANOID_G
    gait_pattern = np.asarray(gait_pattern)
    out = np.zeros((HUMANOID_NF, gait_pattern.shape[1]), dtype=np.float64)
    for k in range(gait_pattern.shape[1]):
        for foot in range(4):
            if int(gait_pattern[foot, k]) == 1:
                out[3 * foot + 2, k] = weight / 0.5 / 4.0
    return out


def yaw_body_to_world_xy(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    out = np.zeros(yaw.shape + (2, 2), dtype=np.float64)
    out[..., 0, 0] = cy
    out[..., 0, 1] = -sy
    out[..., 1, 0] = sy
    out[..., 1, 1] = cy
    return out


def humanoid_make_references(
    parameters: HumanoidMPCParameters,
    current_q: np.ndarray,
    command: BodyReferenceCommand | np.ndarray | None = None,
    discretization_times: np.ndarray | None = None,
    *,
    command_dt: float = HUMANOID_HIGH_LEVEL_TIMESTEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Create batched ``qref, dqref`` arrays with body-frame velocity commands.

    ``command`` may be a ``BodyReferenceCommand`` for all batch elements or an
    array with columns ``vx, vy, yaw_rate, body_height``.
    """

    current_q = np.asarray(current_q, dtype=np.float64)
    if current_q.ndim == 1:
        current_q = current_q[None, :]
    batch = current_q.shape[0]
    n_nodes = parameters.n_nodes
    dt = parameters.discretization_times if discretization_times is None else discretization_times
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    if dt.size < n_nodes:
        raise ValueError(f"discretization_times must contain at least {n_nodes} entries")

    if command is None:
        command_arr = np.zeros((batch, 4), dtype=np.float64)
        command_arr[:, 3] = parameters.qhome[2]
    elif isinstance(command, BodyReferenceCommand):
        command_arr = np.array(
            [[command.velocity_x, command.velocity_y, command.yaw_rate, command.body_height]],
            dtype=np.float64,
        )
        command_arr = np.broadcast_to(command_arr, (batch, 4)).copy()
    else:
        command_arr = np.asarray(command, dtype=np.float64)
        if command_arr.ndim == 1:
            command_arr = command_arr[None, :]
        if command_arr.shape != (batch, 4):
            command_arr = np.broadcast_to(command_arr, (batch, 4)).copy()

    command_arr[:, 0] = np.clip(
        command_arr[:, 0],
        -HUMANOID_REFERENCE_VELOCITY_X_LIMIT,
        HUMANOID_REFERENCE_VELOCITY_X_LIMIT,
    )
    command_arr[:, 1] = np.clip(
        command_arr[:, 1],
        -HUMANOID_REFERENCE_VELOCITY_Y_LIMIT,
        HUMANOID_REFERENCE_VELOCITY_Y_LIMIT,
    )
    command_arr[:, 2] = np.clip(
        command_arr[:, 2],
        -HUMANOID_REFERENCE_YAW_RATE_LIMIT,
        HUMANOID_REFERENCE_YAW_RATE_LIMIT,
    )
    command_arr[:, 3] = np.clip(
        command_arr[:, 3],
        HUMANOID_REFERENCE_BODY_HEIGHT_MIN,
        HUMANOID_REFERENCE_BODY_HEIGHT_MAX,
    )

    qref = np.tile(parameters.qhome[None, :, None], (batch, 1, n_nodes))
    dqref = np.zeros((batch, HUMANOID_NDQ, n_nodes), dtype=np.float64)
    velocity_xy_body = command_arr[:, :2]
    yaw_rate = command_arr[:, 2]
    yaw_ref = current_q[:, 5] + command_dt * yaw_rate
    xy_ref = current_q[:, :2] + np.einsum(
        "bij,bj->bi",
        yaw_body_to_world_xy(yaw_ref),
        velocity_xy_body,
    ) * command_dt
    qref[:, :2, 0] = xy_ref
    qref[:, 2, :] = command_arr[:, 3, None]
    qref[:, 5, 0] = yaw_ref
    dqref[:, 2, :] = yaw_rate[:, None]
    dqref[:, 3:5, :] = velocity_xy_body[:, :, None]

    for k in range(1, n_nodes):
        yaw_ref = yaw_ref + yaw_rate * dt[k - 1]
        qref[:, 5, k] = yaw_ref
        xy_ref = xy_ref + np.einsum(
            "bij,bj->bi",
            yaw_body_to_world_xy(yaw_ref),
            velocity_xy_body,
        ) * dt[k - 1]
        qref[:, :2, k] = xy_ref
    return qref, dqref


def humanoid_default_parameters(
    *,
    n_nodes: int = HUMANOID_N_NODES,
    phi: float = 0.0,
    phi_vel: float = 1.0 / 0.6,
    parameters: HumanoidMPCParameters | None = None,
) -> dict[str, np.ndarray]:
    params = parameters or _default_parameter_values()
    dt = humanoid_dt_schedule(
        n_nodes,
        dt_start=params.dt_start,
        dt_exponent=params.dt_exponent,
    )
    gait, foot_height = gait_schedule_walking(phi, phi_vel, dt, n_nodes)
    f_ref = generate_desired_ground_forces(gait)
    qhome = params.qhome
    return {
        "q0": qhome.copy(),
        "dq0": np.zeros(HUMANOID_NDQ),
        "tau0": np.zeros(HUMANOID_NTAU),
        "ncontactseq": gait,
        "foot_height_des": foot_height,
        "qref": np.tile(qhome[:, None], n_nodes),
        "dqref": np.zeros((HUMANOID_NDQ, n_nodes)),
        "Fref": f_ref,
        "dt": dt,
        "Q_q": params.cost_q.copy(),
        "Q_dq": params.cost_dq.copy(),
        "Q_ddq": params.cost_ddq.copy(),
        "Q_F": params.cost_f.copy(),
        "Q_tau_first": params.cost_tau.copy(),
    }


def _load_humanoid_casadi_function(name: str) -> ca.Function:
    path = HUMANOID_CASADI_FUNCTION_DIR / f"{name}.casadi"
    if not path.exists():
        raise FileNotFoundError(f"missing generated humanoid CasADi function: {path}")
    return ca.Function.load(str(path))


def _diag_quadratic(weights: Any, residual: Any) -> ca.SX:
    out = ca.SX(0)
    for i in range(int(residual.numel())):
        if isinstance(weights, np.ndarray):
            weight = float(weights[i])
            if weight == 0.0:
                continue
        else:
            weight = weights[i]
        out = out + weight * residual[i] ** 2
    return out


def _split_torque_stage(z: Any) -> tuple[Any, Any, Any, Any]:
    q = z[:HUMANOID_NQ]
    dq = z[HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ]
    tau = z[HUMANOID_NQ + HUMANOID_NDQ : HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU]
    force = z[HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU : HUMANOID_NZ_TORQUE]
    return q, dq, tau, force


def _split_force_stage(z: Any) -> tuple[Any, Any, Any]:
    q = z[:HUMANOID_NQ]
    dq = z[HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ]
    force = z[HUMANOID_NQ + HUMANOID_NDQ : HUMANOID_NZ_FORCE]
    return q, dq, force


def _split_q_dq(z: Any) -> tuple[Any, Any]:
    return z[:HUMANOID_NQ], z[HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ]


def _rx(roll: Any) -> Any:
    return ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll)),
    )


def _ry(pitch: Any) -> Any:
    return ca.vertcat(
        ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch)),
    )


def _rz(yaw: Any) -> Any:
    return ca.vertcat(
        ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
        ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
        ca.horzcat(0, 0, 1),
    )


def _plus_jacobian_fb(q_fb: Any) -> Any:
    roll = q_fb[3]
    pitch = q_fb[4]
    yaw = q_fb[5]
    r_body_to_world = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    b_inv = ca.vertcat(
        ca.horzcat(ca.cos(yaw) / ca.cos(pitch), ca.sin(yaw) / ca.cos(pitch), 0),
        ca.horzcat(-ca.sin(yaw), ca.cos(yaw), 0),
        ca.horzcat(ca.cos(yaw) * ca.tan(pitch), ca.sin(yaw) * ca.tan(pitch), 1),
    )
    return ca.vertcat(
        ca.horzcat(ca.SX.zeros(3, 3), r_body_to_world),
        ca.horzcat(b_inv @ r_body_to_world, ca.SX.zeros(3, 3)),
    )


def _configuration_integrator(q: Any, dq: Any, ddq: Any, dt: Any) -> Any:
    base_delta = dq[:6] * dt + 0.5 * ddq[:6] * dt**2
    joint_delta = dq[6:] * dt + 0.5 * ddq[6:] * dt**2
    return ca.vertcat(q[:6] + _plus_jacobian_fb(q[:6]) @ base_delta, q[6:] + joint_delta)


def _single_output(result: Any) -> Any:
    return result[0] if isinstance(result, (list, tuple)) else result


def _robot_contact_terms(contact_pos_jac: ca.Function, q: Any) -> tuple[Any, Any]:
    out = contact_pos_jac(q)
    if not isinstance(out, (list, tuple)) or len(out) != 8:
        raise ValueError("contact position/Jacobian function must have eight outputs")
    p_contact = ca.vertcat(out[0], out[1], out[2], out[3])
    j_contact = ca.vertcat(out[4], out[5], out[6], out[7])
    return p_contact, j_contact


def _robot_collision_distances(signed_distance: ca.Function, q: Any) -> Any:
    out = signed_distance(q, HUMANOID_COLLISION_SPHERE_RADIUS)
    if not isinstance(out, (list, tuple)):
        out = [out]
    return ca.vertcat(*out)


def _full_inverse_dynamics_residual(
    inverse_dynamics: ca.Function,
    friction_model: ca.Function,
    q: Any,
    dq: Any,
    tau: Any,
    force: Any,
    ddq: Any,
    j_contact: Any,
) -> Any:
    leg_friction = _single_output(
        friction_model(ca.SX.zeros(HUMANOID_NTAU), q[6:], dq[6:])
    )[:HUMANOID_N_LEG]
    tau_friction = ca.vertcat(
        ca.SX.zeros(HUMANOID_N_FB),
        leg_friction,
        ca.SX.zeros(HUMANOID_N_ARM),
    )
    generalized_tau = ca.vertcat(ca.SX.zeros(HUMANOID_N_FB), tau)
    return _single_output(inverse_dynamics(q, dq, ddq)) - tau_friction - j_contact.T @ force - generalized_tau


def _floating_base_dynamics_residual(
    inverse_dynamics: ca.Function,
    q: Any,
    dq: Any,
    force: Any,
    ddq: Any,
    j_contact: Any,
) -> Any:
    return (_single_output(inverse_dynamics(q, dq, ddq)) - j_contact.T @ force)[:HUMANOID_N_FB]


def _actuator_voltage_torque_bounds(dq_joint: Any) -> tuple[Any, Any]:
    lower: list[Any] = [None] * HUMANOID_NTAU
    upper: list[Any] = [None] * HUMANOID_NTAU

    def set_single(indices: tuple[int, ...], offset: float, slope: float) -> None:
        for idx in indices:
            lower[idx] = -offset - slope * dq_joint[idx]
            upper[idx] = offset - slope * dq_joint[idx]

    set_single((0, 1, 5, 6, 10, 11, 12, 14, 15, 16), 78.2608695652, 1.6305652174)
    set_single((2, 7), 125.0000000000, 3.2062500000)
    set_single((13, 17), 117.3913043478, 3.6687717391)

    for knee_idx, ankle_idx in ((3, 4), (8, 9)):
        dq_knee = dq_joint[knee_idx]
        dq_ankle = dq_joint[ankle_idx]
        lower[knee_idx] = -406.5217391304 - 19.3472608696 * dq_knee - 6.5222608696 * dq_ankle
        upper[knee_idx] = 406.5217391304 - 19.3472608696 * dq_knee - 6.5222608696 * dq_ankle
        lower[ankle_idx] = -156.5217391304 - 6.5222608696 * dq_knee - 6.5222608696 * dq_ankle
        upper[ankle_idx] = 156.5217391304 - 6.5222608696 * dq_knee - 6.5222608696 * dq_ankle

    if any(value is None for value in lower + upper):
        raise RuntimeError("incomplete actuator voltage torque bound map")
    return ca.vertcat(*lower), ca.vertcat(*upper)


def _append_bound(parts: list[Any], lower: list[Any], upper: list[Any], expr: Any, lo: Any, hi: Any) -> None:
    expr = ca.reshape(expr, int(expr.numel()), 1)
    parts.append(expr)
    lower.append(ca.SX.ones(expr.numel()) * lo if np.isscalar(lo) else lo)
    upper.append(ca.SX.ones(expr.numel()) * hi if np.isscalar(hi) else hi)


def _append_stage_constraints(
    parts: list[Any],
    lower: list[Any],
    upper: list[Any],
    *,
    signed_distance: ca.Function,
    q: Any,
    dq: Any,
    gamma: Any,
    foot_height_des: Any,
    p_contact: Any,
    j_contact: Any,
) -> None:
    q_min, q_max, dq_min, dq_max = humanoid_joint_limits()
    _append_bound(parts, lower, upper, q[6:], q_min, q_max)
    _append_bound(parts, lower, upper, dq[6:], dq_min, dq_max)
    distances = _robot_collision_distances(signed_distance, q)
    _append_bound(parts, lower, upper, distances, 0.0, HUMANOID_INFTY)
    _append_bound(parts, lower, upper, q[11] - q[6], -HUMANOID_INFTY, np.deg2rad(45.0))
    _append_bound(
        parts,
        lower,
        upper,
        ca.vertcat(p_contact[2], p_contact[5], p_contact[8], p_contact[11]) - foot_height_des,
        0.0,
        0.0,
    )
    contact_velocity = j_contact @ dq
    _append_bound(parts, lower, upper, contact_velocity[0::3] * gamma, 0.0, 0.0)
    _append_bound(parts, lower, upper, contact_velocity[1::3] * gamma, 0.0, 0.0)


def _append_force_constraints(
    parts: list[Any],
    lower: list[Any],
    upper: list[Any],
    *,
    force: Any,
    gamma: Any,
) -> None:
    mu = 0.7
    tol_f = 1.0
    fx = force[0::3]
    fy = force[1::3]
    fz = force[2::3]
    friction = gamma * ca.sqrt(fx**2 + fy**2 + tol_f) * 0.005 - gamma * mu * fz * 0.005
    _append_bound(parts, lower, upper, friction, -HUMANOID_INFTY, 0.0)
    _append_bound(parts, lower, upper, fx * (1.0 - gamma), 0.0, 0.0)
    _append_bound(parts, lower, upper, fy * (1.0 - gamma), 0.0, 0.0)
    _append_bound(parts, lower, upper, fz * (1.0 - gamma), 0.0, 0.0)


def _append_torque_constraints(
    parts: list[Any],
    lower: list[Any],
    upper: list[Any],
    *,
    tau: Any,
    dq_joint: Any,
) -> None:
    _append_bound(parts, lower, upper, tau, ACTUATOR_STATIC_TAU_MIN_NM, ACTUATOR_STATIC_TAU_MAX_NM)
    tau_voltage_min, tau_voltage_max = _actuator_voltage_torque_bounds(dq_joint)
    _append_bound(parts, lower, upper, tau, tau_voltage_min, tau_voltage_max)


def _stage_cost(
    q: Any,
    dq: Any,
    force: Any | None,
    qref: Any,
    dqref: Any,
    fref: Any | None,
    dt: Any,
    weights: dict[str, Any],
    *,
    tau: Any | None = None,
    tau0: Any | None = None,
    ddq: Any | None = None,
) -> Any:
    cost = _diag_quadratic(weights["Q_q"], q - qref)
    cost += _diag_quadratic(weights["Q_dq"], dq - dqref)
    if ddq is not None and force is not None and fref is not None:
        cost += _diag_quadratic(weights["Q_ddq"], ddq[6:])
        cost += _diag_quadratic(weights["Q_F"], force - fref)
    if tau is not None and tau0 is not None:
        cost += _diag_quadratic(weights["Q_tau_first"], tau0 - tau)
    return dt * cost


def _pack_stage_output(cost: Any, parts: list[Any], lower: list[Any], upper: list[Any]) -> tuple[Any, Any, Any, Any]:
    return cost, ca.vertcat(*parts), ca.vertcat(*lower), ca.vertcat(*upper)


def make_humanoid_sqp_problem(
    n_nodes: int = HUMANOID_N_NODES,
    *,
    model_name: str = HUMANOID_MODEL_NAME,
) -> SparseMPCProblem:
    """Create the tuned sparse humanoid SQP MPC problem."""

    if n_nodes < HUMANOID_N_TORQUE_STAGES + 1:
        raise ValueError("humanoid MPC requires at least three shooting nodes")

    inverse_dynamics = _load_humanoid_casadi_function(f"{model_name}_inverse_dynamics")
    contact_pos_jac = _load_humanoid_casadi_function(f"{model_name}_contact_points_pos_jac")
    signed_distance = _load_humanoid_casadi_function(f"{model_name}_signed_distance")
    friction_model = _load_humanoid_casadi_function("friction_model")

    first_param_dim = 231
    running_param_dim = 147
    terminal_param_dim = 105

    def unpack_first(p: Any) -> tuple[Any, ...]:
        offset = 0
        q0 = p[offset : offset + HUMANOID_NQ]
        offset += HUMANOID_NQ
        dq0 = p[offset : offset + HUMANOID_NDQ]
        offset += HUMANOID_NDQ
        tau0 = p[offset : offset + HUMANOID_NTAU]
        offset += HUMANOID_NTAU
        gamma = p[offset : offset + 4]
        offset += 4
        foot = p[offset : offset + 4]
        offset += 4
        qref = p[offset : offset + HUMANOID_NQ]
        offset += HUMANOID_NQ
        dqref = p[offset : offset + HUMANOID_NDQ]
        offset += HUMANOID_NDQ
        fref = p[offset : offset + HUMANOID_NF]
        offset += HUMANOID_NF
        dt = p[offset]
        offset += 1
        weights = {
            "Q_q": p[offset : offset + HUMANOID_NQ],
            "Q_dq": p[offset + HUMANOID_NQ : offset + HUMANOID_NQ + HUMANOID_NDQ],
            "Q_ddq": p[offset + 48 : offset + 66],
            "Q_F": p[offset + 66 : offset + 78],
            "Q_tau_first": p[offset + 78 : offset + 96],
        }
        return q0, dq0, tau0, gamma, foot, qref, dqref, fref, dt, weights

    def unpack_running(p: Any) -> tuple[Any, ...]:
        offset = 0
        gamma = p[offset : offset + 4]
        offset += 4
        foot = p[offset : offset + 4]
        offset += 4
        qref = p[offset : offset + HUMANOID_NQ]
        offset += HUMANOID_NQ
        dqref = p[offset : offset + HUMANOID_NDQ]
        offset += HUMANOID_NDQ
        fref = p[offset : offset + HUMANOID_NF]
        offset += HUMANOID_NF
        dt = p[offset]
        offset += 1
        weights = {
            "Q_q": p[offset : offset + HUMANOID_NQ],
            "Q_dq": p[offset + HUMANOID_NQ : offset + HUMANOID_NQ + HUMANOID_NDQ],
            "Q_ddq": p[offset + 48 : offset + 66],
            "Q_F": p[offset + 66 : offset + 78],
        }
        return gamma, foot, qref, dqref, fref, dt, weights

    def unpack_terminal(p: Any) -> tuple[Any, ...]:
        offset = 0
        gamma = p[offset : offset + 4]
        offset += 4
        foot = p[offset : offset + 4]
        offset += 4
        qref = p[offset : offset + HUMANOID_NQ]
        offset += HUMANOID_NQ
        dqref = p[offset : offset + HUMANOID_NDQ]
        offset += HUMANOID_NDQ
        dt = p[offset]
        offset += 1
        weights = {
            "Q_q": p[offset : offset + HUMANOID_NQ],
            "Q_dq": p[offset + HUMANOID_NQ : offset + HUMANOID_NQ + HUMANOID_NDQ],
        }
        return gamma, foot, qref, dqref, dt, weights

    def make_first() -> ca.Function:
        z = ca.SX.sym("humanoid_z0", HUMANOID_NZ_TORQUE)
        zn = ca.SX.sym("humanoid_z1", HUMANOID_NZ_TORQUE)
        p = ca.SX.sym("humanoid_p0", first_param_dim)
        q, dq, tau, force = _split_torque_stage(z)
        q_next, dq_next = _split_q_dq(zn)
        q0, dq0, tau0, gamma, _foot, qref, dqref, fref, dt, weights = unpack_first(p)
        ddq = (dq_next - dq) / dt
        p_contact, j_contact = _robot_contact_terms(contact_pos_jac, q)
        parts = [
            q - q0,
            dq - dq0,
            _full_inverse_dynamics_residual(
                inverse_dynamics, friction_model, q, dq, tau, force, ddq, j_contact
            ),
            q_next - _configuration_integrator(q, dq, ddq, dt),
        ]
        lower = [
            ca.SX.zeros(HUMANOID_NQ),
            ca.SX.zeros(HUMANOID_NDQ),
            ca.SX.zeros(HUMANOID_NQ),
            ca.SX.zeros(HUMANOID_NQ),
        ]
        upper = [
            ca.SX.zeros(HUMANOID_NQ),
            ca.SX.zeros(HUMANOID_NDQ),
            ca.SX.zeros(HUMANOID_NQ),
            ca.SX.zeros(HUMANOID_NQ),
        ]
        _append_force_constraints(parts, lower, upper, force=force, gamma=gamma)
        _append_torque_constraints(parts, lower, upper, tau=tau, dq_joint=dq[6:])
        cost = _stage_cost(
            q,
            dq,
            force,
            qref,
            dqref,
            fref,
            dt,
            weights,
            tau=tau,
            tau0=tau0,
            ddq=ddq,
        )
        return ca.Function("humanoid_first", [z, zn, p], list(_pack_stage_output(cost, parts, lower, upper)))

    def make_second(next_z_dim: int) -> ca.Function:
        z = ca.SX.sym("humanoid_z_second", HUMANOID_NZ_TORQUE)
        zn = ca.SX.sym("humanoid_zn_second", next_z_dim)
        p = ca.SX.sym("humanoid_p_second", running_param_dim)
        q, dq, tau, force = _split_torque_stage(z)
        q_next, dq_next = _split_q_dq(zn)
        gamma, foot, qref, dqref, fref, dt, weights = unpack_running(p)
        ddq = (dq_next - dq) / dt
        p_contact, j_contact = _robot_contact_terms(contact_pos_jac, q)
        parts = [
            _full_inverse_dynamics_residual(
                inverse_dynamics, friction_model, q, dq, tau, force, ddq, j_contact
            ),
            q_next - _configuration_integrator(q, dq, ddq, dt),
        ]
        lower = [ca.SX.zeros(HUMANOID_NQ), ca.SX.zeros(HUMANOID_NQ)]
        upper = [ca.SX.zeros(HUMANOID_NQ), ca.SX.zeros(HUMANOID_NQ)]
        _append_stage_constraints(
            parts,
            lower,
            upper,
            signed_distance=signed_distance,
            q=q,
            dq=dq,
            gamma=gamma,
            foot_height_des=foot,
            p_contact=p_contact,
            j_contact=j_contact,
        )
        _append_force_constraints(parts, lower, upper, force=force, gamma=gamma)
        _append_torque_constraints(parts, lower, upper, tau=tau, dq_joint=dq[6:])
        cost = _stage_cost(q, dq, force, qref, dqref, fref, dt, weights, ddq=ddq)
        return ca.Function(
            f"humanoid_second_next{next_z_dim}",
            [z, zn, p],
            list(_pack_stage_output(cost, parts, lower, upper)),
        )

    def make_middle(next_z_dim: int) -> ca.Function:
        z = ca.SX.sym("humanoid_z_middle", HUMANOID_NZ_FORCE)
        zn = ca.SX.sym("humanoid_zn_middle", next_z_dim)
        p = ca.SX.sym("humanoid_p_middle", running_param_dim)
        q, dq, force = _split_force_stage(z)
        q_next, dq_next = _split_q_dq(zn)
        gamma, foot, qref, dqref, fref, dt, weights = unpack_running(p)
        ddq = (dq_next - dq) / dt
        p_contact, j_contact = _robot_contact_terms(contact_pos_jac, q)
        parts = [
            _floating_base_dynamics_residual(inverse_dynamics, q, dq, force, ddq, j_contact),
            q_next - _configuration_integrator(q, dq, ddq, dt),
        ]
        lower = [ca.SX.zeros(HUMANOID_N_FB), ca.SX.zeros(HUMANOID_NQ)]
        upper = [ca.SX.zeros(HUMANOID_N_FB), ca.SX.zeros(HUMANOID_NQ)]
        _append_stage_constraints(
            parts,
            lower,
            upper,
            signed_distance=signed_distance,
            q=q,
            dq=dq,
            gamma=gamma,
            foot_height_des=foot,
            p_contact=p_contact,
            j_contact=j_contact,
        )
        _append_force_constraints(parts, lower, upper, force=force, gamma=gamma)
        cost = _stage_cost(q, dq, force, qref, dqref, fref, dt, weights, ddq=ddq)
        return ca.Function(
            f"humanoid_middle_next{next_z_dim}",
            [z, zn, p],
            list(_pack_stage_output(cost, parts, lower, upper)),
        )

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("humanoid_zN", HUMANOID_NZ_TERMINAL)
        p = ca.SX.sym("humanoid_pN", terminal_param_dim)
        q, dq = _split_q_dq(z)
        gamma, foot, qref, dqref, dt, weights = unpack_terminal(p)
        p_contact, j_contact = _robot_contact_terms(contact_pos_jac, q)
        parts: list[Any] = []
        lower: list[Any] = []
        upper: list[Any] = []
        _append_stage_constraints(
            parts,
            lower,
            upper,
            signed_distance=signed_distance,
            q=q,
            dq=dq,
            gamma=gamma,
            foot_height_des=foot,
            p_contact=p_contact,
            j_contact=j_contact,
        )
        cost = dt * (
            _diag_quadratic(weights["Q_q"], q - qref)
            + _diag_quadratic(weights["Q_dq"], dq - dqref)
        )
        return ca.Function("humanoid_terminal", [z, p], list(_pack_stage_output(cost, parts, lower, upper)))

    first = CasadiStageFunction.from_function(make_first(), has_next=True)
    terminal = CasadiStageFunction.from_function(make_terminal(), has_next=False)
    if n_nodes == 3:
        intermediate = [
            CasadiStageFunction.from_function(
                make_second(HUMANOID_NZ_TERMINAL),
                has_next=True,
            )
        ]
    else:
        second = CasadiStageFunction.from_function(make_second(HUMANOID_NZ_FORCE), has_next=True)
        penultimate = CasadiStageFunction.from_function(make_middle(HUMANOID_NZ_TERMINAL), has_next=True)
        if n_nodes == 4:
            intermediate = [second, penultimate]
        else:
            middle = CasadiStageFunction.from_function(make_middle(HUMANOID_NZ_FORCE), has_next=True)
            intermediate = [second, *([middle] * (n_nodes - 4)), penultimate]
    return SparseMPCProblem.from_stage_functions(
        horizon=n_nodes - 1,
        first=first,
        intermediate=intermediate,
        terminal=terminal,
    )


def _pack_first_params(default: dict[str, np.ndarray], q0: np.ndarray, dq0: np.ndarray, tau0: np.ndarray, stage: int) -> np.ndarray:
    return np.concatenate(
        [
            q0,
            dq0,
            tau0,
            default["ncontactseq"][:, stage],
            default["foot_height_des"][:, stage],
            default["qref"][:, stage],
            default["dqref"][:, stage],
            default["Fref"][:, stage],
            default["dt"][stage : stage + 1],
            default["Q_q"],
            default["Q_dq"],
            default["Q_ddq"],
            default["Q_F"],
            default["Q_tau_first"],
        ]
    )


def _pack_running_params(default: dict[str, np.ndarray], stage: int) -> np.ndarray:
    return np.concatenate(
        [
            default["ncontactseq"][:, stage],
            default["foot_height_des"][:, stage],
            default["qref"][:, stage],
            default["dqref"][:, stage],
            default["Fref"][:, stage],
            default["dt"][stage : stage + 1],
            default["Q_q"],
            default["Q_dq"],
            default["Q_ddq"],
            default["Q_F"],
        ]
    )


def _pack_terminal_params(default: dict[str, np.ndarray], stage: int) -> np.ndarray:
    return np.concatenate(
        [
            default["ncontactseq"][:, stage],
            default["foot_height_des"][:, stage],
            default["qref"][:, stage],
            default["dqref"][:, stage],
            default["dt"][stage : stage + 1],
            default["Q_q"],
            default["Q_dq"],
        ]
    )


def _param_stage_offsets(n_nodes: int) -> tuple[np.ndarray, tuple[int, ...]]:
    widths = [231, *([147] * (n_nodes - 2)), 105]
    offsets = np.zeros(len(widths) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.asarray(widths, dtype=np.int32))
    return offsets, tuple(widths)


def humanoid_initial_guess_and_params(
    q0: np.ndarray,
    dq0: np.ndarray | None = None,
    *,
    n_nodes: int = HUMANOID_N_NODES,
    dtype: np.dtype | str = np.float64,
    phi: float = 0.0,
    phi_vel: float = 1.0 / 0.6,
    parameters: HumanoidMPCParameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.dtype(dtype)
    q0 = np.asarray(q0, dtype=dtype)
    if q0.ndim == 1:
        q0 = q0[None, :]
    batch = q0.shape[0]
    if dq0 is None:
        dq0 = np.zeros((batch, HUMANOID_NDQ), dtype=dtype)
    dq0 = np.asarray(dq0, dtype=dtype)
    if dq0.ndim == 1:
        dq0 = dq0[None, :]
    params_obj = parameters or _default_parameter_values()
    default = humanoid_default_parameters(
        n_nodes=n_nodes,
        phi=phi,
        phi_vel=phi_vel,
        parameters=params_obj,
    )
    tau0 = np.zeros((batch, HUMANOID_NTAU), dtype=dtype)

    pieces: list[np.ndarray] = []
    for stage_index, width in enumerate(humanoid_stage_var_dims(n_nodes)):
        stage = np.zeros(width, dtype=dtype)
        stage[:HUMANOID_NQ] = default["qref"][:, min(stage_index, n_nodes - 1)].astype(dtype)
        stage[HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ] = default["dqref"][
            :, min(stage_index, n_nodes - 1)
        ].astype(dtype)
        if width == HUMANOID_NZ_TORQUE:
            stage[HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU :] = default["Fref"][
                :, min(stage_index, n_nodes - 2)
            ].astype(dtype)
        elif width == HUMANOID_NZ_FORCE:
            stage[HUMANOID_NQ + HUMANOID_NDQ :] = default["Fref"][:, stage_index].astype(dtype)
        pieces.append(stage)
    z_stage = np.concatenate(pieces)
    z = np.tile(z_stage[None, :], (batch, 1))
    z[:, :HUMANOID_NQ] = q0
    z[:, HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ] = dq0

    packed = []
    for b in range(batch):
        param_pieces = [_pack_first_params(default, q0[b], dq0[b], tau0[b], 0)]
        for stage_index in range(1, n_nodes - 1):
            param_pieces.append(_pack_running_params(default, stage_index))
        param_pieces.append(_pack_terminal_params(default, n_nodes - 1))
        packed.append(np.concatenate(param_pieces))
    return z.astype(dtype), np.asarray(packed, dtype=dtype)


def update_humanoid_params_initial_state(
    params: np.ndarray,
    q0: np.ndarray,
    dq0: np.ndarray,
    tau0: np.ndarray | None = None,
) -> np.ndarray:
    params = np.array(params, copy=True)
    q0 = np.asarray(q0, dtype=params.dtype)
    dq0 = np.asarray(dq0, dtype=params.dtype)
    if tau0 is None:
        tau0 = np.zeros((params.shape[0], HUMANOID_NTAU), dtype=params.dtype)
    tau0 = np.asarray(tau0, dtype=params.dtype)
    params[:, :HUMANOID_NQ] = q0
    params[:, HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ] = dq0
    start = HUMANOID_NQ + HUMANOID_NDQ
    params[:, start : start + HUMANOID_NTAU] = tau0
    return params


def _as_batched_stage_array(values: np.ndarray, batch: int, rows: int, cols: int, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr[None, :, :], (batch, rows, cols)).copy()
    if arr.shape != (batch, rows, cols):
        raise ValueError(f"expected shape {(batch, rows, cols)}, got {arr.shape}")
    return arr


def update_humanoid_params_mpc(
    params: np.ndarray,
    *,
    q0: np.ndarray,
    dq0: np.ndarray,
    tau0: np.ndarray | None,
    reference_contacts: np.ndarray,
    reference_foot_height: np.ndarray,
    reference_q: np.ndarray,
    reference_dq: np.ndarray,
    discretization_times: np.ndarray,
    weights: HumanoidMPCParameters | None = None,
) -> np.ndarray:
    """Update all runtime humanoid MPC parameters for a batch."""

    out = update_humanoid_params_initial_state(params, q0, dq0, tau0)
    dtype = out.dtype
    batch = out.shape[0]
    offsets, widths = _param_stage_offsets(reference_q.shape[-1])
    n_nodes = len(widths)
    param_weights = weights or _default_parameter_values()
    contacts = np.asarray(reference_contacts, dtype=dtype).reshape(4, n_nodes)
    foot = np.asarray(reference_foot_height, dtype=dtype).reshape(4, n_nodes)
    fref = generate_desired_ground_forces(contacts).astype(dtype)
    qref = _as_batched_stage_array(reference_q, batch, HUMANOID_NQ, n_nodes, dtype)
    dqref = _as_batched_stage_array(reference_dq, batch, HUMANOID_NDQ, n_nodes, dtype)
    dt = np.asarray(discretization_times, dtype=dtype).reshape(n_nodes)

    for stage_index in range(n_nodes):
        base = int(offsets[stage_index])
        if stage_index == 0:
            cursor = base + HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU
            out[:, cursor : cursor + 4] = contacts[:, stage_index]
            cursor += 4
            out[:, cursor : cursor + 4] = foot[:, stage_index]
            cursor += 4
            out[:, cursor : cursor + HUMANOID_NQ] = qref[:, :, stage_index]
            cursor += HUMANOID_NQ
            out[:, cursor : cursor + HUMANOID_NDQ] = dqref[:, :, stage_index]
            cursor += HUMANOID_NDQ
            out[:, cursor : cursor + HUMANOID_NF] = fref[:, stage_index]
            cursor += HUMANOID_NF
            out[:, cursor] = dt[stage_index]
        elif stage_index < n_nodes - 1:
            cursor = base
            out[:, cursor : cursor + 4] = contacts[:, stage_index]
            cursor += 4
            out[:, cursor : cursor + 4] = foot[:, stage_index]
            cursor += 4
            out[:, cursor : cursor + HUMANOID_NQ] = qref[:, :, stage_index]
            cursor += HUMANOID_NQ
            out[:, cursor : cursor + HUMANOID_NDQ] = dqref[:, :, stage_index]
            cursor += HUMANOID_NDQ
            out[:, cursor : cursor + HUMANOID_NF] = fref[:, stage_index]
            cursor += HUMANOID_NF
            out[:, cursor] = dt[stage_index]
        else:
            cursor = base
            out[:, cursor : cursor + 4] = contacts[:, stage_index]
            cursor += 4
            out[:, cursor : cursor + 4] = foot[:, stage_index]
            cursor += 4
            out[:, cursor : cursor + HUMANOID_NQ] = qref[:, :, stage_index]
            cursor += HUMANOID_NQ
            out[:, cursor : cursor + HUMANOID_NDQ] = dqref[:, :, stage_index]
            cursor += HUMANOID_NDQ
            out[:, cursor] = dt[stage_index]

    q_start = None
    # Keep static weights synchronized if a caller provided a non-default object.
    if weights is not None:
        q_start = param_weights.cost_q.astype(dtype)
    del q_start
    return out


def update_humanoid_params_gait(
    params: np.ndarray,
    *,
    phi: float,
    n_nodes: int = HUMANOID_N_NODES,
    phi_vel: float = 1.0 / 0.6,
) -> np.ndarray:
    """Compatibility helper updating the legacy phase-based gait fields."""

    params = np.array(params, copy=True)
    default = humanoid_default_parameters(n_nodes=n_nodes, phi=phi, phi_vel=phi_vel)
    offsets, _widths = _param_stage_offsets(n_nodes)
    for stage_index in range(n_nodes):
        base = int(offsets[stage_index])
        if stage_index == 0:
            cursor = base + HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU
        else:
            cursor = base
        params[:, cursor : cursor + 4] = default["ncontactseq"][:, stage_index]
        cursor += 4
        params[:, cursor : cursor + 4] = default["foot_height_des"][:, stage_index]
        if stage_index < n_nodes - 1:
            cursor += 4 + HUMANOID_NQ + HUMANOID_NDQ
            params[:, cursor : cursor + HUMANOID_NF] = default["Fref"][:, stage_index]
    return params


def humanoid_trajectory_from_solution(
    z: np.ndarray,
    *,
    n_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z)
    if z.ndim == 1:
        z = z[None, :]
    if n_nodes is None:
        n_nodes = humanoid_n_nodes_from_z_width(z.shape[1])
    offsets = humanoid_z_offsets(n_nodes)
    batch = z.shape[0]
    q = np.zeros((batch, HUMANOID_NQ, n_nodes), dtype=z.dtype)
    dq = np.zeros((batch, HUMANOID_NDQ, n_nodes), dtype=z.dtype)
    tau = np.zeros((batch, HUMANOID_NTAU, HUMANOID_N_TORQUE_STAGES), dtype=z.dtype)
    force = np.zeros((batch, HUMANOID_NF, n_nodes - 1), dtype=z.dtype)
    for stage_index in range(n_nodes):
        stage = z[:, offsets[stage_index] : offsets[stage_index + 1]]
        q[:, :, stage_index] = stage[:, :HUMANOID_NQ]
        dq[:, :, stage_index] = stage[:, HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ]
        if stage_index < HUMANOID_N_TORQUE_STAGES:
            tau[:, :, stage_index] = stage[
                :,
                HUMANOID_NQ + HUMANOID_NDQ : HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU,
            ]
            force[:, :, stage_index] = stage[:, HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU :]
        elif stage_index < n_nodes - 1:
            force[:, :, stage_index] = stage[:, HUMANOID_NQ + HUMANOID_NDQ :]
    return q, dq, tau, force


def humanoid_predicted_next_state(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = humanoid_n_nodes_from_z_width(np.asarray(z).shape[-1])
    offsets = humanoid_z_offsets(n_nodes)
    arr = np.asarray(z)
    stage1 = arr[:, offsets[1] : offsets[2]]
    q_next = stage1[:, :HUMANOID_NQ]
    dq_next = stage1[:, HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ]
    tau0_next = arr[
        :,
        HUMANOID_NQ + HUMANOID_NDQ : HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU,
    ]
    return q_next, dq_next, tau0_next


def humanoid_jax_predicted_next_state(z: Any) -> tuple[Any, Any, Any]:
    n_nodes = humanoid_n_nodes_from_z_width(int(z.shape[-1]))
    offsets = humanoid_z_offsets(n_nodes)
    stage1 = z[:, int(offsets[1]) : int(offsets[2])]
    q_next = stage1[:, :HUMANOID_NQ]
    dq_next = stage1[:, HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ]
    tau0_next = z[
        :,
        HUMANOID_NQ + HUMANOID_NDQ : HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU,
    ]
    return q_next, dq_next, tau0_next


def humanoid_jax_trajectory_from_solution(z: Any, *, n_nodes: int | None = None) -> tuple[Any, Any, Any, Any]:
    if n_nodes is None:
        n_nodes = humanoid_n_nodes_from_z_width(int(z.shape[-1]))
    offsets = humanoid_z_offsets(n_nodes)
    q_parts = []
    dq_parts = []
    tau_parts = []
    force_parts = []
    for stage_index in range(n_nodes):
        stage = z[:, int(offsets[stage_index]) : int(offsets[stage_index + 1])]
        q_parts.append(stage[:, :HUMANOID_NQ])
        dq_parts.append(stage[:, HUMANOID_NQ : HUMANOID_NQ + HUMANOID_NDQ])
        if stage_index < HUMANOID_N_TORQUE_STAGES:
            tau_parts.append(
                stage[
                    :,
                    HUMANOID_NQ
                    + HUMANOID_NDQ : HUMANOID_NQ
                    + HUMANOID_NDQ
                    + HUMANOID_NTAU,
                ]
            )
            force_parts.append(stage[:, HUMANOID_NQ + HUMANOID_NDQ + HUMANOID_NTAU :])
        elif stage_index < n_nodes - 1:
            force_parts.append(stage[:, HUMANOID_NQ + HUMANOID_NDQ :])
    q = jnp.stack(q_parts, axis=2)
    dq = jnp.stack(dq_parts, axis=2)
    tau = jnp.stack(tau_parts, axis=2)
    force = jnp.stack(force_parts, axis=2)
    return q, dq, tau, force


def interpolate_humanoid_result(
    q: np.ndarray,
    dq: np.ndarray,
    tau: np.ndarray,
    discretization_times: np.ndarray,
    *,
    interpolation_offset: int,
    n_interp: int,
    interpolation_dt: float = HUMANOID_SIM_TIMESTEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate planned joint commands exactly like the debug controller."""

    batch = q.shape[0]
    dt = np.asarray(discretization_times, dtype=np.float64).reshape(-1)
    elements_per_segment = np.rint(dt / interpolation_dt).astype(np.int64)
    segment_index = 0
    cumulative_prev = 0
    cumulative_next = int(elements_per_segment[0])
    while (
        interpolation_offset >= cumulative_next
        and segment_index < len(elements_per_segment) - 1
    ):
        segment_index += 1
        cumulative_prev = cumulative_next
        cumulative_next += int(elements_per_segment[segment_index])

    q_cmd = np.zeros((batch, HUMANOID_NTAU, n_interp), dtype=q.dtype)
    dq_cmd = np.zeros((batch, HUMANOID_NTAU, n_interp), dtype=dq.dtype)
    tau_cmd = np.zeros((batch, HUMANOID_NTAU, n_interp), dtype=tau.dtype)
    for i in range(n_interp):
        while (
            interpolation_offset + i >= cumulative_next
            and segment_index < len(elements_per_segment) - 1
        ):
            segment_index += 1
            cumulative_prev = cumulative_next
            cumulative_next += int(elements_per_segment[segment_index])
        alpha = (
            (interpolation_offset + i - cumulative_prev)
            * interpolation_dt
            / float(dt[segment_index])
        )
        next_segment = min(segment_index + 1, q.shape[2] - 1)
        tau_segment = min(segment_index, tau.shape[2] - 1)
        next_tau_segment = min(segment_index + 1, tau.shape[2] - 1)
        q0 = q[:, 6:, segment_index]
        q1 = q[:, 6:, next_segment]
        dq0 = dq[:, 6:, segment_index]
        dq1 = dq[:, 6:, next_segment]
        q_cmd[:, :, i] = q0 + alpha * (q1 - q0)
        dq_cmd[:, :, i] = dq0 + alpha * (dq1 - dq0)
        tau_alpha = HUMANOID_TORQUE_FEEDFORWARD_LOOKAHEAD_S / float(dt[tau_segment])
        tau_cmd[:, :, i] = tau[:, :, tau_segment] + tau_alpha * (
            tau[:, :, next_tau_segment] - tau[:, :, tau_segment]
        )
    return q_cmd, dq_cmd, tau_cmd


def gains_for_contact(parameters: HumanoidMPCParameters, gait_pattern_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right_stance = int(gait_pattern_col[0]) == 1
    left_stance = int(gait_pattern_col[1]) == 1
    right_p = parameters.gains_leg_stance_p if right_stance else parameters.gains_leg_swing_p
    right_d = parameters.gains_leg_stance_d if right_stance else parameters.gains_leg_swing_d
    left_p = parameters.gains_leg_stance_p if left_stance else parameters.gains_leg_swing_p
    left_d = parameters.gains_leg_stance_d if left_stance else parameters.gains_leg_swing_d
    kp = np.concatenate(
        [right_p, left_p, parameters.gains_arm_p, parameters.gains_arm_p]
    ).astype(np.float64)
    kd = np.concatenate(
        [right_d, left_d, parameters.gains_arm_d, parameters.gains_arm_d]
    ).astype(np.float64)
    return kp, kd


__all__ = [
    "ACTUATOR_STATIC_TAU_MAX_NM",
    "ACTUATOR_STATIC_TAU_MIN_NM",
    "BodyReferenceCommand",
    "GaitSchedule",
    "HUMANOID_CASADI_FUNCTION_DIR",
    "HUMANOID_HIGH_LEVEL_TIMESTEP",
    "HUMANOID_INFTY",
    "HUMANOID_MPC_PERIOD_S",
    "HUMANOID_MPC_PERIOD_STEPS",
    "HUMANOID_MPC_STANDING_REFERENCE_S",
    "HUMANOID_MODEL_NAME",
    "HUMANOID_NDQ",
    "HUMANOID_NF",
    "HUMANOID_NQ",
    "HUMANOID_NTAU",
    "HUMANOID_NZ",
    "HUMANOID_NZ_FORCE",
    "HUMANOID_NZ_TERMINAL",
    "HUMANOID_NZ_TORQUE",
    "HUMANOID_N_NODES",
    "HUMANOID_N_STEPS",
    "HUMANOID_PARAMETERS_PATH",
    "HUMANOID_PHI_VEL",
    "HUMANOID_SIM_TIMESTEP",
    "HumanoidMPCParameters",
    "InPlaceWalkingGait",
    "gait_schedule_walking",
    "gains_for_contact",
    "generate_desired_ground_forces",
    "humanoid_default_parameters",
    "humanoid_dt_schedule",
    "humanoid_initial_guess_and_params",
    "humanoid_jax_predicted_next_state",
    "humanoid_jax_trajectory_from_solution",
    "humanoid_joint_limits",
    "humanoid_make_references",
    "humanoid_n_nodes_from_z_width",
    "humanoid_predicted_next_state",
    "humanoid_qhome",
    "humanoid_stage_var_dims",
    "humanoid_trajectory_from_solution",
    "humanoid_walking_gait_schedule",
    "humanoid_z_offsets",
    "interpolate_humanoid_result",
    "load_humanoid_mpc_parameters",
    "make_humanoid_sqp_problem",
    "standing_gait_schedule",
    "update_humanoid_params_gait",
    "update_humanoid_params_initial_state",
    "update_humanoid_params_mpc",
    "yaw_body_to_world_xy",
]
