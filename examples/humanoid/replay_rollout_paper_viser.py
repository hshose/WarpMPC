#!/usr/bin/env python3
"""Paper-only humanoid rollout replay with robot clipping and a cinematic camera."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from replay_rollout_viser import (
        DEFAULT_XML,
        _VIEWER_CAMERA_AZIMUTH,
        _VIEWER_CAMERA_DISTANCE,
        _VIEWER_CAMERA_ELEVATION,
        HumanoidRolloutReplay,
        rpy_to_quat,
        rpy_to_rotmat,
    )
except ModuleNotFoundError:
    from examples.humanoid.replay_rollout_viser import (
        DEFAULT_XML,
        _VIEWER_CAMERA_AZIMUTH,
        _VIEWER_CAMERA_DISTANCE,
        _VIEWER_CAMERA_ELEVATION,
        HumanoidRolloutReplay,
        rpy_to_quat,
        rpy_to_rotmat,
    )

_HIDDEN_Z = -1000.0


def _camera_offset(distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return distance * np.array(
        [
            -math.cos(elevation) * math.cos(azimuth),
            -math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def _smootherstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _continuous_accel_distance(s: float, total: float, initial_velocity: float, duration: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    duration = max(float(duration), 1e-9)
    linear = max(0.0, float(initial_velocity)) * duration
    if total <= linear:
        return float(total) * s
    return linear * s + (float(total) - linear) * s * s


class PaperHumanoidRolloutReplay(HumanoidRolloutReplay):
    def __init__(
        self,
        *,
        failure_hide_lead_s: float,
        cinematic_camera: bool,
        camera_lift_m: float,
        camera_downward_pan_deg: float,
        camera_diagonal_drift_m: float,
        camera_traversal_rise_m: float,
        camera_intro_lift_fraction: float,
        trajectory_clip_start_s: float,
        **kwargs: Any,
    ) -> None:
        self.failure_hide_lead_s = float(failure_hide_lead_s)
        self.cinematic_camera_enabled = bool(cinematic_camera)
        self.camera_lift_m = float(camera_lift_m)
        self.camera_downward_pan_deg = float(camera_downward_pan_deg)
        self.camera_diagonal_drift_m = float(camera_diagonal_drift_m)
        self.camera_traversal_rise_m = float(camera_traversal_rise_m)
        self.camera_intro_lift_fraction = float(np.clip(camera_intro_lift_fraction, 0.05, 0.95))
        self.trajectory_clip_start_s = float(trajectory_clip_start_s)
        self.cinematic_camera_checkbox = None
        self.hide_time = np.empty(0, dtype=np.float64)
        self.hidden_count = 0
        super().__init__(start_time_s=self.trajectory_clip_start_s, **kwargs)
        self.hide_time = self._make_hide_times()
        self._hidden_body_xpos = np.zeros((self.model.nbody, 3), dtype=np.float64)
        self._hidden_body_xpos[:, 2] = _HIDDEN_Z
        self._hidden_body_xmat = np.broadcast_to(
            np.eye(3, dtype=np.float64),
            (self.model.nbody, 3, 3),
        ).copy()
        self._hidden_qpos = np.zeros(self.model.nq, dtype=np.float64)
        if self.model.nq >= 3:
            self._hidden_qpos[2] = _HIDDEN_Z
        self._hidden_qvel = np.zeros(self.model.nv, dtype=np.float64)
        self._sync_ui()

    def _make_hide_times(self) -> np.ndarray:
        hide_time = np.full(self.q.shape[1], np.inf, dtype=np.float64)
        failed = np.asarray(self.data.get("failed", np.zeros(self.q.shape[1], dtype=bool)), dtype=bool)
        failure_time = np.asarray(
            self.data.get("failure_time", np.full(self.q.shape[1], np.nan)),
            dtype=np.float64,
        )
        if failed.shape[0] != self.q.shape[1] or failure_time.shape[0] != self.q.shape[1]:
            return hide_time[self.robot_ids]
        valid = failed & np.isfinite(failure_time)
        hide_time[valid] = np.maximum(0.0, failure_time[valid] - self.failure_hide_lead_s)
        return hide_time[self.robot_ids]

    def _setup_gui(self) -> None:
        if hasattr(self.scene, "camera_tracking_enabled"):
            self.scene.camera_tracking_enabled = False
        super()._setup_gui()
        with self.server.gui.add_folder("Paper Camera"):
            self.cinematic_camera_checkbox = self.server.gui.add_checkbox(
                "Cinematic Camera",
                initial_value=self.cinematic_camera_enabled,
            )

            @self.cinematic_camera_checkbox.on_update
            def _(_) -> None:
                self.cinematic_camera_enabled = bool(self.cinematic_camera_checkbox.value)
                self.apply_cinematic_camera_to_clients()

    def _setup_camera_focus(self) -> None:
        if hasattr(self.server, "on_client_connect"):
            @self.server.on_client_connect
            def _(client: Any) -> None:
                self.apply_cinematic_camera(client)
        self.apply_cinematic_camera_to_clients()

    def _sequence_progress(self) -> float:
        if self.time.size < 2:
            return 0.0
        duration = max(float(self.time[-1] - self.time[0]), 1e-9)
        return float(np.clip((float(self.time[self.frame]) - float(self.time[0])) / duration, 0.0, 1.0))

    def cinematic_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        u = self._sequence_progress()
        intro_fraction = self.camera_intro_lift_fraction
        traversal_fraction = max(1.0 - intro_fraction, 1e-9)
        diagonal_distance = math.sqrt(2.0) * self.camera_diagonal_drift_m
        traversal_rise_total = max(0.0, self.camera_traversal_rise_m)
        traversal_path_length = math.hypot(diagonal_distance, traversal_rise_total)
        intro_u = min(u / intro_fraction, 1.0)
        intro_lift = self.camera_lift_m * intro_u * intro_u
        join_velocity = 0.0
        if intro_fraction > 0.0 and self.camera_lift_m > 0.0:
            join_velocity = 2.0 * self.camera_lift_m / intro_fraction
        if traversal_path_length > 1e-12:
            join_velocity = min(join_velocity, 0.8 * traversal_path_length / traversal_fraction)
        diagonal_entry_velocity = 0.0
        traversal_rise_entry_velocity = 0.0
        if traversal_path_length > 1e-12:
            diagonal_entry_velocity = join_velocity * diagonal_distance / traversal_path_length
            traversal_rise_entry_velocity = join_velocity * traversal_rise_total / traversal_path_length
        traversal_u = max(0.0, (u - intro_fraction) / traversal_fraction)
        diagonal_distance_now = _continuous_accel_distance(
            traversal_u,
            diagonal_distance,
            diagonal_entry_velocity,
            traversal_fraction,
        )
        traversal_rise = _continuous_accel_distance(
            traversal_u,
            traversal_rise_total,
            traversal_rise_entry_velocity,
            traversal_fraction,
        )
        pan = traversal_u
        diagonal_component = diagonal_distance_now / math.sqrt(2.0)
        drift = np.array(
            [
                -diagonal_component,
                diagonal_component,
                0.0,
            ],
            dtype=np.float64,
        )
        rise = intro_lift + traversal_rise
        elevation = _VIEWER_CAMERA_ELEVATION + self.camera_downward_pan_deg * pan
        rig_lift = np.array([0.0, 0.0, rise], dtype=np.float64)
        look_at = self.initial_camera_focus + drift + rig_lift
        position = (
            look_at
            + _camera_offset(_VIEWER_CAMERA_DISTANCE, _VIEWER_CAMERA_AZIMUTH, elevation)
        )
        return position, look_at

    def apply_cinematic_camera(self, client: Any) -> None:
        if not self.cinematic_camera_enabled:
            return
        position, look_at = self.cinematic_camera_pose()
        client.camera.position = tuple(position.tolist())
        client.camera.look_at = tuple(look_at.tolist())
        if hasattr(client.camera, "up_direction"):
            client.camera.up_direction = (0.0, 0.0, 1.0)

    def apply_cinematic_camera_to_clients(self) -> None:
        for client in self.server.get_clients().values():
            self.apply_cinematic_camera(client)

    def _hidden_mask(self) -> np.ndarray:
        if self.hide_time.size != self.render_count:
            return np.zeros(self.render_count, dtype=bool)
        return float(self.time[self.frame]) + 1e-12 >= self.hide_time

    def _sync_ui(self) -> None:
        super()._sync_ui()
        if self.status_html is None or self.hide_time.size != self.render_count:
            return
        hidden = self._hidden_mask()
        self.hidden_count = int(hidden.sum())
        self.status_html.content = self.status_html.content.replace(
            "</div>",
            f"<br/><strong>Hidden:</strong> {self.hidden_count}/{self.render_count}</div>",
        )

    def update_scene(self) -> None:
        q_frame = self.q[self.frame, self.robot_ids]
        dq_frame = self.dq[self.frame, self.robot_ids]
        hidden = self._hidden_mask()
        for i, (q, dq) in enumerate(zip(q_frame, dq_frame, strict=True)):
            if bool(hidden[i]):
                self.body_xpos[i] = self._hidden_body_xpos
                self.body_xmat[i] = self._hidden_body_xmat
                self.qpos[i] = self._hidden_qpos
                self.qvel[i] = self._hidden_qvel
                continue
            self.mjdata.qpos[:3] = q[:3]
            self.mjdata.qpos[3:7] = rpy_to_quat(q[3:6])
            self.mjdata.qpos[7 : 7 + 18] = q[6:24]
            if self.mjdata.qvel.size >= 24:
                rot = rpy_to_rotmat(q[3:6])
                self.mjdata.qvel[:3] = rot @ dq[3:6]
                self.mjdata.qvel[3:6] = rot @ dq[:3]
                self.mjdata.qvel[6 : 6 + 18] = dq[6:24]
            self.mujoco.mj_forward(self.model, self.mjdata)
            self.body_xpos[i] = self.mjdata.xpos
            self.body_xmat[i] = self.mjdata.xmat.reshape(self.model.nbody, 3, 3)
            self.qpos[i] = self.mjdata.qpos
            self.qvel[i] = self.mjdata.qvel
        with self.server.atomic():
            self.scene.paused = self.paused
            self.scene.update_from_arrays(
                self.body_xpos,
                self.body_xmat,
                env_idx=self.focus_env_idx,
                qpos=self.qpos,
                qvel=self.qvel,
            )
            self.server.flush()
        self.apply_cinematic_camera_to_clients()
        self.maybe_capture_recording_frame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--render-count", type=int, default=256)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--playback-fps", type=float, default=60.0)
    parser.add_argument("--failure-hide-lead-s", type=float, default=0.5)
    parser.add_argument("--cinematic-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-lift-m", type=float, default=2.0)
    parser.add_argument("--camera-downward-pan-deg", type=float, default=10.0)
    parser.add_argument("--camera-diagonal-drift-m", type=float, default=8.0)
    parser.add_argument("--camera-traversal-rise-m", type=float, default=1.5)
    parser.add_argument("--camera-intro-lift-fraction", type=float, default=0.4)
    parser.add_argument("--trajectory-clip-start-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xml.exists():
        raise FileNotFoundError(f"Humanoid XML not found: {args.xml}")
    PaperHumanoidRolloutReplay(
        rollout_path=args.rollout,
        xml_path=args.xml,
        render_count=args.render_count,
        host=args.host,
        port=args.port,
        playback_fps=args.playback_fps,
        failure_hide_lead_s=args.failure_hide_lead_s,
        cinematic_camera=args.cinematic_camera,
        camera_lift_m=args.camera_lift_m,
        camera_downward_pan_deg=args.camera_downward_pan_deg,
        camera_diagonal_drift_m=args.camera_diagonal_drift_m,
        camera_traversal_rise_m=args.camera_traversal_rise_m,
        camera_intro_lift_fraction=args.camera_intro_lift_fraction,
        trajectory_clip_start_s=args.trajectory_clip_start_s,
    ).run()


if __name__ == "__main__":
    main()
