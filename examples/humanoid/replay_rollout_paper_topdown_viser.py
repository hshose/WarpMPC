#!/usr/bin/env python3
"""Paper-only top-down humanoid rollout replay."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from replay_rollout_paper_viser import PaperHumanoidRolloutReplay
    from replay_rollout_viser import DEFAULT_XML, _VIEWER_LOOKAT_Z, select_square_robot_ids
except ModuleNotFoundError:
    from examples.humanoid.replay_rollout_paper_viser import PaperHumanoidRolloutReplay
    from examples.humanoid.replay_rollout_viser import DEFAULT_XML, _VIEWER_LOOKAT_Z, select_square_robot_ids


def _median_grid_spacing(xy: np.ndarray) -> float:
    spacings: list[float] = []
    for axis in range(2):
        values = np.unique(np.round(xy[:, axis], 6))
        if values.size > 1:
            diffs = np.diff(np.sort(values))
            diffs = diffs[diffs > 1e-9]
            if diffs.size:
                spacings.append(float(np.median(diffs)))
    if not spacings:
        return 1.0
    return float(np.median(spacings))


def _topdown_height_for_visible_width(visible_width: float, vertical_fov_deg: float, aspect: float) -> float:
    vertical_fov = math.radians(vertical_fov_deg)
    horizontal_fov = 2.0 * math.atan(math.tan(vertical_fov * 0.5) * aspect)
    return float(visible_width / max(2.0 * math.tan(horizontal_fov * 0.5), 1e-9))


def _ease_in_out_smootherstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


class TopDownPaperHumanoidRolloutReplay(PaperHumanoidRolloutReplay):
    def __init__(
        self,
        *,
        topdown_move_s: float,
        topdown_fov_deg: float,
        topdown_aspect: float,
        topdown_start_height_m: float,
        topdown_end_fill_fraction: float,
        **kwargs: Any,
    ) -> None:
        self.topdown_move_s = float(topdown_move_s)
        self.topdown_fov_deg = float(topdown_fov_deg)
        self.topdown_aspect = float(topdown_aspect)
        self.topdown_start_height_m = float(topdown_start_height_m)
        self.topdown_end_fill_fraction = float(np.clip(topdown_end_fill_fraction, 0.1, 0.98))
        self.topdown_center = np.zeros(3, dtype=np.float64)
        self.topdown_start_height = self.topdown_start_height_m
        self.topdown_end_height = self.topdown_start_height_m
        super().__init__(**kwargs)
        self._configure_topdown_grid_from_initial_frame()
        self.apply_cinematic_camera_to_clients()
        self._sync_ui()

    def _configure_topdown_grid_from_initial_frame(self) -> None:
        original_q = np.asarray(self.data["q"], dtype=np.float64)
        self.robot_ids = select_square_robot_ids(original_q, self.render_count)
        initial_xy = np.asarray(original_q[0, self.robot_ids, :2], dtype=np.float64)
        finite = np.all(np.isfinite(initial_xy), axis=1)
        if not np.any(finite):
            initial_xy = np.zeros((1, 2), dtype=np.float64)
        else:
            initial_xy = initial_xy[finite]
        xy_min = np.min(initial_xy, axis=0)
        xy_max = np.max(initial_xy, axis=0)
        spacing = _median_grid_spacing(initial_xy)
        square_width = max(float(np.max(xy_max - xy_min) + spacing), spacing)
        center_xy = (xy_min + xy_max) * 0.5
        self.topdown_center = np.array([center_xy[0], center_xy[1], _VIEWER_LOOKAT_Z], dtype=np.float64)
        self.initial_camera_focus = self.topdown_center
        self.model.stat.center[:] = self.topdown_center
        center_distance = np.sum((np.asarray(original_q[0, self.robot_ids, :2], dtype=np.float64) - center_xy) ** 2, axis=1)
        self.focus_env_idx = int(np.argmin(center_distance))
        self.scene.env_idx = self.focus_env_idx
        self.hide_time = self._make_hide_times()

        end_visible_width = square_width / self.topdown_end_fill_fraction
        self.topdown_end_height = _topdown_height_for_visible_width(
            end_visible_width,
            self.topdown_fov_deg,
            self.topdown_aspect,
        )
        self.topdown_start_height = self.topdown_start_height_m
        if self.topdown_start_height > self.topdown_end_height:
            self.topdown_end_height = self.topdown_start_height

    def _setup_gui(self) -> None:
        super()._setup_gui()
        with self.server.gui.add_folder("Top Down Camera"):
            self.server.gui.add_html(
                '<div style="font-size:0.85em;line-height:1.25;">'
                f"<strong>FOV:</strong> {self.topdown_fov_deg:.1f} deg<br/>"
                f"<strong>Rise:</strong> {self.topdown_move_s:.1f}s"
                "</div>"
            )

    def cinematic_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        elapsed = max(0.0, float(self.time[self.frame] - self.time[0]))
        progress = _ease_in_out_smootherstep(elapsed / max(self.topdown_move_s, 1e-9))
        height = self.topdown_start_height + (self.topdown_end_height - self.topdown_start_height) * progress
        look_at = self.topdown_center.copy()
        position = look_at + np.array([0.0, 0.0, height], dtype=np.float64)
        return position, look_at

    def apply_cinematic_camera(self, client: Any) -> None:
        if not self.cinematic_camera_enabled:
            return
        position, look_at = self.cinematic_camera_pose()
        client.camera.position = tuple(position.tolist())
        client.camera.look_at = tuple(look_at.tolist())
        client.camera.fov = math.radians(self.topdown_fov_deg)
        if hasattr(client.camera, "up_direction"):
            client.camera.up_direction = (1.0, 0.0, 0.0)


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
    parser.add_argument("--trajectory-clip-start-s", type=float, default=5.0)
    parser.add_argument("--topdown-move-s", type=float, default=5.0)
    parser.add_argument("--topdown-fov-deg", type=float, default=60.0)
    parser.add_argument("--topdown-aspect", type=float, default=16.0 / 9.0)
    parser.add_argument("--topdown-start-height-m", type=float, default=2.73)
    parser.add_argument("--topdown-end-fill-fraction", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xml.exists():
        raise FileNotFoundError(f"Humanoid XML not found: {args.xml}")
    TopDownPaperHumanoidRolloutReplay(
        rollout_path=args.rollout,
        xml_path=args.xml,
        render_count=args.render_count,
        host=args.host,
        port=args.port,
        playback_fps=args.playback_fps,
        failure_hide_lead_s=args.failure_hide_lead_s,
        cinematic_camera=args.cinematic_camera,
        trajectory_clip_start_s=args.trajectory_clip_start_s,
        topdown_move_s=args.topdown_move_s,
        topdown_fov_deg=args.topdown_fov_deg,
        topdown_aspect=args.topdown_aspect,
        topdown_start_height_m=args.topdown_start_height_m,
        topdown_end_fill_fraction=args.topdown_end_fill_fraction,
        camera_lift_m=0.0,
        camera_downward_pan_deg=0.0,
        camera_diagonal_drift_m=0.0,
        camera_traversal_rise_m=0.0,
        camera_intro_lift_fraction=0.4,
    ).run()


if __name__ == "__main__":
    main()
