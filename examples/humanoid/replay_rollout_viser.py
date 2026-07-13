#!/usr/bin/env python3
"""Replay saved humanoid headless rollout data in a Viser GUI."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MJLAB_SRC = ROOT / "resources" / "mjlab" / "src"
for path in (ROOT, MJLAB_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_XML = ROOT / "examples" / "humanoid" / "mit_humanoid" / "xmls" / "mit_humanoid.xml"
RECORDING_FPS = 30.0
RECORDING_WIDTH = 1920
RECORDING_HEIGHT = 1080
_GRID_ROUND_DECIMALS = 6
_VIEWER_CAMERA_DISTANCE = 2.0
_VIEWER_CAMERA_AZIMUTH = 135.0
_VIEWER_CAMERA_ELEVATION = 25.0
_VIEWER_LOOKAT_Z = 0.55
_VISER_FOG_NEAR = 4.0
_VISER_FOG_FAR = 150.0
_VISER_FOG_COLOR = (237, 237, 237)


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return quat / max(np.linalg.norm(quat), 1e-12)


def rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def configure_mit_humanoid_viser_scene(server: Any) -> None:
    server.scene.configure_fog(
        near=_VISER_FOG_NEAR,
        far=_VISER_FOG_FAR,
        color=_VISER_FOG_COLOR,
        enabled=True,
    )


def create_render_model(mujoco: Any, xml_path: Path) -> Any:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    terrain_geom = next((geom for geom in spec.geoms if geom.name == "terrain"), None)
    if terrain_geom is None:
        terrain_geom = spec.worldbody.add_geom()
        terrain_geom.name = "terrain"
        terrain_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        terrain_geom.size[:] = (20.0, 20.0, 0.1)
    terrain_geom.rgba[:] = (0.93, 0.93, 0.93, 1.0)
    terrain_geom.contype = 0
    terrain_geom.conaffinity = 0
    light = spec.worldbody.add_light()
    light.name = "key_light"
    light.pos[:] = (0.0, 0.0, 4.0)
    light.dir[:] = (0.0, 0.0, -1.0)
    light.diffuse[:] = (0.8, 0.8, 0.8)
    return spec.compile()


def load_rollout(path: Path) -> dict[str, Any]:
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def metadata_from_rollout(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def select_square_robot_ids(q: np.ndarray, render_count: int) -> np.ndarray:
    """Select a compact square-ish subset from the robots' initial global xy positions."""

    xy = np.asarray(q[0, :, :2], dtype=np.float64)
    finite = np.all(np.isfinite(xy), axis=1)
    valid_ids = np.flatnonzero(finite)
    if valid_ids.size == 0:
        raise ValueError("rollout has no finite robot base xy positions")
    render_count = max(1, min(int(render_count), valid_ids.size))
    if render_count == valid_ids.size:
        return valid_ids

    valid_xy = xy[valid_ids]
    center = np.median(valid_xy, axis=0)
    target_side = int(math.ceil(math.sqrt(render_count)))
    rounded_x = np.round(valid_xy[:, 0], _GRID_ROUND_DECIMALS)
    rounded_y = np.round(valid_xy[:, 1], _GRID_ROUND_DECIMALS)
    unique_x = np.unique(rounded_x)
    unique_y = np.unique(rounded_y)

    best: tuple[float, float, np.ndarray] | None = None
    # The saved batch is grid-like; use a square grid window when that structure
    # is visible. Fall back to a center-nearest Chebyshev ball for arbitrary xy.
    if unique_x.size * unique_y.size <= 4 * valid_ids.size:
        for ix in range(max(0, unique_x.size - target_side + 1)):
            x_min = unique_x[ix]
            x_max = unique_x[ix + target_side - 1]
            x_mask = (rounded_x >= x_min) & (rounded_x <= x_max)
            if int(x_mask.sum()) < render_count:
                continue
            for iy in range(max(0, unique_y.size - target_side + 1)):
                y_min = unique_y[iy]
                y_max = unique_y[iy + target_side - 1]
                mask = x_mask & (rounded_y >= y_min) & (rounded_y <= y_max)
                if int(mask.sum()) < render_count:
                    continue
                span = max(float(x_max - x_min), float(y_max - y_min))
                window_center = np.array(
                    [(float(x_min) + float(x_max)) * 0.5, (float(y_min) + float(y_max)) * 0.5]
                )
                center_dist = float(np.linalg.norm(window_center - center))
                candidate_ids = valid_ids[mask]
                key = (span, center_dist)
                if best is None or key < best[:2]:
                    best = (span, center_dist, candidate_ids)

    if best is None:
        score = np.max(np.abs(valid_xy - center[None, :]), axis=1)
        candidate_ids = valid_ids[np.argsort(score, kind="stable")[:render_count]]
    else:
        candidate_ids = best[2]
        candidate_xy = xy[candidate_ids]
        candidate_center = np.mean(candidate_xy, axis=0)
        score = np.sum((candidate_xy - candidate_center[None, :]) ** 2, axis=1)
        candidate_ids = candidate_ids[np.argsort(score, kind="stable")[:render_count]]

    # Deterministic row-major ordering keeps the visualized square coherent.
    candidate_xy = xy[candidate_ids]
    order = np.lexsort((candidate_xy[:, 1], candidate_xy[:, 0]))
    return candidate_ids[order]


def select_front_right_render_index(q: np.ndarray, robot_ids: np.ndarray) -> int:
    """Pick the selected robot within the rendered square.

    Humanoid convention is x-forward and y-left, so front-right means maximum x
    and minimum y. For the 2x2 diagnostic grid this changes focus from the
    back-right robot to the front-right robot.
    """

    xy = np.asarray(q[0, robot_ids, :2], dtype=np.float64)
    finite = np.all(np.isfinite(xy), axis=1)
    if not np.any(finite):
        return 0
    valid_ids = np.flatnonzero(finite)
    valid_xy = xy[valid_ids]
    order = np.lexsort((valid_xy[:, 1], -valid_xy[:, 0]))
    return int(valid_ids[order[0]])


def camera_offset_from_angles() -> np.ndarray:
    azimuth = math.radians(_VIEWER_CAMERA_AZIMUTH)
    elevation = math.radians(_VIEWER_CAMERA_ELEVATION)
    return _VIEWER_CAMERA_DISTANCE * np.array(
        [
            -math.cos(elevation) * math.cos(azimuth),
            -math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


@dataclass
class CanvasRecording:
    client: Any
    directory: Path
    start_time: float
    next_frame_time: float
    frame_count: int = 0

    @property
    def frame_glob(self) -> str:
        return str(self.directory / "frame_%06d.jpg")

    @property
    def video_path(self) -> Path:
        return self.directory / f"{self.directory.name}.mp4"


class HumanoidRolloutReplay:
    def __init__(
        self,
        *,
        rollout_path: Path,
        xml_path: Path,
        render_count: int,
        host: str,
        port: int,
        playback_fps: float,
        start_time_s: float = 0.0,
    ) -> None:
        import mujoco
        import viser
        from mjlab.viewer.viser.scene import MjlabViserScene

        self.mujoco = mujoco
        self.viser = viser
        self.MjlabViserScene = MjlabViserScene
        self.data = load_rollout(rollout_path)
        self.metadata = metadata_from_rollout(self.data)
        self.q = np.asarray(self.data["q"], dtype=np.float64)
        self.dq = np.asarray(self.data.get("dq", np.zeros_like(self.q)), dtype=np.float64)
        self.time = np.asarray(self.data["time"], dtype=np.float64)
        if self.q.ndim != 3 or self.q.shape[2] < 24:
            raise ValueError("rollout q must have shape (frames, robots, 24)")
        if self.dq.shape != self.q.shape:
            self.dq = np.zeros_like(self.q)
        if self.time.shape[0] != self.q.shape[0]:
            raise ValueError("rollout time must have one entry per q frame")
        if start_time_s > 0.0:
            start_index = int(np.searchsorted(self.time, float(start_time_s), side="left"))
            start_index = max(0, min(start_index, self.q.shape[0] - 1))
            self.q = self.q[start_index:]
            self.dq = self.dq[start_index:]
            self.time = self.time[start_index:]
        self.render_count = max(1, min(int(render_count), self.q.shape[1]))
        self.robot_ids = select_square_robot_ids(self.q, self.render_count)
        self.focus_env_idx = select_front_right_render_index(self.q, self.robot_ids)
        focus_q0 = self.q[0, self.robot_ids[self.focus_env_idx]]
        focus_xy0 = focus_q0[:2] if np.all(np.isfinite(focus_q0[:2])) else np.zeros(2, dtype=np.float64)
        self.initial_camera_focus = np.array(
            [focus_xy0[0], focus_xy0[1], _VIEWER_LOOKAT_Z],
            dtype=np.float64,
        )
        self.frame = 0
        self.paused = False
        self.single_step = False
        self.speed = 1.0
        self.playback_dt = 1.0 / playback_fps
        self.model = create_render_model(mujoco, xml_path)
        self.model.stat.center[:] = self.initial_camera_focus
        self.mjdata = mujoco.MjData(self.model)
        self.body_xpos = np.zeros((self.render_count, self.model.nbody, 3), dtype=np.float64)
        self.body_xmat = np.zeros((self.render_count, self.model.nbody, 3, 3), dtype=np.float64)
        self.qpos = np.zeros((self.render_count, self.model.nq), dtype=np.float64)
        self.qvel = np.zeros((self.render_count, self.model.nv), dtype=np.float64)
        self.server = viser.ViserServer(host=host, port=port, label="mjlab")
        configure_mit_humanoid_viser_scene(self.server)
        self.scene = MjlabViserScene(
            server=self.server,
            mj_model=self.model,
            num_envs=self.render_count,
            sim_model=None,
            expanded_fields=set(),
            max_extra_envs=self.render_count - 1,
        )
        self.scene.env_idx = self.focus_env_idx
        self.scene.show_all_envs = True
        self.scene.debug_visualization_enabled = False
        self.status_html = None
        self.pause_button = None
        self.frame_slider = None
        self.record_button = None
        self.record_download_button = None
        self.record_status_html = None
        self.recording: CanvasRecording | None = None
        self.recording_finalizing = False
        self.recording_status = "Idle"
        self.last_recording_path: Path | None = None
        self.last_recording_client: Any | None = None
        self._setup_gui()
        self._setup_camera_focus()

    def _setup_gui(self) -> None:
        tabs = self.server.gui.add_tab_group()
        with tabs.add_tab("Controls", icon=self.viser.Icon.SETTINGS):
            with self.server.gui.add_folder("Info"):
                self.status_html = self.server.gui.add_html("")
            with self.server.gui.add_folder("Playback"):
                self.pause_button = self.server.gui.add_button(
                    "Pause",
                    icon=self.viser.Icon.PLAYER_PAUSE,
                )

                @self.pause_button.on_click
                def _(_) -> None:
                    self.paused = not self.paused
                    self._sync_ui()

                step_button = self.server.gui.add_button("Step", icon=self.viser.Icon.PLAYER_TRACK_NEXT)

                @step_button.on_click
                def _(_) -> None:
                    self.single_step = True
                    self.paused = True
                    self._sync_ui()

                reset_button = self.server.gui.add_button("Reset")

                @reset_button.on_click
                def _(_) -> None:
                    self.frame = 0
                    self._sync_ui()
                    self.update_scene()

                speed_buttons = self.server.gui.add_button_group(
                    "Speed",
                    options=["Slower", "1x", "Faster"],
                )

                @speed_buttons.on_click
                def _(event) -> None:
                    if event.target.value == "Slower":
                        self.speed = max(0.05, self.speed * 0.5)
                    elif event.target.value == "1x":
                        self.speed = 1.0
                    else:
                        self.speed = min(16.0, self.speed * 2.0)
                    self._sync_ui()

                self.frame_slider = self.server.gui.add_slider(
                    "Frame",
                    min=0,
                    max=self.q.shape[0] - 1,
                    step=1,
                    initial_value=0,
                )

                @self.frame_slider.on_update
                def _(_) -> None:
                    self.frame = int(self.frame_slider.value)
                    self.update_scene()
                    self._sync_ui()

            with self.server.gui.add_folder("Recording"):
                self.record_status_html = self.server.gui.add_html("")
                self.record_button = self.server.gui.add_button(
                    "Start Recording",
                    icon=self.viser.Icon.PLAYER_RECORD,
                )

                @self.record_button.on_click
                def _(event) -> None:
                    if self.recording is None:
                        self.start_recording(event.client)
                    else:
                        self.stop_recording()

                self.record_download_button = self.server.gui.add_button(
                    "Download Last Video",
                    icon=self.viser.Icon.DOWNLOAD,
                )
                self.record_download_button.visible = False

                @self.record_download_button.on_click
                def _(event) -> None:
                    self.send_recording_download(event.client)

            with self.server.gui.add_folder("Scene"):
                self.scene.create_scene_gui(
                    camera_distance=_VIEWER_CAMERA_DISTANCE,
                    camera_azimuth=_VIEWER_CAMERA_AZIMUTH,
                    camera_elevation=_VIEWER_CAMERA_ELEVATION,
                    show_debug_viz_control=True,
                )

        with tabs.add_tab("Visualization", icon=self.viser.Icon.EYE):
            self.scene.create_overlay_gui()
        with tabs.add_tab("Groups", icon=self.viser.Icon.LAYERS_INTERSECT):
            self.scene.create_groups_gui()
        self._sync_ui()

    def camera_focus(self) -> np.ndarray:
        return self.initial_camera_focus

    def _set_client_camera_focus(self, client: Any) -> None:
        focus = self.camera_focus()
        fallback_offset = camera_offset_from_angles()
        try:
            old_position = np.asarray(client.camera.position, dtype=np.float64)
            old_look_at = np.asarray(client.camera.look_at, dtype=np.float64)
            offset = old_position - old_look_at
            if offset.shape != (3,) or not np.all(np.isfinite(offset)) or np.linalg.norm(offset) < 1e-6:
                offset = fallback_offset
        except Exception:
            offset = fallback_offset
        client.camera.look_at = tuple(focus.tolist())
        client.camera.position = tuple((focus + offset).tolist())
        if hasattr(client.camera, "up_direction"):
            client.camera.up_direction = (0.0, 0.0, 1.0)

    def _setup_camera_focus(self) -> None:
        if hasattr(self.server, "on_client_connect"):
            @self.server.on_client_connect
            def _(client: Any) -> None:
                self._set_client_camera_focus(client)
        for client in self.server.get_clients().values():
            self._set_client_camera_focus(client)

    def _sync_ui(self) -> None:
        if self.pause_button is not None:
            self.pause_button.label = "Play" if self.paused else "Pause"
            self.pause_button.icon = (
                self.viser.Icon.PLAYER_PLAY if self.paused else self.viser.Icon.PLAYER_PAUSE
            )
        if self.frame_slider is not None and int(self.frame_slider.value) != self.frame:
            self.frame_slider.value = self.frame
        if self.status_html is not None:
            failed_count = self.metadata.get("failed_count", "-")
            self.status_html.content = (
                '<div style="font-size:0.85em;line-height:1.25;">'
                f"<strong>Frame:</strong> {self.frame}/{self.q.shape[0] - 1}<br/>"
                f"<strong>Time:</strong> {self.time[self.frame]:.3f}s<br/>"
                f"<strong>Robots:</strong> {self.render_count}/{self.q.shape[1]}<br/>"
                f"<strong>Focus:</strong> {int(self.robot_ids[self.focus_env_idx])}<br/>"
                f"<strong>Speed:</strong> {self.speed:.2f}x<br/>"
                f"<strong>Failures:</strong> {failed_count}"
                "</div>"
            )
        self.sync_recording_ui()

    def _recording_directory(self) -> Path:
        root = Path(tempfile.gettempdir()) / "humanoid_mpc_recordings"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"viser_recording_{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1.0) * 1000):03d}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def recording_client_or_first(self, client: Any | None) -> Any | None:
        if client is not None:
            return client
        clients = self.server.get_clients()
        if not clients:
            return None
        return next(iter(clients.values()))

    def start_recording(self, client: Any | None) -> None:
        if self.recording_finalizing:
            self.recording_status = "Waiting for ffmpeg to finish..."
            self.sync_recording_ui()
            return
        client = self.recording_client_or_first(client)
        if client is None:
            self.recording_status = "Open a Viser browser client before recording."
            self.sync_recording_ui()
            return
        sim_time = float(self.time[self.frame])
        self.recording = CanvasRecording(
            client=client,
            directory=self._recording_directory(),
            start_time=sim_time,
            next_frame_time=sim_time,
        )
        self.last_recording_path = None
        self.last_recording_client = client
        self.recording_status = "Recording..."
        self.sync_recording_ui()

    def stop_recording(self) -> None:
        recording = self.recording
        if recording is None:
            return
        self.recording = None
        self.recording_finalizing = True
        self.recording_status = f"Encoding {recording.frame_count} frames with ffmpeg..."
        self.sync_recording_ui()
        threading.Thread(target=self.encode_recording_worker, args=(recording,), daemon=True).start()

    def encode_recording_worker(self, recording: CanvasRecording) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ok = False
        message = ""
        if ffmpeg is None:
            message = "ffmpeg was not found on PATH."
        elif recording.frame_count == 0:
            message = "No frames were captured."
        else:
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                f"{RECORDING_FPS:g}",
                "-i",
                recording.frame_glob,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(recording.video_path),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            ok = result.returncode == 0
            message = (result.stderr or result.stdout or "ffmpeg failed").strip()
        self.recording_finalizing = False
        if ok:
            self.last_recording_path = recording.video_path
            self.last_recording_client = recording.client
            duration = recording.frame_count / RECORDING_FPS
            self.recording_status = (
                f"Ready: {recording.video_path.name} "
                f"({recording.frame_count} frames, {duration:.2f}s video)"
            )
            self.send_recording_download(recording.client)
        else:
            self.recording_status = f"Recording failed: {message}"
        self.sync_recording_ui()

    def send_recording_download(self, client: Any | None) -> None:
        if self.last_recording_path is None or not self.last_recording_path.exists():
            self.recording_status = "No encoded video is available yet."
            self.sync_recording_ui()
            return
        target_client = self.recording_client_or_first(client)
        if target_client is None:
            self.recording_status = f"Video ready on disk: {self.last_recording_path}"
            self.sync_recording_ui()
            return
        target_client.send_file_download(
            self.last_recording_path.name,
            self.last_recording_path.read_bytes(),
            save_immediately=False,
        )

    def maybe_capture_recording_frame(self) -> None:
        recording = self.recording
        if recording is None:
            return
        sim_time = float(self.time[self.frame])
        if sim_time + 1e-12 < recording.next_frame_time:
            return
        frame_path = recording.directory / f"frame_{recording.frame_count:06d}.jpg"
        try:
            import imageio.v3 as iio

            image = recording.client.get_render(
                height=RECORDING_HEIGHT,
                width=RECORDING_WIDTH,
                transport_format="jpeg",
            )
            iio.imwrite(frame_path, image, extension=".jpg", quality=95)
        except Exception as exc:
            self.recording = None
            self.recording_status = f"Recording stopped: {exc}"
            self.sync_recording_ui()
            return
        elapsed_intervals = max(
            1,
            int(math.floor((sim_time - recording.next_frame_time) * RECORDING_FPS + 1e-12)) + 1,
        )
        recording.next_frame_time += elapsed_intervals / RECORDING_FPS
        recording.frame_count += 1

    def sync_recording_ui(self) -> None:
        if self.record_button is not None:
            self.record_button.disabled = self.recording_finalizing
            if self.recording is None:
                self.record_button.label = "Encoding..." if self.recording_finalizing else "Start Recording"
                self.record_button.icon = self.viser.Icon.PLAYER_RECORD
            else:
                self.record_button.label = "Stop Recording"
                self.record_button.icon = self.viser.Icon.PLAYER_STOP
        if self.record_download_button is not None:
            self.record_download_button.visible = self.last_recording_path is not None
            self.record_download_button.disabled = self.recording_finalizing
        if self.record_status_html is not None:
            if self.recording is None:
                self.record_status_html.content = (
                    '<div style="font-size:0.85em;line-height:1.25;">'
                    f"<strong>Recording:</strong> {self.recording_status}<br/>"
                    f"<strong>Target:</strong> {RECORDING_FPS:.0f} fps, {RECORDING_WIDTH}x{RECORDING_HEIGHT}"
                    "</div>"
                )
            else:
                duration = self.recording.frame_count / RECORDING_FPS
                self.record_status_html.content = (
                    '<div style="font-size:0.85em;line-height:1.25;">'
                    "<strong>Recording:</strong> Active<br/>"
                    f"<strong>Frames:</strong> {self.recording.frame_count}<br/>"
                    f"<strong>Video Time:</strong> {duration:.2f}s"
                    "</div>"
                )

    def update_scene(self) -> None:
        q_frame = self.q[self.frame, self.robot_ids]
        dq_frame = self.dq[self.frame, self.robot_ids]
        for i, (q, dq) in enumerate(zip(q_frame, dq_frame, strict=True)):
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
        self.maybe_capture_recording_frame()

    def run(self) -> None:
        print(f"Viser: http://localhost:{self.server.get_port()}")
        print(f"frames={self.q.shape[0]} robots={self.q.shape[1]} rendering={self.render_count}")
        self.update_scene()
        next_wall = time.perf_counter()
        while True:
            if self.single_step:
                self.frame = min(self.frame + 1, self.q.shape[0] - 1)
                self.single_step = False
                self.update_scene()
                self._sync_ui()
            elif not self.paused:
                now = time.perf_counter()
                if now >= next_wall:
                    self.frame = (self.frame + 1) % self.q.shape[0]
                    self.update_scene()
                    self._sync_ui()
                    dt = max(1e-4, self.playback_dt / max(self.speed, 1e-6))
                    next_wall = now + dt
            time.sleep(0.002)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--render-count", type=int, default=256)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--playback-fps", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xml.exists():
        raise FileNotFoundError(f"Humanoid XML not found: {args.xml}")
    HumanoidRolloutReplay(
        rollout_path=args.rollout,
        xml_path=args.xml,
        render_count=args.render_count,
        host=args.host,
        port=args.port,
        playback_fps=args.playback_fps,
    ).run()


if __name__ == "__main__":
    main()
