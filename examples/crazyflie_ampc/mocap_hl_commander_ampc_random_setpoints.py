#!/usr/bin/env python3
"""Run random AMPC setpoint experiments and record setpoint-relative errors."""

from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import time
from threading import Event, Thread

import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator
import motioncapture


STATE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")
SETPOINT_LABELS = ("x", "y", "z", "yaw")


def wrap_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def wrap_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def deg_to_rad(degrees: float) -> float:
    return degrees * math.pi / 180.0


class SegmentRecorder:
    def __init__(
        self,
        *,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
    ) -> None:
        self.rows: list[dict[str, float | int | str]] = []
        self.segments: list[dict[str, float | int | str]] = []
        self.active_segment: dict[str, float | int | str] | None = None
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

    def start_segment(
        self,
        *,
        index: int,
        setpoint: tuple[float, float, float, float],
        delta: tuple[float, float, float, float],
    ) -> None:
        now = time.time()
        segment = {
            "index": index,
            "name": f"random_{index:03d}",
            "start_host_time_s": now,
            "setpoint_x": setpoint[0],
            "setpoint_y": setpoint[1],
            "setpoint_z": setpoint[2],
            "setpoint_yaw_deg": setpoint[3],
            "delta_x": delta[0],
            "delta_y": delta[1],
            "delta_z": delta[2],
            "delta_yaw_deg": delta[3],
        }
        self.segments.append(segment)
        self.active_segment = segment

    def stop_segment(self) -> None:
        self.active_segment = None

    def log_callback(self, timestamp, data, logconf) -> None:
        del logconf
        segment = self.active_segment
        if segment is None:
            return

        state_x = float(data["stateEstimate.x"])
        state_y = float(data["stateEstimate.y"])
        state_z = float(data["stateEstimate.z"])
        state_roll_deg = float(data["stateEstimate.roll"])
        state_pitch_deg = float(data["stateEstimate.pitch"])
        state_yaw_deg = float(data["stateEstimate.yaw"])

        setpoint_x = float(segment["setpoint_x"])
        setpoint_y = float(segment["setpoint_y"])
        setpoint_z = float(segment["setpoint_z"])
        setpoint_yaw_deg = float(segment["setpoint_yaw_deg"])

        self.rows.append(
            {
                "timestamp_ms": float(timestamp),
                "host_time_s": time.time(),
                "segment_index": int(segment["index"]),
                "segment_elapsed_s": time.time() - float(segment["start_host_time_s"]),
                "setpoint_x": setpoint_x,
                "setpoint_y": setpoint_y,
                "setpoint_z": setpoint_z,
                "setpoint_yaw_deg": setpoint_yaw_deg,
                "state_x": state_x,
                "state_y": state_y,
                "state_z": state_z,
                "state_roll_deg": state_roll_deg,
                "state_pitch_deg": state_pitch_deg,
                "state_yaw_deg": state_yaw_deg,
                "error_x": state_x - setpoint_x,
                "error_y": state_y - setpoint_y,
                "error_z": state_z - setpoint_z,
                "error_roll_rad": deg_to_rad(state_roll_deg),
                "error_pitch_rad": deg_to_rad(state_pitch_deg),
                "error_yaw_rad": wrap_rad(deg_to_rad(wrap_deg(state_yaw_deg - setpoint_yaw_deg))),
                "oot_error_x": float(data.get("oot.pos_x", np.nan)),
                "oot_error_y": float(data.get("oot.pos_y", np.nan)),
                "oot_error_z": float(data.get("oot.pos_z", np.nan)),
                "oot_error_roll": float(data.get("oot.phi_x", np.nan)),
                "oot_error_pitch": float(data.get("oot.phi_y", np.nan)),
                "oot_error_yaw": float(data.get("oot.phi_z", np.nan)),
            }
        )

    def save(self, pickle_path: pathlib.Path, npz_path: pathlib.Path) -> None:
        payload = {
            "rows": self.rows,
            "segments": self.segments,
            "state_labels": STATE_LABELS,
            "setpoint_labels": SETPOINT_LABELS,
            "notes": (
                "error_* columns are state minus the active segment setpoint; "
                "roll and pitch setpoints are zero, yaw setpoint is in setpoint_yaw_deg."
            ),
        }

        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with pickle_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Saved {len(self.rows)} error samples to {pickle_path}")

        if not self.rows:
            print("No error samples received; skipping NPZ export.")
            return

        rows = self.rows
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            timestamps_ms=np.asarray([row["timestamp_ms"] for row in rows], dtype=np.float64),
            host_time_s=np.asarray([row["host_time_s"] for row in rows], dtype=np.float64),
            segment_index=np.asarray([row["segment_index"] for row in rows], dtype=np.int32),
            segment_elapsed_s=np.asarray([row["segment_elapsed_s"] for row in rows], dtype=np.float64),
            setpoints=np.asarray(
                [[row["setpoint_x"], row["setpoint_y"], row["setpoint_z"], row["setpoint_yaw_deg"]] for row in rows],
                dtype=np.float64,
            ),
            states=np.asarray(
                [
                    [
                        row["state_x"],
                        row["state_y"],
                        row["state_z"],
                        deg_to_rad(float(row["state_roll_deg"])),
                        deg_to_rad(float(row["state_pitch_deg"])),
                        deg_to_rad(float(row["state_yaw_deg"])),
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            ),
            errors=np.asarray(
                [
                    [
                        row["error_x"],
                        row["error_y"],
                        row["error_z"],
                        row["error_roll_rad"],
                        row["error_pitch_rad"],
                        row["error_yaw_rad"],
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            ),
            oot_errors=np.asarray(
                [
                    [
                        row["oot_error_x"],
                        row["oot_error_y"],
                        row["oot_error_z"],
                        row["oot_error_roll"],
                        row["oot_error_pitch"],
                        row["oot_error_yaw"],
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            ),
            segment_setpoints=np.asarray(
                [
                    [s["setpoint_x"], s["setpoint_y"], s["setpoint_z"], s["setpoint_yaw_deg"]]
                    for s in self.segments
                ],
                dtype=np.float64,
            ),
            segment_deltas=np.asarray(
                [[s["delta_x"], s["delta_y"], s["delta_z"], s["delta_yaw_deg"]] for s in self.segments],
                dtype=np.float64,
            ),
            bounds=np.asarray(
                [
                    [self.x_bounds[0], self.x_bounds[1]],
                    [self.y_bounds[0], self.y_bounds[1]],
                    [self.z_bounds[0], self.z_bounds[1]],
                ],
                dtype=np.float64,
            ),
            error_labels=np.asarray(STATE_LABELS),
            setpoint_labels=np.asarray(SETPOINT_LABELS),
        )
        print(f"Saved plotting arrays to {npz_path}")


class MocapWrapper(Thread):
    def __init__(self, *, system_type: str, host_name: str, body_name: str):
        super().__init__(daemon=True)
        self.system_type = system_type
        self.host_name = host_name
        self.body_name = body_name
        self.on_pose = None
        self._stay_open = True
        self.start()

    def close(self) -> None:
        self._stay_open = False

    def run(self) -> None:
        mc = motioncapture.connect(self.system_type, {"hostname": self.host_name})
        while self._stay_open:
            mc.waitForNextFrame()
            body = mc.rigidBodies.get(self.body_name)
            if body is not None and self.on_pose is not None:
                pos = body.position
                self.on_pose(pos[0], pos[1], pos[2], body.rotation)


def send_extpose(cf: Crazyflie, send_full_pose: bool, x: float, y: float, z: float, quat) -> None:
    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, quat.x, quat.y, quat.z, quat.w)
    else:
        cf.extpos.send_extpos(x, y, z)


def start_emergency_stop_watchdog(cf: Crazyflie, interval_s: float = 0.1):
    stop_event = Event()

    def watchdog_loop() -> None:
        while not stop_event.is_set():
            try:
                cf.loc.send_emergency_stop_watchdog()
            except Exception:
                pass
            time.sleep(interval_s)

    thread = Thread(target=watchdog_loop, daemon=True)
    thread.start()
    return stop_event, thread


def set_controller_rates(cf: Crazyflie, *, nn_hz: int, controller_hz: int) -> None:
    # The current AMPC firmware evaluates the policy directly in controllerOutOfTree.
    del nn_hz
    cf.param.set_value("ctrl.rate_hz", str(int(controller_hz)))


def set_sd_logging(cf: Crazyflie, enabled: bool) -> None:
    cf.param.set_value("usd.logging", "1" if enabled else "0")


def console_callback(text: str) -> None:
    print(text, end="")


def send_position_for(
    cf: Crazyflie,
    setpoint: tuple[float, float, float, float],
    duration_s: float,
    rate_hz: float,
) -> None:
    dt = 1.0 / rate_hz
    end = time.time() + duration_s
    while time.time() < end:
        cf.commander.send_position_setpoint(*setpoint)
        time.sleep(dt)


def make_log_config(args: argparse.Namespace, recorder: SegmentRecorder) -> LogConfig:
    logconf = LogConfig(name="AMPCRandomErrors", period_in_ms=args.log_period_ms)
    for name in (
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stateEstimate.roll",
        "stateEstimate.pitch",
        "stateEstimate.yaw",
        "oot.pos_x",
        "oot.pos_y",
        "oot.pos_z",
        "oot.phi_x",
        "oot.phi_y",
        "oot.phi_z",
    ):
        logconf.add_variable(name, "FP16")
    logconf.data_received_cb.add_callback(recorder.log_callback)
    return logconf


def sample_training_delta(
    rng: np.random.Generator,
    args: argparse.Namespace,
    previous: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    for _ in range(args.max_resample_attempts):
        dx, dy = rng.uniform(-args.xy_delta_max, args.xy_delta_max, size=2)
        dz = rng.uniform(-args.z_delta_max, args.z_delta_max)
        dyaw = 0.0
        candidate = (
            previous[0] + float(dx),
            previous[1] + float(dy),
            previous[2] + float(dz),
            wrap_deg(previous[3] + float(dyaw)),
        )
        if (
            args.x_min <= candidate[0] <= args.x_max
            and args.y_min <= candidate[1] <= args.y_max
            and args.z_min <= candidate[2] <= args.z_max
        ):
            return candidate, (float(dx), float(dy), float(dz), float(dyaw))

    raise RuntimeError(
        "Could not sample an in-bounds setpoint after "
        f"{args.max_resample_attempts} attempts from previous={previous}. "
        "Increase --max-resample-attempts or reduce the setpoint delta ranges."
    )


def validate_setpoint_bounds(
    setpoint: tuple[float, float, float, float],
    args: argparse.Namespace,
    name: str,
) -> None:
    if not (args.x_min <= setpoint[0] <= args.x_max):
        raise ValueError(f"{name} x={setpoint[0]} is outside [{args.x_min}, {args.x_max}]")
    if not (args.y_min <= setpoint[1] <= args.y_max):
        raise ValueError(f"{name} y={setpoint[1]} is outside [{args.y_min}, {args.y_max}]")
    if not (args.z_min <= setpoint[2] <= args.z_max):
        raise ValueError(f"{name} z={setpoint[2]} is outside [{args.z_min}, {args.z_max}]")


def run_experiment(cf: Crazyflie, args: argparse.Namespace, recorder: SegmentRecorder, logconf: LogConfig) -> None:
    rng = np.random.default_rng(args.seed)

    cf.param.set_value("locSrv.extQuatStdDev", str(args.orientation_std_dev))
    cf.param.set_value("stabilizer.estimator", "2")
    cf.param.set_value("locSrv.extQuatStdDev", "0.06")
    cf.param.set_value("stabilizer.controller", "1")
    reset_estimator(cf)

    cf.platform.send_arming_request(True)
    time.sleep(1.0)
    logconf.start()

    current = (args.x, args.y, args.z, args.yaw)
    validate_setpoint_bounds(current, args, "initial setpoint")
    cf.high_level_commander.takeoff(args.z, args.takeoff_s)
    time.sleep(args.takeoff_s + 1.0)
    send_position_for(cf, current, args.pid_settle_s, args.setpoint_rate_hz)

    time.sleep(0.5)
    set_controller_rates(cf, nn_hz=args.nn_hz, controller_hz=args.controller_hz)
    cf.param.set_value("stabilizer.controller", "6")
    send_position_for(cf, current, args.pre_random_hover_s, args.setpoint_rate_hz)

    if args.sd_log:
        print("Starting SD card logging...")
        set_sd_logging(cf, True)

    try:
        for index in range(args.num_segments):
            next_setpoint, delta = sample_training_delta(rng, args, current)
            validate_setpoint_bounds(next_setpoint, args, f"random segment {index}")
            recorder.start_segment(index=index, setpoint=next_setpoint, delta=delta)
            print(
                f"segment {index:03d}: setpoint={next_setpoint}, "
                f"delta={delta}, duration={args.segment_s:.2f}s"
            )
            send_position_for(cf, next_setpoint, args.segment_s, args.setpoint_rate_hz)
            recorder.stop_segment()
            current = next_setpoint
    finally:
        recorder.stop_segment()
        if args.sd_log:
            print("Stopping SD card logging...")
            set_sd_logging(cf, False)

    print(f"segment return: setpoint={(args.x, args.y, args.z, args.yaw)}, duration={args.return_s:.2f}s")
    send_position_for(cf, (args.x, args.y, args.z, args.yaw), args.return_s, args.setpoint_rate_hz)

    cf.param.set_value("stabilizer.controller", "1")
    cf.commander.send_stop_setpoint()
    cf.commander.send_notify_setpoint_stop()
    cf.high_level_commander.land(0.0, args.land_s)
    time.sleep(args.land_s)
    cf.high_level_commander.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=uri_helper.uri_from_env(default="radio://0/0/2M/E7E7E7E7E7"))
    parser.add_argument("--mocap-host", default="134.130.192.68")
    parser.add_argument("--mocap-system", default="vicon")
    parser.add_argument("--rigid-body", default="cf_hsh_psd")
    parser.add_argument("--position-only", action="store_true", help="Send mocap position without attitude quaternion.")
    parser.add_argument("--orientation-std-dev", type=float, default=8.0e-3)
    parser.add_argument("--x", type=float, default=-2.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw setpoint in degrees.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-segments", type=int, default=30)
    parser.add_argument("--segment-s", type=float, default=2.0)
    parser.add_argument("--xy-delta-max", type=float, default=0.8)
    parser.add_argument("--z-delta-max", type=float, default=0.5)
    parser.add_argument("--x-min", type=float, default=-3.0)
    parser.add_argument("--x-max", type=float, default=3.0)
    parser.add_argument("--y-min", type=float, default=-1.5)
    parser.add_argument("--y-max", type=float, default=1.5)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=1.5)
    parser.add_argument("--max-resample-attempts", type=int, default=100)
    parser.add_argument("--nn-hz", type=int, default=100)
    parser.add_argument("--controller-hz", type=int, default=500)
    parser.add_argument("--setpoint-rate-hz", type=float, default=10.0)
    parser.add_argument("--log-period-ms", type=int, default=10)
    parser.add_argument("--log-pickle", type=pathlib.Path, default=pathlib.Path("random_setpoint_errors.pkl"))
    parser.add_argument("--log-npz", type=pathlib.Path, default=pathlib.Path("random_setpoint_errors.npz"))
    parser.add_argument("--sd-log", action="store_true", help="Also enable SD-card logging during random segments.")
    parser.add_argument("--takeoff-s", type=float, default=2.0)
    parser.add_argument("--pid-settle-s", type=float, default=5.0)
    parser.add_argument("--pre-random-hover-s", type=float, default=5.0)
    parser.add_argument("--return-s", type=float, default=3.0)
    parser.add_argument("--land-s", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cflib.crtp.init_drivers()

    recorder = SegmentRecorder(
        x_bounds=(args.x_min, args.x_max),
        y_bounds=(args.y_min, args.y_max),
        z_bounds=(args.z_min, args.z_max),
    )
    mocap = MocapWrapper(
        system_type=args.mocap_system,
        host_name=args.mocap_host,
        body_name=args.rigid_body,
    )
    try:
        cf = Crazyflie(rw_cache="./cache")
        cf.console.receivedChar.add_callback(console_callback)
        with SyncCrazyflie(args.uri, cf=cf) as scf:
            cf = scf.cf
            mocap.on_pose = lambda x, y, z, quat: send_extpose(cf, not args.position_only, x, y, z, quat)
            logconf = make_log_config(args, recorder)
            cf.log.add_config(logconf)
            watchdog_stop, _ = start_emergency_stop_watchdog(cf)
            try:
                run_experiment(cf, args, recorder, logconf)
            except KeyboardInterrupt:
                try:
                    print("SENDING EMERGENCY STOP... ", end="")
                    set_sd_logging(cf, False)
                    cf.loc.send_emergency_stop()
                    print("DONE.")
                except Exception:
                    pass
            finally:
                watchdog_stop.set()
                set_sd_logging(cf, False)
                try:
                    logconf.stop()
                except Exception:
                    pass
                recorder.save(args.log_pickle, args.log_npz)
    finally:
        mocap.close()


if __name__ == "__main__":
    main()
