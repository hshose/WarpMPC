#!/usr/bin/env python3
"""Run hover and setpoint-change experiments for the AMPC OOT controller."""

from __future__ import annotations

import argparse
import pathlib
import pickle
import time
from threading import Event, Thread

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator
import motioncapture


LOG_BUFFER = {
    "timestamps": [],
    "data": [],
}


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
    # cf.param.set_value("ctrl.nn_hz", str(int(nn_hz)))
    cf.param.set_value("ctrl.rate_hz", str(int(controller_hz)))


def send_position_for(cf: Crazyflie, setpoint: tuple[float, float, float, float], duration_s: float, rate_hz: float) -> None:
    dt = 1.0 / rate_hz
    end = time.time() + duration_s
    while time.time() < end:
        cf.commander.send_position_setpoint(*setpoint)
        time.sleep(dt)


def set_sd_logging(cf: Crazyflie, enabled: bool) -> None:
    cf.param.set_value("usd.logging", "1" if enabled else "0")


def console_callback(text: str) -> None:
    """Forward Crazyflie console/debug output to this terminal."""

    print(text, end="")


def log_callback(timestamp, data, logconf) -> None:
    LOG_BUFFER["timestamps"].append(timestamp)
    LOG_BUFFER["data"].append(data)


def make_log_config(args: argparse.Namespace) -> LogConfig:
    logconf = LogConfig(name="AMPController", period_in_ms=args.log_period_ms)

    logconf.add_variable("stateEstimate.vx", "FP16")
    logconf.add_variable("stateEstimate.vy", "FP16")
    logconf.add_variable("stateEstimate.vz", "FP16")
    logconf.add_variable("gyro.x", "FP16")
    logconf.add_variable("gyro.y", "FP16")
    logconf.add_variable("gyro.z", "FP16")
    logconf.add_variable("oot.pos_x", "FP16")
    logconf.add_variable("oot.pos_y", "FP16")
    logconf.add_variable("oot.pos_z", "FP16")
    logconf.add_variable("oot.phi_x", "FP16")
    logconf.add_variable("oot.phi_y", "FP16")
    logconf.add_variable("oot.phi_z", "FP16")
    logconf.data_received_cb.add_callback(log_callback)
    return logconf


def save_log_buffer(path: pathlib.Path) -> None:
    if not LOG_BUFFER["timestamps"]:
        print("No radio log entries received.")
        return

    log_data = {
        "timestamps": np.asarray(LOG_BUFFER["timestamps"]),
        "data": LOG_BUFFER["data"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(log_data, f)
    print(f"Saved {len(LOG_BUFFER['timestamps'])} radio log entries to {path}")


def run_experiment(cf: Crazyflie, args: argparse.Namespace, logconf: LogConfig | None = None) -> None:
    cf.param.set_value("locSrv.extQuatStdDev", str(args.orientation_std_dev))
    cf.param.set_value("stabilizer.estimator", "2")
    cf.param.set_value("locSrv.extQuatStdDev", "0.06")
    cf.param.set_value("stabilizer.controller", "1")
    reset_estimator(cf)

    cf.platform.send_arming_request(True)
    time.sleep(1.0)
    if logconf is not None:
        logconf.start()

    start = (args.x, args.y, args.z, args.yaw)
    cf.high_level_commander.takeoff(args.z, args.takeoff_s)
    time.sleep(args.takeoff_s + 1.0)
    send_position_for(cf, start, args.pid_settle_s, args.setpoint_rate_hz)

    time.sleep(0.5)
    set_controller_rates(cf, nn_hz=args.nn_hz, controller_hz=args.controller_hz)
    cf.param.set_value("stabilizer.controller", "6")
    send_position_for(cf, start, args.pre_step_hover_s, args.setpoint_rate_hz)

    print("Starting SD card logging...")
    set_sd_logging(cf, True)

    setpoint_x = (start[0] + args.dx, start[1], start[2], start[3])
    print(f"segment x_step: setpoint={setpoint_x}, duration={args.step_s:.2f}s")
    send_position_for(cf, setpoint_x, args.step_s, args.setpoint_rate_hz)

    print("Stopping SD card logging...")
    set_sd_logging(cf, False)
    time.sleep(0.1)

    cf.param.set_value("stabilizer.controller", "1")
    cf.commander.send_stop_setpoint()
    cf.commander.send_notify_setpoint_stop()
    cf.high_level_commander.land(0.0, args.land_s)
    time.sleep(args.land_s)
    cf.high_level_commander.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--uri", default=uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7"))
    parser.add_argument("--uri", default=uri_helper.uri_from_env(default="radio://0/0/2M/E7E7E7E7E7"))
    parser.add_argument("--mocap-host", default="134.130.192.68")
    parser.add_argument("--mocap-system", default="vicon")
    parser.add_argument("--rigid-body", default="cf_hsh_psd")
    parser.add_argument("--position-only", action="store_true", help="Send mocap position without attitude quaternion.")
    parser.add_argument("--orientation-std-dev", type=float, default=8.0e-3)
    parser.add_argument("--x", type=float, default=-2.6)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw setpoint in degrees.")
    parser.add_argument("--dx", type=float, default=0.6)
    parser.add_argument("--dy", type=float, default=0.5)
    parser.add_argument("--dz", type=float, default=0.2)
    parser.add_argument("--yaw-z-offset", type=float, default=0.5)
    parser.add_argument("--dyaw", type=float, default=30.0)
    parser.add_argument("--nn-hz", type=int, default=100)
    parser.add_argument("--controller-hz", type=int, default=500)
    parser.add_argument("--setpoint-rate-hz", type=float, default=10.0)
    parser.add_argument("--log-period-ms", type=int, default=10)
    parser.add_argument("--log-pickle", type=pathlib.Path, default=pathlib.Path("log_data.pkl"))
    parser.add_argument("--no-radio-log", action="store_true", help="Disable cflib radio logging to a pickle file.")
    parser.add_argument("--takeoff-s", type=float, default=2.0)
    parser.add_argument("--pid-settle-s", type=float, default=5.0)
    parser.add_argument("--pre-step-hover-s", type=float, default=5.0)
    parser.add_argument("--step-s", type=float, default=3.0)
    parser.add_argument("--return-s", type=float, default=3.0)
    parser.add_argument("--land-s", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cflib.crtp.init_drivers()

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
            logconf = None
            if not args.no_radio_log:
                logconf = make_log_config(args)
                cf.log.add_config(logconf)
            watchdog_stop, _ = start_emergency_stop_watchdog(cf)
            try:
                run_experiment(cf, args, logconf)
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
                if logconf is not None:
                    logconf.stop()
                    save_log_buffer(args.log_pickle)
    finally:
        mocap.close()


if __name__ == "__main__":
    main()
