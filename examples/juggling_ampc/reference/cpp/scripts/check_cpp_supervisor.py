#!/usr/bin/env python3
"""Exercise native supervisor with C++ device doubles, without hardware/network.

The real supervisor, estimator, model, controller and typed IPC run unchanged.
Only Vicon and robot executables are replaced by explicitly offline C++ helpers.
The ball is held until RELEASE BALL NOW, dropped, then deliberately occluded.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def rows(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--devices', type=Path)
    parser.add_argument('--factors', type=Path)
    parser.add_argument('--calibration', type=Path, default=project / 'config/paddle_hand_eye.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=45.)
    args = parser.parse_args()
    if args.profile_dir:
        profile = args.profile_dir.resolve()
        args.runtime = args.runtime or profile / 'build/runtime/juggling_ball_cpp'
        args.devices = args.devices or profile / 'build/runtime/juggling_mock_experiment_devices'
        args.factors = args.factors or profile / 'factors'
    if not all((args.runtime, args.devices, args.factors)):
        parser.error('Provide --profile-dir or all of --runtime, --devices and --factors')
    if not 10 <= args.timeout <= 120:
        parser.error('--timeout must be between 10 and 120 seconds')
    output = (args.output or project / 'runs' / ('cpp_supervisor_smoke_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    trial = output / 'trial'
    config = dict(factors=str(args.factors.resolve()), calibration=str(args.calibration.resolve()), output=str(trial), world=str(output / 'mock_world.ipc'))
    (output / 'mock_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'mock_timing.json').write_text(json.dumps(dict(vicon_host='mock', extra_latency_s=-.002)) + '\n')
    env = dict(os.environ, JUGGLING_MOCK_DEVICES='1', JUGGLING_MOCK_CONFIG=str(output / 'mock_config.json'), OPENBLAS_NUM_THREADS='1', JUGGLING_EVAL_THREADS='4')
    # Same lock as compilation/physical launchers: offline timing never competes
    # with compilation or a live hardware experiment.
    (project / 'build').mkdir(exist_ok=True)
    lock = (project / 'build/cpp_experiment.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A compilation or experiment is active; smoke check not started.')
    capability = json.loads(subprocess.check_output([str(args.devices.resolve()), '--describe'], env=env, text=True))
    expected = dict(mock_devices=True, robot_connected=False, network_capability=False, protocol=2)
    if any(capability.get(key) != value for key, value in expected.items()):
        raise SystemExit('Device double did not attest offline capability.')
    subprocess.run([str(args.devices.resolve()), '--init'], env=env, check=True)
    command = [str(args.runtime.resolve()), '--execute', '--factors', str(args.factors.resolve()),
               '--robot-binary', str(args.devices.resolve()), '--vicon-binary', str(args.devices.resolve()),
               '--robot', 'mock', '--vicon-host', 'mock', '--calibration', str(args.calibration.resolve()),
               '--timing-calibration', str(output / 'mock_timing.json'), '--output', str(trial),
               '--seconds', '0', '--release-timeout', '10']
    if args.profile_dir and (args.profile_dir / 'juggling_home.json').exists():
        command += ['--home-config', str((args.profile_dir / 'juggling_home.json').resolve())]
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    began = time.monotonic()
    with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except BaseException as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                raise SystemExit(f'Offline smoke timed out; inspect {output}') from error
            raise
    duration = time.monotonic() - began
    messages = (output / 'stdout.log').read_text()
    failure = json.loads((trial / 'failure.json').read_text()) if (trial / 'failure.json').exists() else {}
    native = json.loads((trial / 'robot/session.json').read_text()) if (trial / 'robot/session.json').exists() else {}
    timing = json.loads((trial / 'runtime_timing.json').read_text()) if (trial / 'runtime_timing.json').exists() else {}
    frames = rows(trial / 'vicon.jsonl') if (trial / 'vicon.jsonl').exists() else []
    plans = rows(trial / 'mpc.jsonl') if (trial / 'mpc.jsonl').exists() else []
    updates = rows(trial / 'observer_updates.jsonl') if (trial / 'observer_updates.jsonl').exists() else []
    reason = failure.get('error', '').lower()
    expected_loss = any(phrase in reason for phrase in ('ball tracking lost', 'raw marker missing', 'raw ball', 'ball observer measurements stale'))
    checks = dict(
        robot_connected=False,
        native_devices_only=True,
        selected_ball='BALL LOCATED' in messages,
        ready_before_release='RELEASE BALL NOW' in messages,
        performed_fixture_move=native.get('moves_received', 0) >= 1,
        detected_release=(trial / 'release.json').exists(),
        controller_sent_plan=bool(plans) and native.get('plans_received', 0) > 0,
        observed_positions=any(update.get('accepted') for update in updates),
        expected_ball_loss_abort=code != 0 and expected_loss,
        clean_robot_stop=native.get('fault') == 0 and native.get('control_completed') is True and native.get('stop_seen') == 1,
        no_acquisition_error=timing.get('reader_error') == '',
        no_recording_error=timing.get('writer_error') == '',
        archived_post_loss_frames=bool(frames) and not frames[-1].get('unlabeled_markers'),
        retained_stop_tail=bool(frames and plans) and frames[-1]['receive_time'] - plans[-1]['time'] >= .65,
    )
    required = [value for key, value in checks.items() if key != 'robot_connected']
    report = dict(passed=all(required), checks=checks, seconds=duration, failure=failure, native=native,
                  vicon_frames=len(frames), mpc_plans=len(plans), accepted_updates=sum(bool(row.get('accepted')) for row in updates),
                  runtime_timing=timing, artifact_directory=str(output))
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
