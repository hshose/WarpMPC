"""Exercise the actual session supervisor with synthetic, in-process transports.

No SDK, sockets, subprocesses or robot connection. Only transport and clock are
replaced: Session.start/pump/move/collect and calibrate/fixture run unchanged.
Native SessionControl/JointVelocityTracking execute on the offline 0.5 ms plant.
Simulation pauses for Python, observer and MPC computation; this is not a
realtime, libfranka callback, Newton or physical contact validation.
"""
import argparse
from collections import deque
import ctypes as ct
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from arm.model import HOME
from cpp.python_controller import NativeController, DOUBLE
from . import calibration as cal
from . import session as live
from .fixture import ReleaseGate


class SyntheticSession(live.Session):
    calibration_source_kind = 'synthetic'

    def load_calibration(self):
        fixed = cal.load(self.args.calibration, require_live=False)
        if fixed['source_kind'] != 'synthetic':
            raise ValueError('Offline check requires a synthetic calibration')
        return fixed


class Stream:
    def __init__(self, env, name):
        self.env = env; self.name = name; self.queue = deque(); self.closed = False
        self.process = SimpleNamespace(returncode=None)

    def drain(self):
        if self.closed: raise RuntimeError('Synthetic stream closed')
        items = list(self.queue); self.queue.clear(); return items

    def send(self, text):
        if self.closed: raise RuntimeError('Synthetic stream closed')
        words = text.split(); env = self.env
        env.commands[words[0]] = env.commands.get(words[0], 0)+1
        if words[0] == 'heartbeat':
            env.heartbeat = env.t
        elif words[0] == 'move':
            q = np.ascontiguousarray(list(map(float, words[3:])))
            env.call(env.lib.fixture_plant_move, int(words[1]), float(words[2])-env.epoch, q)
            if env.fault == 'early_release' and int(words[1]) == 2:
                env.release_at(env.t+.2)
        elif words[0] == 'plan':
            trajectory = np.ascontiguousarray(list(map(float, words[2:])))
            env.call(env.lib.fixture_plant_plan, float(words[1])-env.epoch, trajectory)
            if env.first_plan is None: env.first_plan = env.t
        elif words[0] != 'stop':
            raise ValueError('Unexpected supervisor command')

    def close(self):
        self.closed = True; self.process.returncode = 0


class Environment:
    epoch = 1000.
    dt = .0005

    def __init__(self, args, robot, X, Y, fault=None):
        self.args = args; self.model = robot; self.X = X; self.Y = Y; self.fault = fault
        self.rng = np.random.default_rng(481); self.tick = 0; self.t = 0.
        self.robot = None; self.vicon = None; self.pending = deque(); self.frame = 0
        self.heartbeat = None; self.max_heartbeat_gap = 0.; self.commands = {}
        self.release = None; self.first_plan = None; self.trace = []; self.injected = False
        self.injected_frames = dict(dropout=0, outlier=0, ambiguous=0)
        self.F_T_EE = cal.pose([.02, -.01, .07], Rotation.from_euler('xyz', [.1, -.2, .05]).as_quat())
        self.err = ct.create_string_buffer(4096); errors = [ct.c_char_p, ct.c_size_t]; ptr = ct.c_void_p
        self.lib = ct.CDLL(str(args.plant.resolve()))
        self.lib.fixture_plant_create.argtypes = [ct.c_char_p, DOUBLE, DOUBLE, ct.c_double, *errors]
        self.lib.fixture_plant_create.restype = ptr
        self.lib.fixture_plant_destroy.argtypes = [ptr]
        self.lib.fixture_plant_step.argtypes = [ptr, DOUBLE, DOUBLE, ct.POINTER(ct.c_int), *errors]
        self.lib.fixture_plant_plan.argtypes = [ptr, ct.c_double, DOUBLE, *errors]
        self.lib.fixture_plant_move.argtypes = [ptr, ct.c_int, ct.c_double, DOUBLE, *errors]
        self.lib.fixture_plant_status.argtypes = [ptr, *([ct.POINTER(ct.c_int)]*3)]
        self.lib.fixture_plant_release_at.argtypes = [ptr, ct.c_double]
        ball = np.array(getattr(args, 'fixture_ball', [.04, -.03, .67])); self.x = np.r_[ball, np.zeros(3), HOME, np.zeros(10)]
        self.acc = np.zeros(7); self.impacts = ct.c_int()
        self.handle = self.lib.fixture_plant_create(str(args.factors.resolve()).encode(), HOME, ball, 1e6, self.err, len(self.err))
        if not self.handle: raise RuntimeError(self.err.value.decode())

    def call(self, function, *args):
        if function(self.handle, *args, self.err, len(self.err)):
            raise RuntimeError(self.err.value.decode())

    def release_at(self, t):
        self.release = t; self.lib.fixture_plant_release_at(self.handle, t)

    def monotonic(self): return self.epoch+self.t

    def process(self, command, output, name, writable=False):
        # This factory deliberately never launches a process or opens a socket.
        if name == 'vicon': self.vicon = Stream(self, name); return self.vicon
        if name == 'robot_stream':
            if self.args.mode == 'prepare': assert '--homing-only' in command
            self.robot = Stream(self, name); self.heartbeat = self.t; return self.robot
        raise ValueError('Unexpected process request')

    def sleep(self, duration):
        for _ in range(max(1, round(duration/self.dt))):
            self.call(self.lib.fixture_plant_step, self.x, self.acc, ct.byref(self.impacts))
            self.tick += 1; self.t = self.tick*self.dt
            if self.robot and not self.robot.closed:
                gap = self.t-self.heartbeat; self.max_heartbeat_gap = max(gap, self.max_heartbeat_gap)
                if gap > .15: raise RuntimeError('Synthetic native heartbeat expired')
            if self.tick % 20 == 0: self.sample()
            while self.pending and self.pending[0]['receive_time'] <= self.monotonic()+1e-10:
                item = self.pending.popleft()
                if self.vicon and not self.vicon.closed: self.vicon.queue.append(item)

    def sample(self):
        q = self.x[6:13]; dq = self.x[13:20]
        pin.framesForwardKinematics(self.model.pin, self.model.data, q)
        A = self.model.data.oMf[self.model.pin.getFrameId('panda_link8')].homogeneous.copy()
        if self.robot and not self.robot.closed:
            mid = ct.c_int(); moving = ct.c_int(); mpc = ct.c_int()
            self.lib.fixture_plant_status(self.handle, ct.byref(mid), ct.byref(moving), ct.byref(mpc))
            self.robot.queue.append(dict(receive_time=self.monotonic(), q=q.tolist(), dq=dq.tolist(),
                q_d=q.tolist(), ddq_d=self.acc.tolist(), move_id=mid.value, moving=bool(moving.value),
                mpc=bool(mpc.value), stopping=False, O_T_EE=(A@self.F_T_EE).ravel(order='F').tolist(),
                F_T_EE=self.F_T_EE.ravel(order='F').tolist()))
        C = self.Y@A@self.X
        C[:3, 3] += self.rng.normal(0, .00015, 3)
        C[:3, :3] = C[:3, :3]@Rotation.from_rotvec(self.rng.normal(0, np.deg2rad(.015), 3)).as_matrix()
        b = self.Y[:3, :3]@(self.x[:3]+self.model.origin)+self.Y[:3, 3]+self.rng.normal(0, .0008, 3)
        self.frame += 1; number = self.frame
        if self.fault == 'duplicate_frame' and self.commands.get('move', 0) and not self.injected:
            number -= 1; self.injected = True
        if self.fault == 'ball_dropout' and self.first_plan is not None and self.t-self.first_plan > .3:
            b = None; self.injected = True
        ambiguous = False
        if self.fault == 'short_gaps_outliers' and self.first_plan is not None:
            # Deterministic 50 ms gaps, 30 cm false detections, and a second
            # plausible point. Applied throughout flight, including impacts.
            phase = number % 100
            if 20 <= phase < 25:
                b = None; self.injected_frames['dropout'] += 1
            elif phase == 55:
                b += np.array([.3,0,0]); self.injected_frames['outlier'] += 1
            elif phase == 75:
                ambiguous = True; self.injected_frames['ambiguous'] += 1
        raw=[] if b is None else [dict(id=number%13, position_m=b.tolist())]
        if ambiguous: raw.append(dict(id=987,position_m=(b+np.array([.001,0,0])).tolist()))
        self.pending.append(dict(receive_time=self.monotonic()+.005, latency_s=.005, frame=number, frame_rate_hz=100,
            paddle=dict(subject='hsh_pingpong_paddle', segment='root', position_m=C[:3, 3].tolist(),
                        quaternion_xyzw=Rotation.from_matrix(C[:3, :3]).as_quat().tolist()),
            markers=[dict(subject='hsh_pingpong_ball', name='hsh_pingpong_ball', position_m=None if b is None else b.tolist())],
            unlabeled_markers=raw))
        self.trace.append(np.r_[self.t, self.x, self.impacts.value])

    def close(self):
        if self.handle: self.lib.fixture_plant_destroy(self.handle); self.handle = None


def plot_case(output, robot):
    """One-page record of startup motion, ball flight and solver calls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.load(output/'trace.npz')['data']; t = data[:, 0]
    report = json.loads((output/'summary.json').read_text())
    face = np.array([robot.numeric_fk(q)[0]-robot.origin for q in data[:, 7:14]])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(t, data[:, 3]-.02, label='Ball bottom')
    axes[0, 0].plot(t, face[:, 2], label='Paddle center')
    axes[0, 0].axhline(.75, color='k', ls='--', label='Target apex')
    axes[0, 0].set(ylabel='Height above nominal strike plane [m]')
    for j, coordinate in enumerate(['x', 'y']):
        axes[0, 1].plot(t, 100*data[:, 1+j], label='Ball '+coordinate)
        axes[0, 1].plot(t, 100*face[:, j], ls='--', label='Paddle '+coordinate)
    axes[0, 1].set(ylabel='Lateral position from nominal HOME [cm]')
    axes[1, 0].plot(t, data[:, 7:14]-HOME)
    axes[1, 0].set(ylabel='Joints relative to nominal HOME [rad]')
    path = output/'mpc.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if rows:
        axes[1, 1].plot([r['time']-Environment.epoch for r in rows], [r['metrics'][0] for r in rows])
        axes[1, 1].set(ylabel='Native MPC computation [ms]')
    else:
        axes[1, 1].plot(t, data[:, 14:21])
        axes[1, 1].set(ylabel='Measured joint velocities [rad/s]')
    for ax in axes.flat:
        ax.set(xlabel='Simulated time [s]'); ax.grid(alpha=.25)
        if report['release_time_s'] is not None: ax.axvline(report['release_time_s'], color='grey', ls=':')
    axes[0, 0].legend(); axes[0, 1].legend(ncol=2)
    outcome = report['error'] or f"{report['impacts']} impacts; completed"
    fig.suptitle(f'OFFLINE actual supervisor: {output.name}\n{outcome}; computation pauses simulation')
    fig.savefig(output/'report.pdf'); plt.close(fig)


def exercise(args, robot, meta, X, Y, mode, name, fault=None):
    options = SimpleNamespace(**vars(args)); options.output = args.output/name; options.mode = mode
    options.vicon_host = 'OFFLINE'; options.robot = 'OFFLINE'; options.allow_nonrealtime = False
    options.vicon_binary = Path('OFFLINE'); options.robot_binary = Path('OFFLINE')
    options.extra_latency = 0.; options.ball = getattr(args, 'ball', 'hsh_pingpong_ball'); options.release_timeout = 3.
    env = Environment(options, robot, X, Y, fault); session = SyntheticSession(options)
    controller = NativeController(args.library, args.factors) if mode in ('fixture', 'prepare') else None
    original_arm = ReleaseGate.arm
    def arm_gate(gate):
        original_arm(gate)
        if env.release is None: env.release_at(env.t+.25)
    description = json.dumps(dict(model_sha256=hashlib.sha256((args.factors/'model_config.hpp').read_bytes()).hexdigest()))
    error = None
    try:
        with patch.object(live, 'time', env), patch.object(live, 'JsonProcess', env.process), \
                patch.object(live.subprocess, 'check_output', return_value=description), patch.object(ReleaseGate, 'arm', arm_gate):
            session.start()
            if mode == 'calibrate': live.calibrate(session, options)
            else: live.fixture(session, options, robot, meta, controller)
    except Exception as e:
        error = str(e)
    finally:
        session.close()
        if controller is not None: controller.close()
        env.close()
    trace = np.array(env.trace)
    np.savez_compressed(options.output/'trace.npz', data=trace)
    report = dict(error=error, physical_robot_used=False, transport='synthetic in-process; actual Python supervisor',
                  simulator='native SessionControl/velocity servo; 5 ms servo lag; impulse ball; smooth wind',
                  simulation_pauses_for_computation=True, simulation_dt_s=env.dt,
                  calibration_source_kind=session.calibration_source_kind, vicon_rate_hz=100, vicon_delay_s=.005,
                  elapsed_simulation_s=env.t, commands=env.commands, impacts=env.impacts.value,
                  max_simulated_heartbeat_gap_s=env.max_heartbeat_gap,
                  release_time_s=env.release, first_plan_time_s=env.first_plan,
                  injected_frames=env.injected_frames,
                  raw_rejections=dict(session.raw_ball.rejections) if session.raw_ball else {},
                  first_command_delay_s=None if env.first_plan is None else env.first_plan-env.release)
    if (options.output/'world_alignment.json').exists():
        fitted = np.array(json.loads((options.output/'world_alignment.json').read_text())['V_T_B'])
        pos, angle = cal.errors(Y, fitted)
        report.update(world_translation_error_m=pos, world_rotation_error_deg=float(np.rad2deg(angle)))
    if (options.output/'mpc.jsonl').exists():
        rows = [json.loads(line) for line in (options.output/'mpc.jsonl').read_text().splitlines()]
        report['mpc_solves'] = len(rows)
        report['osqp_iterations'] = sorted(set(round(row['metrics'][12]) for row in rows))
        stamps = [row['time'] for row in rows]
        report['median_mpc_interval_s'] = float(np.median(np.diff(stamps))) if len(stamps)>1 else None
        report['max_measurement_age_s'] = max(row['measurement_age'] for row in rows)
        truth=np.array([np.interp(np.array(stamps)-Environment.epoch,trace[:,0],trace[:,j+1]) for j in range(6)]).T
        estimated=np.array([row['state'][:6] for row in rows])
        report['position_rmse_m']=float(np.sqrt(np.mean(np.sum((estimated[:,:3]-truth[:,:3])**2,axis=1))))
        report['max_position_error_m']=float(np.max(np.linalg.norm(estimated[:,:3]-truth[:,:3],axis=1)))
    if env.release is not None:
        apices = [float(max(a[3], b[3])-.02) for a,b in zip(trace[:-1], trace[1:])
                  if a[0] > env.release+.4 and a[6]>0>=b[6]]
        report['apices_m'] = apices
    cal.write_new(options.output/'summary.json', report)
    plot_case(options.output, robot)
    return report


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.calibration = args.output/'synthetic_hand_eye.json'
    robot, meta = live.model(args.factors)
    X = cal.pose([.007, -.155, .045], Rotation.from_euler('xyz', [.13, -.07, 1.3]).as_quat())
    Y = cal.pose([1.2, -.4, .07], Rotation.from_euler('z', .6).as_quat())
    reports = {}
    reports['calibration'] = exercise(args, robot, meta, X, Y, 'calibrate', 'calibration')
    assert reports['calibration']['error'] is None, reports['calibration']
    fixed = cal.load(args.calibration, require_live=False)
    pos, angle = cal.errors(X, np.array(fixed['F_T_P']))
    assert pos < .002 and angle < np.deg2rad(.3), (pos, angle)
    try: live.Session(args).load_calibration()
    except ValueError: pass
    else: raise AssertionError('Hardware accepted synthetic calibration')
    # A different base/world transform on restart must be inferred at HOME.
    moved = cal.pose([.4, .8, .03], Rotation.from_euler('z', -.8).as_quat())
    for fault in [None, 'early_release', 'ball_dropout', 'duplicate_frame']:
        name = fault or 'fixture'
        reports[name] = exercise(args, robot, meta, X, moved, 'fixture', name, fault)
        print(name, json.dumps(reports[name]), flush=True)
    good = reports['fixture']
    assert good['error'] is None and good['impacts'] >= int(args.seconds), good
    assert good['osqp_iterations'] == [50] and abs(good['median_mpc_interval_s']-.02)<1e-6, good
    assert good['world_translation_error_m'] < .003 and good['world_rotation_error_deg'] < .3, good
    assert .04 < good['first_command_delay_s'] < .12, good
    assert 'before' in (reports['early_release']['error'] or '') and not reports['early_release']['commands'].get('plan'), reports['early_release']
    assert 'Ball tracking lost' in (reports['ball_dropout']['error'] or ''), reports['ball_dropout']
    assert 'counter' in (reports['duplicate_frame']['error'] or '') and not reports['duplicate_frame']['commands'].get('plan'), reports['duplicate_frame']
    report = dict(passed=True, physical_robot_used=False, synthetic_calibration_rejected_by_hardware=True,
                  hand_eye_translation_error_m=pos, hand_eye_rotation_error_deg=float(np.rad2deg(angle)), cases=reports)
    cal.write_new(args.output/'checks.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--factors', type=Path, default=live.DEFAULT_FACTORS)
    p.add_argument('--library', type=Path, default=live.DEFAULT_FACTORS.parent/'controller/libjuggling_controller.so')
    p.add_argument('--plant', type=Path, default=live.DEFAULT_FACTORS.parent/'controller/libjuggling_fixture_plant.so')
    run(p.parse_args())
