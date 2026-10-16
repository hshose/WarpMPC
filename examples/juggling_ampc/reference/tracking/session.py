"""Vicon calibration and release-fixture supervisor, explicit --execute only.

Run this process, its Vicon reader, native controller and libfranka bridge on
the robot workstation. Network timestamps from different hosts are not mixed.
"""
import argparse
from collections import deque
from dataclasses import asdict
import json
import hashlib
import shutil
from pathlib import Path
import subprocess
import time
import numpy as np

from arm.model import HOME, Panda
from estimation.ball import BallObserver, ObserverConfig
from cpp.python_controller import NativeController
from . import calibration as cal
from .fixture import ReleaseGate, fixture_home, predicted_landing
from .streams import JsonProcess, ball_position, paddle_pose, capture_time
from .raw_ball import RawBall, MAX_BALL_GAP
from .reach import within_landing_reach, nominal_return_fraction

# Updated physics/tuning; Newton validated. Offline hardware-servo mock still
# trips its reference-acceleration guard: see runs/20261002_model_update/report.txt.
DEFAULT_FACTORS = Path('build/variants/mesh24_identified_servo/factors')


def calibration_waypoints():
    points = [HOME.copy()]
    for j, delta in [(4, .18), (5, .18), (6, .22), (1, .10), (3, .10)]:
        for sign in [-1, 1]:
            q = HOME.copy(); q[j] += sign*delta; points.append(q)
    for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        q = HOME.copy(); q[4] += a*.14; q[5] += b*.14; points.append(q)
    points.append(HOME.copy())
    return points


def model(factors):
    meta = json.loads((Path(factors)/'metadata.json').read_text()); spec = meta['spec']
    robot = Panda(assembly=spec.get('assembly'), **spec.get('geometry', {}))
    if not np.allclose(robot.origin, meta['arrays']['origin'], atol=1e-10):
        raise ValueError('Robot geometry differs from compiled factors')
    return robot, meta


class Session:
    calibration_source_kind = 'live_vicon_robot'

    def __init__(self, args):
        self.args = args; self.robot = None; self.vicon = None; self.arm = None
        self.arms = deque(maxlen=200); self.latest_frame = None; self.last_paddle = None
        self.move_id = 0; self.on_frame = None; self.observer = None; self.mpc_started = False
        self.paddle_segment = None
        self.postroll_done = False
        self.output_created = False
        self.output = args.output
        self.raw_ball = None
        self.virtual_ball = None
        self.ball_frame = None
        self.pending_frames = deque()
        self.frame_sync = dict(buffered=0, processed=0, max_depth=0, max_wait_s=0.)

    def load_calibration(self):
        # Hardware sessions always require measured acquisition provenance.
        return cal.load(self.args.calibration)

    def ball_position(self, frame):
        if self.raw_ball is not None:
            return frame.get('_selected_ball_position')
        return ball_position(frame, self.args.ball)

    def raw_ball_position(self, frame):
        prediction = None
        if self.observer is not None:
            t = capture_time(frame, self.args.extra_latency)
            # The position observer already models gravity and paddle impacts.
            # Use its predicted uncertainty for raw association in Vicon axes.
            if t < self.observer.t or t > self.observer.history[-1].t:
                return None  # No causal prediction; do not invent a measurement.
            x, P, info = self.observer.estimate(t)
            B_T_V, origin = self.ball_frame; R = B_T_V[:3, :3].T
            mean = np.r_[R@(x[:3]+origin-B_T_V[:3, 3]), R@x[3:6]]
            rotation = np.kron(np.eye(2), R)
            std = (self.observer.cfg.contact_measurement_std
                   if abs(t-info['last_impact']) < self.observer.cfg.contact_measurement_window
                   else self.observer.cfg.position_std)
            prediction = (mean, rotation@P[:6, :6]@rotation.T, std)
        return self.raw_ball.update(frame, prediction=prediction)

    def start(self):
        description = json.loads(subprocess.check_output([str(self.args.robot_binary), '--describe'], text=True, timeout=3))
        if description.get('protocol')!=2:
            raise RuntimeError('Rebuild robot session: endpoint-acceleration protocol 2 required')
        expected = hashlib.sha256((self.args.factors/'model_config.hpp').read_bytes()).hexdigest()
        if description.get('model_sha256') != expected:
            raise RuntimeError('Robot session was built against a different MPC model_config.hpp')
        self.output.mkdir(parents=True, exist_ok=False)
        self.output_created = True
        cal.write_new(self.output/'native_description.json',description)
        # Freeze the exact extrinsic/model used by this trajectory for fitting.
        if self.args.calibration.exists(): shutil.copy2(self.args.calibration,self.output/'calibration_used.json')
        timing_path=getattr(self.args, 'timing_calibration', None)
        if timing_path is not None and timing_path.exists(): shutil.copy2(timing_path,self.output/'timing_calibration_used.json')
        for name in ['gravity_calibration','paddle_normal_calibration']:
            calibration=self.args.factors.parent/(name+'.json')
            if calibration.exists():shutil.copy2(calibration,self.output/(name+'_used.json'))
        shutil.copy2(self.args.factors/'metadata.json',self.output/'model_metadata.json')
        shutil.copy2(self.args.factors/'model_config.hpp',self.output/'model_config.hpp')
        cal.write_new(self.output/'clock.json',dict(monotonic=time.monotonic(),unix=__import__('time').time(),
            convention='same-host receive_monotonic minus per-frame SDK latency (cadence fallback); signed residual correction',
            vicon_extra_latency_s=self.args.extra_latency))
        needs_ball = self.args.mode in ('fixture', 'prepare', 'drop', 'rehearse')
        if needs_ball and self.args.ball == 'auto':
            fixed = self.load_calibration()
            # A bounded stop can leave the paddle away from the held fixture.
            # This is a marker search radius, not a robot workspace expansion:
            # fixture_home still checks the ball against the nominal base pose.
            self.raw_ball = RawBall(fixed['P_T_CAD_face'], lateral=.5, extra_latency=self.args.extra_latency)
        self.vicon = JsonProcess([self.args.vicon_binary, self.args.vicon_host], self.output, 'vicon')
        deadline = time.monotonic()+10; visible = 0
        preflight=dict(fresh_frames=0,paddle_visible_frames=0,frames_with_unlabeled_points=0,
            marker_search_radius_m=.5 if self.raw_ball is not None else None)
        while visible < 10:
            for frame in self.vicon.drain():
                if time.monotonic()-frame['receive_time'] > .1: continue
                preflight['fresh_frames']+=1
                preflight['paddle_visible_frames']+=int(paddle_pose(frame) is not None)
                preflight['frames_with_unlabeled_points']+=int(bool(frame.get('unlabeled_markers')))
                if self.raw_ball is not None: frame['_selected_ball_position'] = self.raw_ball_position(frame)
                if paddle_pose(frame) is not None and (not needs_ball or self.ball_position(frame) is not None):
                    visible += 1; self.latest_frame = frame; self.last_paddle = frame['receive_time']
                    self.paddle_segment = frame['paddle']['segment']
            if time.monotonic() > deadline:
                cal.write_new(self.output/'preflight_tracking.json',dict(preflight,acquired=False))
                raise RuntimeError('Stable Vicon paddle/ball acquisition failed; inspect preflight_tracking.json; no robot connection made')
            time.sleep(.002)
        cal.write_new(self.output/'preflight_tracking.json',dict(preflight,acquired=True))
        if self.raw_ball is not None:
            cal.write_new(self.output/'ball_selection.json', self.raw_ball.selection)
            print('BALL LOCATED: stable Vicon point acquired. Keep holding the ball.',flush=True)
        command = [self.args.robot_binary, '--execute', '--robot', self.args.robot, '--output', self.output/'robot', '--seconds', '600']
        if getattr(self.args,'require_realtime_priority',False):command.append('--require-realtime-priority')
        if getattr(self.args,'compact_robot_readback',False):command.append('--compact-readback')
        if getattr(self.args,'robot_control_cpu',None) is not None:command+=['--control-cpu',str(self.args.robot_control_cpu)]
        if self.args.mode not in ('fixture','rehearse'): command.append('--homing-only')
        if self.args.mode in ('fixture','rehearse'):command += ['--max-plan-delivery',str(getattr(self.args,'rehearsal_delivery_budget',.015))]
        if self.args.allow_nonrealtime: command.append('--allow-nonrealtime')
        self.robot = JsonProcess(command, self.output, 'robot_stream', writable=True)
        deadline = time.monotonic()+10
        while self.arm is None:
            self.pump(starting=True)
            if time.monotonic() > deadline: raise RuntimeError('No robot readback')

    def pump(self, starting=False):
        fresh_arm = False
        if self.robot and (not starting or self.arm is not None): self.robot.send('heartbeat')
        for arm in self.robot.drain() if self.robot else []:
            if self.arms and arm['receive_time'] <= self.arms[-1]['receive_time']:
                raise RuntimeError('Robot measurement time is not increasing')
            if arm['stopping']:
                raise RuntimeError(f"Native session stopped (fault {arm.get('fault', 'unknown')}); inspect robot/session.json")
            self.arm = arm; self.arms.append(arm)
            fresh_arm = True
            if self.observer is not None:
                self.observer.record_arm(arm['receive_time'], arm['q'], arm['dq'])
            if self.virtual_ball is not None:
                self.virtual_ball.record_arm(arm['receive_time'],arm['q'],arm['dq'])
        frames = []
        for frame in self.vicon.drain():
            if self.latest_frame is not None and frame['frame'] <= self.latest_frame['frame']:
                raise RuntimeError('Vicon frame counter restarted or duplicated')
            self.latest_frame = frame
            if paddle_pose(frame) is not None:
                if frame['paddle']['segment'] != self.paddle_segment:
                    raise RuntimeError('Vicon paddle root segment changed within this session')
                self.last_paddle = frame['receive_time']
            if len(self.pending_frames)>=32:
                raise RuntimeError('Vicon/robot synchronization queue overflow')
            self.pending_frames.append(frame)
            self.frame_sync['max_depth']=max(self.frame_sync['max_depth'],len(self.pending_frames))
        while self.pending_frames:
            frame=self.pending_frames[0]
            if self.observer is not None:
                # Both capture and original arrival must be covered: the EKF
                # validates causal arm history at update_position(arrival_time).
                needed=max(capture_time(frame,self.args.extra_latency),frame['receive_time'])
                if needed>self.observer.history[-1].t:
                    if not frame.get('_waiting_for_arm'):
                        frame['_waiting_for_arm']=True;self.frame_sync['buffered']+=1
                    if time.monotonic()-frame['receive_time']>.05:
                        raise RuntimeError('Robot history unavailable for queued Vicon frame')
                    break
            self.pending_frames.popleft()
            self.frame_sync['processed']+=1
            self.frame_sync['max_wait_s']=max(self.frame_sync['max_wait_s'],time.monotonic()-frame['receive_time'])
            if self.raw_ball is not None: frame['_selected_ball_position']=self.raw_ball_position(frame)
            if self.on_frame is not None: self.on_frame(frame)
            frames.append(frame)
            # Let the fixture loop apply this measurement before associating
            # the next queued frame against the updated observer posterior.
            if self.observer is not None: break
        now = time.monotonic()
        if not starting and (self.arm is None or now-self.arm['receive_time'] > .05):
            raise RuntimeError('Robot readback stale')
        if self.latest_frame is None or now-self.latest_frame['receive_time'] > .10:
            raise RuntimeError('Vicon stream stale')
        if not self.mpc_started and (self.last_paddle is None or now-self.last_paddle > .15):
            raise RuntimeError('Paddle tracking lost during setup')
        # Do not spend a millisecond sleeping on a fresh MPC source sample.
        # Idle polling remains bounded; measured timestamps are unchanged.
        if not fresh_arm: time.sleep(.0002 if self.virtual_ball is not None else .001)
        return frames

    def move(self, target):
        self.move_id += 1
        self.robot.send('move '+str(self.move_id)+' '+str(time.monotonic())+' '+' '.join(map(str, target)))
        deadline = time.monotonic()+65; settled = None
        while True:
            self.pump(); a = self.arm
            good = (a['move_id'] == self.move_id and not a['moving'] and max(abs(np.array(a['q'])-target)) < .01
                    and max(abs(np.array(a['dq']))) < .015)
            if good:
                settled = time.monotonic() if settled is None else settled
                if time.monotonic()-settled >= .4: return
            else: settled = None
            if time.monotonic() > deadline: raise RuntimeError('Robot failed to settle at target')

    def collect(self, duration=.7):
        start = time.monotonic(); observations = []
        while time.monotonic()-start < duration:
            for frame in self.pump():
                P = paddle_pose(frame)
                if P is None: continue
                t = capture_time(frame, self.args.extra_latency)
                a = min(self.arms, key=lambda a: abs(a['receive_time']-t))
                if abs(a['receive_time']-t) > .015 or max(abs(np.array(a['dq']))) > .015:
                    continue
                observations.append(dict(B_T_F=cal.flange_from_state(a).tolist(), V_T_P=P.tolist(),
                                         capture_time=t, robot_time=a['receive_time'], frame=frame['frame'], q=a['q']))
        if len(observations) < 30: raise RuntimeError('Fewer than 30 fresh settled Vicon/robot pairs')
        A = cal.mean_pose([o['B_T_F'] for o in observations]); C = cal.mean_pose([o['V_T_P'] for o in observations])
        for o in observations:
            for mean, value in [(A, o['B_T_F']), (C, o['V_T_P'])]:
                e = cal.errors(mean, np.array(value))
                if e[0] > .002 or e[1] > np.deg2rad(.4): raise RuntimeError('Capture was not stationary')
        return observations, dict(B_T_F=A.tolist(), V_T_P=C.tolist(), frames=len(observations))

    def close(self):
        if self.robot: self.robot.close()
        if self.output_created and not (self.output/'frame_sync.json').exists():
            cal.write_new(self.output/'frame_sync.json',dict(self.frame_sync,pending_at_stop=len(self.pending_frames)))
        # Recording continues after the bounded robot stop, including on faults.
        # This is read-only Vicon capture; no heartbeat, plan or move is sent.
        if self.vicon and not self.postroll_done and getattr(self.args,'mode',None) in ('drop','rehearse','fixture'):
            self.postroll_done=True
            start=time.monotonic(); count=0; error=None
            try:
                while time.monotonic()-start < .75:
                    count+=len(self.vicon.drain()); time.sleep(.01)
            except Exception as exc: error=str(exc)
            if self.output_created:
                cal.write_new(self.output/'postroll.json',dict(duration_s=time.monotonic()-start,
                    drained_frames=count,error=error,robot_stop_requested=True,control_commands_sent=0))
        if self.vicon: self.vicon.close()
        return self.robot.process.returncode if self.robot else 0


def calibrate(session, args):
    points = calibration_waypoints(); cal.write_new(args.output/'waypoints.json', [p.tolist() for p in points])
    means = []; raw = []
    for i, q in enumerate(points):
        print(f'Calibration pose {i+1}/{len(points)}', flush=True)
        session.move(q); captures, mean = session.collect(); raw.extend(captures); means.append(mean)
        cal.write_new(args.output/f'pose_{i:02}.json', dict(mean=mean, observations=captures))
    # Finish the motion session at HOME before nonlinear fitting/disk output.
    session.robot.close()
    if session.robot.process.returncode != 0:
        raise RuntimeError('Native calibration motion session reported a fault; calibration not saved')
    result = cal.solve(means); result['source_run'] = str(args.output.resolve())
    result['paddle_segment'] = session.paddle_segment
    result['source_kind'] = session.calibration_source_kind
    # Derive the marker-body-to-face transform, preserving the CAD contact face.
    robot, _ = model(args.factors)
    p, n, J, N, rot = robot.fk(HOME)
    B_T_face = np.eye(4); B_T_face[:3, :3] = np.array(rot); B_T_face[:3, 3] = np.array(p).ravel()
    import pinocchio as pin
    pin.framesForwardKinematics(robot.pin, robot.data, HOME)
    B_T_F = robot.data.oMf[robot.pin.getFrameId('panda_link8')].homogeneous
    result['P_T_CAD_face'] = (cal.inverse(np.array(result['F_T_P'])) @ cal.inverse(B_T_F) @ B_T_face).tolist()
    result['cad_geometry_spec'] = json.loads((args.factors/'metadata.json').read_text())['spec']
    cal.write_new(args.calibration, result)
    print(f'Saved fixed flange-to-paddle calibration: {args.calibration}', flush=True)


def home_alignment(session, args, current_pose=False):
    fixed = session.load_calibration()
    if fixed.get('paddle_segment') != session.paddle_segment:
        raise RuntimeError('Vicon paddle root segment differs from the saved calibration')
    if not current_pose: session.move(HOME)
    samples, _ = session.collect()
    cal.write_new(args.output/'home_observations.json', samples)
    alignment = cal.align_home(fixed, samples)
    alignment['acquisition_pose']='current_stationary_pose' if current_pose else 'HOME'
    cal.write_new(args.output/'world_alignment.json', alignment)
    return alignment


def fixture(session, args, robot, meta, controller):
    # Detect accidental early release even during initial HOME motion.
    preliminary = ReleaseGate()
    def early(frame):
        p = session.ball_position(frame)
        if p is None: preliminary.missing(capture_time(frame, args.extra_latency)); return
        preliminary.update(capture_time(frame, args.extra_latency), p, frame['frame'])
    session.on_frame = early
    deadline = time.monotonic()+5
    while preliminary.anchor is None:
        session.pump()
        if time.monotonic() > deadline: raise RuntimeError('Ball is not stationary in its fixture')
    alignment = home_alignment(session, args, current_pose=True); B_T_V = np.array(alignment['B_T_V'])
    session.ball_frame = (B_T_V, robot.origin.copy())
    anchor = B_T_V[:3, :3] @ preliminary.anchor+B_T_V[:3, 3]
    preparation_only = args.mode == 'prepare'
    stationary_drop = args.mode == 'drop'
    home = fixture_home(robot, anchor, preparation_only=preparation_only,
        strike_plane_offset=getattr(args,'strike_plane_offset',0.)); cal.write_new(args.output/'fixture_home.json', home)
    gravity=[meta['config'].get('gravity_'+axis,default) for axis,default in zip('xyz',(0.,0.,-9.81))]
    gate = ReleaseGate(gravity=gravity); gate.anchor = anchor.copy()
    last_ball = None; pending = deque()
    def receive(frame):
        nonlocal last_ball
        t = capture_time(frame, args.extra_latency); p = session.ball_position(frame)
        if p is None:
            if not session.mpc_started: gate.missing(t)
            return
        p = B_T_V[:3, :3] @ p+B_T_V[:3, 3]; last_ball = frame['receive_time']
        if not session.mpc_started: gate.update(t, p, frame['frame'])
        pending.append((t, p, frame['receive_time']))
        if len(pending) > 100: pending.popleft()  # Held samples unused by the observer.
    session.on_frame = receive; session.move(np.array(home['q_home']))
    # Check the saved mounting at the new settled posture as well as at HOME.
    captures, _ = session.collect(.4); X = np.array(session.load_calibration()['F_T_P'])
    for observation in captures:
        expected = np.array(alignment['V_T_B']) @ np.array(observation['B_T_F']) @ X
        translation, angle = cal.errors(expected, np.array(observation['V_T_P']))
        if translation > .005 or angle > np.deg2rad(1):
            raise RuntimeError('Paddle mounting/world alignment inconsistent at fixture HOME')
    try:
        if controller is not None: controller.set_reference(home['face_offset_m'], home['q_home'])
        # Warm load/factor caches without actuating the predicted falling ball.
        tau = np.asarray(robot.rnea(session.arm['q'], session.arm['dq'], session.arm['ddq_d'])).ravel()
        seed = np.r_[anchor-robot.origin, np.zeros(3), session.arm['q'], session.arm['dq'], np.zeros(3)]
        if controller is not None and home['release_height_supported']:
            session.robot.send('heartbeat'); controller.step(seed, tau, 0); session.pump()
        if preparation_only:
            cal.write_new(args.output/'ready.json', dict(status='positioned_below_held_ball',
                release_armed=False, mpc_plans_sent=0, mpc_warmup_performed=home['release_height_supported'],
                release_height_supported=home['release_height_supported'], free_fall_gap_m=home['free_fall_gap_m'],
                note='Preparation-only: stopping before arming release or executing any MPC plan.'))
            print('PREPARED: paddle below held ball; stopping without arming release or sending MPC plans.', flush=True)
            return
        if args.mode=='rehearse':
            from .rehearsal import run
            run(session,args,robot,meta,controller,home,anchor,gate)
            return
        if getattr(args,'rehearsal_project_reference',False):
            from .rehearsal import executable_trajectory
        if session.raw_ball is not None: session.raw_ball.allow_motion = True
        gate.arm()
        cal.write_new(args.output/'ready.json',dict(release_armed=True,ready_time=time.monotonic(),
            status='paddle_settled_below_ball',ball_base_m=anchor.tolist(),logging_active=True,
            stationary_drop=stationary_drop,post_release_duration_s=args.seconds))
        print(('RELEASE BALL NOW: stationary paddle; recording the drop and rebound; no MPC motion.' if stationary_drop
               else 'RELEASE BALL NOW: paddle settled below ball; tracking healthy; recording active.'), flush=True)
        wait_start = time.monotonic()
        while gate.released is None:
            session.pump(); gate.missing(time.monotonic()-args.extra_latency)
            if time.monotonic()-wait_start > args.release_timeout: raise RuntimeError('Release wait timed out')
        event = gate.released; cal.write_new(args.output/'release.json', event)
        cfg = ObserverConfig(**{k: meta['config'][k] for k in ['radius', 'restitution', 'friction', 'paddle_radius', 'inertia_ratio']})
        cfg.drag = meta['config'].get('drag', .02)
        cfg.gravity_x,cfg.gravity_y,cfg.gravity_z=gravity
        cfg.ball_mass = meta['config']['mass']; cfg.max_measurement_age = MAX_BALL_GAP+.02
        cfg.contact_measurement_window = .015  # 100 Hz: span a full frame after impact.
        cal.write_new(args.output/'observer_config.json', asdict(cfg))
        observer = BallObserver(robot, cfg)
        # Start posterior at the confirmed capture time using measured velocity
        # from the release fit. Interpolate actual encoder history at that time.
        t0 = event['capture_time']; past = [a for a in session.arms if a['receive_time'] <= t0]
        future = [a for a in session.arms if a['receive_time'] >= t0]
        deadline = time.monotonic()+.03
        while not future:
            session.pump(); future = [a for a in session.arms if a['receive_time'] >= t0]
            if time.monotonic() > deadline: raise RuntimeError('No arm history bracketing release')
        if not past: raise RuntimeError('Missing pre-release robot history')
        a, b = past[-1], future[0]; weight=(t0-a['receive_time'])/max(b['receive_time']-a['receive_time'], 1e-12)
        q0=(1-weight)*np.array(a['q'])+weight*np.array(b['q']); dq0=(1-weight)*np.array(a['dq'])+weight*np.array(b['dq'])
        observer.record_arm(t0, q0, dq0); observer.initialize(np.array(event['position'])-robot.origin, t0)
        observer.x[3:6] = event['velocity']; observer.P[3:6, 3:6] = np.eye(3)*.08**2
        for a in session.arms:
            if a['receive_time'] > t0: observer.record_arm(a['receive_time'], a['q'], a['dq'])
        session.observer=observer; session.mpc_started=True
        previous=None; start=time.monotonic(); next_solve=0; return_start=float("inf")
        with (args.output/('ball_state.jsonl' if stationary_drop else 'mpc.jsonl')).open('x') as log:
            while time.monotonic()-start < args.seconds:
                session.pump(); now=time.monotonic(); arm=session.arm; t=arm['receive_time']
                if last_ball is None or now-last_ball > MAX_BALL_GAP: raise RuntimeError('Ball tracking lost during '+('stationary drop' if stationary_drop else 'MPC'))
                while pending and pending[0][2] <= t:
                    capture, p, arrival=pending.popleft()
                    if capture<=t0: continue
                    # Position posterior uses measured arm history; no virtual ball.
                    observer.update_position(capture, p-robot.origin, arrival)
                if t<next_solve: continue
                source_to_compute=time.monotonic()-t
                x, covariance, info=observer.mpc_state(t, arm['q'], arm['dq'])
                if info['stale']: raise RuntimeError('Ball observer measurements are stale or rejected')
                if stationary_drop:
                    if np.max(np.abs(np.array(arm['q'])-np.array(home['q_home'])))>.02:
                        raise RuntimeError('Stationary paddle drifted from the held pose')
                    log.write(json.dumps(dict(time=t,state=x.tolist(),covariance=covariance.tolist(),
                        measurement_age=info['measurement_age'],last_impact=info['last_impact'] if np.isfinite(info['last_impact']) else None,
                        mode='stationary_drop',mpc_plans_sent=0))+'\n');log.flush()
                    next_solve=t+.009
                    continue
                # Stop when the expected crossing of the strike plane lies
                # outside the reachable ball-following region. This uses the
                # estimated trajectory, not a request to chase a lost marker.
                if not np.isfinite(return_start) and np.isfinite(info['last_impact']): return_start=info['last_impact']
                return_fraction=nominal_return_fraction(t,return_start,meta['config'].get('return_to_nominal_s',0.))
                reference_offset=(1-return_fraction)*np.array(home['face_offset_m'])
                if meta['config'].get('return_to_nominal_s',0.)>0:
                    controller.set_reference(reference_offset,(1-return_fraction)*np.array(home['q_home'])+return_fraction*HOME)
                strike_height=meta['config']['radius']+reference_offset[2]
                height=x[2]-strike_height
                if x[2] < -.15 or x[2] > 1.5: raise RuntimeError('Ball left MPC workspace')
                if height>0:
                    landing=predicted_landing(x,meta['config']['drag'],strike_height,gravity)
                    if not within_landing_reach(landing, home['face_offset_m'][:2], meta['config']):
                        cal.write_new(args.output/'workspace_abort.json',dict(time=t,state=x.tolist(),
                            landing_m=landing.tolist(),fixture_offset_m=home['face_offset_m'],
                            nominal_radius_m=float(np.linalg.norm(landing)),fixture_radius_m=float(np.linalg.norm(landing-np.array(home['face_offset_m'])[:2]))))
                        raise RuntimeError('Predicted ball landing is out of reach')
                tau=np.asarray(robot.rnea(arm['q'],arm['dq'],arm['ddq_d'])).ravel()
                if previous is None:
                    controller.initialize(x,tau)
                    primal, dual=controller.warmstart(); controller.set_warmstart(primal,np.zeros_like(dual))
                Y, _, stats=controller.step(x,tau,0 if previous is None else t-previous)
                if time.monotonic()-t > .05: raise RuntimeError('Computed plan is too old to execute')
                if getattr(args,'rehearsal_project_reference',False):
                    # Execute the same acceleration-consistent reference as HIL.
                    trajectory,raw_acceleration_ratio,projection_ratio=executable_trajectory(
                        robot,meta,Y,arm['q'],arm['dq'],arm['ddq_d'])
                else:
                    endpoint_acceleration=np.array([np.linalg.solve(np.asarray(robot.mass(y[6:13])),
                        y[23:30]-np.asarray(robot.bias(y[6:13],y[13:20])).ravel()) for y in Y[:4]])
                    trajectory=np.c_[Y[:4,6:20],Y[:4,23:30],endpoint_acceleration]
                    trajectory[0]=np.r_[arm['q'],arm['dq'],tau,arm['ddq_d']]
                    raw_acceleration_ratio=None;projection_ratio=0.
                budget=float(getattr(args,'rehearsal_delivery_budget',.015))
                if time.monotonic()-t>budget:
                    cal.write_new(args.output/'plan_timing_abort.json',dict(measurement_time=t,
                        age_s=time.monotonic()-t,source_to_compute_s=source_to_compute,
                        native_metrics=stats.tolist(),first_plan=previous is None))
                    raise RuntimeError(f'Measured state exceeded {1000*budget:g} ms delivery budget')
                session.robot.send('plan '+str(t)+' '+' '.join(map(str,trajectory.ravel())))
                log.write(json.dumps(dict(time=t,state=x.tolist(),covariance=covariance.tolist(),
                    plan=Y[:4].tolist(),metrics=stats.tolist(),measurement_age=info['measurement_age'],
                    ball_source='vicon',reference_offset_m=reference_offset.tolist(),nominal_return_fraction=return_fraction,source_to_compute_s=source_to_compute,
                    source_to_send_s=time.monotonic()-t,raw_acceleration_ratio=raw_acceleration_ratio,
                    acceleration_projection_ratio=projection_ratio,executed_trajectory=trajectory.tolist()))+'\n')
                log.flush()
                previous=t; next_solve=t+meta['config']['control_dt']-.001
        cal.write_new(args.output/'observer.json', dict(updates=observer.updates, impacts=observer.events))
        if stationary_drop:
            cal.write_new(args.output/'drop_complete.json',dict(duration_s=time.monotonic()-start,mpc_plans_sent=0,
                model_predicted_impacts=observer.events,warning='Impact events are observer estimates; fit the saved raw measurements.'))
            print('DROP RECORDING COMPLETE: stopping the robot; keeping Vicon recording briefly.',flush=True)
    finally:
        if controller is not None: controller.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['calibrate','align','fixture','prepare','drop','rehearse'])
    p.add_argument('--execute',action='store_true',help='Connect to and move the physical robot')
    p.add_argument('--robot',required=True); p.add_argument('--allow-nonrealtime',action='store_true')
    p.add_argument('--vicon-host',required=True); p.add_argument('--ball',default='auto',help='auto: persistent unlabeled point above paddle; otherwise Subject/Marker or unique label')
    p.add_argument('--vicon-binary',type=Path,default=Path('build/vicon/juggling_vicon'))
    p.add_argument('--robot-binary',type=Path,default=Path('build/session/juggling_franka_session'))
    p.add_argument('--factors',type=Path,default=DEFAULT_FACTORS)
    p.add_argument('--library',type=Path,default=DEFAULT_FACTORS.parent/'controller/libjuggling_controller.so')
    p.add_argument('--calibration',type=Path,default=Path('config/paddle_hand_eye.json'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--extra-latency',type=float,default=None)
    p.add_argument('--timing-calibration',type=Path,default=Path('config/vicon_timing.json'))
    p.add_argument('--seconds',type=float,default=20.); p.add_argument('--release-timeout',type=float,default=120.)
    p.add_argument('--rehearsal-prediction',type=float,default=0.,help='Predict rehearsal state ahead by this many seconds')
    p.add_argument('--rehearsal-project-reference','--project-reference',action='store_true',help='Project small acceleration defects and integrate a consistent MPC reference')
    p.add_argument('--strike-plane-offset',type=float,default=0.,help='Vertical offset of the fixture strike plane in metres, within +/-0.05')
    p.add_argument('--rehearsal-delivery-budget','--plan-delivery-budget',type=float,default=.015,help='Maximum measured-state-to-command age for MPC, in seconds')
    p.add_argument('--require-realtime-priority',action='store_true',help='Refuse robot control without SCHED_FIFO permission')
    p.add_argument('--compact-robot-readback',action='store_true',help='Publish compact feedback; preserve complete native robot logging')
    p.add_argument('--robot-control-cpu',type=int,default=None,help='Optional CPU affinity for the native control thread only')
    args=p.parse_args()
    if not .01<=args.rehearsal_delivery_budget<=.035:p.error('MPC delivery budget must be between 10 and 35 ms')
    if args.robot_control_cpu is not None and args.robot_control_cpu<0:p.error('Control CPU must be nonnegative')
    if not np.isfinite(args.strike_plane_offset) or abs(args.strike_plane_offset)>.05:p.error('Strike-plane offset must be within +/-5 cm')
    if not 0<=args.rehearsal_prediction<=.015:p.error('Rehearsal prediction must be between 0 and 15 ms')
    if args.extra_latency is None:
        timing=json.loads(args.timing_calibration.read_text())
        if timing['vicon_host'] != args.vicon_host: p.error('Timing calibration is for a different Vicon server; provide --extra-latency explicitly')
        args.extra_latency=float(timing['extra_latency_s'])
    if not 0<args.seconds<=(120 if args.mode in ('fixture','rehearse') else 60) or not 0<args.release_timeout<=300 or not -.01<=args.extra_latency<.05:
        p.error('Invalid session duration or extra latency')
    if args.mode=='drop' and args.seconds>2: p.error('Drop must be bounded to at most 2 seconds')
    if args.mode=='rehearse' and args.seconds>120: p.error('Rehearsal must be bounded to at most 120 seconds')
    if args.mode=='calibrate' and args.calibration.exists(): p.error('Calibration output already exists; use a new filename')
    if args.execute and args.output.exists(): p.error('Run output must be a new directory')
    robot, meta=model(args.factors)
    if not args.execute:
        print('Prepared only; no Vicon/robot connection. Supply --execute when hardware is ready.');return
    if args.mode!='calibrate': cal.load(args.calibration)  # Reject before any connection.
    session=Session(args)
    controller=None;failure=None
    try:
        # Library loading, worker creation and sparse-pattern construction are
        # completed before opening FCI (the held-pose watchdog is only 150 ms).
        if args.mode in ('fixture','prepare','rehearse'): controller=NativeController(args.library,args.factors)
        session.start()
        cal.write_new(args.output/'configuration.json',dict(mode=args.mode, vicon_host=args.vicon_host, ball=args.ball,
                      extra_latency_s=args.extra_latency, timestamp_convention='receive_monotonic - per-frame SDK latency (cadence fallback) - signed extra_latency; approximate capture time',
                      factors=str(args.factors.resolve()), calibration=str(args.calibration.resolve()), physical_robot_used=True,
                      seconds=args.seconds,release_timeout=args.release_timeout,robot_binary=str(args.robot_binary.resolve()),
                      strike_plane_offset_m=args.strike_plane_offset,rehearsal_prediction_s=args.rehearsal_prediction,
                      rehearsal_delivery_budget_s=args.rehearsal_delivery_budget,
                      compact_robot_readback=args.compact_robot_readback,robot_control_cpu=args.robot_control_cpu,
                      rehearsal_project_reference=args.rehearsal_project_reference,
                      controller_library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest() if controller is not None else None,
                      rehearsal_module_sha256=hashlib.sha256(Path(__file__).with_name('rehearsal.py').read_bytes()).hexdigest(),
                      fixture_module_sha256=hashlib.sha256(Path(__file__).with_name('fixture.py').read_bytes()).hexdigest(),
                      robot_binary_sha256=hashlib.sha256(args.robot_binary.read_bytes()).hexdigest(),
                      supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        if args.mode=='calibrate': calibrate(session,args)
        elif args.mode=='align': home_alignment(session,args)
        else: fixture(session,args,robot,meta,controller)
    except BaseException as error:
        failure=error
        # Request the native bounded stop before doing any error-file I/O.
        if session.robot is not None:
            try: session.robot.send('stop')
            except (RuntimeError,BrokenPipeError): pass
        print('ABORT: '+(str(error) or type(error).__name__)+'. Stopping robot; do not release.',flush=True)
        raise
    finally:
        status=session.close()
        native={}
        native_path=args.output/'robot'/'session.json'
        if session.output_created and native_path.exists():
            native=json.loads(native_path.read_text())
            if native.get('error'):
                print('NATIVE STOP REASON: '+native['error'],flush=True)
        if controller is not None: controller.close()
        if session.output_created:
            virtual=getattr(session,'virtual_ball',None)
            if virtual is not None:
                cal.write_new(args.output/'virtual_ball_final.json',dict(time=virtual.t,
                    state=virtual.x.tolist(),impacts=virtual.events,external_disturbance=False))
            if session.observer is not None and not (args.output/'observer.json').exists():
                cal.write_new(args.output/'observer.json',dict(updates=session.observer.updates,impacts=session.observer.events))
            if failure is not None: cal.write_new(args.output/'failure.json',dict(error=str(failure) or type(failure).__name__,stop_requested=True,native_session=native))
        if status and failure is None:
            message='Native robot session reported a fault; inspect robot/session.json'
            cal.write_new(args.output/'failure.json',dict(error=message))
            raise RuntimeError(message)


if __name__=='__main__': main()
