"""Read-only Vicon ball tracking, with no robot connection or motion commands.

The six-state association KF predicts through brief missing measurements.
The hardware supervisor instead supplies its contact-aware ball EKF prediction
once juggling starts. This monitor can detect a held ball's release, but does
not model paddle contacts or start any robot policy.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time
import numpy as np
from . import calibration as cal
from .fixture import ReleaseGate
from .raw_ball import RawBall
from .streams import JsonProcess, capture_time


def latest(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


class Monitor:
    def __init__(self, fixed, output):
        self.tracker = RawBall(fixed['P_T_CAD_face']); self.tracker.allow_motion = True
        self.output = output; self.log = (output/'ball.jsonl').open('x')
        self.gate = None; self.flight = False; self.frames = 0; self.rows = []
        self.selected_frame = None; self.started = None; self.last_receive = None
        self.counts = Counter(); self.marker_ids = Counter()

    def process(self, frame):
        self.frames += 1; self.last_receive = frame['receive_time']
        self.started = self.last_receive if self.started is None else self.started
        p = self.tracker.update(frame)
        if self.tracker.selection is None: return
        t = capture_time(frame)
        if self.selected_frame is None:
            self.selected_frame = frame['frame']
            cal.write_new(self.output/'ball_selection.json', self.tracker.selection)
            self.gate = ReleaseGate(); self.gate.anchor = np.array(self.tracker.selection['position_vicon_m'])
            self.gate.arm()  # Only enables this read-only release detector.
        if p is not None and not self.flight:
            event = self.gate.update(t, p, frame['frame'])
            if event is not None:
                self.flight = True
                self.tracker.acceleration = np.array([0., 0., -9.81])
                self.tracker.x[3:] = event['velocity']
                self.tracker.P[3:, 3:] += np.eye(3)*.08**2
                cal.write_new(self.output/'observed_release.json', event)
        x, P, info = self.tracker.estimate(t)
        self.counts['measured' if p is not None else 'predicted'] += 1
        association = self.tracker.last_association
        if association and association['accepted']: self.marker_ids[str(association['marker_id'])] += 1
        row = dict(frame=frame['frame'], capture_time=t, receive_time=frame['receive_time'],
            mode='flight' if self.flight else 'held', measurement_vicon_m=None if p is None else p.tolist(),
            position_vicon_m=x[:3].tolist(), velocity_vicon_m_s=x[3:].tolist(),
            position_std_m=np.sqrt(np.diag(P)[:3]).tolist(), **info,
            raw_points=len(frame.get('unlabeled_markers') or []), paddle_visible=frame['paddle'] is not None,
            association=association)
        self.rows.append(row); self.log.write(json.dumps(row)+'\n')
        self.publish(t)

    def publish(self, now):
        estimate = self.tracker.estimate(now)
        state = dict(status='acquiring', robot_connected=False)
        if estimate is not None:
            x, P, info = estimate
            state.update(info, position_vicon_m=x[:3].tolist(), velocity_vicon_m_s=x[3:].tolist(),
                         position_std_m=np.sqrt(np.diag(P)[:3]).tolist())
            if info['stale']: state['status'] = 'lost'
        latest(self.output/'latest.json', dict(time=now, running=True, **state))
        return state

    def finish(self, error=None):
        self.log.close()
        last_path=self.output/'latest.json'
        if last_path.exists():
            state=json.loads(last_path.read_text()); state['running']=False
            latest(last_path,state)
        summary = dict(frames=self.frames, selected_frame=self.selected_frame, selection=self.tracker.selection,
            counts=dict(self.counts), rejections=dict(self.tracker.rejections), marker_ids=dict(self.marker_ids),
            startup_resets=self.tracker.startup_resets, error=error, robot_connected=False,
            max_gap_s=self.tracker.max_gap, observed_release=self.flight)
        if self.rows:
            positions=np.array([r['position_vicon_m'] for r in self.rows])
            summary.update(tracked_seconds=self.rows[-1]['capture_time']-self.rows[0]['capture_time'],
                max_estimated_step_m=float(np.max(np.linalg.norm(np.diff(positions,axis=0),axis=1),initial=0)),
                max_measurement_age_s=max(r['measurement_age'] for r in self.rows),
                simultaneous_paddle_frames=sum(r['paddle_visible'] for r in self.rows))
        cal.write_new(self.output/'summary.json', summary)
        self.plot()
        return summary

    def plot(self):
        if not self.rows: return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t=np.array([r['capture_time'] for r in self.rows]); t-=t[0]
        p=np.array([r['position_vicon_m'] for r in self.rows]); anchor=self.tracker.selection['position_vicon_m']
        v=np.array([r['velocity_vicon_m_s'] for r in self.rows])
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for i,label in enumerate('xyz'):
            axes[0,0].plot(t,(p[:,i]-anchor[i])*1000,label=label)
            axes[0,1].plot(t,v[:,i],label=label)
        axes[0,0].set(ylabel='Estimated displacement from selected point [mm]')
        axes[0,1].set(ylabel='Estimated velocity [m/s]')
        axes[1,0].plot(t,[1000*r['measurement_age'] for r in self.rows],label='Measurement age')
        axes[1,0].axhline(self.tracker.max_gap*1000,ls='--',color='r',label='Loss threshold')
        axes[1,0].set(ylabel='Age [ms]')
        axes[1,1].plot(t,[r['raw_points'] for r in self.rows],label='Raw points')
        axes[1,1].plot(t,[int(r['measurement_vicon_m'] is not None) for r in self.rows],label='Accepted ball')
        axes[1,1].set(ylabel='Count / accepted flag')
        for ax in axes.flat: ax.set(xlabel='Time since selection [s]'); ax.grid(alpha=.25); ax.legend()
        fig.suptitle('Read-only Vicon ball tracking; no robot control')
        fig.savefig(self.output/'report.pdf'); plt.close(fig)


def run(args):
    fixed=cal.load(args.calibration); args.output.mkdir(parents=True,exist_ok=False)
    monitor=Monitor(fixed,args.output); stream=None; error=None
    try:
        if args.input:
            opener=gzip.open if args.input.suffix=='.gz' else open
            with opener(args.input,'rt') as data:
                for line in data: monitor.process(json.loads(line))
        else:
            stream=JsonProcess([args.vicon_binary,args.host],args.output,'vicon')
            end=time.monotonic()+args.seconds; deadline=time.monotonic()+10
            while time.monotonic()<end:
                for frame in stream.drain(): monitor.process(frame)
                now=time.monotonic()
                if monitor.tracker.selection is None:
                    if now>deadline: raise RuntimeError('No unambiguous stable ball above paddle')
                elif monitor.publish(now)['stale']:
                    raise RuntimeError('Ball tracking lost; new acquisition required')
                if monitor.last_receive is not None and now-monitor.last_receive>.10:
                    raise RuntimeError('Vicon stream stale')
                time.sleep(.005)
    except (RuntimeError,ValueError,KeyboardInterrupt) as exc:
        error=str(exc) or type(exc).__name__
        latest(args.output/'latest.json',dict(status='lost',error=error,robot_connected=False))
    finally:
        if stream: stream.close()
        summary=monitor.finish(error)
        print(json.dumps(summary,indent=2),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Replay a saved JSONL or gzip stream instead of connecting')
    parser.add_argument('--calibration',type=Path,default=Path('config/paddle_hand_eye.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--host',required=True)
    parser.add_argument('--vicon-binary',type=Path,default=Path('build/vicon/juggling_vicon'))
    args=parser.parse_args()
    if not 3<args.seconds<=3600: parser.error('Duration must be between 3 and 3600 seconds')
    raise SystemExit(run(args))
