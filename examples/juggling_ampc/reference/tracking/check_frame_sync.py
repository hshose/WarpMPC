"""Replay recorded arrival cadence through Session.pump, raw association and EKF.

No subprocesses or hardware; original timestamps are retained. Recorded stopping
flags are cleared to examine observations after the historical supervisor abort.
"""
import argparse,json,importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from . import session as live
from .streams import capture_time
from .raw_ball import RawBall,MAX_BALL_GAP
from estimation.ball import BallObserver,ObserverConfig

class Clock:
    def __init__(self,t):self.t=t
    def monotonic(self):return self.t
    def sleep(self,dt):self.t+=dt

class Stream:
    def __init__(self,rows,clock,delay=0.):self.rows=deque(rows);self.clock=clock;self.delay=delay
    def drain(self):
        out=[]
        while self.rows and self.rows[0]['receive_time']+self.delay<=self.clock.t:out.append(self.rows.popleft())
        return out
    def send(self,message):assert message=='heartbeat'


def replay(path,legacy=None,fault=None,delay=0.,robot_delay=0.):
    read=lambda n:json.loads((path/n).read_text())
    rows=lambda n:[json.loads(l) for l in (path/n).read_text().splitlines()]
    release=read('release.json');t0=release['capture_time'];extra=read('configuration.json')['extra_latency_s']
    arm=rows('robot_stream.jsonl');vicon=rows('vicon.jsonl')
    from arm.model import Panda
    spec=read('model_metadata.json')['spec'];robot=Panda(assembly=spec['assembly'],**spec['geometry'])
    before=[a for a in arm if a['receive_time']<=t0][-1];after=next(a for a in arm if a['receive_time']>=t0)
    clock=Clock(after['receive_time']);args=SimpleNamespace(output=path,extra_latency=extra,ball='auto')
    module=live
    if legacy is not None:
        spec=importlib.util.spec_from_file_location('tracking._legacy_session',legacy)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    s=module.Session(args);s.arm=after;s.arms.extend([before,after]);s.mpc_started=True
    s.latest_frame=next(f for f in vicon if f['frame']==release['frame']);s.paddle_segment=s.latest_frame['paddle']['segment'];s.last_paddle=s.latest_frame['receive_time']
    B=np.array(read('world_alignment.json')['B_T_V']);s.ball_frame=(B,robot.origin.copy())
    observer=BallObserver(robot,ObserverConfig(**read('observer_config.json')))
    w=(t0-before['receive_time'])/(after['receive_time']-before['receive_time'])
    q=(1-w)*np.array(before['q'])+w*np.array(after['q']);dq=(1-w)*np.array(before['dq'])+w*np.array(after['dq'])
    observer.record_arm(t0,q,dq);observer.initialize(np.array(release['position'])-robot.origin,t0)
    observer.x[3:6]=release['velocity'];observer.P[3:6,3:6]=np.eye(3)*.08**2
    if after['receive_time']>t0:observer.record_arm(after['receive_time'],after['q'],after['dq'])
    s.observer=observer
    tracker=RawBall(read('calibration_used.json')['P_T_CAD_face'],extra_latency=extra)
    tracker.selection=read('ball_selection.json');tracker.allow_motion=True
    tracker.position=B[:3,:3].T@(np.array(release['position'])-B[:3,3]);tracker.x=np.r_[tracker.position,B[:3,:3].T@release['velocity']]
    tracker.P=np.diag([.001**2]*3+[.08**2]*3);tracker.seen_time=t0;tracker.filter_time=t0;tracker.last_time=t0;tracker.last_frame=release['frame'];s.raw_ball=tracker
    ar=[dict(a,stopping=False) for a in arm if a['receive_time']>clock.t]
    vr=[f for f in vicon if f['frame']>release['frame']]
    if fault=='robot_loss':ar=[a for a in ar if a['receive_time']-t0<.70]
    if fault=='ball_loss':
        vr=[dict(f,unlabeled_markers=[]) if capture_time(f,extra)-t0>.70 else f for f in vr]
    if fault=='duplicate':
        i=next(i for i,f in enumerate(vr) if capture_time(f,extra)-t0>.70);vr[i]=dict(vr[i],frame=vr[i-1]['frame'])
    s.robot=Stream(ar,clock,delay+robot_delay);s.vicon=Stream(vr,clock,delay)
    accepted=[];last=[s.latest_frame['receive_time']];pending=deque();max_age=0.;error=None
    def receive(frame):
        p=s.ball_position(frame)
        if p is not None:
            last[0]=frame['receive_time'];pending.append((capture_time(frame,extra),B[:3,:3]@p+B[:3,3]-robot.origin,frame['receive_time'],frame['frame']))
    s.on_frame=receive
    try:
        with patch.object(module,'time',clock):
            while clock.t<t0+1.0:
                s.pump()
                if clock.t-last[0]>MAX_BALL_GAP:raise RuntimeError('Ball tracking lost')
                while pending and pending[0][2]<=s.arm['receive_time']:
                    t,p,arrival,frame=pending.popleft();info=observer.update_position(t,p,arrival)
                    if info['accepted']:accepted.append((t-t0,frame))
                _,_,health=observer.estimate(s.arm['receive_time'])
                max_age=max(max_age,health['measurement_age'])
                if health['stale']:raise RuntimeError('Observer stale')
    except RuntimeError as exc:error=str(exc)
    return dict(error=error,end_s=clock.t-t0,accepted=len(accepted),last_accepted_s=accepted[-1][0] if accepted else None,
                recovered_interval_frames=sum(.775<t<.845 for t,_ in accepted),max_measurement_age_s=max_age,
                sync=getattr(s,'frame_sync',{}),raw_rejections=dict(tracker.rejections))


def main():
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--legacy-supervisor',type=Path,required=True);a=p.parse_args()
    results={'legacy':replay(a.recording,legacy=a.legacy_supervisor,robot_delay=.002),'fixed':replay(a.recording,robot_delay=.002),'delayed_delivery':replay(a.recording,delay=.015,robot_delay=.002)}
    for f in ['ball_loss','robot_loss','duplicate']:results[f]=replay(a.recording,fault=f)
    print(json.dumps(results,indent=2))
    assert results['legacy']['error']=='Ball tracking lost',results
    for name in ['fixed','delayed_delivery']:
        assert results[name]['error'] is None and results[name]['recovered_interval_frames']==7,results
    assert 'lost' in results['ball_loss']['error'],results
    assert 'stale' in results['robot_loss']['error'],results
    assert 'duplicated' in results['duplicate']['error'],results
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,results=results),indent=2)+'\n')

if __name__=='__main__':main()
