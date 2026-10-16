"""Offline wall-clock transport/solver benchmark. Never connects to hardware.

Replays a stationary recorded robot pose and Vicon frame through subprocess
pipes and the real supervisor. Command receipt is measured in the receiver;
1 ms is added as a conservative callback scheduling allowance. This tests
latency, not closed-loop dynamics or actual libfranka/network scheduling.
"""
import argparse,json,sys,time,threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tracking import session as live
from tracking.rehearsal import run
from tracking.streams import JsonProcess
from cpp.python_controller import NativeController


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--factors',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--feed',choices=['robot','vicon'])
    args=p.parse_args()
    if args.feed:
        name='robot_stream.jsonl' if args.feed=='robot' else 'vicon.jsonl'
        rows=[json.loads(s) for s in (args.recording/name).read_text().splitlines()]
        row=next(r for r in reversed(rows) if (not r['stopping'] if args.feed=='robot' else r.get('paddle')))
        done=threading.Event()
        def receive():
            with (args.output/'receipts.jsonl').open('w') as log:
                for line in sys.stdin:
                    words=line.split()
                    if words[0]=='stop':done.set();return
                    if words[0]=='plan':
                        log.write(json.dumps(dict(age_s=time.monotonic()-float(words[1])))+'\n');log.flush()
        if args.feed=='robot':threading.Thread(target=receive,daemon=True).start()
        deadline=time.monotonic();number=0
        while not done.is_set():
            now=time.monotonic()
            if now<deadline:time.sleep(min(.0002,deadline-now));continue
            row['receive_time']=now
            if args.feed=='vicon':
                number+=1;row['frame']=number;row.pop('capture_time_s',None)
            print(json.dumps(row),flush=True);deadline+=.01
        return
    args.output.mkdir(parents=True,exist_ok=False)
    robot,meta=live.model(args.factors)
    home=json.loads((args.recording/'fixture_home.json').read_text())
    opts=SimpleNamespace(output=args.output,extra_latency=0.,seconds=.8,mode='rehearse',ball='auto')
    s=live.Session(opts);s.output_created=True
    base=[sys.executable,'-m','tracking.check_rehearsal_timing','--recording',str(args.recording),
          '--factors',str(args.factors),'--library',str(args.library),'--output',str(args.output)]
    s.robot=JsonProcess(base+['--feed','robot'],args.output,'robot_stream',writable=True)
    s.vicon=JsonProcess(base+['--feed','vicon'],args.output,'vicon')
    # Pump preserves stream freshness and source timestamps. The physical held
    # ball gate is omitted here: this benchmark has no physical ball or robot.
    gate=SimpleNamespace(missing=lambda t:None)
    controller=NativeController(args.library,args.factors)
    error=None
    try:
        deadline=time.monotonic()+5
        while s.arm is None or s.latest_frame is None:
            for a in s.robot.drain():s.arm=a;s.arms.append(a)
            for f in s.vicon.drain():s.latest_frame=f;s.paddle_segment=f['paddle']['segment'];s.last_paddle=f['receive_time']
            if time.monotonic()>deadline:raise RuntimeError('Replay startup timeout')
            time.sleep(.001)
        controller.set_reference(home['face_offset_m'],home['q_home'])
        x=np.r_[np.array(home['ball_fixture_base_m'])-robot.origin,np.zeros(3),s.arm['q'],s.arm['dq'],np.zeros(3)]
        tau=np.asarray(robot.rnea(s.arm['q'],s.arm['dq'],s.arm['ddq_d'])).ravel()
        controller.step(x,tau,0)
        s.pump()
        run(s,opts,robot,meta,controller,home,np.array(home['ball_fixture_base_m']),gate)
    except Exception as exc:error=str(exc)
    finally:s.close();controller.close()
    receipts=[json.loads(l)['age_s']*1000+1 for l in (args.output/'receipts.jsonl').read_text().splitlines()]
    result=dict(robot_connected=False,error=error,plans=len(receipts),
        receipt_plus_1ms_allowance_ms=dict(median=float(np.median(receipts)),p95=float(np.percentile(receipts,95)),maximum=max(receipts)),
        over_15ms=sum(v>15 for v in receipts),
        limitation='Offline stationary replay: excludes real SDK/network scheduling, 1kHz native logging, and physical ball association.')
    (args.output/'timing.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
