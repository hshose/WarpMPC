"""Replay live held-ball clutter with injected missing/jumping/ambiguous data."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from . import calibration as cal
from .track_ball import Monitor


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    monitor=Monitor(cal.load(args.calibration),args.output)
    opener=gzip.open if args.input.suffix=='.gz' else open
    injected=dict(missing=0,outlier=0,ambiguous=0); rng=np.random.default_rng(819)
    with opener(args.input,'rt') as data:
        for k,line in enumerate(data):
            frame=json.loads(line)
            if monitor.tracker.selection is not None:
                anchor=np.array(monitor.tracker.selection['position_vicon_m'])
                points=frame['unlabeled_markers']
                ball=[p for p in points if np.linalg.norm(np.array(p['position_m'])-anchor)<.003]
                assert len(ball)==1, 'Recording must contain one continuously visible held ball'
                if 800<=k<805:
                    points.remove(ball[0]); injected['missing']+=1
                elif k==1200:
                    ball[0]['position_m']=(np.array(ball[0]['position_m'])+[.3,0,0]).tolist()
                    injected['outlier']+=1
                elif k==1600:
                    points.append(dict(id=-1,position_m=(np.array(ball[0]['position_m'])+[.001,0,0]).tolist()))
                    injected['ambiguous']+=1
            # Deliberately unstable IDs/order cannot determine ball identity.
            for j,p in enumerate(frame.get('unlabeled_markers') or []): p['id']=k*100+j
            rng.shuffle(frame['unlabeled_markers'])
            monitor.process(frame)
    summary=monitor.finish()
    assert injected==dict(missing=5,outlier=1,ambiguous=1),injected
    assert summary['counts']['predicted']==7,summary
    assert summary['rejections']['ambiguous']==1,summary
    assert summary['max_estimated_step_m']<.001 and not summary['observed_release'],summary
    # No prediction-only frame is disguised as a real camera observation.
    assert all(r['measurement_vicon_m'] is None for r in monitor.rows if r['status']=='predicted')
    result=dict(passed=True,injected=injected,summary=summary)
    cal.write_new(args.output/'checks.json',result)
    print(json.dumps({k:v for k,v in summary.items() if k!='marker_ids'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path('config/paddle_hand_eye.json'))
    run(p.parse_args())
