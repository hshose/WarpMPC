"""Offline full supervisor/velocity-servo test at a recorded release fixture.

Uses an impulse-contact plant and illustrative servo lag; no SDK or network.
This complements, and does not replace, Newton contact validation.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from tracking.check_supervisor import exercise
from tracking.session import model
from tracking import calibration as cal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--factors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['fixture','drop','rehearse'],default='fixture')
    p.add_argument('--fault',choices=['ball_dropout','short_gaps_outliers','duplicate_frame','early_release','rehearsal_release'])
    p.add_argument('--plan-delay',type=float,default=0.,help='Deterministic simulated command delivery delay in seconds')
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--rehearsal-prediction',type=float,default=0.)
    p.add_argument('--rehearsal-project-reference',action='store_true')
    p.add_argument('--strike-plane-offset',type=float,default=0.)
    p.add_argument('--rehearsal-delivery-budget',type=float,default=.015)
    p.add_argument('--start-from-recorded-final',action='store_true',help='Start stationary at the recording last measured joint position')
    p.add_argument('--synthetic-calibration',type=Path,default=Path('runs/20261002_model_update/servo_supervisor/synthetic_calibration.json'))
    args=p.parse_args()
    robot,meta=model(args.factors)
    fixed=cal.load(args.synthetic_calibration,require_live=False)
    if fixed['source_kind']!='synthetic':raise ValueError('Synthetic calibration required')
    X=np.array(fixed['F_T_P']);Y=cal.pose([.4,.8,.03],Rotation.from_euler('z',-.8).as_quat())
    fixture=json.loads((args.recording/'fixture_home.json').read_text())
    options=SimpleNamespace(output=args.output,calibration=args.synthetic_calibration,
        factors=args.factors,library=args.factors.parent/'controller/libjuggling_controller.so',
        plant=args.factors.parent/'controller/libjuggling_fixture_plant.so',seconds=args.seconds,
        ball='auto',plan_delay=args.plan_delay,rehearsal_prediction=args.rehearsal_prediction,
        rehearsal_project_reference=args.rehearsal_project_reference,
        strike_plane_offset=args.strike_plane_offset,
        rehearsal_delivery_budget=args.rehearsal_delivery_budget,
        fixture_ball=(np.array(fixture['ball_fixture_base_m'])-robot.origin).tolist())
    if args.start_from_recorded_final:
        options.initial_q=json.loads((args.recording/'robot/robot.jsonl').read_text().splitlines()[-1])['q']
    result=exercise(options,robot,meta,X,Y,args.mode,'normal',fault=args.fault)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
