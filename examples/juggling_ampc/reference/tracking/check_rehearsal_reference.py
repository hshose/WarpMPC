"""Replay a rejected plan through the bounded reference generator."""
import json
from pathlib import Path
import numpy as np
from tracking.session import model
from tracking.rehearsal import executable_trajectory

def main():
 root=Path('runs/hardware_moving_mpc_20261005/moving_mpc_20261005_154920')
 robot,meta=model(Path('build/variants/fixture39_gain50/factors'))
 row=json.loads((root/'mpc.jsonl').read_text().splitlines()[-1]);Y=np.array(row['plan'])
 commands=[json.loads(s)['command'] for s in (root/'robot_stream_commands.jsonl').read_text().splitlines()]
 sent=np.array(list(map(float,[s for s in commands if s.startswith('plan ')][-1].split()[2:]))).reshape(4,28)
 trajectory,raw,change=executable_trajectory(robot,meta,Y,sent[0,:7],sent[0,7:14],sent[0,21:28])
 limits=np.array(meta['arrays']['acceleration_limit'])
 assert raw>1.1 and change>0 and np.all(abs(trajectory[1:,21:28])<=.95*limits+1e-10)
 for k in range(3):
  h=meta['stage_steps'][k];a=trajectory[k,21:28];b=trajectory[k+1,21:28]
  np.testing.assert_allclose(trajectory[k+1,7:14],trajectory[k,7:14]+h*(a+b)/2)
  np.testing.assert_allclose(trajectory[k+1,:7],trajectory[k,:7]+h*trajectory[k,7:14]+h*h*(2*a+b)/6)
 bad=Y.copy();bad[1,23:30]=np.asarray(robot.rnea(Y[1,6:13],Y[1,13:20],2*limits)).ravel()
 try:executable_trajectory(robot,meta,bad,sent[0,:7],sent[0,7:14],sent[0,21:28])
 except RuntimeError:pass
 else:raise AssertionError('Large acceleration defect accepted')
 print('PASS: recorded excess projected; ramp integration consistent; large defect rejected')
if __name__=='__main__':main()
