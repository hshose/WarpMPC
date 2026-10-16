"""Verify captured native Newton stepping against ordinary stepping."""
import bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
from arm.model import Panda,HOME
from empirical import EmpiricalBatch
from empirical_newton import EmpiricalNewton
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).parent;spec=json.loads((root/'reference/config/cpp_ball_spec.json').read_text());robot=Panda(assembly=spec['assembly'],**spec['geometry']);draws=EmpiricalBatch(root/'calibration/model.json',4)
ball=np.tile(robot.origin,(4,1));ball[:,2]+=.06;q=np.tile(HOME,(4,1));parameters=draws.parameter_draws()
plants=[EmpiricalNewton(root/'calibration/model.json',root/'calibration/newton_contact_fit.json',4,a.output/str(i),spec,q,ball,parameters) for i in range(2)]
torque=wp.array(np.tile(np.asarray(robot.bias(HOME,np.zeros(7))).ravel(),(4,1)).astype('float32'),device='cuda:0');acc=wp.zeros((4,3),device='cuda:0');residual=wp.array(np.tile([.02,-.01,.015],(4,1)).astype('float32'),device='cuda:0')
plants[1].enable_graph();timings=[];states=[];counts=[]
for plant in plants:
    begin=time.monotonic()
    for _ in range(400):plant.step(torque,acc,residual)
    states.append(plant.observe().numpy());counts.append(plant.counts.numpy());timings.append(time.monotonic()-begin)
error=float(np.max(abs(states[0]-states[1])));passed=bool(error<1e-5 and np.array_equal(*counts) and np.all(counts[0]>0))
r=dict(passed=passed,max_state_error=error,contacts=counts[0].tolist(),ordinary_wall_s=timings[0],captured_wall_s=timings[1],seconds=.2)
(a.output/'graph.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));assert passed
