"""Exercise per-world measured geometry and native contacts on the GPU."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from arm.model import Panda,HOME
from empirical import EmpiricalBatch
from empirical_newton import EmpiricalNewton
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).parent;spec=json.loads((root/'reference/config/cpp_ball_spec.json').read_text())
robot=Panda(assembly=spec['assembly'],**spec['geometry']);draws=EmpiricalBatch(root/'calibration/model.json',4)
ball=np.tile(robot.origin,(4,1));ball[:,0]+=np.arange(4)*.015;ball[:,2]+=.06
plant=EmpiricalNewton(root/'calibration/model.json',root/'calibration/newton_contact_fit.json',4,a.output,spec,np.tile(HOME,(4,1)),ball,draws.parameter_draws())
torque=wp.array(np.tile(np.asarray(robot.bias(HOME,np.zeros(7))).ravel(),(4,1)).astype('float32'),device='cuda:0');zero=wp.zeros((4,3),device='cuda:0')
for _ in range(600):plant.step(torque,zero,zero)
x=plant.observe().numpy();counts=plant.counts.numpy();result=dict(passed=bool(np.isfinite(x).all() and (counts>=1).all()),native_contacts_per_world=counts.tolist(),seconds=.3,batch=4)
(a.output/'empirical_smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['passed']
