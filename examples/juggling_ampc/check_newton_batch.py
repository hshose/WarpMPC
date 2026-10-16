"""Native Newton batch smoke check: independent worlds and ballistic release."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from newton_batch import NewtonBatch
from newton_sim.scene import SceneConfig
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=4);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=NewtonBatch(a.batch,SceneConfig(device=a.device,visuals=False),a.output)
x=s.observe().numpy();tau=np.tile(np.asarray(s.robot.bias(x[0,6:13],x[0,13:20])).ravel(),(a.batch,1)).astype('float32');tau=wp.array(tau,device=a.device);wind=wp.zeros((a.batch,3),device=a.device)
for _ in range(20):s.step(tau,wind)
y=s.observe().numpy();fall=float(np.mean(y[:,2]-x[:,2]));identical=float(np.max(abs(y-y[:1])));passed=bool(np.isfinite(y).all() and -.0006<fall<-.0004 and identical<1e-6)
r=dict(passed=passed,batch=a.batch,device=a.device,seconds=.01,ball_height_change_m=fall,max_world_difference=identical,native_contacts=True)
a.output.mkdir(parents=True,exist_ok=True);(a.output/'smoke.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert passed
