"""Verify calibrated native-Newton free flight against an independent ODE solve."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from scipy.integrate import solve_ivp
from newton_batch import NewtonBatch
from empirical_scene import author_plant_geometry
from newton_sim.scene import SceneConfig
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path(__file__).parent/'calibration/model.json');p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
model=json.loads(a.model.read_text());plant=model['plant'];g=np.array(plant['gravity_m_s2'])
cfg=SceneConfig(device=a.device,visuals=False,drag=plant['drag_s_inv'],gravity_x=g[0],gravity_y=g[1],gravity_z=g[2],ball_mass=plant['ball_mass_kg'],radius=plant['ball_radius_m'])
v0=np.array([[.2,-.15,-2.5]],dtype=np.float32)
scene=NewtonBatch(1,cfg,a.output,quadratic_drag=plant['quadratic_drag_m_inv'],scene_transform=lambda path,robot:author_plant_geometry(path,robot,plant),ball_velocity=v0)
x0=scene.observe().numpy()[0].copy();torque=np.array(scene.robot.bias(x0[6:13],x0[13:20])).reshape(1,7).astype('float32')
force=wp.array(torque,device=a.device);acceleration=wp.zeros((1,3),device=a.device)
for _ in range(100):scene.step(force,acceleration)
x1=scene.observe().numpy()[0]
def ode(t,x):return np.r_[x[3:],g-plant['drag_s_inv']*x[3:]-plant['quadratic_drag_m_inv']*np.linalg.norm(x[3:])*x[3:]]
truth=solve_ivp(ode,(0,.05),x0[:6],rtol=1e-12,atol=1e-12).y[:,-1]
error=float(np.max(abs(x1[:6]-truth)));result=dict(passed=error<1e-4,seconds=.05,max_state_error=error,native_newton=True,device=a.device)
(a.output/'flight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['passed']
