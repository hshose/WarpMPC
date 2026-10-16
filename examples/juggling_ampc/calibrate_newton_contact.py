"""Fit native compliant-contact damping to the empirical restitution curve.

The measured e is not a MuJoCo damping coefficient. Calibrate their mapping at
0.5 ms with multiple incident speeds and contact phases, then validate holdouts.
"""
import bootstrap
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from newton_sim.contact_consistency import bounce
from scipy.optimize import brentq
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path(__file__).parent/'calibration/model.json');p.add_argument('--stiffness',type=float,default=5e5);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
plant=json.loads(a.model.read_text())['plant'];stiffness=a.stiffness

def evaluate(damping,speeds,phases):
    return [bounce(stiffness,damping,speed,phase=phase,ball_mass=plant['ball_mass_kg'],friction=plant['friction'],friction_damping=1500.)[0] for speed in speeds for phase in phases]
rows=[]
for radius in np.linspace(0,max(plant['paddle_axes_m']),9):
    target=plant['restitution']-plant['restitution_edge_loss']*(radius/plant['restitution_reference_radius_m'])**4
    damping=brentq(lambda d:np.mean([v['restitution'] for v in evaluate(d,[2.5,4.5],[0.,.00025])])-target,50,3000,xtol=.01)
    cases=evaluate(damping,[2.,3.,4.,5.],[.000125,.000375])
    rows.append(dict(radius_m=float(radius),target_restitution=target,damping=damping,validation=cases))
    print(radius,target,damping,flush=True)
error=max(abs(c['restitution']-r['target_restitution']) for r in rows for c in r['validation'])
report=dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),stiffness=stiffness,dt=.0005,rows=rows,max_heldout_restitution_error=error,passed=bool(error<.04))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}));assert report['passed']
