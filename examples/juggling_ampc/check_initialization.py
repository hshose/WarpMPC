import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from factors import Factors
from initialization import FixtureIK
from sampling import drop_positions
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--randomized-count',type=int,default=0);p.add_argument('--native-api',action='store_true');a=p.parse_args()
f=Factors(a.directory/'gpu_factors');ik=FixtureIK(f,backend='jax')
model=json.loads((Path(__file__).parent/'calibration/model.json').read_text())
targets=np.array([x['ball_base_m'] for x in model['initial_conditions']],dtype=np.float32)
if a.randomized_count:
    targets,_=drop_positions(np.random.default_rng(1002026),a.randomized_count,f.arrays['origin'],f.cfg,targets,nominal_strike_z=model['nominal_reference']['nominal_face_base_m'][2])
    targets=targets.astype('float32')
targets[:,2]=f.arrays['origin'][2]+model['nominal_reference']['initial_offset_m'][2]
q,valid=ik.solve(targets)
q=np.asarray(q);valid=np.asarray(valid)
native_checked=0
if a.native_api:
    from cpp.python_controller import NativeController
    native=NativeController(a.directory/'controller/libjuggling_controller.so',a.directory/'factors',compiled=False)
    for target,posture in zip(targets[valid],q[valid]):
        offset=target-f.arrays['origin'];offset[2]=model['nominal_reference']['initial_offset_m'][2]
        native.set_reference(offset,posture);native_checked+=1
    native.close()
r=dict(passed=bool(valid.any() if a.randomized_count else valid.all()),count=len(q),valid=int(valid.sum()),native_api_checked=native_checked)
print(json.dumps(r));assert r['passed']
