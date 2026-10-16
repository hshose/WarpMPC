"""Replay saved in-flight Warp solves with the independent native controller."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from factors import Factors
from plan_parity import metrics
from cpp.python_controller import NativeController
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('fixture',type=Path);p.add_argument('--world',type=int,required=True);a=p.parse_args()
f=Factors(a.directory/'factors');data=np.load(a.fixture);i=a.world
native=NativeController(a.directory/'controller/libjuggling_controller.so',a.directory/'factors',compiled=False)
param=data['p'][i];native.set_reference(param[30:33],param[33:]);native.set_warmstart(data['z'][i],data['dual'][i]);expected=native.step(param[:23],param[23:30],0.)[0]
r=metrics(f,data['plan'][i:i+1],expected[None]);r.update(world=i,fixture=str(a.fixture),time_s=float(data['time_s']));print(json.dumps(r,indent=2));native.close()
a.fixture.with_suffix(f'.world{i}.parity.json').write_text(json.dumps(r,indent=2)+'\n')
