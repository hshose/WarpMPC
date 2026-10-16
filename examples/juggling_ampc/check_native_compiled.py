"""Validate compiled native C factors against independent serialized factors."""
import bootstrap
import argparse, json
from pathlib import Path
import numpy as np
from cpp.python_controller import NativeController
from factors import Factors
from plan_parity import metrics

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
data=np.load(a.directory/'native_fixtures.npz');plans=[];expected=[]
controllers=[NativeController(a.directory/'controller/libjuggling_controller.so',a.directory/'factors',compiled=c) for c in (False,True)]
for i in range(len(data['p'])):
    p=data['p'][i]
    for controller,output in zip(controllers,(expected,plans)):
        controller.set_reference(p[30:33],p[33:]);controller.set_warmstart(data['z'][i],data['dual'][i])
        output.append(controller.step(p[:23],p[23:30],0.)[0])
for c in controllers:c.close()
result=metrics(Factors(a.directory/'factors'),np.asarray(plans),np.asarray(expected))
print(json.dumps(result,indent=2));(a.directory/'native_compiled_parity.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['passed']
