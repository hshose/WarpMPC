"""Compare independent assembly against the duplicated deployed native controller."""
import bootstrap
import argparse,json,os
from pathlib import Path
import numpy as np
from factors import Factors
from cpp.python_controller import NativeController


def check(directory,factor_name='factors'):
    trace=directory/'first_event_trace.jsonl'
    trace.unlink(missing_ok=True)
    os.environ['JUGGLING_FIRST_EVENT_CENTER_TRACE']=str(trace)
    f=Factors(directory/factor_name);native=NativeController(library=directory/'controller/libjuggling_controller.so',factors=directory/factor_name,compiled=False)
    rng=np.random.default_rng(427);errors={k:0. for k in ['H','gradient','A','constraints','cost']}
    for k in range(3):
        x=np.r_[[.01,-.01,.65],[0.,0.,-.1],f.arrays['home'],np.zeros(10)]
        tau=np.zeros(7);native.initialize(x,tau);z,_=native.warmstart()
        reference=np.r_[[.01,-.02,-.03],f.arrays['home']];native.set_reference(reference[:3],reference[3:])
        z+=rng.normal(0,1e-4,z.shape)
        p=np.r_[x,tau,reference]
        event=np.zeros(6)
        if f.meta['spec'].get('first_event_center',{}).get('weight',0.):
            record=json.loads(trace.read_text().splitlines()[-1])
            event=np.r_[record['active'],record['relative_event_time_s'],record['stage'],record['target']]
            assert event[0], 'Validation fixture must exercise the active first-event objective'
        lhs=f.numpy(z,np.r_[p,event]);rhs=native.linearize(z,p[:30])
        for name,a,b in zip(errors,lhs,rhs):
            delta=a-b;err=float(np.max(abs(delta.data),initial=0)) if hasattr(delta,'tocsc') else float(np.max(abs(delta)))
            errors[name]=max(errors[name],err)
    report=dict(passed=max(errors.values())<1e-5,errors=errors,variables=f.n,constraints=f.m,qp_constraints=f.m+f.n,hessian_nnz=f.P.nnz,A_nnz=f.A.nnz)
    print(json.dumps(report,indent=2));(directory/'factor_parity.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--factor-name',default='factors');a=p.parse_args();check(a.directory.resolve(),a.factor_name)
