"""Replay nontrivial in-flight GPU updates through the independent native MPC."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from factors import Factors
from plan_parity import metrics
from cpp.python_controller import NativeController


def main(a):
    f=Factors(a.directory/'factors');scale=np.asarray(f.arrays['scale']);offset=np.asarray(f.arrays['offset'])
    candidates=[];seen=set()
    for path in sorted(a.rollout.glob('loss_*_preceding_*.npz')):
        with np.load(path) as data:
            seed=data['z'].reshape(-1,f.cfg['horizon']+1,34)*scale+offset
            delta=np.max(abs((data['plan'][:,:4]-seed[:,:4])[:,:,np.r_[6:20,23:30]]/scale[np.r_[6:20,23:30]]),axis=(1,2))
            p=data['p'];valid=np.isfinite(p).all(axis=1)&(np.linalg.norm(p[:,:2],axis=1)>1e-5)
            valid&=(np.max(abs(p[:,33:]-f.arrays['home']),axis=1)<=1.)
            delta=np.where(valid,delta,-1.);i=int(np.argmax(delta));key=(float(data['time_s']),i)
            if delta[i]>1e-5 and key not in seen:candidates.append((float(delta[i]),path,i,key[0]));seen.add(key)
    candidates=sorted(candidates,reverse=True)[:a.cases]
    if a.selection:
        candidates=[(float(r['scaled_control_update']),a.rollout/r['file'],int(r['world']),float(r['time_s'])) for r in json.loads(a.selection.read_text())['cases'][:a.cases]]
    if len(candidates)<a.cases:raise ValueError('Not enough distinct nontrivial admissible in-flight updates')
    replay=None
    if a.resolve or a.solve_batch:
        from mpc import MPC
        inputs={k:[] for k in ('z','dual','p','projector')}
        for _,path,i,_ in candidates:
            with np.load(path) as data:
                for k in inputs:inputs[k].append(data[k][i])
        mpc=MPC(a.directory/'gpu_factors')
        size=a.solve_batch or len(candidates)
        if size<len(candidates):raise ValueError('Solve batch must include every independent case')
        index=np.arange(size)%len(candidates)
        replay=np.asarray(mpc.solve(*[np.asarray(v,np.float32)[index] for v in inputs.values()])[2])
    rows=[]
    for case,(movement,path,i,time) in enumerate(candidates):
        native=NativeController(a.directory/'controller/libjuggling_controller.so',a.directory/'factors',compiled=True)
        with np.load(path) as data:
            p=data['p'][i];native.set_reference(p[30:33],p[33:]);native.set_warmstart(data['z'][i],data['dual'][i])
            expected=native.step(p[:23],p[23:30],0.)[0]
            actual=data['plan'][i:i+1] if replay is None else replay[case::len(candidates)]
            row=metrics(f,actual,np.broadcast_to(expected,actual.shape));row.update(file=path.name,world=i,time_s=time,scaled_control_update=movement);rows.append(row)
        native.close()
    report=dict(passed=all(r['passed'] for r in rows),cases=rows,resolved_current_gpu=bool(a.resolve or a.solve_batch),solve_batch=(len(replay) if replay is not None else None))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],cases=len(rows),max_first4_scaled_error=max(r['max_first4_scaled_error'] for r in rows),max_acceleration_limit_fraction=max(r['max_first4_acceleration_limit_fraction'] for r in rows)),indent=2))
    if not report['passed']:raise RuntimeError('In-flight MPC parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('rollout',type=Path)
    p.add_argument('--selection',type=Path);p.add_argument('--resolve',action='store_true');p.add_argument('--solve-batch',type=int,default=0);p.add_argument('--cases',type=int,default=16);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
