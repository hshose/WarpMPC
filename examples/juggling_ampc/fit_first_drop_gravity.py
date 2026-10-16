"""Fit only raw Vicon positions between release and the first paddle impact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def fit(flights, fixed=None):
    count=len(flights); rows=[]; targets=[]
    for i,f in enumerate(flights):
        t=f['t'];d=f['drag'];a=-np.expm1(-d*t)/d if d else t;b=(t-a)/d if d else .5*t*t
        D=np.zeros((len(t),2*count+(1 if fixed is None else 0)))
        D[:,2*i]=1;D[:,2*i+1]=a
        if fixed is None:D[:,-1]=b
        y=f['p'] if fixed is None else f['p']-b[:,None]*fixed
        rows.append(D/np.sqrt(len(t)));targets.append(y/np.sqrt(len(t)))
    D=np.vstack(rows);y=np.vstack(targets);initial=np.linalg.lstsq(D,y,rcond=None)[0]
    result=least_squares(lambda z:(D@z.reshape(initial.shape)-y).ravel(),initial.ravel(),loss='soft_l1',f_scale=.00015,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    beta=result.x.reshape(initial.shape);g=beta[-1] if fixed is None else np.asarray(fixed)
    residuals=[(d@beta-y)*np.sqrt(len(f['t'])) for d,y,f in zip(rows,targets,flights)]
    return g,residuals,bool(result.success)


def main(paths,output):
    flights=[];records=[];prior=None;hashes={}
    for p in paths:
        def read(name):
            data=(p/name).read_bytes();hashes[str(p/name)]=hashlib.sha256(data).hexdigest();return json.loads(data)
        cfg=read('configuration.json');meta=read('model_metadata.json');release=read('release.json');observer=read('observer.json');B=np.asarray(read('world_alignment.json')['B_T_V'])
        current=np.array(read('effective_gravity.json')['gravity_m_s2']) if (p/'effective_gravity.json').exists() else np.array([meta['config']['gravity_'+a] for a in 'xyz'])
        if prior is None:prior=current
        if not np.allclose(current,prior,rtol=0,atol=1e-12):raise ValueError('Trials have different prior gravity')
        if meta['config'].get('quadratic_drag',meta['config'].get('drag_quadratic',0.)):raise ValueError('Expected deployed linear-drag model')
        start=release['capture_time'];impact=min(t for t in observer['impacts'] if t>start)
        plans=[json.loads(s) for s in (p/'mpc.jsonl').read_text().splitlines()];pt=np.array([v['time'] for v in plans]);px=np.array([v['state'][:3] for v in plans])+meta['arrays']['origin']
        rows=[];seen=set();raw=(p/'vicon.jsonl').read_bytes();hashes[str(p/'vicon.jsonl')]=hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            v=json.loads(line);t=v.get('capture_time_s',v['receive_time']-v['latency_s'])-cfg['extra_latency_s']
            if not start+.02<t<impact-.04 or v['frame'] in seen:continue
            seen.add(v['frame']);guess=np.array([np.interp(t,pt,px[:,j]) for j in range(3)])
            candidates=[]
            for marker in v['unlabeled_markers']:
                if marker.get('position_m') is None:continue
                position=B[:3,:3]@marker['position_m']+B[:3,3];candidates.append((np.linalg.norm(position-guess),position))
            candidates.sort(key=lambda c:c[0])
            if not candidates or candidates[0][0]>.15:continue
            if len(candidates)>1 and candidates[1][0]-candidates[0][0]<.025:continue
            rows.append([t,*candidates[0][1]])
        raw=np.asarray(sorted(rows,key=lambda r:r[0]));
        if len(raw)<16 or np.ptp(raw[:,0])<.22:raise ValueError(f'Insufficient initial flight in {p}')
        flight=dict(t=raw[:,0]-raw[0,0],p=raw[:,1:4],drag=meta['config']['drag']);flights.append(flight)
        single,_,_=fit([flight]);records.append(dict(run=str(p),release_time_s=start,first_impact_time_s=impact,first_used_time_s=float(raw[0,0]),last_used_time_s=float(raw[-1,0]),samples=len(raw),gravity_individual_m_s2=single.tolist(),drag_s_inv=flight['drag']))
    g,residuals,success=fit(flights);_,old_residuals,_=fit(flights,prior)
    for record,res,old in zip(records,residuals,old_residuals):record.update(position_rms_m=float(np.sqrt(np.mean(res**2))),previous_gravity_position_rms_m=float(np.sqrt(np.mean(old**2))))
    report=dict(schema_version=1,frame='robot_base',gravity_m_s2=g.tolist(),previous_gravity_m_s2=prior.tolist(),difference_m_s2=(g-prior).tolist(),direction_change_deg=float(np.rad2deg(np.arccos(np.clip(g@prior/(np.linalg.norm(g)*np.linalg.norm(prior)),-1,1)))),success=success,trials=records,input_sha256=hashes,fit='Equal weight per drop; robust fixed-linear-drag position fit with independent initial position and velocity',window='20 ms after detected release to 40 ms before first observer impact; no impact or outgoing-flight samples',unchanged=['paddle-to-arm calibration','paddle normal','drag','neural-network weights','robot-base input frame'],policy_note='Gravity override updates observer, release and landing prediction; neural policy retains its learned dynamics. No ball-only frame rotation is applied.')
    if not success or not np.isfinite(g).all():raise ValueError('Gravity fit did not converge')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs=3,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.runs,a.output)
