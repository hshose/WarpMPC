"""Paired closed-loop uptime comparison; regression loss alone cannot pass this gate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from uncertainty_settings import require_matching,from_report

def compare(expert,student,checkpoint,output,minimum_ratio=.95):
    reference=json.loads(expert.read_text());learned=json.loads(student.read_text())
    for field in ('seeds','calibration_sha256','noise_disturbance_multiplier','rollout_seconds'):
        if reference[field]!=learned[field]:raise ValueError('Paired evaluation mismatch: '+field)
    require_matching(reference,learned)
    if reference['rollout_seconds']!=30.:raise ValueError('Qualification requires 30-second rollouts')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if learned['checkpoint_sha256']!=digest:raise ValueError('Evaluation belongs to another checkpoint')
    a=np.asarray(reference['uptime_s'],float);b=np.asarray(learned['uptime_s'],float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<100 or len(a)!=len(reference['seeds']):raise ValueError('Need at least 100 paired held-out trajectories')
    if not np.isfinite(np.r_[a,b]).all() or np.any(np.r_[a,b]<0) or np.any(np.r_[a,b]>30):raise ValueError('Invalid uptime data')
    rng=np.random.default_rng(812);boot=[]
    for _ in range(2000):
        ix=rng.integers(len(a),size=len(a));boot.append(float(np.mean(b[ix]-minimum_ratio*a[ix])))
    lower,upper=np.quantile(boot,[.025,.975]);ratio=float(np.mean(b)/max(np.mean(a),1e-12))
    result=dict(checkpoint_sha256=digest,calibration_sha256=reference['calibration_sha256'],expert_mean_uptime_s=float(a.mean()),policy_mean_uptime_s=float(b.mean()),mean_uptime_ratio=ratio,
        expert_survival_30s=float(np.mean(a>=30.)),policy_survival_30s=float(np.mean(b>=30.)),paired_margin_95ci_s=[float(lower),float(upper)],minimum_ratio=minimum_ratio,
        passed=bool(reference.get('expert_stabilization_passed') and ratio>=minimum_ratio and lower>=0),n=len(a))
    result['uncertainty_multipliers']=from_report(reference)
    output.mkdir(parents=True,exist_ok=True);(output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    marker=output/'closed_loop_passed.json'
    if result['passed']:marker.write_text(json.dumps(result,indent=2)+'\n')
    else:marker.unlink(missing_ok=True)
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('expert',type=Path);p.add_argument('student',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=compare(a.expert,a.student,a.checkpoint,a.output);raise SystemExit(0 if r['passed'] else 1)
