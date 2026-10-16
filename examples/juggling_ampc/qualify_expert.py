"""Qualify sustained juggling against the unchanged native MPC in the same plant.

The empirical source's native controller is itself failure prone. Qualification
therefore uses paired native uptime and repeated contacts, not an invented 80%
30-second survival requirement. All failure rates remain visible in the report.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from uncertainty_settings import require_matching,from_report


def compatible(report, certificate):
    require_matching(report,certificate)
    for key in ('calibration_sha256', 'noise_disturbance_multiplier', 'drop_bounds'):
        if report[key] != certificate[key]:
            raise ValueError('Expert qualification mismatch: ' + key)


def qualifies(report, certificate, spec):
    if certificate.get('reuse_completed_gates'):
        # The user explicitly accepted the completed native/Warp gates. Do not
        # rerun them per device or uncertainty setting. Observe basic sustained
        # juggling directly in the collection/evaluation that is already needed.
        if certificate['mpc_spec']!=spec or certificate['calibration_sha256']!=report['calibration_sha256']:
            raise ValueError('Completed gates belong to a different MPC or fitted plant')
        return bool(certificate['passed'] and report['policy_kind']=='warp_mpc'
                    and report['mean_uptime_s']>=min(3.,report['rollout_seconds'])
                    and np.median(report['contacts_before_loss'])>=3)
    compatible(report, certificate)
    if report['policy_kind']=='warp_mpc':
        for key in ('solver_numerics','implementation_sha256'):
            if key in certificate and report.get(key)!=certificate[key]:
                raise ValueError('Expert implementation differs from qualification: '+key)
    if certificate['mpc_spec'] != spec:
        raise ValueError('Expert qualification belongs to a different MPC specification')
    duration=report['rollout_seconds']
    if duration not in (5,10,20,30):raise ValueError('Unsupported qualified duration')
    if duration!=30 and 'warp_uptime_s' not in certificate:raise ValueError('Short collection requires the censored baseline uptime distribution')
    baseline=float(np.minimum(certificate['warp_uptime_s'],duration).mean()) if duration!=30 else certificate['warp_mean_uptime_s']
    contacts=np.asarray(report['contacts_before_loss'])
    return bool(certificate['passed'] and report['policy_kind']=='warp_mpc'
                and len(report['uptime_s'])>=100
                and report['mean_uptime_s']>=.9*baseline
                and np.median(contacts)>=3)


def qualify(a):
    if not 0<a.minimum_ratio<=1:raise ValueError('Native uptime ratio must be in (0, 1]')
    native=json.loads(a.native.read_text());warp=json.loads(a.warp.read_text())
    for key in ('seeds','calibration_sha256','noise_disturbance_multiplier','drop_bounds','rollout_seconds'):
        if native[key]!=warp[key]:raise ValueError('Unpaired native/Warp qualification: '+key)
    require_matching(native,warp)
    if native['policy_kind']!='native_diagnostic' or warp['policy_kind']!='warp_mpc':
        raise ValueError('Need an independent native baseline and a Warp MPC rollout')
    n=np.asarray(native['uptime_s']);w=np.asarray(warp['uptime_s'])
    if n.shape!=w.shape or len(n)<100 or warp['rollout_seconds']!=30:
        raise ValueError('Qualification requires at least 100 paired 30-second trials')
    if not np.isfinite(np.r_[n,w]).all() or np.any(np.r_[n,w]<0) or np.any(np.r_[n,w]>30):
        raise ValueError('Invalid observed uptime')
    inflight=json.loads(a.inflight_parity.read_text())
    if not inflight['passed'] or len(inflight.get('cases',[]))<16:
        raise ValueError('At least 16 independent in-flight MPC parity cases must pass')
    parity=json.loads(a.parity.read_text())
    if not parity['passed']:raise ValueError('Independent full MPC parity has not passed')
    rng=np.random.default_rng(4816)
    margin=w-a.minimum_ratio*n
    boot=np.array([margin[rng.integers(len(n),size=len(n))].mean() for _ in range(5000)])
    interval=np.quantile(boot,[.025,.975])
    native_contacts=float(np.median(native['contacts_before_loss']))
    warp_contacts=float(np.median(warp['contacts_before_loss']))
    result={k:warp[k] for k in ('calibration_sha256','noise_disturbance_multiplier','drop_bounds','rollout_seconds')}
    result['uncertainty_multipliers']=from_report(warp)
    result.update(mpc_spec=json.loads(a.metadata.read_text())['spec'],n=len(n),
        native_mean_uptime_s=float(n.mean()),warp_mean_uptime_s=float(w.mean()),warp_uptime_s=w.tolist(),
        native_survival_30s=float(np.mean(n>=30)),warp_survival_30s=float(np.mean(w>=30)),
        native_median_contacts_before_loss=native_contacts,warp_median_contacts_before_loss=warp_contacts,
        minimum_native_uptime_ratio=a.minimum_ratio,paired_margin_95ci_s=interval.tolist(),
        passed=bool(interval[0]>=0 and native_contacts>=3 and warp_contacts>=3 and n.mean()>=3 and w.mean()>=3),
        interpretation='Sustained repeated juggling and native-relative uptime; not guaranteed 30-second survival',
        evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.native,a.warp,a.parity,a.inflight_parity,a.metadata)})
    for key in ('solver_numerics','implementation_sha256'):
        if key in warp:result[key]=warp[key]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--warp',type=Path,required=True)
    p.add_argument('--inflight-parity',type=Path,required=True);p.add_argument('--parity',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--minimum-ratio',type=float,default=.9)
    raise SystemExit(0 if qualify(p.parse_args())['passed'] else 1)
