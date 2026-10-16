"""Compare offline uncertainty batches without touching controllers or hardware.

Example: python -m tracking.empirical_report --output runs/comparison \
    runs/baseline runs/center_weight_16 runs/horizon_24

Uptime is the restricted observed duration, capped at the requested horizon.
Paired bootstrap intervals describe seed sensitivity; they are not guarantees
about physical hardware or probabilities of success on the real robot.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path):return json.loads(Path(path).read_text())
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def physical_hash(model):
    disturbance=dict(model.get('uncertainty',{}))
    # Candidate solve-time benchmarks legitimately change total delivery delay.
    disturbance.pop('plan_delay_samples_s',None)
    content=dict(plant=model.get('plant'),uncertainty=disturbance,
                 initial_conditions=model.get('initial_conditions'),vicon_rate_hz=model.get('vicon_rate_hz',100.))
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


def summarize(values):
    values=np.asarray(values,float)
    if not len(values):return dict(n=0)
    return dict(n=len(values),mean=np.mean(values,axis=0).tolist(),median=np.median(values,axis=0).tolist(),
                p05=np.quantile(values,.05,axis=0).tolist(),p95=np.quantile(values,.95,axis=0).tolist())


def completion_interval(completed,trials):
    # Descriptive Monte Carlo interval, conditional on this fitted simulator.
    z=1.959963984540054;p=completed/trials;denominator=1+z*z/trials
    center=(p+z*z/(2*trials))/denominator
    half=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return dict(fraction=p,wilson_95=[max(0.,center-half),min(1.,center+half)],
                interpretation='Finite-seed simulation frequency; not a physical-robot success probability.')


def survival_curve(records):
    """Kaplan–Meier display: completed horizons are censored, not failures."""
    rows=list(records.values());times=sorted({r['survival_s'] for r in rows})
    x=[0.];y=[1.];survival=1.
    for t in times:
        at_risk=sum(r['survival_s']>=t for r in rows)
        failed=sum(r['survival_s']==t and not r['completed'] for r in rows)
        if at_risk:survival*=1-failed/at_risk
        x.append(t);y.append(survival)
    return x,y


def read_batch(path):
    path=Path(path);model=load(path/'empirical_model_used.json');records={};contacts={}
    provenance=load(path/'provenance.json')
    for folder in sorted(path.glob('seed_*')):
        if not (folder/'summary.json').exists():continue
        record=load(folder/'summary.json');seed=record['seed'];records[seed]=record
        contacts[seed]=load(folder/'contacts.json') if (folder/'contacts.json').exists() else []
    if not records:raise ValueError('No finished trial artifacts in '+str(path))
    hits=[c for seed in records for c in contacts[seed] if c['hit']]
    strike_errors=[]
    for seed,events in contacts.items():
        # New summaries already contain per-trial strike-height statistics; the
        # raw base-Z distribution remains useful without guessing the origin.
        strike_errors.extend(c['face_base_m'][2] for c in events if c['hit'])
    completed=sum(r['completed'] for r in records.values())
    strata={}
    for record in records.values():
        key=record['initial_condition']['recording'];strata.setdefault(key,[]).append(record)
    return dict(path=str(path.resolve()),label=path.name,model=model,physical_model_sha256=physical_hash(model),
        execution_semantics=provenance.get('plan_delivery_order','legacy delivery-before-advance'),
        supervisor_overrides=model.get('supervisor_overrides',{}),
        records=records,contacts=contacts,summary=dict(trials=len(records),completed=completed,
        completion=completion_interval(completed,len(records)),
        restricted_uptime_s=summarize([r['survival_s'] for r in records.values()]),
        requested_horizons_s=sorted({r['requested_seconds'] for r in records.values()}),
        failure_counts=dict(Counter(r['failure'] or 'completed' for r in records.values())),
        impact_counts=summarize([r['impacts'] for r in records.values()]),
        pooled_contact_xy_local_m=summarize([c['local_contact_m'][:2] for c in hits]),
        pooled_contact_radius_m=summarize([np.linalg.norm(c['local_contact_m'][:2]) for c in hits]),
        pooled_strike_face_base_z_m=summarize(strike_errors),
        equal_trial_contact_xy_local_m=summarize([r['contact_xy_local_m']['mean'] for r in records.values() if r['contact_xy_local_m']['n']]),
        equal_trial_mean_strike_height_error_m=summarize([r['strike_height_error_m']['mean'] for r in records.values() if r['strike_height_error_m']['n']]),
        equal_trial_mean_apex_m=summarize([r['apex_above_nominal_plane_m']['mean'] for r in records.values() if r['apex_above_nominal_plane_m']['n']]),
        by_initial_fixture={key:dict(trials=len(rows),completed=sum(r['completed'] for r in rows),
            restricted_uptime_s=summarize([r['survival_s'] for r in rows])) for key,rows in strata.items()}))


def compare(baseline,candidate):
    seeds=sorted(set(baseline['records']) & set(candidate['records']))
    if not seeds:return dict(common_seeds=[],paired_comparison_available=False)
    differences=np.array([candidate['records'][s]['survival_s']-baseline['records'][s]['survival_s'] for s in seeds])
    rng=np.random.default_rng(7341);boot=differences[rng.integers(len(seeds),size=(5000,len(seeds)))].mean(axis=1)
    horizons_match=all(baseline['records'][s]['requested_seconds']==candidate['records'][s]['requested_seconds'] for s in seeds)
    physics_match=baseline['physical_model_sha256']==candidate['physical_model_sha256']
    return dict(common_seeds=seeds,paired_comparison_available=True,physical_model_matches=physics_match,
        execution_semantics_match=baseline['execution_semantics']==candidate['execution_semantics'],
        supervisor_settings_match=baseline['supervisor_overrides']==candidate['supervisor_overrides'],
        requested_horizons_match=horizons_match,paired_improved=int(np.sum(differences>1e-9)),
        paired_unchanged=int(np.sum(abs(differences)<=1e-9)),paired_worse=int(np.sum(differences< -1e-9)),
        paired_mean_uptime_difference_s=float(differences.mean()),
        paired_bootstrap_95_interval_s=np.quantile(boot,[.025,.975]).tolist(),
        caveat='Finite Monte Carlo seed sensitivity, not a hardware validation or success guarantee.')


def plot(batches,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
    colors=plt.get_cmap('tab10').colors
    shared=sorted(set.intersection(*(set(b['records']) for b in batches)))
    for index,batch in enumerate(batches):
        color=colors[index%len(colors)];records=batch['records'];label=f"{batch['label']} (n={len(records)})"
        sx,sy=survival_curve(records)
        axes[0,0].step(sx,sy,where='post',label=label,color=color)
        if shared:axes[0,1].plot(shared,[records[s]['survival_s'] for s in shared],'.-',color=color,label=label)
        hits=[c for seed in records for c in batch['contacts'][seed] if c['hit']]
        if hits:
            xy=np.array([c['local_contact_m'][:2] for c in hits])*1000
            axes[1,0].scatter(xy[:,0],xy[:,1],s=7,alpha=.3,color=color)
            axes[1,1].hist(xy[:,0],bins=np.linspace(-85,85,35),histtype='step',density=True,color=color,label=label)
            axes[2,0].hist([c['face_base_m'][2] for c in hits],bins=25,histtype='step',density=True,color=color)
    axes[0,0].set(xlabel='Observed uptime [s]',ylabel='Fraction reaching this time',ylim=(0,1.03))
    axes[0,0].legend(fontsize=7)
    axes[0,1].set(xlabel='Common random seed',ylabel='Observed uptime [s]')
    theta=np.linspace(0,2*np.pi,200)
    for index,batch in enumerate(batches):
        radii=np.asarray(batch['model']['plant'].get('paddle_axes_m',[.07,.07]))*1000
        axes[1,0].plot(radii[0]*np.cos(theta),radii[1]*np.sin(theta),color=colors[index%len(colors)],alpha=.7,lw=.8)
    axes[1,0].set(xlabel='Paddle-local contact X [mm]',ylabel='Paddle-local contact Y [mm]',aspect='equal')
    axes[1,1].set(xlabel='Paddle-local contact X [mm]',ylabel='Hit-weighted density')
    axes[2,0].set(xlabel='Physical face base Z at contact [m]',ylabel='Hit-weighted density')
    reasons=sorted(set().union(*(set(b['summary']['failure_counts']) for b in batches)))
    bottom=np.zeros(len(batches));labels=[b['label'] for b in batches]
    for reason in reasons:
        counts=np.array([b['summary']['failure_counts'].get(reason,0) for b in batches]);axes[2,1].barh(labels,counts,left=bottom,label=reason)
        bottom+=counts
    axes[2,1].set(xlabel='Trials');axes[2,1].legend(fontsize=6,loc='lower right')
    for axis in axes.flat:axis.grid(alpha=.2)
    fig.suptitle('Empirical offline robustness comparison\nCompleted horizons are censored; candidate improvements require later hardware validation')
    fig.savefig(output/'comparison.pdf');plt.close(fig)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);batches=[read_batch(path) for path in args.batches]
    results=[]
    for batch in batches:
        report={key:batch[key] for key in ('path','label','physical_model_sha256','summary')}
        report['comparison_to_first']=compare(batches[0],batch);results.append(report)
    save(args.output/'comparison.json',dict(physical_robot_used=False,batches=results,
        interpretation='Restricted mean uptime uses the configured maximum duration. Compare matching model hashes and common seeds.'))
    plot(batches,args.output)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batches',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
