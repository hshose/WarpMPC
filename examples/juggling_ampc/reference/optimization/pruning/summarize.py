"""Summarize physical tests and independently check omitted prediction rows."""
import hashlib,json
from pathlib import Path
import casadi as ca
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    reference=Path('runs/20260924T175306.001124Z_cpp_final_wind/native')
    metadata=json.loads((reference/'factor_metadata.json').read_text());a=metadata['arrays']
    factor=ca.external('stage_value',str((reference/'libjuggling_factors.so').resolve()))
    lower=np.array(a['stage_lower']);upper=np.array(a['stage_upper'])
    results=[]
    for directory in sorted(Path('runs').glob('*prun*')):
        if not (directory/'metrics.json').exists() or not (directory/'pruning.json').exists():continue
        cfg=json.loads((directory/'config.json').read_text());m=json.loads((directory/'metrics.json').read_text())
        options=json.loads((directory/'pruning.json').read_text());prefix=options['prefix_nodes']
        Y=np.load(directory/'predictions.npz')['Y'];Z=(Y-np.array(a['offset']))/np.array(a['scale'])
        C=np.array([np.asarray(factor(np.r_[z[k],z[k+1]],np.zeros(30))[1]).ravel()
            for z in Z for k in range(Y.shape[1]-1)]).reshape(len(Y),Y.shape[1]-1,-1)
        slack=np.minimum(C-lower,upper-C)
        groups={}
        for name,(start,end) in metadata['groups'].items():
            if name in ('ball','spin','position','dynamics','clock'):continue
            s=slack[:,:,start:end]
            groups[name]={label:dict(minimum_margin=float(x.min()),maximum_violation=float(max(0.,-x.min())),
                fraction_plan_stages_within_1pct=float(np.mean(np.any(x<=.01,axis=2))))
                for label,x in [('prefix',s[:,:prefix]),('tail',s[:,prefix:])]}
        (directory/'pruned_constraint_audit.json').write_text(json.dumps(dict(
            baseline_reference_library=str(reference/'libjuggling_factors.so'),
            baseline_library_sha256=hashlib.sha256((reference/'libjuggling_factors.so').read_bytes()).hexdigest(),
            sampled_predictions=len(Y),groups=groups),indent=2)+'\n')
        apices=np.array(m['measured_apices_m'])
        results.append(dict(run=str(directory),name=cfg['arguments']['name'],options=options,
            duration_s=m['duration_s'],horizon_nodes=cfg['mpc']['horizon'],wind=cfg['arguments']['wind'],joint_damping=cfg['arguments']['damping'],
            task_pass=m['task_pass'],contacts=m['contacts'],limits=m['limit_ratios'],
            apex_min_m=float(apices.min()),apex_max_m=float(apices.max()),
            apex_rmse_m=float(np.sqrt(np.mean((apices-.75)**2))),
            max_lateral_distance_m=m['max_lateral_distance_m'],
            solve_median_ms=m['solve_median_ms'],data_median_ms=m['data_median_ms'],qp_median_ms=m['qp_median_ms'],
            other_contact_samples=m['other_contact_samples'],max_torque_clipping_Nm=m['max_torque_clipping_Nm'],
            warmstart_fallback_count=m['warmstart_fallback_count'],prediction_constraint_audit=groups))
    Path('optimization/pruning/results.json').write_text(json.dumps(results,indent=2)+'\n')
    labels=[r['name'].replace('prun_','').replace('_','\n') for r in results];x=np.arange(len(results))
    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    ax=axes[0,0]
    for i,r in enumerate(results):
        ax.plot([i,i],[r['apex_min_m'],r['apex_max_m']],lw=4,color='C0' if r['task_pass'] else 'C3')
    ax.axhline(.75,ls=':',color='k');ax.set_ylabel('Measured apex range [m]');ax.set_ylim(.5,1.)
    ax=axes[0,1]
    for j,key in enumerate(['velocity','acceleration','torque_rate','jerk']):
        ax.plot(x,[r['limits'][key] for r in results],'.-',label=key)
    ax.axhline(1,ls=':',color='k');ax.set_ylabel('Maximum measured limit ratio');ax.legend()
    ax=axes[1,0];ax.bar(x,[r['apex_rmse_m']*1000 for r in results]);ax.set_ylabel('Apex RMSE about 0.75 m [mm]')
    ax=axes[1,1]
    ax.bar(x,[r['data_median_ms'] for r in results],label='Python generated data')
    ax.bar(x,[r['qp_median_ms'] for r in results],bottom=[r['data_median_ms'] for r in results],label='OSQP update + solve')
    ax.set_ylabel('Python prototype component median [ms]');ax.legend()
    for ax in axes.ravel():ax.set_xticks(x,labels,fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Pruning trials (uniform 40 nodes and geometric 24 nodes): 0.5 ms Newton, position-only observer, 1 SQP / 50 OSQP\n'
        'Python timings are screening measurements, not optimized native benchmarks')
    fig.savefig('optimization/pruning/report.pdf');plt.close(fig)
    print(json.dumps([{k:r[k] for k in ['name','duration_s','task_pass','contacts','apex_min_m','apex_max_m','apex_rmse_m','limits']} for r in results],indent=2))


if __name__=='__main__':main()
