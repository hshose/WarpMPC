"""Build a complete trial inventory and one-page selected-run comparison."""
import argparse,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def incomplete_plot(directory):
    rows=[json.loads(s) for s in (directory/'iterations.jsonl').read_text().splitlines()]
    t=[r['t'] for r in rows];fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    values=[('total_ms','MPC time [ms]'),('violation','Scaled constraint defect'),('bound_violation','Scaled bound defect'),
            ('torque_clipping_Nm','Torque limiting [Nm]')]
    for ax,(key,label) in zip(axes.flat,values):ax.plot(t,[r[key] for r in rows]);ax.set_ylabel(label)
    axes[1,1].plot(t,[r['iterations'][0]['alpha'] for r in rows]);axes[1,1].set_ylabel('SQP step size')
    axes[1,2].plot(t,[r['iterations'][0]['iter'] for r in rows]);axes[1,2].set_ylabel('OSQP iterations')
    for ax in axes.flat:ax.set_xlabel('Recorded simulation time [s]');ax.grid(alpha=.2)
    fig.suptitle(directory.name+'\nINCOMPLETE EXPORT: controller logs only; no physical trace or success claim')
    fig.savefig(directory/'incomplete_report.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('selected',nargs='+',type=Path);args=parser.parse_args()
    rows=[];incomplete=[]
    for directory in sorted(Path('runs').iterdir()):
        if not (directory/'config.json').exists():continue
        cfg=json.loads((directory/'config.json').read_text())
        if 'engine' not in cfg:continue
        if not (directory/'metrics.json').exists():
            if (directory/'iterations.jsonl').exists():incomplete_plot(directory);incomplete.append(str(directory))
            continue
        m=json.loads((directory/'metrics.json').read_text());peaks=m['measured_apices_m']
        rows.append(dict(run=str(directory),name=cfg['arguments']['name'],seconds=m['duration_s'],contacts=m['contacts'],
            task_pass=m['task_pass'],apex_range_m=[min(peaks),max(peaks)] if peaks else None,
            lateral_cm=100*m['max_lateral_distance_m'],mpc_ms=m['solve_median_ms'],codegen=cfg['mpc']['codegen'],
            limit_ratios=m['limit_ratios'],clipping_Nm=m['max_torque_clipping_Nm'],other_contacts=m['other_contact_samples'],
            prediction_defect=m['max_violation'],prediction_bound_defect=m['max_bound_violation'],
            fallback_count=m.get('warmstart_fallback_count',0),extended_updates=m.get('extended_update_intervals',0),
            max_update_s=m.get('max_update_interval_s',cfg['mpc']['control_dt']),
            contact_model=cfg['arguments'].get('contact_model','normal'),wind=cfg['arguments']['wind'],
            source_revision=cfg['git_revision']))
    result=dict(selected=[str(p) for p in args.selected],trials=rows,incomplete_exports=incomplete)
    Path('docs/stage3_trial_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    table=['# Stage 3 trial inventory','',
        'All completed Newton trials are retained, including failed controllers and superseded contact assumptions. '
        'A pass uses actual traces and the documented actuated-motion criteria; it does not imply that contact-inclusive acceleration/jerk stays below motion-planning bounds.','',
        '| Trial | Duration / impacts | Apex range [m] | Max lateral [cm] | MPC median [ms] | Result |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        peaks=r['apex_range_m'];apex=f'{peaks[0]:.3f}–{peaks[1]:.3f}' if peaks else '—'
        table.append(f"| [{r['name']}](../{r['run']}/report.pdf) | {r['seconds']:.3f} s / {r['contacts']} | {apex} | {r['lateral_cm']:.2f} | {r['mpc_ms']:.1f}{' compiled' if r['codegen'] else ' interpreted'} | {'PASS' if r['task_pass'] else 'FAIL'} |")
    if incomplete:
        table+=['','Incomplete exports have controller-log PDFs only; their physical results are unavailable:','']
        table += [f'- [{Path(p).name}](../{p}/incomplete_report.pdf)' for p in incomplete]
    Path('docs/stage3_trial_inventory.md').write_text('\n'.join(table)+'\n')
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for directory in args.selected:
        cfg=json.loads((directory/'config.json').read_text());m=json.loads((directory/'metrics.json').read_text());p=np.load(directory/'plant.npz');a=np.load(directory/'arrays.npz')
        label=cfg['arguments']['name'];step=max(1,len(p['t'])//4000);t=p['t'][::step]
        axes[0,0].plot(m['measured_apex_times_s'],m['measured_apices_m'],'.-',ms=3,label=label)
        axes[0,1].plot(t,100*np.linalg.norm(p['x'][::step,:2],axis=1),label=label)
        axes[1,0].plot(p['t'][:-1:step],np.linalg.norm(p['spin'][::step],axis=1),label=label)
        axes[1,1].plot(a['t'][:-1],a['solve_ms'],lw=.6,label=label)
        axes[2,0].plot(a['t'][:-1],a['violation'],lw=.6,label=label)
        keys=['velocity','torque','acceleration','torque_rate','jerk'];axes[2,1].plot(keys,[m['limit_ratios'][k] for k in keys],'.-',label=label)
    axes[0,0].axhspan(.5,1,alpha=.08,color='green');axes[0,0].axhline(.75,ls=':',c='black')
    axes[2,0].set_yscale('symlog',linthresh=1e-5);axes[2,1].axhline(1,ls=':',c='black');axes[2,1].tick_params(axis='x',rotation=20)
    for ax,label in zip(axes.flat,['Measured apex [m]','Lateral distance [cm]','Ball spin [rad/s]',
                                 'MPC wall time [ms]','Scaled prediction defect','Maximum actuated-motion limit ratio']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
        if ax is not axes[2,1]:ax.set_xlabel('Simulated time [s]')
    axes[0,0].legend(fontsize=6)
    fig.suptitle('Newton full-arm juggling · 0.5 ms physics · 1 SQP / 50 OSQP\nSimulated time pauses during computation; finite compliant contact and friction/spin retained')
    fig.savefig('docs/stage3_comparison.pdf');plt.close(fig)
    print(f'{len(rows)} completed trials and {len(incomplete)} incomplete exports catalogued.')


if __name__=='__main__':main()
