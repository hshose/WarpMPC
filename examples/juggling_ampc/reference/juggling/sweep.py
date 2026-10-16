"""Sequential, reproducible fixed-budget comparisons with persistent smooth wind."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def summarize(run):
    m=json.loads((run/'metrics.json').read_text());cfg=json.loads((run/'config.json').read_text())
    a=np.load(run/'arrays.npz');u=a['u'];t=a['t']
    physical=np.load(run/'plant.npz') if (run/'plant.npz').exists() else a
    x=physical['x']
    logs=[json.loads(line) for line in (run/'iterations.jsonl').read_text().splitlines()]
    budget=cfg['mpc']['osqp_iters'];n=len(t)-1
    assert n==len(logs)
    assert all(len(l['iterations'])==1 and l['iterations'][0]['iter']==budget for l in logs)
    assert cfg['mpc']['mass']==cfg['sim']['ball_mass']==.0027
    assert cfg['mpc']['radius']==cfg['sim']['ball_radius']==.02
    assert cfg['mpc']['paddle_radius']==cfg['sim']['paddle_radius']==.075
    wind=physical['wind_accel'];assert np.min(np.linalg.norm(wind,axis=1))>0
    pose=np.array([.4,.4,.18,.4,.4]);vel=np.array([1.5,1.5,1.5,3.,3.])
    limits_ok=bool(np.max(np.abs(x[:,6:11])-pose)<1e-6 and np.max(np.abs(x[:,11:16])-vel)<1e-6)
    peaks=m['measured_apices_m']
    task_pass=bool(m['stop_reason']=='duration completed' and len(peaks)>=8 and
        m['measured_apex_band_fraction']==1. and limits_ok and m['max_control_clipping']<=1e-6)
    tail=[l for l in logs if l['t']>=1.]
    qps=[l['iterations'][0] for l in logs]
    contacts=json.loads((run/'contacts.json').read_text())
    return dict(run=str(run),mode='variable' if cfg['mpc']['variable_time'] else 'fixed',
        osqp_iterations=budget,sqp_iterations=1,task_pass=task_pass,actual_state_limits_ok=limits_ok,
        completed_apices=len(peaks),apex_min_m=min(peaks) if peaks else None,
        apex_max_m=max(peaks) if peaks else None,
        max_contact_radius_m=max((c['radial_distance'] for c in contacts),default=None),
        max_jerk_m_s3=float(np.max(np.abs(np.diff(u[:,:3],axis=0)))/cfg.get('timing',{}).get('control_dt_s',cfg['mpc']['h_min'])),
        max_stationarity_residual=max(q['dual_res'] for q in qps),
        line_search_acceptance_fraction=float(np.mean([q['alpha']>0 for q in qps])),
        post_1s_hard_violation=max((l['violation'] for l in tail),default=None),
        post_1s_contact_gap_m=max((l['contact_gap_active_m'] for l in tail),default=None),
        post_1s_restitution_defect_m_s=max((l['restitution_active_m_s'] for l in tail),default=None),
        **m)


def report(directory,rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (directory/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,ax=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    metrics=[('solve_median_ms','Controller median / 95th percentile [ms]'),('apex_min_m','Measured apex range [m]'),
             ('max_lateral_distance_m','Maximum ball lateral excursion [m]'),
             ('max_paddle_speed_m_s','Maximum paddle speed [m/s]'),
             ('post_1s_hard_violation','Maximum scaled hard violation after 1 s'),
             ('post_1s_contact_gap_m','Maximum active contact gap after 1 s [m]')]
    for mode,marker in [('fixed','o'),('variable','s')]:
        r=sorted([r for r in rows if r['mode']==mode],key=lambda r:r['osqp_iterations'])
        if not r:continue
        for axis,(key,title) in zip(ax.flat,metrics):
            budgets=np.array([v['osqp_iterations'] for v in r])
            if key=='apex_min_m':
                low=np.array([v['apex_min_m'] for v in r]);high=np.array([v['apex_max_m'] for v in r])
                axis.errorbar(budgets+(-1 if mode=='fixed' else 1),(low+high)/2,
                              yerr=(high-low)/2,fmt=marker+'-',capsize=4,label=mode)
            else:
                line,=axis.plot(budgets,[v[key] for v in r],marker=marker,label=mode)
                if key=='solve_median_ms':
                    axis.fill_between(budgets,[v[key] for v in r],[v['solve_p95_ms'] for v in r],
                                      color=line.get_color(),alpha=.15)
            axis.set(xlabel='Fixed OSQP iterations',ylabel=title)
            axis.set_xticks([25,40,50,60,80,100])
            axis.grid(alpha=.25);axis.legend()
    ax[0,0].axhline(10,color='black',ls=':',label='10 ms MPC update')
    ax[0,0].legend()
    ax[0,1].axhspan(.5,1.,color='green',alpha=.08,label='Target band')
    ax[0,1].legend()
    fig.suptitle('One SQP iteration · standard equipment · persistent smooth wind\n'
                 'Sequential 10-second trials; original traces and individual PDFs linked in summary.md')
    fig.savefig(directory/'report.pdf');plt.close(fig)
    lines=['# One-SQP iteration budget sweep','',
           'Each run uses the same deterministic smooth wind, 2.7 g / 40 mm ball, 75 mm-radius paddle, and 10 ms MPC updates. Plant timesteps are recorded in each configuration.',
           'Task pass requires full duration, at least eight completed flights, all measured apices in [0.5,1] m, actual paddle state limits, and no acceleration clipping. It does not claim an exactly feasible NLP.','',
           '| Mode | OSQP | Task pass | Impacts | Apex range [m] | Median / p95 [ms] | Max hard violation | PDF |',
           '| --- | ---: | --- | ---: | --- | --- | ---: | --- |']
    for r in rows:
        ap='none' if r['apex_min_m'] is None else f"{r['apex_min_m']:.3f}–{r['apex_max_m']:.3f}"
        link=Path('..')/Path(r['run']).name/'report.pdf'
        lines.append(f"| {r['mode']} | {r['osqp_iterations']} | {r['task_pass']} | {r['contacts']} | {ap} | {r['solve_median_ms']:.2f} / {r['solve_p95_ms']:.2f} | {r['max_violation']:.3g} | [plot]({link}) |")
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--iterations',nargs='+',type=int,default=[25,40,50,60,80,100])
    p.add_argument('--modes',nargs='+',choices=['fixed','variable'],default=['fixed','variable'])
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--sim-dt',type=float,default=.0005)
    a=p.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory=Path('runs')/(stamp+'_one_sqp_sweep');directory.mkdir(parents=True)
    (directory/'sweep_config.json').write_text(json.dumps(dict(timestamp=stamp,arguments=vars(a),
        task_pass_criteria='full duration; >=8 measured apices; all apices in [.5,1]m; actual state bounds; no clipping',
        execution='sequential to avoid cross-trial CPU contention'),indent=2)+'\n')
    rows=[]
    for mode in a.modes:
        for budget in a.iterations:
            name=f'one_sqp_{mode}_osqp{budget}_smooth'
            before=set(Path('runs').glob('*'))
            cmd=[sys.executable,'-m','juggling.run_experiment','--seconds',str(a.seconds),'--name',name,
                 '--sqp','1','--osqp',str(budget),'--wind','smooth','--codegen','--sim-dt',str(a.sim_dt)]
            if mode=='variable':cmd.append('--variable-time')
            with (directory/f'{mode}_{budget}.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            matches=[v for v in set(Path('runs').glob('*'))-before if v.name.endswith('_'+name)]
            assert len(matches)==1,matches
            row=summarize(matches[0]);rows.append(row);report(directory,rows)
            print(json.dumps({k:row[k] for k in ['run','mode','osqp_iterations','task_pass','contacts',
                'apex_min_m','apex_max_m','solve_median_ms','max_violation']}),flush=True)
    print('SUMMARY',directory,flush=True)

if __name__=='__main__':main()
