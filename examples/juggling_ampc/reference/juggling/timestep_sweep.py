"""Compare plant integration intervals with an unchanged 100 Hz MPC controller."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .run_experiment import CONTROL_DT
from .sweep import summarize


def verify(run):
    """Audit the recorded plant trajectory, command hold, wind and solve clocks."""
    cfg=json.loads((run/'config.json').read_text())
    with np.load(run/'arrays.npz') as a, np.load(run/'plant.npz') as p:
        dt=cfg['timing']['plant_dt_s'];n=cfg['timing']['plant_steps_per_update']
        t=p['t'];x=p['x'];ci=p['control_index'];u=a['u'];ct=a['t']
        assert np.allclose(np.diff(t),dt,rtol=0,atol=2e-15)
        assert np.allclose(ct[:-1],np.arange(len(u))*CONTROL_DT,rtol=0,atol=2e-15)
        assert np.array_equal(ci,np.arange(len(t)-1)//n)
        assert np.array_equal(a['x'][:-1],x[np.arange(len(u))*n])
        assert np.array_equal(a['x'][-1],x[-1])
        observed_acceleration=np.diff(x[:,11:16],axis=0)/dt
        hold_error=float(np.max(np.abs(observed_acceleration-u[ci])))
        pose_error=float(np.max(np.abs(np.diff(x[:,6:11],axis=0)-
                         dt*x[:-1,11:16]-.5*dt*dt*u[ci])))
        assert hold_error<1e-8 and pose_error<1e-12
        midpoint=(np.arange(len(ci))+.5)*dt
        expected_wind=np.column_stack((.12*np.cos(.8*midpoint)+.04*np.sin(1.7*midpoint),
                         .10*np.sin(.8*midpoint)+.03*np.cos(1.3*midpoint),np.zeros(len(ci))))
        wind_error=float(np.max(np.abs(p['wind_accel']-expected_wind)))
        assert wind_error<1e-14
        logs=[json.loads(line) for line in (run/'iterations.jsonl').read_text().splitlines()]
        assert len(logs)==len(u)
        assert np.allclose([l['t'] for l in logs],ct[:-1],rtol=0,atol=2e-15)
        assert all(len(l['iterations'])==1 and l['iterations'][0]['iter']==25 for l in logs)
        assert cfg['mpc']['variable_time'] is False
        result=dict(controller_updates=len(u),plant_steps=len(ci),
                    observed_steps_per_command=sorted(set(np.bincount(ci).tolist())),
                    max_command_hold_acceleration_error=hold_error,
                    max_paddle_integration_position_error_m_or_rad=pose_error,
                    max_wind_profile_error_m_s2=wind_error,
                    sqp_iterations=1,osqp_iterations=25,control_period_s=CONTROL_DT)
    (run/'timing_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def report(directory,rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for row in rows:
        run=Path(row['run']);label=f"plant {1000*row['plant_dt_s']:g} ms"
        with np.load(run/'plant.npz') as p, np.load(run/'arrays.npz') as a:
            indices=np.unique(np.linspace(0,len(p['t'])-1,min(len(p['t']),5000),dtype=int))
            t=p['t'][indices];x=p['x'][indices]
            axes[0,0].plot(t,x[:,2]-.02,label=label,lw=1)
            axes[0,1].plot(t,100*np.linalg.norm(x[:,:2],axis=1),label=label,lw=1)
            contacts=json.loads((run/'contacts.json').read_text())
            apex_t=[c['t_global'] for c in contacts[:row['completed_apices']]]
            axes[1,0].plot(apex_t,row['measured_apices_m'],'o-',label=label,ms=3)
            axes[1,1].plot(t,np.linalg.norm(x[:,11:14],axis=1),label=label,lw=1)
            axes[2,0].plot(a['t'][:-1],a['solve_ms'],label=label,alpha=.6,lw=.7)
            axes[2,1].plot(a['t'][:-1],a['violation'],label=label,lw=1)
    titles=[('Ball height above strike plane [m]','Time [s]'),
            ('Ball lateral excursion [cm]','Time [s]'),
            ('Measured apex [m]','Preceding impact time [s]'),
            ('Paddle speed [m/s]','Time [s]'),
            ('Controller computation [ms]','Time [s]'),
            ('Scaled prediction constraint violation','Time [s]')]
    for ax,(ylabel,xlabel) in zip(axes.flat,titles):
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[1,0].axhspan(.5,1.,color='green',alpha=.08)
    axes[2,0].axhline(10,color='black',ls=':',label='10 ms MPC update');axes[2,0].legend(fontsize=8)
    axes[2,1].set_yscale('symlog',linthresh=1e-5)
    fig.suptitle('Plant timestep refinement · 1 SQP / 25 OSQP iterations · continuous wind\n'
                 f"MPC updates 10 ms; prediction nodes {1000*rows[0]['prediction_dt_s']:g} ms; unchanged controller per sweep")
    fig.savefig(directory/'report.pdf');plt.close(fig)
    (directory/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Plant timestep refinement','',
           'Sequential trials; fixed 10 ms MPC updates, one SQP and exactly 25 OSQP iterations per update.',
           f"Fixed prediction nodes: {1000*rows[0]['prediction_dt_s']:g} ms. Smooth wind is sampled at every plant-step midpoint.",'',
           '| Plant step [ms] | Pass | Plant steps / MPC solves | Impacts | Apex range [m] | Max lateral [cm] | Controller median / p95 [ms] | PDF |',
           '| ---: | --- | ---: | ---: | --- | ---: | --- | --- |']
    for r in rows:
        apex='none' if r['apex_min_m'] is None else f"{r['apex_min_m']:.4f}–{r['apex_max_m']:.4f}"
        lines.append(f"| {1000*r['plant_dt_s']:g} | {r['task_pass']} | {r['plant_steps']} / {r['controller_updates']} | "
                     f"{r['contacts']} | {apex} | {100*r['max_lateral_distance_m']:.3f} | "
                     f"{r['solve_median_ms']:.2f} / {r['solve_p95_ms']:.2f} | [plot](../{Path(r['run']).name}/report.pdf) |")
    lines+=['','Pass: full duration, at least eight measured apices, all in [0.5,1] m, paddle state limits on the full plant trace, and no command clipping.',
            'Each timing_verification.json independently checks the update clock, command hold from measured paddle motion, exact iteration budgets, and substep wind samples.',
            'Computation delay is not applied to the plant. Physical task success does not establish a 10 ms computation deadline or exact nonlinear prediction feasibility.']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sim-dts',type=float,nargs='+',default=[.01,.0005,.0001])
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--h',type=float,default=.03)
    parser.add_argument('--horizon',type=int,default=40)
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory=Path('runs')/(stamp+'_plant_timestep_sweep');directory.mkdir(parents=True)
    shutil.copyfile(__file__,directory/'sweep_source.py')
    (directory/'sweep_config.json').write_text(json.dumps(dict(timestamp=stamp,arguments=vars(args),
        control_dt_s=CONTROL_DT,sqp_iters=1,osqp_iters=25,execution='sequential'),indent=2)+'\n')
    rows=[]
    for dt in args.sim_dts:
        name=f'plant_{dt*1000:g}ms_sqp1_osqp25'
        before=set(Path('runs').glob('*'))
        cmd=[sys.executable,'-m','juggling.run_experiment','--name',name,'--seconds',str(args.seconds),
             '--sim-dt',str(dt),'--h',str(args.h),'--horizon',str(args.horizon),
             '--sqp','1','--osqp','25','--wind','smooth','--codegen']
        with (directory/f'{name}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        matches=[p for p in set(Path('runs').glob('*'))-before if p.name.endswith('_'+name)]
        assert len(matches)==1,matches
        run=matches[0];row=summarize(run)
        row['timing_verification']=verify(run);row['prediction_dt_s']=args.h
        rows.append(row);report(directory,rows)
        print(json.dumps({k:row[k] for k in ['run','plant_dt_s','task_pass','contacts','apex_min_m',
                        'apex_max_m','max_lateral_distance_m','solve_median_ms','solve_p95_ms']}),flush=True)
    print('SUMMARY',directory,flush=True)


if __name__=='__main__':main()
