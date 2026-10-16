"""Audit Stage 2 traces and make a single-page comparison of retained experiments."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .model import Panda,HOME,PADDLE_OFFSET


def audit(run):
    cfg=json.loads((run/'config.json').read_text());m=json.loads((run/'metrics.json').read_text())
    a=np.load(run/'arrays.npz');p=np.load(run/'plant.npz');t=p['t'];x=p['x'];dt=cfg['mpc']['sim_dt'];control_dt=cfg['mpc']['control_dt']
    log=[json.loads(s) for s in (run/'iterations.jsonl').read_text().splitlines()];n=len(log);substeps=round(control_dt/dt)
    assert n==m['controller_updates']==len(a['t'])-1
    assert len(t)-1==m['plant_steps']
    assert np.allclose(np.diff(t),dt,rtol=0,atol=4e-15)
    assert np.allclose([r['t'] for r in log],np.arange(n)*control_dt,rtol=0,atol=4e-15)
    assert all(len(r['iterations'])==1 and r['iterations'][0]['iter']==cfg['mpc']['osqp_iters'] for r in log)
    assert np.array_equal(p['control_index'],np.arange(len(t)-1)//substeps)
    assert np.array_equal(a['x'][:-1],x[np.arange(n)*substeps])
    assert np.array_equal(a['x'][-1],x[-1])
    mid=np.arange(len(t)-1)*dt+.5*dt
    if cfg['arguments']['wind']=='smooth':
        wind=np.column_stack((.12*np.cos(.8*mid)+.04*np.sin(1.7*mid),.10*np.sin(.8*mid)+.03*np.cos(1.3*mid),np.zeros(len(mid))))
        assert np.allclose(p['wind'],wind,atol=1e-14,rtol=0)
    observed_rate=np.diff(p['tau'],axis=0)/dt
    torque_error=float(np.max(abs(observed_rate-p['torque_rate'][:-1])))
    assert torque_error<1e-8
    midpoint_limits=cfg['limits'];pose_ok=bool(np.all(x[:,6:13]>=np.array(midpoint_limits['joint_lower'])-1e-6) and np.all(x[:,6:13]<=np.array(midpoint_limits['joint_upper'])+1e-6))
    ratio_checks=dict(velocity=float(np.max(abs(x[:,13:20])/midpoint_limits['velocity'])),
                      torque=float(np.max(abs(p['tau'])/midpoint_limits['torque'])),
                      acceleration=float(np.max(abs(p['acceleration'])/midpoint_limits['acceleration'])),
                      torque_rate=float(np.max(abs(p['torque_rate'])/midpoint_limits['torque_rate'])),
                      jerk=float(np.max(abs(p['jerk'])/midpoint_limits['jerk'])))
    for key,value in ratio_checks.items():assert abs(value-m['limit_ratios'][key])<1e-12
    peaks=np.flatnonzero((x[:-1,5]>0)&(x[1:,5]<=0));peak_height=np.maximum(x[peaks,2],x[peaks+1,2])-.02
    assert np.allclose(peak_height,m['measured_apices_m'],rtol=0,atol=1e-14)
    info=subprocess.check_output(['pdfinfo',str(run/'report.pdf')],text=True)
    assert next(s.split(':')[1].strip() for s in info.splitlines() if s.startswith('Pages:'))=='1'
    contacts=json.loads((run/'contacts.json').read_text())
    result=dict(controller_updates=n,plant_steps=len(t)-1,plant_dt_s=dt,control_dt_s=control_dt,
                one_sqp_exact_osqp_budget=True,wind_samples_verified=True,
                maximum_torque_ramp_error_Nm_s=torque_error,full_trace_limits_verified=True,
                physical_joint_position_limits_ok=pose_ok,one_page_pdf=True,
                constraint_violation_percentiles=np.percentile(a['violation'],[50,95,99]).tolist(),
                constraint_violation_over_001_fraction=float(np.mean(a['violation']>.01)),
                bound_violation_p95=float(np.percentile(a['bound_violation'],95)),
                max_contact_impulse_Ns=max((c['impulse_Ns'] for c in contacts),default=0),
                max_contact_radius_m=max((c['radial_distance'] for c in contacts),default=0),
                max_impact_joint_velocity_jump_rad_s=float(np.max(np.abs([c['joint_velocity_jump'] for c in contacts]))) if contacts else 0.)
    (run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(run=str(run),control_dt_ms=1000*control_dt,osqp_iters=cfg['mpc']['osqp_iters'],
                warmstart=cfg['arguments'].get('warmstart','acceleration'),compiled=cfg['mpc']['codegen'],
                actuator_margin=cfg['mpc'].get('actuator_margin',1),
                rate_margin=cfg['mpc'].get('rate_margin',1),damping=cfg['sim']['damping'],
                **m,audit=result)


def motion_plot(run):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    robot=Panda();p=np.load(run/'plant.npz');x=p['x'];t=p['t'];origin=robot.origin
    fig=plt.figure(figsize=(11,8));ax=fig.add_subplot(111,projection='3d')
    for i,color,alpha in [(0,'grey',.4),(int(np.argmax(x[:,2])),'tab:orange',.6),(len(x)-1,'tab:blue',.9)]:
        points=np.column_stack((np.zeros(3),np.asarray(robot.geometry(x[i,6:13]))))
        ax.plot(*points,lw=4,color=color,alpha=alpha,label=f'Arm at {t[i]:.2f} s')
        center,n,J,N,R=robot.fk(x[i,6:13]);R=np.asarray(R);center=np.asarray(center).ravel()
        angles=np.linspace(0,2*np.pi,50);disk=center[:,None]+.075*(R[:,0,None]*np.cos(angles)+R[:,1,None]*np.sin(angles))
        ax.add_collection3d(Poly3DCollection([disk.T],facecolors=color,alpha=.3,edgecolor=color))
    indices=np.arange(0,len(x),max(1,len(x)//3000));ball=x[indices,:3]+origin
    ax.plot(ball[:,0],ball[:,1],ball[:,2],lw=.8,color='tab:green',alpha=.65,label='Ball trajectory')
    paddle=p['paddle_pos'][indices]+origin
    ax.plot(paddle[:,0],paddle[:,1],paddle[:,2],color='tab:red',lw=1.2,label='Racket center')
    ax.set(xlabel='Base x [m]',ylabel='Base y [m]',zlabel='Base z [m]',xlim=(-.15,.65),ylim=(-.45,.35),zlim=(0,1.45))
    ax.set_box_aspect((.8,.8,1.45));ax.view_init(elev=18,azim=-50);ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Panda + directly attached racket: simulated motion\nKinematic skeleton and disk, not a collision mesh or Newton rendering')
    fig.tight_layout();fig.savefig(run/'motion.pdf');fig.savefig(run/'motion.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path);parser.add_argument('--motion',type=Path)
    parser.add_argument('--output',type=Path,help='Regenerate derived summaries in this directory')
    args=parser.parse_args();rows=[audit(p) for p in args.runs]
    directory=args.output or Path('runs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_stage2_summary')
    directory.mkdir(exist_ok=bool(args.output))
    shutil.copyfile(__file__,directory/'summarize.py')
    (directory/'generation.json').write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,'-m','arm.summarize',*sys.argv[1:]]),indent=2)+'\n')
    (directory/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-matplotlib')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    labels=[f"{r['osqp_iters']} QP\n{r['control_dt_ms']:g} ms\nb={r['damping']:g}\nr={100*r['rate_margin']:g}%\n"+
            ('C' if r['compiled'] else 'Python') for r in rows]
    x=np.arange(len(rows));colors=['tab:green' if r['task_pass'] else 'tab:red' for r in rows]
    for ax,key,title in [(axes[0,0],'duration_s','Completed simulation [s]'),(axes[0,1],'contacts','Physical impacts'),
                         (axes[1,0],'max_lateral_distance_m','Maximum lateral excursion [m]'),
                         (axes[1,1],'solve_median_ms','Controller median [ms]')]:
        ax.bar(x,[r[key] for r in rows],color=colors);ax.set_ylabel(title)
    ax=axes[0,2]
    for k,r in enumerate(rows):
        if r['measured_apices_m']:ax.plot([k,k],[min(r['measured_apices_m']),max(r['measured_apices_m'])],color=colors[k],marker='_',ms=10,lw=3)
    ax.axhspan(.5,1.,color='green',alpha=.1);ax.set_ylabel('Measured apex range [m]')
    ax=axes[1,2];ax.bar(x,[max(r['limit_ratios'].values()) for r in rows],color=colors);ax.axhline(1,color='black',ls=':');ax.set_ylabel('Largest actual research-limit ratio')
    for ax in axes.flat:ax.set_xticks(x,labels,fontsize=7);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Stage 2: retained full-arm comparisons\nGreen requires full duration, apex band, actuator/state limits and no clipping.\n'
                 'b: unmodeled joint damping [Nm s/rad]; r: predicted torque-rate limit multiplier; C/Python: compiled/interpreted.')
    fig.savefig(directory/'report.pdf');plt.close(fig)
    lines=['# Full-arm comparison','', '| Trial | Update | OSQP | Pass | Duration | Impacts | Apex range | Median MPC | PDF |',
           '| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |']
    for r in rows:
        peaks=r['measured_apices_m'];apex=f'{min(peaks):.3f}–{max(peaks):.3f} m' if peaks else 'none'
        lines.append(f"| {Path(r['run']).name} | {r['control_dt_ms']:g} ms | {r['osqp_iters']} | {r['task_pass']} | {r['duration_s']:.3f} s | {r['contacts']} | {apex} | {r['solve_median_ms']:.2f} ms | [plot](../{Path(r['run']).name}/report.pdf) |")
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')
    if args.motion:motion_plot(args.motion)
    print(directory)

if __name__=='__main__':main()
