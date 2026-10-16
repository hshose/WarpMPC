"""Report measured homing/MPC tracking, including aborted physical trials."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args();r=args.run
    session=json.loads((r/'hardware_session.json').read_text());home=session['home_only']
    data=np.genfromtxt(r/('homing.csv' if home else 'hardware_tracking.csv'),delimiter=',',names=True)
    t=data['time'];joints=lambda name:np.stack([data[name+str(j)] for j in range(7)],axis=1)
    active=np.ones(len(t),dtype=bool) if home else data['stopping']==0
    qerr=joints('q')-joints('q_ref');verr=joints('dq')-joints('dq_ref')
    faults=[] if home else np.flatnonzero(data['fault'])
    metrics=dict(physical_robot_used=True,physical_ball_used=False,session=session,
        first_fault=None if len(faults)==0 else dict(time_s=float(t[faults[0]]),code=int(data['fault'][faults[0]])),
        max_active_joint_error_rad=float(abs(qerr[active]).max()),rms_active_joint_error_rad=float(np.sqrt(np.mean(qerr[active]**2))),
        max_active_velocity_error_rad_s=float(abs(verr[active]).max()),
        final_home_error_rad=float(abs(qerr[-1]).max()) if home else None,
        max_callback_gap_ms=float(data['dt'].max()*1000),max_callback_us=float(data['callback_us'].max()),
        min_command_success_rate_after_100ms=float(data['success_rate'][t>.1].min()))
    fig,ax=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    ax[0,0].plot(t,joints('q'));ax[0,0].plot(t,joints('q_ref'),'--',alpha=.4);ax[0,0].set_title('Measured joint angles / dashed references [rad]')
    ax[0,1].plot(t,qerr*1000);ax[0,1].set_title('Joint tracking error [mrad]')
    ax[1,0].plot(t,joints('dq'));ax[1,0].plot(t,joints('dq_ref'),'--',alpha=.4);ax[1,0].set_title('Measured joint velocity / references [rad/s]')
    ax[1,1].plot(t,verr);ax[1,1].set_title('Joint velocity error [rad/s]')
    if home:
        ax[2,0].plot(t,data['dt']*1000);ax[2,0].set_title('Robot callback period [ms]')
        ax[2,1].plot(t,data['success_rate']);ax[2,1].set_title('Robot command success rate')
    else:
        ball=np.genfromtxt(r/'hil_ball.csv',delimiter=',',names=True);mpc=np.genfromtxt(r/'hil_mpc.csv',delimiter=',',names=True)
        runtime=json.loads((r/'hil_runtime.json').read_text());peaks=np.flatnonzero((ball['x5'][:-1]>0)&(ball['x5'][1:]<=0))
        metrics.update(runtime=runtime,measured_apices_m=(ball['x2'][peaks]-.02).tolist(),
            cpp_timing_ms=dict(median=float(np.median(mpc['total_ms'][1:])),mean=float(np.mean(mpc['total_ms'][1:])),p95=float(np.percentile(mpc['total_ms'][1:],95)),maximum=float(mpc['total_ms'][1:].max()),initial=float(mpc['total_ms'][0])),
            fault_note='First callback fault is authoritative. Legacy trials may overwrite it in the final summary; updated runtimes latch it.')
        active_peaks=peaks[ball['time'][peaks]<=session['requested_duration_s']]
        active_apices=ball['x2'][active_peaks]-.02
        metrics['active_apices_m']=active_apices.tolist()
        metrics['juggling_task_pass']=bool(session['control_completed'] and session['fault']==0 and runtime['fault']==0
            and session['elapsed_s']>=session['requested_duration_s'] and len(active_peaks)>=int(.8*session['requested_duration_s'])
            and np.all((active_apices>=.5)&(active_apices<=1.)))
        ax[2,0].plot(ball['time'],ball['x2']-.02);ax[2,0].axhspan(.5,1,alpha=.1,color='green');ax[2,0].set_title('Virtual ball height [m] from measured arm')
        ax[2,1].plot(mpc['time'],mpc['total_ms'],label='MPC solve');ax[2,1].plot(t,data['command_age']*1000,label='plan age',alpha=.6);ax[2,1].legend();ax[2,1].set_title('Computation / age [ms]')
        if len(faults):
            for a in ax.flat:a.axvspan(t[faults[0]],t[-1],color='orange',alpha=.1)
    for a in ax.flat:a.set_xlabel('Time [s]');a.grid(alpha=.25)
    title='Physical homing completed' if home else 'Physical arm / virtual ball: juggling task pass='+str(metrics['juggling_task_pass'])
    fig.suptitle(r.name+'\n'+title+'; orange = stop after a fault'*(not home))
    fig.savefig(r/'report.pdf');plt.close(fig)
    (r/'tracking_analysis.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
