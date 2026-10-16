"""One-page reports for C++ timing and unpaused mock-arm HIL experiments."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm.model import ACCEL_LIMIT,VELOCITY_LIMIT,JERK_LIMIT,TORQUE_LIMIT,TORQUE_RATE_LIMIT,Panda


def timing(v):
    return dict(count=len(v),median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(max(v)))


def mock_report(directory,seconds=20.):
    r=Path(directory);runtime=json.loads((r/'hil_runtime.json').read_text())
    s=np.genfromtxt(r/'hil_ball.csv',delimiter=',',names=True);c=np.genfromtxt(r/'hil_callback.csv',delimiter=',',names=True);mpc=np.genfromtxt(r/'hil_mpc.csv',delimiter=',',names=True)
    t=s['time'];x=np.array([s['x'+str(j)] for j in range(23)]).T;tau=np.array([s['tau'+str(j)] for j in range(7)]).T
    dt=np.diff(t);acc=np.diff(x[:,13:20],axis=0)/dt[:,None];jerk=np.diff(acc,axis=0)/dt[1:,None];rate=np.diff(tau,axis=0)/dt[:,None]
    peaks=np.flatnonzero((x[:-1,5]>0)&(x[1:,5]<=0));apices=x[peaks,2]-.02
    ratios=dict(velocity=float(np.max(abs(x[:,13:20])/VELOCITY_LIMIT)),acceleration=float(np.max(abs(acc)/ACCEL_LIMIT)),jerk=float(np.max(abs(jerk)/JERK_LIMIT)),torque=float(np.max(abs(tau)/TORQUE_LIMIT)),torque_rate=float(np.max(abs(rate)/TORQUE_RATE_LIMIT)))
    valid=not (r/'invalid_run.json').exists();robot=Panda()
    joint_ok=bool(np.all(x[:,6:13]>=robot.q_min-1e-6) and np.all(x[:,6:13]<=robot.q_max+1e-6))
    metrics=dict(duration_s=float(t[-1]),runtime=runtime,measured_apices_m=apices.tolist(),limit_ratios=ratios,
        callback_max_us=float(max(c['callback_us'])),command_age_max_s=float(max(c['command_age'])),
        scheduling_lateness=timing(c['lateness_us']/1000),joint_position_limits_ok=joint_ok,
        cpp_all_calls=timing(mpc['total_ms']),cpp_steady_calls=timing(mpc['total_ms'][1:]),
        one_sqp_50_osqp=bool(np.all(mpc['iterations']==50)),valid_executable=valid,
        task_pass=bool(valid and joint_ok and np.all(mpc['iterations']==50) and runtime['fault']==0 and t[-1]>=seconds-.002 and len(peaks)>=int(seconds*.8)
            and np.all((apices>=.5)&(apices<=1)) and max(ratios.values())<=1.0001),
        note='C++ wall-clock mock arm with virtual ball and asynchronous workers; no physical robot. Acceleration/jerk derived from 1 kHz joint samples. No CAD collision model in this mock; see separate Newton tests.')
    (r/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    fig,ax=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    ax[0,0].plot(t,x[:,2]-.02,lw=.8);ax[0,0].scatter(t[peaks],apices,s=10);ax[0,0].axhspan(.5,1,color='green',alpha=.07);ax[0,0].set_title('Virtual ball height / apices [m]')
    ax[0,1].plot(t,x[:,:2],lw=.8);ax[0,1].set_title('Virtual lateral position [m]')
    ax[1,0].plot(t,np.max(abs(x[:,13:20])/VELOCITY_LIMIT,axis=1),label='velocity');ax[1,0].plot(t[1:],np.max(abs(acc)/ACCEL_LIMIT,axis=1),label='acceleration',lw=.7);ax[1,0].plot(t[2:],np.max(abs(jerk)/JERK_LIMIT,axis=1),label='jerk',lw=.6);ax[1,0].axhline(1,color='black',ls='--');ax[1,0].legend(fontsize=7);ax[1,0].set_title('Measured motion / research bounds')
    ax[1,1].plot(mpc['time'],mpc['total_ms'],lw=.8,label='total');ax[1,1].plot(mpc['time'],mpc['data_ms'],lw=.7,label='data');ax[1,1].plot(mpc['time'],mpc['qp_ms'],lw=.7,label='OSQP');ax[1,1].axhline(20,color='black',ls='--');ax[1,1].legend(fontsize=7);ax[1,1].set_title('C++ computation [ms]')
    ax[2,0].plot(c['time'],1000*c['command_age'],lw=.7);ax[2,0].axhline(runtime['max_command_age_s']*1000,color='black',ls='--');ax[2,0].set_title('Plan age / watchdog limit [ms]')
    ax[2,1].plot(c['time'],c['callback_us'],lw=.7);ax[2,1].set_title('Recorded callback section [µs] (scope in run config)')
    for a in ax.flat:a.grid(alpha=.25);a.set_xlabel('Time [s]')
    fig.suptitle(f'{r.name}\nUnpaused C++ mock-arm HIL; task pass={metrics["task_pass"]}, fault={runtime["fault"]}, impacts={runtime["virtual_impacts"]}',fontsize=11)
    fig.savefig(r/'report.pdf');plt.close(fig);return metrics


def benchmark_report(directory):
    r=Path(directory);files=sorted(r.glob('*steps.csv'));results={}
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for file in files:
        data=np.genfromtxt(file,delimiter=',',names=True);name=file.stem.replace('_steps','').replace('remote_','')
        steady=data['index']>0;v=data['total_ms'][steady]
        results[name]=dict(all_calls=timing(data['total_ms']),steady_calls=timing(v),first_calls=timing(data['total_ms'][~steady]),
                           exactly_50_osqp=bool(np.all(data['iterations']==50)))
        ax[0,0].plot(data['total_ms'],lw=.5,label=name)
        ax[0,1].plot(np.sort(v),np.linspace(0,1,len(v)),label=name)
        ax[1,0].plot(['median','mean','p95','maximum'],[np.median(v),np.mean(v),np.percentile(v,95),max(v)],marker='o',label=name)
        ax[1,1].plot(['warmstart','data','QP','merit'],[np.mean(data[k][steady]) for k in ['shift_ms','data_ms','qp_ms','merit_ms']],marker='o',label=name)
    ax[0,0].set(xlabel='Call index (includes first calls)',ylabel='C++ time [ms]',title='Every timed step; no outlier removal')
    ax[0,1].set(xlabel='C++ steady-call time [ms]',ylabel='Cumulative fraction',title='Empirical distribution')
    ax[1,0].set(ylabel='C++ steady-call time [ms]',title='Summary statistics');ax[1,1].set(ylabel='Mean component time [ms]',title='Where the step spends time')
    for a in ax.flat:a.grid(alpha=.25);a.legend(fontsize=7)
    fig.suptitle(r.name+'\nSame 999-update recorded wind trajectory; one SQP / 50 OSQP; constructor/build excluded',fontsize=11)
    fig.savefig(r/'report.pdf');plt.close(fig);(r/'timing_audit.json').write_text(json.dumps(results,indent=2)+'\n');return results


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['mock','benchmark']);p.add_argument('directories',nargs='+');p.add_argument('--seconds',type=float,default=20.);a=p.parse_args()
    for r in a.directories:
        result=mock_report(r,a.seconds) if a.kind=='mock' else benchmark_report(r)
        print(r,json.dumps(result))


if __name__=='__main__':main()
