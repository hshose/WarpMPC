"""One-page arm, ball, solver and actuator-limit diagnostics."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .model import HOME,TORQUE_LIMIT,VELOCITY_LIMIT,ACCEL_LIMIT,TORQUE_RATE_LIMIT,JERK_LIMIT


def report(directory,a,p,config,metrics,robot):
    fig,axes=plt.subplots(4,3,figsize=(16,13),constrained_layout=True)
    ix=np.unique(np.linspace(0,len(p['t'])-1,min(5000,len(p['t'])),dtype=int));t=p['t'][ix];x=p['x'][ix]
    ct=a['t'][:-1];sample=ix[ix<len(p['tau'])];st=p['t'][sample]
    ax=axes[0,0];ax.plot(t,x[:,2]-.02,label='Ball lower pole');ax.plot(t,p['paddle_pos'][ix,2],label='Paddle')
    ax.scatter(metrics['measured_apex_times_s'],metrics['measured_apices_m'],s=10,label='Measured apex')
    ax.axhspan(.5,1.,alpha=.1,color='green');ax.axhline(.75,ls=':',color='green');ax.set_ylabel('Height above nominal strike [m]');ax.legend(fontsize=6)
    ax=axes[0,1]
    for i,label in enumerate(['x','y']):ax.plot(t,100*x[:,i],label='Ball '+label);ax.plot(t,100*p['paddle_pos'][ix,i],ls='--',label='Paddle '+label)
    ax.set_ylabel('Lateral position [cm]');ax.legend(fontsize=6)
    ax=axes[0,2]
    for i in range(2):ax.plot(st,p['wind'][sample,i],label=['ax','ay'][i])
    ax.set_ylabel('Persistent wind acceleration [m/s²]');ax.legend(fontsize=6)
    ax=axes[1,0]
    for i in range(7):ax.plot(t,x[:,6+i]-HOME[i],label=f'J{i+1}')
    ax.set_ylabel('Joint deviation from home [rad]');ax.legend(fontsize=6,ncol=4)
    ax=axes[1,1];ax.plot(t,x[:,13:20]/VELOCITY_LIMIT);ax.axhline(1,c='black',ls=':');ax.axhline(-1,c='black',ls=':');ax.set_ylabel('Joint velocity / research limit')
    ax=axes[1,2];ax.plot(st,p['tau'][sample]/TORQUE_LIMIT);ax.axhline(1,c='black',ls=':');ax.axhline(-1,c='black',ls=':');ax.set_ylabel('Total joint torque / research limit')
    ax=axes[2,0];ax.plot(st,p['acceleration'][sample]/ACCEL_LIMIT);ax.axhline(1,c='black',ls=':');ax.axhline(-1,c='black',ls=':');ax.set_ylabel('Actuated acceleration / research limit')
    ax=axes[2,1];ax.plot(st,p['torque_rate'][sample]/TORQUE_RATE_LIMIT);ax.axhline(1,c='black',ls=':');ax.axhline(-1,c='black',ls=':');ax.set_ylabel('Torque rate / research limit')
    ax=axes[2,2];ax.plot(st,p['jerk'][sample]/JERK_LIMIT);ax.axhline(1,c='black',ls=':');ax.axhline(-1,c='black',ls=':');ax.set_ylabel('Actuated joint jerk / research limit')
    ax=axes[3,0]
    for key,label in [('solve_ms','MPC including shift'),('data_ms','SQP data'),('qp_ms','OSQP')]:ax.plot(ct,a[key],label=label)
    ax.axhline(1000*config['mpc']['control_dt'],c='black',ls=':',label='Update period');ax.set_ylabel('Controller computation [ms]');ax.legend(fontsize=6)
    ax=axes[3,1];ax.plot(ct,a['violation'],label='Constraints');ax.plot(ct,a['bound_violation'],label='Variable bounds');ax.set_yscale('symlog',linthresh=1e-5);ax.set_ylabel('Scaled prediction violation');ax.legend(fontsize=6)
    ax=axes[3,2];ax.plot(t,p['paddle_speed'][ix]);ax.set_ylabel('Paddle translational speed [m/s]')
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('Time [s]');ax.tick_params(labelsize=7)
    fig.suptitle(f"{directory.name}\n{metrics['duration_s']:.2f} s · {metrics['contacts']} impacts · "
                 f"1 SQP / {config['mpc']['osqp_iters']} OSQP · plant 0.5 ms · "
                 f"MPC {1000*config['mpc']['control_dt']:g} ms / nodes {1000*config['mpc']['h']:g} ms\n" +
                 ('Actuated-motion bounds shown here; contact-inclusive acceleration/jerk and forces are in contact.pdf.'
                  if 'engine' in config else 'Limit lines are conservative research bounds; impact velocity jumps are recorded separately from actuator jerk.'),fontsize=11)
    fig.savefig(directory/'report.pdf');plt.close(fig)
