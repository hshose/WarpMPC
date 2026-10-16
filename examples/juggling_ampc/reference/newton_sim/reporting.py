"""Newton contact, spin and contact-inclusive acceleration diagnostics."""
import numpy as np
from arm.reporting import plt
from arm.model import ACCEL_LIMIT,JERK_LIMIT


def contact_report(directory,p,m):
    t=p['t'][:-1];fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    axes[0,0].plot(t,p['force']);axes[0,0].set_ylabel('Ball/racket normal force [N]')
    axes[0,1].plot(t,1000*p['penetration']);axes[0,1].set_ylabel('Contact penetration [mm]')
    axes[0,2].plot(t,p['spin']);axes[0,2].set_ylabel('Ball angular velocity [rad/s]')
    axes[1,0].plot(t,p['total_acceleration']/ACCEL_LIMIT);axes[1,0].set_ylabel('Contact-inclusive acceleration / research bound')
    axes[1,1].plot(t,p['total_jerk']/JERK_LIMIT);axes[1,1].set_ylabel('Contact-inclusive jerk / research bound')
    axes[1,2].plot(t,p['plant_step_ms']);axes[1,2].set_ylabel('Newton step computation [ms]')
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('Time [s]')
    fig.suptitle(f'{directory.name}\nNewton native compliant/frictional contact · no scripted impacts\n'
                 'Contact-inclusive acceleration/jerk are shown explicitly; controller PDF separates actuated motion.')
    fig.savefig(directory/'contact.pdf');plt.close(fig)
