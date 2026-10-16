"""Standalone survival and paired-uptime figure for completed simulation reports."""
import argparse,json
from pathlib import Path
import numpy as np
from uncertainty_settings import require_matching,from_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(reference,policy,output):
    reports=[json.loads(path.read_text()) for path in (reference,policy)]
    for key in ('seeds','calibration_sha256','noise_disturbance_multiplier','rollout_seconds','drop_bounds'):
        if reports[0][key]!=reports[1][key]:raise ValueError('Unpaired comparison: '+key)
    require_matching(*reports)
    labels={'native_diagnostic':'Native OSQP MPC','warp_mpc':'WarpMPC','network':'AMPC (3×256)'}
    names=[labels[r['policy_kind']] for r in reports]
    uptime=[np.asarray(r['uptime_s'],float) for r in reports]
    duration=reports[0]['rollout_seconds'];colors=['#2866a3','#d56a24']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for values,name,color in zip(uptime,names,colors):
        events=np.unique(np.r_[0,values,duration])
        survival=np.array([np.mean(values>=t) for t in events])
        axes[0].step(events,survival,where='pre',label=f'{name}: mean {values.mean():.2f} s',color=color,lw=2)
    axes[0].set(xlabel='Simulated time (s)',ylabel='Fraction of balls still in play',xlim=(0,duration),ylim=(0,1.02))
    axes[0].legend(frameon=False,loc='upper right')
    axes[1].scatter(*uptime,s=13,alpha=.45,color=colors[1],edgecolors='none')
    axes[1].plot([0,duration],[0,duration],color='#777777',ls='--',lw=1)
    axes[1].set(xlabel=f'{names[0]} ball uptime (s)',ylabel=f'{names[1]} ball uptime (s)',xlim=(0,duration),ylim=(0,duration))
    for ax in axes:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    uncertainty=from_report(reports[0])
    fig.suptitle(f"Franka juggling: {len(uptime[0]):,} paired {duration:g} s trials\nUncertainty: ball/contact {uncertainty['ball_contact']:g}×, arm {uncertainty['arm_following']:g}×, sensing {uncertainty['sensing']:g}×")
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)
    print(str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('policy',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    plot(a.reference,a.policy,a.output)
