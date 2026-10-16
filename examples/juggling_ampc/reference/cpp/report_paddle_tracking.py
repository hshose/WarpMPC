"""Evaluate a recorded HIL trial using the mounting actually used in that trial."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm.model import Panda


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();r=args.run
    spec=json.loads(args.spec.read_text())
    robot=Panda(assembly=spec.get('assembly'),**spec.get('geometry',{}))
    trace=np.genfromtxt(r/'hardware_tracking.csv',delimiter=',',names=True)
    ball=np.genfromtxt(r/'hil_ball.csv',delimiter=',',names=True)
    t=trace['time'];active=(trace['stopping']==0)&(t<=ball['time'][-1])
    ball_position=np.column_stack([np.interp(t,ball['time'],ball['x'+str(j)]) for j in range(3)])+robot.origin
    results={}
    for kind,suffix in [('measured',''),('reference','_ref')]:
        q=np.column_stack([trace['q'+suffix+str(j)] for j in range(7)])
        dq=np.column_stack([trace['dq'+suffix+str(j)] for j in range(7)])
        position=[];normal=[];normal_speed=[];radius=[];gap=[]
        for i,(qi,vi) in enumerate(zip(q,dq)):
            p,n,J,N,_=[np.asarray(v) for v in robot.fk(qi)]
            p=p.ravel();n=n.ravel();relative=ball_position[i]-p
            position.append(p);normal.append(n)
            # Includes paddle rotation at the ball's location, matching the
            # virtual-ball gap derivative, not only face-center velocity.
            normal_speed.append(float((n@J-relative@N)@vi))
            radius.append(float(np.linalg.norm(relative-n*(n@relative))))
            gap.append(float(n@relative-.02))
        results[kind]=dict(position=np.array(position),normal=np.array(normal),
                           normal_speed=np.array(normal_speed),radius=np.array(radius),gap=np.array(gap))
    measured=results['measured'];reference=results['reference']
    error=measured['position']-reference['position']
    angle=np.rad2deg(np.arccos(np.clip(np.sum(measured['normal']*reference['normal'],axis=1),-1,1)))
    hits=np.flatnonzero(np.diff(ball['x5'])>1.)+1
    impacts=[]
    for hit in hits:
        index=int(np.argmin(abs(t-ball['time'][hit])))
        impacts.append(dict(first_post_impact_sample_s=float(ball['time'][hit]),callback_time_s=float(t[index]),
            incoming_ball_vz_m_s=float(ball['x5'][hit-1]),outgoing_ball_vz_m_s=float(ball['x5'][hit]),
            measured_paddle_normal_speed_m_s=float(measured['normal_speed'][index]),
            reference_paddle_normal_speed_m_s=float(reference['normal_speed'][index]),
            paddle_position_error_m=float(np.linalg.norm(error[index])),normal_error_deg=float(angle[index]),
            radial_contact_offset_m=float(measured['radius'][index])))
    metrics=dict(spec=str(args.spec),physical_motion_commanded_by_analysis=False,impacts=impacts,
        max_active_position_error_m=float(np.linalg.norm(error[active],axis=1).max()),
        max_active_normal_error_deg=float(angle[active].max()),
        max_active_normal_speed_error_m_s=float(abs(measured['normal_speed'][active]-reference['normal_speed'][active]).max()),
        scope='Kinematics evaluated from recorded q/dq and the historical model. Impact values are nearest 1 kHz samples, not reconstructed event times. This does not attribute the failure to any single cause.')
    (r/'paddle_tracking.json').write_text(json.dumps(metrics,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    axes[0,0].plot(t[active],1000*error[active]);axes[0,0].set_title('Face-center position error XYZ [mm]')
    for kind in results:
        axes[0,1].plot(t[active],results[kind]['normal_speed'][active],label=kind)
        axes[1,1].plot(t[active],1000*results[kind]['radius'][active],label=kind)
    axes[0,1].set_title('Paddle normal speed at virtual ball [m/s]');axes[0,1].legend()
    axes[1,0].plot(t[active],angle[active]);axes[1,0].set_title('Face-normal tracking error [degrees]')
    axes[1,1].set_title('Virtual ball radial offset from face center [mm]');axes[1,1].legend()
    axes[1,1].axhline(75,color='black',ls='--')
    for axis in axes.flat:
        axis.set_xlabel('Time [s]');axis.grid(alpha=.25)
        for hit in hits:axis.axvline(ball['time'][hit],color='gray',ls=':',alpha=.7)
    fig.suptitle(r.name+'\nRecorded physical tracking, using that trial’s modeled mounting; dotted line = virtual impact')
    fig.savefig(r/'paddle_tracking.pdf');plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
