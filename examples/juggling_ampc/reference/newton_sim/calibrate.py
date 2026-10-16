"""Measure native Newton/MuJoCo contact response at the requested 0.5 ms step."""
from datetime import datetime,timezone
import json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('WARP_CACHE_PATH','/tmp/franka-warp')
os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-matplotlib')
import warp as wp
import newton
from newton.solvers import SolverMuJoCo


def bounce(stiffness,damping,speed,phase=0.,dt=.0005):
    builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder)
    z=.024+speed*phase
    xml=f'''<mujoco><option gravity="0 0 0" integrator="RK4"/><default><geom
        solref="{-stiffness} {-damping}" solimp=".95 .95 .001 .5 2" friction=".3 .0001 .00001" condim="3"/></default>
      <worldbody><geom name="racket" type="cylinder" size=".075 .006" pos="0 0 -.006"/>
      <body name="ball" pos="0 0 {z}"><freejoint/><inertial pos="0 0 0" mass=".0027" diaginertia=".00000072 .00000072 .00000072"/>
      <geom name="ball_shape" type="sphere" size=".02"/></body></worldbody></mujoco>'''
    builder.add_mjcf(xml);builder.joint_qd[2]=-speed
    model=builder.finalize(device='cpu');solver=SolverMuJoCo(model,use_mujoco_cpu=True,integrator='RK4',
        solver='Newton',iterations=100,tolerance=1e-12,cone='elliptic',update_data_interval=0)
    state=model.state();out=model.state();control=model.control();solver.reset(state)
    newton.eval_fk(model,state.joint_q,state.joint_qd,state)
    trace=[]
    for k in range(100):
        solver.step(state,out,control,None,dt);state,out=out,state
        q=state.joint_q.numpy();v=state.joint_qd.numpy();trace.append([(k+1)*dt,q[2]-.02,v[2]])
        if v[2]>0 and q[2]>.025:break
    a=np.array(trace)
    return dict(stiffness=stiffness,damping=damping,speed=speed,phase_s=phase,
        restitution=float(a[-1,2]/speed),max_penetration_m=float(max(0,-min(a[:,1]))),
        contact_samples=int(np.sum(a[:,1]<0))),a


def main():
    wp.init();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory=Path('runs')/(stamp+'_newton_contact_calibration');directory.mkdir()
    rows=[];traces=[]
    for damping in [60.,100.,140.,180.,220.]:
        for speed in [1.,3.5,4.5]:
            for phase in [0.,.000125,.00025,.000375]:
                row,a=bounce(2e6,damping,speed,phase);rows.append(row);traces.append(a)
    (directory/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(directory/'traces.npz',**{str(i):a for i,a in enumerate(traces)})
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for damping in sorted(set(r['damping'] for r in rows)):
        rr=[r for r in rows if r['damping']==damping]
        axes[0,0].scatter([r['speed'] for r in rr],[r['restitution'] for r in rr],s=14,label=f'D={damping:g} /s')
        axes[0,1].scatter([r['speed'] for r in rr],[1000*r['max_penetration_m'] for r in rr],s=14)
    axes[0,0].axhline(.86,c='black',ls=':');axes[0,0].set(xlabel='Incoming speed [m/s]',ylabel='Measured restitution');axes[0,0].legend(fontsize=7)
    axes[0,1].set(xlabel='Incoming speed [m/s]',ylabel='Maximum penetration [mm]')
    for i in [12,16,20,24,28,32]:
        a=traces[i];axes[1,0].plot(1000*a[:,0],1000*a[:,1]);axes[1,1].plot(1000*a[:,0],a[:,2])
    axes[1,0].set(xlabel='Time [ms]',ylabel='Ball-to-face gap [mm]');axes[1,1].set(xlabel='Time [ms]',ylabel='Normal velocity [m/s]')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Newton SolverMuJoCo contact calibration · 0.5 ms RK4\n2e6 /s² native reference stiffness; phase and impact-speed sensitivity; no scripted impulses')
    fig.savefig(directory/'report.pdf');plt.close(fig)
    with Path('EXPERIMENTS.md').open('a') as f:f.write(f'\n## {datetime.now(timezone.utc).isoformat()} — Newton contact calibration\n\nNative soft-contact bounce measurements over damping, speed and impact phase. [PDF]({directory}/report.pdf), [data]({directory}/results.json). No MPC or manually applied impulses.\n')
    for d in sorted(set(r['damping'] for r in rows)):
        e=[r['restitution'] for r in rows if r['damping']==d];print(d,min(e),max(e),np.mean(e),flush=True)
    print(directory,flush=True)

if __name__=='__main__':main()
