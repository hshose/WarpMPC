"""Native contact sweeps against the reduced hollow-ball impulse law.

This is numerical calibration/validation, not a fit to measured rubber data.
The stationary racket frame covers translating rackets by Galilean invariance.
"""
from datetime import datetime,timezone
import argparse,json,os,shutil
from pathlib import Path
import numpy as np
os.environ.setdefault('WARP_CACHE_PATH','/tmp/franka-warp')
os.environ.setdefault('MPLCONFIGDIR','/tmp/franka-matplotlib')
import warp as wp
import newton
from newton.solvers import SolverMuJoCo


def bounce(stiffness,damping,speed,tangent=0.,spin=0.,phase=0.,dt=.0005,friction=.3,friction_damping=0.):
    builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder)
    xml=f'''<mujoco><option gravity="0 0 0" integrator="RK4"/><default><geom
        solref="{-stiffness} {-damping}" solimp=".95 .95 .001 .5 2" friction="{friction} .0001 .00001" condim="3"/></default>
      <worldbody><geom name="racket" type="cylinder" size=".075 .006" pos="0 0 -.006"/>
      <body name="ball" pos="0 0 {.024+speed*phase}"><freejoint/><inertial pos="0 0 0" mass=".0027" diaginertia=".00000072 .00000072 .00000072"/>
      <geom name="ball_shape" type="sphere" size=".02"/></body></worldbody></mujoco>'''
    if friction_damping:
        pair=f'''<contact><pair geom1="racket" geom2="ball_shape" condim="3"
          solref="{-stiffness} {-damping}" solreffriction="0 {-friction_damping}"
          solimp=".95 .95 .001 .5 2" friction="{friction} {friction} .0001 .00001 .00001"/></contact>'''
        xml=xml.replace('</mujoco>',pair+'</mujoco>')
    builder.add_mjcf(xml);builder.joint_qd[:6]=[tangent,0.,-speed,0.,spin,0.]
    model=builder.finalize(device='cpu');solver=SolverMuJoCo(model,use_mujoco_cpu=True,integrator='RK4',
        solver='Newton',iterations=100,tolerance=1e-12,cone='elliptic',update_data_interval=0)
    state=model.state();out=model.state();control=model.control();solver.reset(state)
    newton.eval_fk(model,state.joint_q,state.joint_qd,state)
    trace=[]
    for k in range(round(.05/dt)):
        solver.step(state,out,control,None,dt);state,out=out,state
        q=state.joint_q.numpy();v=state.joint_qd.numpy();trace.append(np.r_[(k+1)*dt,q[2]-.02,v[:6]])
        if v[2]>0 and q[2]>.025:break
    a=np.array(trace);v=a[-1,2:8];r=.02;kappa=2/3
    jn=(1+.88)*speed;desired=-kappa/(1+kappa)*(tangent-r*spin);capacity=friction*jn
    jt=desired*capacity/np.sqrt(capacity**2+desired**2+1e-10)
    prediction=np.array([tangent+jt,0.,.88*speed,0.,spin-jt/(kappa*r),0.])
    before=.5*.0027*(tangent*tangent+speed*speed)+.5*.00000072*spin*spin
    after=.5*.0027*(v[:3]@v[:3])+.5*.00000072*(v[3:]@v[3:])
    return dict(stiffness=stiffness,damping=damping,speed=speed,tangent_m_s=tangent,spin_rad_s=spin,
        phase_s=phase,dt=dt,friction=friction,friction_damping=friction_damping,restitution=float(v[2]/speed),
        maximum_penetration_m=float(max(0,-min(a[:,1]))),contact_duration_s=float(np.count_nonzero(a[:,1]<0)*dt),
        outgoing_velocity=v[:3].tolist(),outgoing_spin=v[3:].tolist(),
        impulse_prediction=prediction.tolist(),prediction_law='MPC smooth Coulomb saturation, infinite racket mass',
        linear_error_m_s=float(np.linalg.norm(v[:3]-prediction[:3])),
        tangential_error_m_s=float(abs(v[0]-prediction[0])),spin_error_rad_s=float(np.linalg.norm(v[3:]-prediction[3:])),
        energy_ratio=float(after/before)),a


def main():
    p=argparse.ArgumentParser();p.add_argument('--stiffness',type=float,default=2e6);p.add_argument('--damping',type=float,default=110.)
    p.add_argument('--friction-damping',type=float,default=0.)
    p.add_argument('--normal-sweep',action='store_true');args=p.parse_args();wp.init()
    directory=Path('runs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_contact_consistency');directory.mkdir()
    shutil.copyfile(__file__,directory/'source.py');shutil.copyfile('requirements.lock',directory/'requirements.lock')
    rows=[];traces={}
    if args.normal_sweep:
        cases=[(K,D,s,0.,0.,phase,.0005) for K in [5e5,1e6,2e6] for D in [.055*np.sqrt(K),.075*np.sqrt(K),.095*np.sqrt(K)]
               for s in [2.5,3.5,4.5] for phase in [0.,.000125,.00025,.000375]]
    else:
        cases=[(args.stiffness,args.damping,s,s*np.tan(np.deg2rad(angle)),spin,phase,dt)
               for s in [2.5,3.5,4.5] for angle,spin in [(0,0),(5,0),(15,0),(5,2),(5,-2),(5,20),(5,-20)]
               for phase in [0.,.000125,.00025,.000375] for dt in [.0005,.0001]]
    for i,case in enumerate(cases):
        row,trace=bounce(*case,friction_damping=args.friction_damping);rows.append(row);traces[str(i)]=trace
    (directory/'results.json').write_text(json.dumps(rows,indent=2)+'\n');np.savez_compressed(directory/'traces.npz',**traces)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for (K,D,dt) in sorted(set((r['stiffness'],r['damping'],r['dt']) for r in rows)):
        rr=[r for r in rows if (r['stiffness'],r['damping'],r['dt'])==(K,D,dt)]
        label=f'K={K/1e6:g}M D={D:.1f} dt={dt*1000:g}ms'
        for ax,key,mult in [(axes[0,0],'restitution',1),(axes[0,1],'maximum_penetration_m',1000),
                            (axes[0,2],'contact_duration_s',1000),(axes[1,0],'tangential_error_m_s',1),
                            (axes[1,1],'spin_error_rad_s',1),(axes[1,2],'energy_ratio',1)]:
            ax.scatter([r['speed'] for r in rr],[mult*r[key] for r in rr],s=10,alpha=.6,label=label)
    axes[0,0].axhline(.88,c='black',ls=':');axes[0,0].legend(fontsize=5)
    for ax,label in zip(axes.flat,['Normal restitution','Equivalent compression [mm]','Contact duration [ms]',
                                 'Tangential impulse-model error [m/s]','Spin impulse-model error [rad/s]','Energy after / before']):
        ax.set(xlabel='Incident normal speed [m/s]',ylabel=label);ax.grid(alpha=.2)
    fig.suptitle('Newton compliant contact vs hollow-ball frictional impulse law\nNumerical consistency study; parameters are nominal assumptions, not measured rubber identification')
    fig.savefig(directory/'report.pdf');plt.close(fig)
    with Path('EXPERIMENTS.md').open('a') as f:
        f.write(f'\n## {datetime.now(timezone.utc).isoformat()} — contact model consistency\n\nCommand: `.venv/bin/python -m newton_sim.contact_consistency {" ".join(__import__("sys").argv[1:])}`. {len(rows)} isolated native-contact cases. [PDF]({directory}/report.pdf), [data]({directory}/results.json).\n')
    for K,D,dt in sorted(set((r['stiffness'],r['damping'],r['dt']) for r in rows)):
        rr=[r for r in rows if (r['stiffness'],r['damping'],r['dt'])==(K,D,dt)]
        print(K,D,dt,{key:(min(r[key] for r in rr),max(r[key] for r in rr)) for key in ['restitution','maximum_penetration_m','tangential_error_m_s','spin_error_rad_s','energy_ratio']},flush=True)
    print(directory,flush=True)


if __name__=='__main__':main()
