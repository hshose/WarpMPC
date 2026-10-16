"""Fixed-budget contact-implicit MPC applied through Newton's solver interface."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.metadata
import json,platform,shutil,subprocess
from pathlib import Path
from time import perf_counter
import numpy as np
from arm.formulation import Config
from arm.model import TORQUE_LIMIT,VELOCITY_LIMIT,ACCEL_LIMIT,TORQUE_RATE_LIMIT,JERK_LIMIT
from arm.solver_ball_rollout import BallRolloutSolver
from arm.solver_impulse_shift import ImpulseShiftSolver
from arm.solver_state_shift import StateShiftSolver
from arm.solver_spin import SpinSolver
from arm.spin_formulation import Config as SpinConfig
from arm.reporting import report
from juggling.disturbances import wind_acceleration,SMOOTH_WIND_DESCRIPTION
from .scene import Scene,SceneConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='newton50');p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--wind',choices=['none','smooth'],default='smooth');p.add_argument('--damping',type=float,default=0.)
    p.add_argument('--contact-damping',type=float,default=650.);p.add_argument('--stiffness',type=float,default=2e6)
    p.add_argument('--friction',type=float,default=.3);p.add_argument('--control-dt',type=float,default=.02)
    p.add_argument('--initial-height',type=float,default=.67,help='Initial ball center above nominal paddle face, metres')
    p.add_argument('--friction-damping',type=float,default=1500.)
    p.add_argument('--predicted-restitution',type=float,default=.54);p.add_argument('--rate-margin',type=float,default=1.)
    p.add_argument('--actuator-margin',type=float,default=.98)
    p.add_argument('--h',type=float,default=.03);p.add_argument('--horizon',type=int,default=40)
    p.add_argument('--rho',type=float,default=.001);p.add_argument('--interpreted',action='store_true')
    p.add_argument('--codegen-opt',choices=['O0','O1'],default='O0')
    p.add_argument('--warmstart',choices=['ball','planned','shifted'],default='ball')
    p.add_argument('--contact-hold',action=argparse.BooleanOptionalAction,default=True,help='Hold the ramp endpoint briefly if a scheduled measurement falls inside compliant contact.')
    p.add_argument('--contact-model',choices=['normal','spin'],default='spin')
    p.add_argument('--seed-guard',action=argparse.BooleanOptionalAction,default=True,help='Keep shifted multiple-shooting seed if auxiliary ball rollout leaves optimizer workspace.')
    p.add_argument('--controller',choices=['python','cpp'],default='python')
    p.add_argument('--feedback',choices=['vicon','exact'],default='vicon')
    p.add_argument('--vicon-rate',type=float,default=200.)
    p.add_argument('--vicon-latency',type=float,default=.005)
    p.add_argument('--vicon-noise',type=float,default=.001,help='Per-axis position standard deviation in metres.')
    p.add_argument('--vicon-seed',type=int,default=7)
    p.add_argument('--vicon-dropout',type=float,default=0.)
    args=p.parse_args()
    if args.feedback=='vicon' and args.contact_model!='spin':p.error('Vicon observer currently supplies the spin formulation.')
    cfg=(SpinConfig if args.contact_model=='spin' else Config)(osqp_iters=50,control_dt=args.control_dt,restitution=args.predicted_restitution,
               rate_margin=args.rate_margin,actuator_margin=args.actuator_margin,h=args.h,horizon=args.horizon,
               rho=args.rho,codegen=not args.interpreted,codegen_optimization=args.codegen_opt)
    sc=SceneConfig(joint_damping=args.damping,contact_damping=args.contact_damping,contact_stiffness=args.stiffness,
                   friction=args.friction,friction_damping=args.friction_damping,gravity_x=getattr(cfg,'gravity_x',0.),gravity_y=getattr(cfg,'gravity_y',0.),gravity_z=getattr(cfg,'gravity_z',-9.81),drag=getattr(cfg,"drag",.34),ball_mass=cfg.mass,radius=cfg.radius,initial_height=args.initial_height)
    steps=round(args.seconds/sc.dt);substeps=round(cfg.control_dt/sc.dt)
    assert np.isclose(steps*sc.dt,args.seconds) and np.isclose(substeps*sc.dt,cfg.control_dt)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');directory=Path('runs')/(stamp+'_'+args.name);directory.mkdir()
    for source in ['arm','newton_sim','juggling','estimation','cpp']:
        shutil.copytree(source,directory/'source'/source,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('models/panda',directory/'model',ignore=shutil.ignore_patterns('meshes'))
    shutil.copyfile('requirements.lock',directory/'requirements.lock')
    start=perf_counter();scene=Scene(sc,directory);robot=scene.robot
    solver_type={'ball':BallRolloutSolver,'planned':ImpulseShiftSolver,'shifted':StateShiftSolver}[args.warmstart]
    nx=20
    if args.contact_model=='spin':
        assert args.warmstart=='ball','Spin model uses its consistent physical ball warmstart.'
        cfg.friction=args.friction;solver_type=SpinSolver;nx=23
    if args.controller=='cpp':
        if args.contact_model!='spin' or args.warmstart!='ball':p.error('C++ backend implements the validated spin/ball-warmstart controller.')
        from cpp.python_controller import CppSolver
        solver_type=CppSolver
    mpc=solver_type(cfg,robot);build_s=perf_counter()-start
    mpc.guard_ball_rollout=args.seed_guard
    feedback=None
    if args.feedback=='vicon':
        from .vicon import PositionFeedback,ViconConfig
        from estimation.ball import ObserverConfig
        initial=scene.observation()
        vc=ViconConfig(args.vicon_rate,args.vicon_latency,args.vicon_noise,args.vicon_seed,args.vicon_dropout)
        oc=ObserverConfig(gravity_x=cfg.gravity_x,gravity_y=cfg.gravity_y,gravity_z=cfg.gravity_z,position_std=args.vicon_noise,restitution=cfg.restitution,friction=cfg.friction,drag=getattr(cfg,"drag",.34),ball_mass=cfg.mass,contact_measurement_window=.015)
        feedback=PositionFeedback(robot,initial[:3],initial[6:13],initial[13:20],vc,oc)
    config=dict(timestamp=datetime.now(timezone.utc).isoformat(),arguments=vars(args),mpc=asdict(cfg),sim=asdict(sc),
        engine='Newton SolverMuJoCo / native CPU MuJoCo / RK4',
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        newton_revision=subprocess.check_output(['git','-C','resources/newton','rev-parse','HEAD'],text=True).strip(),
        versions={s:importlib.metadata.version(s) for s in ['newton','warp-lang','mujoco','mujoco-warp','casadi','osqp','numpy','pin']},
        machine=platform.uname()._asdict(),computation_delay_modeled=False,wind=SMOOTH_WIND_DESCRIPTION if args.wind=='smooth' else 'none',
        nominal_strike_world_m=robot.origin.tolist(),controller_formulation=mpc.formulation.__name__,
        controller_measurements='Exact simulated ball position/velocity, joint position/velocity, and ball spin for the spin model.',
        simulation_clock_note='Newton is advanced synchronously only after each MPC solve; computation consumes no simulated time.',
        limits=dict(torque=TORQUE_LIMIT.tolist(),velocity=VELOCITY_LIMIT.tolist(),acceleration=ACCEL_LIMIT.tolist(),
            torque_rate=TORQUE_RATE_LIMIT.tolist(),jerk=JERK_LIMIT.tolist(),joint_lower=robot.q_min.tolist(),joint_upper=robot.q_max.tolist()),
        model_note='Robot inertias and mounting come from the selected assembly/geometry spec (legacy defaults if absent). Newton uses the same transformed URDF as MPC, with a disk top at the modeled face offset. Fixture collision envelopes remain simplified. Ball mass is nominal, not identified from optical trajectories.',
        contact_note='Native solref gains calibrated in Newton; not material measurements. No scripted impulses, trajectory correction, or position tracking servo.',
        joint_damping_note='Synthetic optional mismatch; neither franka_ampc nor official identified friction.')
    if feedback:
        config.update(observer=asdict(oc),vicon=asdict(vc),observer_initial_measurement=feedback.initial_measurement,
            controller_measurements='Timestamped noisy ball position only; measured joint position/velocity. Ball velocity, spin, contact and wind truth are evaluation-only.',
            observer_state_order='position[0:3], velocity[3:6], spin[6:9], disturbance_acceleration[9:12]',
            observer_contact_hold='Predicted impact phase and estimated ball/racket geometry only; no native contact information.')
    if args.controller=='cpp':
        import hashlib
        paths=[Path('build/controller/libjuggling_controller.so'),Path('build/cpp_factors/libjuggling_factors.so')]
        (directory/'native').mkdir()
        config['native_controller']=dict(evaluator_threads=int(__import__('os').environ.get('JUGGLING_EVAL_THREADS','2')),timer='std::chrono::steady_clock inside the C++ step, including warmstart, data, OSQP, merit search and command extraction',
            compiler=subprocess.check_output(['clang++','--version'],text=True).splitlines()[0],flags='-O3 -DNDEBUG -march=native',
            osqp_source_revision=subprocess.check_output(['git','-C','resources/OSQP','rev-parse','HEAD'],text=True).strip(),
            binaries={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        for path in paths:shutil.copyfile(path,directory/'native'/path.name)
        shutil.copyfile('build/cpp_factors/metadata.json',directory/'native/factor_metadata.json')
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def observation(state):
        return np.r_[state,scene.state.body_qd.numpy()[scene.ball,3:].copy()] if nx==23 else state
    x=observation(scene.observation());tau=np.asarray(robot.bias(x[6:13],x[13:20])).ravel()
    estimate=x.copy();observer_cov=[];observer_info=[]
    if feedback:
        estimate,cov,oi=feedback.state(0.,x[6:13],x[13:20]);observer_cov.append(cov);observer_info.append(oi)
    states=[x.copy()];times=[0.];control_states=[estimate.copy()];control_times=[0.];infos=[];targets=[];events=[];other=[]
    values={k:[] for k in ['tau','torque_rate','wind','acceleration','total_acceleration','jerk','total_jerk','force','penetration','spin','control_index','plant_step_ms']}
    ppos=[];pspeed=[];frames_qpos=[scene.solver.mj_data.qpos.copy()];frame_t=[0.]
    pos,n,J,N=robot.numeric_fk(x[6:13]);ppos.append(pos-robot.origin);pspeed.append(np.linalg.norm(J@x[13:20]))
    previous_a=None;previous_total=None;in_contact=False;previous_contact=False;active=None;stop='duration completed';clipping=[]
    predictions=[];fps=50;frame_steps=round(1/fps/sc.dt);index=0
    with (directory/'iterations.jsonl').open('w') as log:
        for k in range(int(np.ceil(steps/substeps))):
            if index>=steps:break
            update_time=index*sc.dt
            start=perf_counter()
            try:
                if k:mpc.shift(control_times[-1]-control_times[-2],estimate,tau)
                shift_ms=1000*(perf_counter()-start);Y,info=mpc.solve(estimate,tau)
            except (RuntimeError,ValueError) as error:
                stop='controller failure: '+str(error);print(stop,flush=True);break
            info['shift_ms']=shift_ms;info['total_ms']+=shift_ms
            info['warmstart_fallback']=getattr(mpc,'warmstart_fallback',None)
            desired=tau+(Y[1,nx:nx+7]-Y[0,nx:nx+7])*cfg.control_dt/cfg.h
            target=np.clip(desired,np.maximum(-TORQUE_LIMIT,tau-TORQUE_RATE_LIMIT*cfg.control_dt),np.minimum(TORQUE_LIMIT,tau+TORQUE_RATE_LIMIT*cfg.control_dt))
            clipped=float(np.max(abs(target-desired)));clipping.append(clipped);info['torque_clipping_Nm']=clipped
            log.write(json.dumps(dict(t=update_time,**info))+'\n');log.flush()
            j=0
            while index<steps:
                t=index*sc.dt
                midpoint_tau=tau+(target-tau)*min((j+.5)/substeps,1.);wind=wind_acceleration(t+.5*sc.dt,args.wind)
                before=x.copy();start=perf_counter();x=observation(scene.step(midpoint_tau,wind));plant_ms=1000*(perf_counter()-start)
                if feedback:feedback.step(t+sc.dt,x[:3],x[6:13],x[13:20])
                ci=scene.contact_info();contact=ci['normal_force_N']>1e-8
                a=ci['acceleration'];total=ci['total_acceleration']
                jerk=np.zeros(7) if previous_a is None or contact or previous_contact else (a-previous_a)/sc.dt
                total_jerk=np.zeros(7) if previous_total is None else (total-previous_total)/sc.dt
                if contact and not in_contact:
                    active=dict(start_time=t,incoming_ball_velocity=before[3:6].tolist(),incoming_joint_velocity=before[13:20].tolist(),
                        impulse_Ns=0.,peak_force_N=0.,max_penetration_m=0.,samples=0)
                if contact:
                    active['impulse_Ns']+=ci['normal_force_N']*sc.dt;active['peak_force_N']=max(active['peak_force_N'],ci['normal_force_N'])
                    active['max_penetration_m']=max(active['max_penetration_m'],ci['penetration_m']);active['samples']+=1
                if not contact and in_contact:
                    active.update(end_time=t,outgoing_ball_velocity=x[3:6].tolist(),outgoing_joint_velocity=x[13:20].tolist())
                    events.append(active);active=None
                in_contact=contact;previous_contact=contact;previous_a=a.copy();previous_total=total.copy()
                for c in ci['other_contacts']:other.append(dict(time=t,**c))
                for key,val in dict(tau=midpoint_tau,torque_rate=(target-tau)/cfg.control_dt if j<substeps else np.zeros(7),wind=wind,acceleration=a,total_acceleration=total,
                    jerk=jerk,total_jerk=total_jerk,force=ci['normal_force_N'],penetration=ci['penetration_m'],
                    spin=scene.state.body_qd.numpy()[scene.ball,3:].copy(),control_index=k,plant_step_ms=plant_ms).items():values[key].append(val)
                states.append(x.copy());times.append(t+sc.dt)
                pos,n,J,N=robot.numeric_fk(x[6:13]);ppos.append(pos-robot.origin);pspeed.append(np.linalg.norm(J@x[13:20]))
                if (index+1)%frame_steps==0:frames_qpos.append(scene.solver.mj_data.qpos.copy());frame_t.append(t+sc.dt)
                index+=1;j+=1
                if not np.all(np.isfinite(x)) or x[2]<-.3 or x[2]>2 or np.max(abs(x[13:20]))>8:
                    stop='ball or arm left admissible volume';break
                if j>=substeps:
                    rel=x[:3]-(pos-robot.origin);gap=float(n@rel)-cfg.radius
                    radial=np.linalg.norm(rel-n*(n@rel))
                    pending_contact=feedback.observer.contact_pending(t+sc.dt) if feedback else contact or (-.005<gap<5e-5 and radial<cfg.paddle_radius)
                    if not (args.contact_hold and pending_contact and j<substeps+10):break
            tau=(tau+(target-tau)*min(j,substeps)/substeps).copy()
            estimate=x.copy()
            if feedback:
                estimate,cov,oi=feedback.state(times[-1],x[6:13],x[13:20]);observer_cov.append(cov);observer_info.append(oi)
                if oi['stale']:stop='ball position measurements stale'
            infos.append(info);targets.append(target);control_states.append(estimate.copy());control_times.append(times[-1])
            if k%10==0:predictions.append(Y)
            if k%50==0:print(f'{directory.name} t={times[-1]:.2f} z={x[2]:.3f} contacts={len(events)} other={len(other)} mpc={info["total_ms"]:.1f}ms clip={clipped:.3g}',flush=True)
            if stop!='duration completed':break
    if not infos:
        (directory/'failure.json').write_text(json.dumps(dict(stop=stop,elapsed_s=times[-1]),indent=2))
        raise RuntimeError(stop)
    if active:events.append(dict(**active,end_time=times[-1],incomplete=True))
    X=np.array(states);T=np.array(times);plant={key:np.array(val) for key,val in values.items()}
    plant.update(t=T,x=X,paddle_pos=np.array(ppos),paddle_speed=np.array(pspeed))
    peaks=np.flatnonzero((X[:-1,5]>0)&(X[1:,5]<=0));pi=peaks+(X[peaks+1,2]>X[peaks,2]);apices=X[pi,2]-cfg.radius
    ratios=dict(velocity=float(np.max(abs(X[:,13:20])/VELOCITY_LIMIT)),torque=float(np.max(abs(plant['tau'])/TORQUE_LIMIT)),
        acceleration=float(np.max(abs(plant['acceleration'])/ACCEL_LIMIT)),torque_rate=float(np.max(abs(plant['torque_rate'])/TORQUE_RATE_LIMIT)),
        jerk=float(np.max(abs(plant['jerk'])/JERK_LIMIT)))
    bounds_ok=bool(np.all(X[:,6:13]>=robot.q_min-1e-6) and np.all(X[:,6:13]<=robot.q_max+1e-6))
    arrays=dict(t=np.array(control_times),x=np.array(control_states),tau_target=np.array(targets),solve_ms=np.array([i['total_ms'] for i in infos]),
        **{key:np.array([i[key] for i in infos]) for key in ['data_ms','qp_ms','violation','bound_violation']})
    if args.controller=='cpp':
        for key in ['cpp_total_ms','cpp_shift_ms','cpp_merit_ms']:arrays[key]=np.array([i[key] for i in infos])
    metrics=dict(duration_s=float(T[-1]),stop_reason=stop,contacts=len(events),other_contact_samples=len(other),
        measured_apices_m=apices.tolist(),measured_apex_times_s=T[pi].tolist(),max_lateral_distance_m=float(np.max(np.linalg.norm(X[:,:2],axis=1))),
        max_paddle_speed_m_s=float(max(pspeed)),max_torque_clipping_Nm=float(max(clipping,default=0)),limit_ratios=ratios,joint_position_limits_ok=bounds_ok,
        max_contact_force_N=float(max(plant['force'])),max_penetration_m=float(max(plant['penetration'])),max_ball_spin_rad_s=float(np.max(np.linalg.norm(plant['spin'],axis=1))),
        max_total_acceleration_rad_s2=np.max(abs(plant['total_acceleration']),axis=0).tolist(),max_total_jerk_rad_s3=np.max(abs(plant['total_jerk']),axis=0).tolist(),
        solve_median_ms=float(np.median(arrays['solve_ms'])),solve_p95_ms=float(np.percentile(arrays['solve_ms'],95)),
        data_median_ms=float(np.median(arrays['data_ms'])),qp_median_ms=float(np.median(arrays['qp_ms'])),
        plant_step_median_ms=float(np.median(plant['plant_step_ms'])),max_violation=float(max(arrays['violation'])),
        max_bound_violation=float(max(arrays['bound_violation'])),plant_steps=len(T)-1,controller_updates=len(infos),build_s=build_s,
        sqp_iterations_observed=sorted(set(len(i['iterations']) for i in infos)),qp_iterations_observed=sorted(set(i['iterations'][0]['iter'] for i in infos)),
        engine_final_time_s=float(scene.solver.mj_data.time),
        warmstart_fallback_count=sum(bool(i.get('warmstart_fallback')) for i in infos),
        extended_update_intervals=int(np.count_nonzero(np.diff(control_times)>cfg.control_dt+sc.dt/2)),
        max_update_interval_s=float(max(np.diff(control_times))),
        task_pass=bool(stop=='duration completed' and len(apices)>=max(1,int(args.seconds*.8)) and np.all((apices>=.5)&(apices<=1.))
            and bounds_ok and max(ratios.values())<=1.0001 and max(clipping,default=0)<1e-6 and not other))
    if args.controller=='cpp':
        values_cpp=arrays['cpp_total_ms']
        metrics['cpp_timing_ms']={name:dict(count=len(v),median=float(np.median(v)),mean=float(np.mean(v)),p95=float(np.percentile(v,95)),maximum=float(np.max(v)))
            for name,v in [('all_calls',values_cpp),('after_first_call',values_cpp[1:])]}
    np.savez_compressed(directory/'plant.npz',**plant);np.savez_compressed(directory/'arrays.npz',**arrays)
    np.savez_compressed(directory/'frames.npz',qpos=np.array(frames_qpos),t=frame_t,fps=fps)
    np.savez_compressed(directory/'predictions.npz',Y=np.array(predictions))
    for name,data in [('metrics',metrics),('contacts',events),('other_contacts',other)]:
        (directory/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    if feedback:
        from .observer_reporting import save_observer_results
        save_observer_results(directory,feedback,arrays,plant,observer_cov,observer_info,metrics)
    (directory/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    report(directory,arrays,plant,config,metrics,robot)
    from .reporting import contact_report
    contact_report(directory,plant,metrics)
    with Path('EXPERIMENTS.md').open('a') as f:
        f.write(f'\n## {config["timestamp"]} — {args.name}\n\n- Command: `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m newton_sim.run_experiment {" ".join(__import__("sys").argv[1:])}`\n')
        f.write(f'- [Controller/arm PDF]({directory}/report.pdf), [Newton contact PDF]({directory}/contact.pdf).\n- Result: `{json.dumps(metrics)}`\n')
    print(directory,flush=True);print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
