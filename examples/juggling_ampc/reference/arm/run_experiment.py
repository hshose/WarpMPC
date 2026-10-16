"""Reproducible Stage 2 full-dynamics closed-loop trials."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import platform
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
import pinocchio as pin
import casadi
import osqp
import scipy
from juggling.disturbances import wind_acceleration,SMOOTH_WIND_DESCRIPTION
from .model import Panda,HOME,TORQUE_LIMIT,VELOCITY_LIMIT,ACCEL_LIMIT,TORQUE_RATE_LIMIT,JERK_LIMIT
from .formulation import Config
from .solver import Solver
from .solver_torque_rollout import TorqueRolloutSolver
from .solver_state_shift import StateShiftSolver
from .solver_ball_rollout import BallRolloutSolver
from .simulator import Simulator,SimConfig
from .reporting import report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='panda_initial25');p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--osqp',type=int,choices=[25,50],default=50)
    p.add_argument('--h',type=float,default=.03);p.add_argument('--horizon',type=int,default=40)
    p.add_argument('--control-dt',type=float,default=.02)
    p.add_argument('--rho',type=float,default=.001);p.add_argument('--accel-weight',type=float,default=.4)
    p.add_argument('--energy-weight',type=float,default=12.);p.add_argument('--velocity-weight',type=float,default=1.)
    p.add_argument('--rate-weight',type=float,default=.15);p.add_argument('--interpreted',action='store_true')
    p.add_argument('--actuator-margin',type=float,default=1.)
    p.add_argument('--rate-margin',type=float,default=1.)
    p.add_argument('--warmstart',choices=['ball','state','torque','acceleration'],default='ball')
    p.add_argument('--wind',choices=['smooth','none'],default='smooth');p.add_argument('--damping',type=float,default=0.)
    args=p.parse_args();cfg=Config(osqp_iters=args.osqp,h=args.h,horizon=args.horizon,control_dt=args.control_dt,
        rho=args.rho,accel_weight=args.accel_weight,energy_weight=args.energy_weight,velocity_weight=args.velocity_weight,
        rate_weight=args.rate_weight,actuator_margin=args.actuator_margin,rate_margin=args.rate_margin,codegen=not args.interpreted)
    substeps=round(cfg.control_dt/cfg.sim_dt);updates=round(args.seconds/cfg.control_dt)
    assert np.isclose(substeps*cfg.sim_dt,cfg.control_dt) and np.isclose(updates*cfg.control_dt,args.seconds)
    assert cfg.control_dt<=cfg.h
    robot=Panda();simcfg=SimConfig(damping=args.damping);sim=Simulator(robot,simcfg)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');directory=Path('runs')/(stamp+'_'+args.name);directory.mkdir(parents=True)
    shutil.copytree('arm',directory/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('models/panda',directory/'model')
    config=dict(timestamp=datetime.now(timezone.utc).isoformat(),mpc=asdict(cfg),sim=asdict(simcfg),arguments=vars(args),
        model='Panda, Gaz identified inertias, no gripper, 180 g racket + 70 g fixture',
        pinocchio_version=pin.__version__,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions=dict(python=platform.python_version(),pinocchio=pin.__version__,casadi=casadi.__version__,
                      osqp=osqp.__version__,numpy=np.__version__,scipy=scipy.__version__),
        machine=platform.uname()._asdict(),
        wind=SMOOTH_WIND_DESCRIPTION if args.wind=='smooth' else 'none',computation_delay_modeled=False,
        command='Piecewise-linear total joint torque, continuous across MPC updates',home_q=HOME.tolist(),
        nominal_strike_world=robot.origin.tolist(),limits=dict(torque=TORQUE_LIMIT.tolist(),velocity=VELOCITY_LIMIT.tolist(),
            acceleration=ACCEL_LIMIT.tolist(),torque_rate=TORQUE_RATE_LIMIT.tolist(),jerk=JERK_LIMIT.tolist(),
            joint_lower=robot.q_min.tolist(),joint_upper=robot.q_max.tolist()))
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    start=perf_counter();solver={'ball':BallRolloutSolver,'state':StateShiftSolver,'torque':TorqueRolloutSolver,'acceleration':Solver}[args.warmstart](cfg,robot);build_ms=1000*(perf_counter()-start)
    x=np.r_[0.,0.,.67,0.,0.,0.,HOME,np.zeros(7)]
    tau=pin.computeGeneralizedGravity(robot.pin,sim.data,HOME).copy()
    xs=[x.copy()];ts=[0.];px=[x.copy()];pt=[0.];infos=[];us=[];contacts=[];predictions=[]
    torques=[];rates=[];winds=[];accels=[];jerks=[];paddle=[];paddlespeed=[];clipping=[];index=[]
    pos,n,J,N=robot.numeric_fk(x[6:13]);paddle.append(pos-robot.origin);paddlespeed.append(np.linalg.norm(J@x[13:20]))
    stop='duration completed';previous_acceleration=None;previous_contact=False
    with (directory/'iterations.jsonl').open('w') as log:
        for k in range(updates):
            shift_start=perf_counter()
            try:
                if k:solver.shift(cfg.control_dt,x,tau)
                shift_ms=1000*(perf_counter()-shift_start)
                Y,info=solver.solve(x,tau)
            except (RuntimeError,ValueError) as error:
                if not infos:raise
                stop='controller failure: '+str(error);break
            info['shift_ms']=shift_ms;info['total_ms']+=shift_ms
            target=tau+(Y[1,20:27]-Y[0,20:27])*cfg.control_dt/cfg.h
            limited=np.clip(target,np.maximum(-TORQUE_LIMIT,tau-TORQUE_RATE_LIMIT*cfg.control_dt),
                                  np.minimum(TORQUE_LIMIT,tau+TORQUE_RATE_LIMIT*cfg.control_dt))
            clipped=float(max(abs(target-limited)));clipping.append(clipped);target=limited
            info['torque_clipping_Nm']=clipped;log.write(json.dumps(dict(t=k*cfg.control_dt,**info))+'\n');log.flush()
            for j in range(substeps):
                t=(k*substeps+j)*cfg.sim_dt
                ta=tau+(target-tau)*(j/substeps);tb=tau+(target-tau)*((j+1)/substeps)
                wind=wind_acceleration(t+.5*cfg.sim_dt,args.wind)
                acceleration=pin.aba(robot.pin,sim.data,x[6:13],x[13:20],ta-simcfg.damping*x[13:20]).copy()
                jerk=np.zeros(7) if previous_acceleration is None or previous_contact else (acceleration-previous_acceleration)/cfg.sim_dt
                torques.append(ta);rates.append((target-tau)/cfg.control_dt);winds.append(wind);accels.append(acceleration);jerks.append(jerk);index.append(k)
                x,cs=sim.step(x,ta,tb,cfg.sim_dt,wind=wind)
                for c in cs:c['t_global']=t+c['time'];contacts.append(c)
                previous_acceleration=acceleration;previous_contact=bool(cs)
                px.append(x.copy());pt.append(t+cfg.sim_dt)
                pos,n,J,N=robot.numeric_fk(x[6:13]);paddle.append(pos-robot.origin);paddlespeed.append(np.linalg.norm(J@x[13:20]))
                if not np.all(np.isfinite(x)) or x[2]<-.3 or x[2]>2. or max(abs(x[13:20]))>8:
                    stop='ball or arm left admissible volume';break
            tau=tb.copy();xs.append(x.copy());ts.append(pt[-1]);infos.append(info);us.append(target)
            if k%10==0:predictions.append(Y)
            if k%50==0:print(f'{directory.name} t={pt[-1]:.2f} z={x[2]:.3f} impacts={len(contacts)} qp={cfg.osqp_iters} mpc={info["total_ms"]:.1f}ms defect={info["violation"]:.3g} clip={clipped:.3g}',flush=True)
            if stop!='duration completed':break
    X=np.array(px);T=np.array(pt);P=np.array(paddle);U=np.array(torques);A=np.array(accels);V=np.array(jerks);R=np.array(rates)
    peaks=np.flatnonzero((X[:-1,5]>0)&(X[1:,5]<=0));peak_index=peaks+(X[peaks+1,2]>X[peaks,2]);apices=X[peak_index,2]-cfg.radius
    a=dict(t=np.array(ts),x=np.array(xs),tau_target=np.array(us),
        **{key:np.array([i[key] for i in infos]) for key in ['data_ms','qp_ms','violation','bound_violation']},
        solve_ms=np.array([i['total_ms'] for i in infos]))
    plant=dict(t=T,x=X,tau=U,torque_rate=R,wind=np.array(winds),acceleration=A,jerk=V,paddle_pos=P,
               paddle_speed=np.array(paddlespeed),control_index=np.array(index))
    ratios=dict(velocity=float(max(np.max(abs(X[:,13:20])/VELOCITY_LIMIT,axis=0))),
                torque=float(np.max(abs(U)/TORQUE_LIMIT)),acceleration=float(np.max(abs(A)/ACCEL_LIMIT)),
                torque_rate=float(np.max(abs(R)/TORQUE_RATE_LIMIT)),jerk=float(np.max(abs(V)/JERK_LIMIT)))
    joint_bounds=bool(np.all(X[:,6:13]>=robot.q_min-1e-6) and np.all(X[:,6:13]<=robot.q_max+1e-6))
    metrics=dict(duration_s=float(T[-1]),stop_reason=stop,contacts=len(contacts),measured_apices_m=apices.tolist(),
        measured_apex_times_s=T[peak_index].tolist(),max_lateral_distance_m=float(np.max(np.linalg.norm(X[:,:2],axis=1))),
        max_paddle_speed_m_s=float(max(paddlespeed)),max_abs_joint_torque_Nm=np.max(abs(U),axis=0).tolist(),
        max_abs_joint_velocity_rad_s=np.max(abs(X[:,13:20]),axis=0).tolist(),
        max_abs_joint_acceleration_rad_s2=np.max(abs(A),axis=0).tolist(),max_abs_joint_jerk_rad_s3=np.max(abs(V),axis=0).tolist(),
        max_abs_torque_rate_Nm_s=np.max(abs(R),axis=0).tolist(),limit_ratios=ratios,joint_position_limits_ok=joint_bounds,
        max_torque_clipping_Nm=float(max(clipping)),controller_updates=len(infos),plant_steps=len(U),
        sqp_iterations_observed=sorted(set(len(i['iterations']) for i in infos)),
        qp_iterations_observed=sorted(set(i['iterations'][0]['iter'] for i in infos)),
        solve_median_ms=float(np.median(a['solve_ms'])),solve_p95_ms=float(np.percentile(a['solve_ms'],95)),
        data_median_ms=float(np.median(a['data_ms'])),qp_median_ms=float(np.median(a['qp_ms'])),
        max_violation=float(max(a['violation'])),max_bound_violation=float(max(a['bound_violation'])),build_ms=build_ms,
        task_pass=bool(stop=='duration completed' and len(apices)>=max(1,int(args.seconds*.8)) and np.all((apices>=.5)&(apices<=1.))
                       and joint_bounds and max(ratios.values())<=1+1e-4 and max(clipping)<1e-6))
    for name,value in [('metrics',metrics),('contacts',contacts)]:
        (directory/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    np.savez_compressed(directory/'arrays.npz',**a);np.savez_compressed(directory/'plant.npz',**plant)
    np.savez_compressed(directory/'predictions.npz',Y=np.array(predictions))
    report(directory,a,plant,config,metrics,robot)
    with Path('EXPERIMENTS.md').open('a') as log:
        log.write(f'\n## {config["timestamp"]} — {args.name}\n\n')
        log.write(f'- Command: `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m arm.run_experiment {" ".join(__import__("sys").argv[1:])}`\n')
        log.write(f'- Artifacts: [{directory.name}]({directory}/report.pdf).\n- Result: `{json.dumps(metrics)}`\n')
    print(directory,flush=True);print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
