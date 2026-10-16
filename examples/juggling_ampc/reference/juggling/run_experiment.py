"""Reproducible single-solve and multi-second closed-loop research runs."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
from time import perf_counter
import casadi
import numpy as np
import osqp
import scipy
from .formulation import Config,NX,NZ,SCALE
from .solver import RTI
from .simulator import SimConfig,step
from .reporting import save_report
from .specs import EQUIPMENT_REFERENCE
from .disturbances import wind_acceleration, SMOOTH_WIND_DESCRIPTION


CONTROL_DT = .01  # MPC measurement/command period, independent of prediction nodes.


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='baseline')
    parser.add_argument('--seconds',type=float,default=6.)
    parser.add_argument('--sim-dt',type=float,default=.0005,
                        help='Plant integration interval [s], default 0.5 ms; must divide the fixed 10 ms MPC update period')
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--variable-time',action='store_true')
    parser.add_argument('--disturb',action='store_true',help='Enable persistent smooth wind')
    parser.add_argument('--wind',choices=['none','pulse','smooth'],default=None,
                        help='Explicit wind profile; --disturb defaults to smooth, pulse reproduces the historical short gust')
    parser.add_argument('--nominal-sim',action='store_true')
    parser.add_argument('--horizon',type=int,default=40)
    parser.add_argument('--sqp',type=int,default=Config.sqp_iters)
    parser.add_argument('--osqp',type=int,default=100)
    parser.add_argument('--h',type=float,default=.03,
                        help='Prediction-node interval [s], independent of the 10 ms MPC update period')
    parser.add_argument('--comp',type=float,default=200.)
    parser.add_argument('--energy',type=float,default=12.)
    parser.add_argument('--rho',type=float,default=Config.rho)
    parser.add_argument('--objective-scale',type=float,default=Config.objective_scale)
    parser.add_argument('--accel-weight',type=float,default=Config.accel_weight)
    parser.add_argument('--velocity-weight',type=float,default=Config.velocity_weight)
    parser.add_argument('--pose-weight',type=float,default=Config.pose_weight)
    parser.add_argument('--jerk-weight',type=float,default=Config.jerk_weight)
    parser.add_argument('--proximal',type=float,default=Config.proximal)
    parser.add_argument('--codegen',action='store_true')
    parser.add_argument('--rollout-shift',action='store_true',default=True,
                        help='Strict rollout of shifted controls (default)')
    parser.add_argument('--resample-shift',dest='rollout_shift',action='store_false',
                        help='Earlier experimental interpolation warmstart')
    args=parser.parse_args()
    if not np.isfinite(args.sim_dt) or not 0 < args.sim_dt <= CONTROL_DT:
        parser.error('--sim-dt must be finite, positive, and at most 0.01 s')
    substeps=round(CONTROL_DT/args.sim_dt)
    if not np.isclose(substeps*args.sim_dt,CONTROL_DT,rtol=0,atol=1e-12):
        parser.error('--sim-dt must divide the fixed 0.01 s MPC update period')
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    updates=round(args.seconds/CONTROL_DT)
    if not np.isclose(updates*CONTROL_DT,args.seconds,rtol=0,atol=1e-12):
        parser.error('--seconds must be a multiple of the fixed 0.01 s MPC update period')
    wind_profile=args.wind or ('smooth' if args.disturb else 'none')
    cfg=Config(horizon=args.horizon,h_nom=args.h,variable_time=args.variable_time,
               sqp_iters=args.sqp,osqp_iters=args.osqp,comp_weight=args.comp,
               energy_weight=args.energy,rho=args.rho,codegen=args.codegen,
               objective_scale=args.objective_scale,accel_weight=args.accel_weight,
               velocity_weight=args.velocity_weight,pose_weight=args.pose_weight,
               jerk_weight=args.jerk_weight,proximal=args.proximal)
    sim=SimConfig(ball_mass=cfg.mass,ball_radius=cfg.radius,paddle_radius=cfg.paddle_radius,
                  restitution=cfg.restitution if args.nominal_sim else .86,
                  drag=0 if args.nominal_sim else .02)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory=Path('runs')/(stamp+'_'+args.name);directory.mkdir(parents=True)
    import shutil,subprocess
    shutil.copytree('juggling',directory/'source',ignore=shutil.ignore_patterns('__pycache__'))
    config=dict(mpc=asdict(cfg),sim=asdict(sim),arguments=vars(args),equipment_reference=EQUIPMENT_REFERENCE,
                timing=dict(control_dt_s=CONTROL_DT,plant_dt_s=args.sim_dt,
                            plant_steps_per_update=substeps,
                            prediction_dt_s=None if cfg.variable_time else cfg.h_nom,
                            command_hold='constant acceleration between MPC updates',
                            computation_delay_modeled=False),
                wind_profile=wind_profile,wind_description=SMOOTH_WIND_DESCRIPTION if wind_profile=='smooth' else wind_profile,
                timestamp=datetime.now(timezone.utc).isoformat(),
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                versions=dict(python=platform.python_version(),casadi=casadi.__version__,
                              osqp=osqp.__version__,numpy=np.__version__,scipy=scipy.__version__),
                machine=platform.uname()._asdict(),target_apex=cfg.apex,
                strike_plane=0.,target_apex_band=[.5,1.])
    (directory/'config.json').write_text(json.dumps(config,indent=2))
    from .solver_rollout import RolloutRTI
    tstart=perf_counter();solver=(RolloutRTI if args.rollout_shift else RTI)(cfg);build_ms=(perf_counter()-tstart)*1000
    x=np.zeros(NX);x[2]=.67
    u,Y,info=solver.solve(x)
    plant=None
    if args.single:
        ts=np.r_[0,np.cumsum(Y[:-1,23])]
        arrays=dict(t=ts,x=Y[:,:NX],u=Y[:-1,16:21],
                    solve_ms=np.full(cfg.horizon,np.nan),data_ms=np.full(cfg.horizon,np.nan),
                    qp_ms=np.full(cfg.horizon,np.nan),violation=np.full(cfg.horizon,np.nan),
                    energy_apex=Y[:,2]-cfg.radius+Y[:,5]**2/19.62)
        arrays['solve_ms'][0]=info['total_ms'];arrays['data_ms'][0]=info['data_ms'];arrays['qp_ms'][0]=info['qp_ms']
        arrays['violation'][0]=info['violation']
        np.savez_compressed(directory/'prediction.npz',Y=Y)
        metrics=dict(kind='single solve prediction, not simulation',build_ms=build_ms,**info)
    else:
        xs=[x.copy()];ts=[0.];us=[];infos=[];contacts=[];pred=[];winds=[]
        plant_x=[x.copy()];plant_t=[0.];plant_wind=[];control_index=[]
        stop='duration completed';dt=CONTROL_DT;plant_ms=0.
        with (directory/'iterations.jsonl').open('w') as log:
            for k in range(updates):
                t=k*dt
                if k:
                    shift_start=perf_counter()
                    solver.shift(dt,x)
                    shift_ms=1000*(perf_counter()-shift_start)
                    u,Y,info=solver.solve(x)
                    info['shift_ms']=shift_ms
                    info['total_ms']+=shift_ms
                applied=np.clip(u,[-18,-18,-18,-35,-35],[18,18,18,35,35])
                info['control_clipping']=float(np.max(np.abs(applied-u)))
                log.write(json.dumps(dict(t=t,**info))+'\n')
                block_wind=[]
                plant_start=perf_counter()
                for j in range(substeps):
                    index=k*substeps+j
                    plant_time=index*args.sim_dt
                    wind=wind_acceleration((index+.5)*args.sim_dt,wind_profile)
                    x,cs=step(x,applied,args.sim_dt,sim,wind=wind)
                    for c in cs:
                        c.update(t_global=plant_time+c['time'],plant_step=index,control_index=k)
                        contacts.append(c)
                    plant_x.append(x.copy());plant_t.append((index+1)*args.sim_dt)
                    plant_wind.append(wind);block_wind.append(wind);control_index.append(k)
                    if not np.all(np.isfinite(x)) or x[2]<-.3 or x[2]>2.:
                        stop='ball left admissible simulation volume';break
                plant_ms+=1000*(perf_counter()-plant_start)
                xs.append(x.copy());ts.append(plant_t[-1]);us.append(applied);infos.append(info)
                winds.append(np.mean(block_wind,axis=0))
                if k%10==0:pred.append(Y.copy())
                if k%100==0:
                    print(f'{directory.name} t={t+dt:.2f} z={x[2]:.3f} contacts={len(contacts)} solve={info["total_ms"]:.1f}ms violation={info["violation"]:.2g}',flush=True)
                if stop!='duration completed':break
        X=np.array(xs);U=np.array(us);T=np.array(ts)
        PX=np.array(plant_x);PT=np.array(plant_t);PW=np.array(plant_wind)
        plant=dict(t=PT,x=PX,wind_accel=PW,control_index=np.array(control_index))
        peak_ids=np.flatnonzero((PX[:-1,5]>0)&(PX[1:,5]<=0))
        # The measured peaks use sampled true simulator states, independently of
        # the controller's energy prediction, at the full plant resolution.
        peaks=np.maximum(PX[peak_ids,2],PX[peak_ids+1,2])-cfg.radius
        ct=np.array([c['t_global'] for c in contacts])
        ca=np.array([c['position'][2]+cfg.radius*(c['normal'][2]-1)+c['ball_velocity_after'][2]**2/19.62 for c in contacts])
        arrays=dict(t=T,x=X,u=U,energy_apex=X[:,2]-cfg.radius+X[:,5]**2/19.62,
                    wind_accel=np.array(winds),
                    contact_t=ct,contact_apex=ca,
                    **{key:np.array([i[key] for i in infos]) for key in ['data_ms','qp_ms','violation',
                        'bound_violation','complementarity_gap_scaled','complementarity_restitution_scaled']},
                    solve_ms=np.array([i['total_ms'] for i in infos]))
        np.savez_compressed(directory/'predictions.npz',Y=np.array(pred))
        (directory/'contacts.json').write_text(json.dumps(contacts,indent=2,default=lambda a:np.asarray(a).tolist()))
        metrics=dict(kind='closed loop',stop_reason=stop,duration_s=float(T[-1]),
                     control_dt_s=dt,plant_dt_s=args.sim_dt,
                     controller_updates=len(infos),plant_steps=len(PT)-1,
                     plant_steps_per_update=substeps,plant_computation_ms=plant_ms,
                     contacts=len(contacts),contact_apices_m=ca.tolist(),
                     measured_apices_m=peaks.tolist(),
                     measured_apex_band_fraction=float(np.mean((peaks>=.5)&(peaks<=1.))) if len(peaks) else 0.,
                     apex_band_fraction=float(np.mean((ca>=.5)&(ca<=1.))) if len(ca) else 0.,
                     max_lateral_distance_m=float(np.max(np.linalg.norm(PX[:,:2],axis=1))),
                     max_paddle_speed_m_s=float(np.max(np.linalg.norm(PX[:,11:14],axis=1))),
                     max_paddle_tilt_rad=float(np.max(np.abs(PX[:,9:11]))),
                     max_paddle_pose_abs=np.max(np.abs(PX[:,6:11]),axis=0).tolist(),
                     max_paddle_velocity_abs=np.max(np.abs(PX[:,11:16]),axis=0).tolist(),
                     max_acceleration_m_s2=float(np.max(np.abs(U[:,:3]))),
                     max_control_clipping=float(max(i['control_clipping'] for i in infos)),
                     wind_acceleration_norm_range_m_s2=[float(np.min(np.linalg.norm(PW,axis=1))),float(np.max(np.linalg.norm(PW,axis=1)))],
                     qp_iterations_observed=sorted(set(q['iter'] for i in infos for q in i['iterations'] if 'iter' in q)),
                     sqp_iterations_observed=sorted(set(len(i['iterations']) for i in infos)),
                     solve_median_ms=float(np.median(arrays['solve_ms'])),
                     solve_p95_ms=float(np.percentile(arrays['solve_ms'],95)),
                     data_median_ms=float(np.median(arrays['data_ms'])),
                     qp_median_ms=float(np.median(arrays['qp_ms'])),
                     max_violation=float(np.max(arrays['violation'])),
                     max_bound_violation=float(np.max(arrays['bound_violation'])),
                     max_contact_gap_active_m=max(i['contact_gap_active_m'] for i in infos),
                     max_restitution_active_m_s=max(i['restitution_active_m_s'] for i in infos),
                     max_small_impulses_m_s=max(i['small_impulses_m_s'] for i in infos),
                     horizon_range_s=[min(i['horizon_s'] for i in infos),max(i['horizon_s'] for i in infos)],
                     build_ms=build_ms)
    save_report(directory,arrays,config,metrics,plant=plant)
    with Path('EXPERIMENTS.md').open('a') as log:
        log.write(f'\n## {config["timestamp"]} — {args.name}\n\n')
        log.write(f'- Command: `.venv/bin/python -m juggling.run_experiment {" ".join(__import__("sys").argv[1:])}`\n')
        log.write(f'- Artifacts: [{directory.name}]({directory}/report.pdf).\n')
        log.write(f'- Result: `{json.dumps(metrics)}`\n')
    print(directory,flush=True);print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
