"""Synchronous batched Newton rollout for expert collection and policy evaluation.

Simulated time is frozen throughout MPC. Sensor/servo clocks advance exclusively
with native Newton steps. Training data are irreversibly censored after ball loss.
"""
import bootstrap
import argparse,json,time,hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import warp as wp
from factors import Factors
from mpc import MPC
from warmstart import WarmStart
from execution import Execution
from initialization import FixtureIK
from projector import Projectors
from empirical import EmpiricalBatch
from empirical_newton import EmpiricalNewton
from observer_batch import ObserverBatch
from sampling import drop_positions
from dataset import BallLossMask,finite_samples
from loss import ball_lost
from snapshot import fingerprint,assert_current
from qualify_expert import qualifies
from uncertainty_settings import settings,require_matching


def run(a, sink=None):
    if a.native_expert and a.policy:raise ValueError('Choose a native diagnostic or a learned policy, not both')
    if a.collect and not a.smoke and not a.expert_qualification:raise ValueError('Qualify the paired native/Warp expert before collection')
    multipliers=settings(a.multiplier,getattr(a,'arm_multiplier',None))
    root=Path(__file__).parent;live_hashes=fingerprint(a.source.resolve(),a.calibration.resolve());model=json.loads(a.calibration.read_text());start=time.monotonic()
    runtime_sources=('rollout.py','qp.py','mpc.py','compensated_residual.py','factors.py',
        'warmstart.py','execution.py','initialization.py','projector.py','empirical.py',
        'empirical_newton.py','empirical_scene.py','newton_batch.py','observer_batch.py',
        'sampling.py','dataset.py','loss.py','chunked_factors.py','bootstrap.py','uncertainty_settings.py')
    implementation_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in runtime_sources}
    if not a.smoke:
        parity=json.loads((a.factors.parent/'gpu_mpc_parity.json').read_text())
        if not parity['passed']:raise RuntimeError('Full MPC parity required before collection')
        if a.seconds!=30 and not (a.collect and a.seconds in (5,10,20)):raise ValueError('Use 5/10/20-second collection or 30-second validation')
    a.output.mkdir(parents=True,exist_ok=False)
    f=Factors(a.factors);c=f.cfg;batch=a.batch;dt=.0005;steps=round(a.seconds/dt)
    if a.expert_qualification:
        certificate=json.loads(a.expert_qualification.read_text())
        if not certificate['passed']:raise ValueError('Expert qualification did not pass')
        if not certificate.get('reuse_completed_gates'):require_matching(certificate,dict(uncertainty_multipliers=multipliers))
        if certificate['mpc_spec']!=f.meta['spec'] or certificate['calibration_sha256']!=hashlib.sha256(a.calibration.read_bytes()).hexdigest() or (not certificate.get('reuse_completed_gates') and certificate['noise_disturbance_multiplier']!=a.multiplier):
            raise ValueError('Expert qualification does not match this MPC/empirical model')
    draws=EmpiricalBatch(a.calibration,batch,a.seed,a.multiplier,getattr(a,'arm_multiplier',None))
    origin=np.asarray(f.arrays['origin']);nominal=model['nominal_reference'];plane=nominal['nominal_face_base_m'][2]
    initial_plane=origin[2]+nominal['initial_offset_m'][2]
    initial=FixtureIK(f);rng=np.random.default_rng(a.seed);positions=[];postures=[];rejected=0
    while sum(len(v) for v in positions)<batch:
        count=batch-sum(len(v) for v in positions)
        ball,bounds=drop_positions(rng,count,origin,c,[r['ball_base_m'] for r in model['initial_conditions']],nominal_strike_z=plane)
        target=ball.copy();target[:,2]=initial_plane
        q,valid=initial.solve(jnp.asarray(target,dtype=jnp.float32));valid=np.asarray(valid);q=np.asarray(q)
        positions.append(ball[valid]);postures.append(q[valid]);rejected+=int((~valid).sum())
        if rejected>20*batch:raise RuntimeError('Cannot sample enough reachable fixture poses')
    ball=np.concatenate(positions);q0=np.concatenate(postures)
    fixture=ball-origin;fixture[:,2]=initial_plane-origin[2]
    parameters=draws.parameter_draws()
    plant=EmpiricalNewton(a.calibration,root/'calibration/newton_contact_fit.json',batch,a.output/'scene',f.meta['spec'],q0,ball,parameters)
    plant.enable_graph()
    warm=WarmStart(f);execution=Execution(f,warm);projectors=Projectors(f,batch)
    teacher=None if a.policy else (True if a.native_expert else MPC(a.factors))
    native=[]
    native_pool=None
    if a.native_expert:
        if a.collect:raise ValueError('Native CPU teacher is diagnostic only; data collection uses WarpMPC')
        from cpp.python_controller import NativeController
        native=[NativeController(a.factors.parent/'controller/libjuggling_controller.so',a.factors.parent/'factors',compiled=a.native_compiled) for _ in range(batch)]
        if a.native_workers>1:native_pool=ThreadPoolExecutor(max_workers=a.native_workers)
    if a.policy:
        weights=np.load(a.policy);layers=[];i=0
        while f'weight_{i}' in weights:
            layers.append((jnp.asarray(weights[f'weight_{i}']),jnp.asarray(weights[f'bias_{i}'])));i+=1
        if [w.shape[1] for w,b in layers]!=[256,256,256,len(weights['output_mean'])] or len(weights['output_mean']) not in (7,84):raise ValueError('Expected requested 3 x 256 policy')
        @jax.jit
        def infer(x):
            y=(x-jnp.asarray(weights['input_mean']))/jnp.asarray(weights['input_scale'])
            for w,b in layers[:-1]:y=jnp.tanh(jnp.matmul(y,w,precision=jax.lax.Precision.HIGHEST)+b)
            w,b=layers[-1];return (jnp.matmul(y,w,precision=jax.lax.Precision.HIGHEST)+b)*jnp.asarray(weights['output_scale'])+jnp.asarray(weights['output_mean'])+jnp.matmul(x,jnp.asarray(weights['output_baseline']),precision=jax.lax.Precision.HIGHEST)
    action_fn=jax.jit(execution.action);project_action=jax.jit(execution.project_action);project=jax.jit(execution.project);callback=jax.jit(execution.callback);torque_fn=jax.jit(warm.torque)
    x=plant.observe().numpy();geometry=np.asarray(execution.geometry(jnp.asarray(x[:,6:13]),jnp.asarray(x[:,13:20]))).reshape(batch,48)
    observer=ObserverBatch(a.observer,c,x[:,:3]+a.multiplier*draws.row('noise_position_samples_m'),np.zeros((batch,3)),geometry)
    estimated=observer.step(0.,geometry,np.full(batch,np.nan),np.zeros((batch,3)))
    mask=BallLossMask(batch);pending=deque();first_impact=np.full(batch,np.inf);contact_counts=np.zeros(batch,int);contacts_before_loss=np.zeros(batch,int)
    command=jnp.zeros((batch,7));command_acc=jnp.zeros_like(command);history=deque()
    lag=jnp.asarray(parameters['servo_lag_s'],dtype=jnp.float32)
    dead=np.broadcast_to(np.asarray(parameters['servo_dead_time_s']).reshape(batch,-1),(batch,7))
    delays=jnp.asarray(np.ceil(dead/dt-1e-9).astype(int));maxdelay=int(np.max(delays));ring=jnp.zeros((maxdelay+1,batch,7))
    @jax.jit
    def motor(ring,tick,command,dq):
        ring=ring.at[tick%(maxdelay+1)].set(command)
        target=ring[(tick-delays)%(maxdelay+1),jnp.arange(batch)[:,None],jnp.arange(7)[None,:]]
        return ring,(target-dq)*(-jnp.expm1(-dt/lag))/dt
    z=dual=None;ramp=None;samples=[];sample_count=0;robot_flags=np.zeros(batch,int);projected_flags=np.zeros(batch,int)
    zero_capture=np.full(batch,np.nan);zero_position=np.zeros((batch,3));last_solve=0.
    timing=dict(mpc_wall_s=0.,plant_wall_s=0.)
    filtering=dict(kept=0,nonfinite=0,projection=0,after_ball_loss=0)
    recent=deque(maxlen=3);loss_counts={k:0 for k in ['stale','nonfinite','height','unreachable']}
    for tick in range(steps+1):
        t=tick*dt
        state=wp.to_jax(plant.observe());q=state[:,6:13];dq=state[:,13:20]
        if tick%20==0:
            host=np.asarray(state);capture,arrival,point=draws.measurement(t,host[:,:3])
            pending.append([capture,arrival,point,np.zeros(batch,bool)])
        if tick%4==0:
            geometry=np.asarray(execution.geometry(q,dq)).reshape(batch,48)
            observer.step(t,geometry,zero_capture,zero_position,predict=False)
            for frame in pending:
                capture,arrival,point,delivered=frame
                ready=(np.maximum(capture,arrival)<=t+1e-10)&~delivered
                if ready.any():
                    observer.step(t,geometry,np.where(ready,capture,np.nan),point,predict=False);delivered|=ready
            while pending and pending[0][3].all():pending.popleft()
        if tick%40==0:
            contacts_before_loss=np.where(mask.lost,contacts_before_loss,contact_counts)
            estimated=observer.step(t,geometry,zero_capture,zero_position)
            x=np.asarray(state).copy();x[:,:6]=estimated[:,:6];x[:,20:23]=estimated[:,6:9]
            first_impact=np.minimum(first_impact,np.where(np.isfinite(estimated[:,9]),estimated[:,9],np.inf))
            u=np.clip((t-first_impact)/c['return_to_nominal_s'],0,1);fraction=u**3*(10+u*(-15+6*u))
            reference=(1-fraction[:,None])*fixture+fraction[:,None]*(np.asarray(nominal['nominal_face_base_m'])-origin)
            posture=(1-fraction[:,None])*q0+fraction[:,None]*np.asarray(nominal['nominal_posture'])
            flags=ball_lost(x,estimated[:,11]>0,reference,fixture,c,return_details=True)
            event=np.logical_or.reduce(list(flags.values()));first=event&~mask.lost
            for key in flags:loss_counts[key]+=int(np.sum(flags[key]&first))
            if a.diagnostic and first.any():
                np.savez_compressed(a.output/f'loss_{tick:06d}.npz',worlds=np.flatnonzero(first),truth=np.asarray(state),estimate=x,observer=estimated,reference=reference,posture=posture,initial_ball=ball,initial_posture=q0,contacts=contact_counts,**{f'flag_{k}':v for k,v in flags.items()})
                for j,frame in enumerate(recent):np.savez_compressed(a.output/f'loss_{tick:06d}_preceding_{j}.npz',**frame)
                print(json.dumps(dict(time_s=t,new_losses=int(first.sum()),loss_causes=loss_counts)),flush=True)
            alive=mask.update(t,event)
            if tick==steps:break
            # Lost worlds keep finite safe inputs but never yield another training row.
            safe=np.nan_to_num(x).astype('float32');safe[~alive,:6]=np.r_[np.zeros(2),.6,np.zeros(3)];safe[~alive,6:13]=q0[~alive];safe[~alive,13:]=0
            tau=torque_fn(jnp.asarray(safe[:,6:13]),jnp.asarray(safe[:,13:20]),command_acc).reshape(batch,7)
            inputs=jnp.concatenate([jnp.asarray(safe),tau,jnp.asarray(reference,dtype=jnp.float32),jnp.asarray(posture,dtype=jnp.float32)],axis=1)
            solve_start=time.monotonic()
            if a.native_expert:
                host_tau=np.asarray(tau)
                def native_step(i):
                    controller=native[i]
                    controller.set_reference(reference[i],posture[i])
                    if tick==0:controller.initialize(safe[i],host_tau[i])
                    return controller.step(safe[i],host_tau[i],0. if tick==0 else .02)[0]
                plans=list(native_pool.map(native_step,range(batch))) if native_pool else [native_step(i) for i in range(batch)]
                plan=jnp.asarray(np.asarray(plans),dtype=jnp.float32)
                outputs=action_fn(plan) if sink is not None else jnp.concatenate([plan[:,:4,6:13],plan[:,:4,13:20],plan[:,:4,23:30]],axis=2).reshape(batch,84)
            elif teacher:
                if z is None:z,dual=warm.initialize(jnp.asarray(safe),tau)
                else:z,dual,_=warm.shift(z,dual,jnp.asarray(safe),tau)
                projector=jnp.asarray(projectors(posture))
                if a.diagnostic:frame=dict(time_s=t,z=np.asarray(z).copy(),dual=np.asarray(dual).copy(),p=np.asarray(inputs).copy(),projector=np.asarray(projector).copy())
                z,dual,plan=teacher.solve(z,dual,inputs,projector,warm.event_parameters)
                if a.diagnostic:frame['plan']=np.asarray(plan).copy();recent.append(frame)
                outputs=action_fn(plan) if sink is not None else jnp.concatenate([plan[:,:4,6:13],plan[:,:4,13:20],plan[:,:4,23:30]],axis=2).reshape(batch,84)
            else:
                outputs=infer(inputs)
                if outputs.shape[1]==84:
                    nodes=outputs.reshape(batch,4,21)
                    plan=jnp.zeros((batch,4,34)).at[:,:,6:20].set(nodes[:,:,:14]).at[:,:,23:30].set(nodes[:,:,14:])
            if not teacher and outputs.shape[1]==7:
                ramp,raw=project_action(outputs,q,dq,command_acc)
            else:
                ramp,raw=project(plan,q,dq,command_acc)
            raw=np.asarray(raw)
            timing['mpc_wall_s']+=time.monotonic()-solve_start
            projected_flags+=(raw>1.25)&alive
            if teacher and a.collect:
                if sink is not None:
                    finite=np.asarray(jnp.all(jnp.isfinite(inputs),axis=1)&jnp.all(jnp.isfinite(outputs),axis=1))&alive&np.isfinite(raw)
                else:
                    inp=np.asarray(inputs);out=np.asarray(outputs);finite=finite_samples(inp,out,alive)&np.isfinite(raw)
                keep=finite&(raw<=1.25)
                filtering['kept']+=int(keep.sum());filtering['nonfinite']+=int((alive&~finite).sum());filtering['projection']+=int((finite&(raw>1.25)).sum());filtering['after_ball_loss']+=int((~alive).sum())
                if sink is not None:sink.append(inputs,outputs,keep)
                else:samples.append((inp[keep],out[keep],np.flatnonzero(keep)));sample_count+=int(keep.sum())
                if sink is None and sample_count>=a.shard_rows:
                    write_shard(a.output,samples);samples=[];sample_count=0
            last_solve=t
            if tick%2000==0:print(json.dumps(dict(time_s=t,alive=int(alive.sum()),wall_s=time.monotonic()-start,**timing)),flush=True)
        if tick%2==0:
            ref=execution.reference(ramp,t-last_solve)
            command,command_acc,violation=callback(ref,q,dq,command,command_acc)
            robot_flags+=np.asarray(violation)&~mask.lost
        ring,motor_acc=motor(ring,tick,command,dq)
        torque=torque_fn(q,dq,motor_acc).reshape(batch,7)
        acceleration=wp.array(draws.acceleration(t).astype('float32'),device='cuda:0')
        residual=wp.array(draws.impact.astype('float32'),device='cuda:0')
        plant_start=time.monotonic()
        plant.step(wp.from_jax(torque),acceleration,residual)
        counts=plant.counts.numpy();hit=np.flatnonzero(counts!=contact_counts)
        if len(hit):draws.new_flight(hit,t+dt)
        contact_counts=counts
        timing['plant_wall_s']+=time.monotonic()-plant_start
    observer.close()
    if native_pool:native_pool.shutdown()
    for controller in native:controller.close()
    if samples:write_shard(a.output,samples)
    uptime=np.minimum(mask.first_loss,a.seconds);stabilized=False
    report=dict(seeds=[f'{a.seed}:{i}' for i in range(batch)],seed_protocol='batch seed plus world index',seed=a.seed,rollout_seconds=a.seconds,uptime_s=uptime.tolist(),mean_uptime_s=float(uptime.mean()),survival_fraction=float(np.mean(uptime>=a.seconds)),expert_stabilization_passed=bool(teacher and stabilized and not a.native_expert),calibration_sha256=hashlib.sha256(a.calibration.read_bytes()).hexdigest(),noise_disturbance_multiplier=a.multiplier,policy_kind=('native_diagnostic' if a.native_expert else 'network' if a.policy else 'warp_mpc'),mpc_feedback='instantaneous paused simulation',wall_s=time.monotonic()-start,robot_velocity_flag_ticks=robot_flags.tolist(),projection_flag_ticks=projected_flags.tolist(),contacts=contact_counts.tolist(),drop_bounds=bounds,rejected_ik_samples=rejected,timing=timing,newton_cuda_graph=True,loss_causes=loss_counts)
    report['uncertainty_multipliers']=multipliers
    report['contacts_before_loss']=contacts_before_loss.tolist()
    report['filtering']=filtering
    report['implementation_sha256']=implementation_hashes
    report['solver_numerics']=teacher.qp.numerics if teacher and not a.native_expert else None
    if a.expert_qualification:
        certificate=json.loads(a.expert_qualification.read_text())
        stabilized=bool(not a.smoke and qualifies(report,certificate,f.meta['spec']))
        report['expert_stabilization_passed']=bool(teacher and stabilized and not a.native_expert)
        report['expert_qualification']=certificate
    if a.policy:report['checkpoint_sha256']=hashlib.sha256(a.policy.read_bytes()).hexdigest()
    (a.output/'rollout.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.collect:
        assert_current(a.source.resolve(),a.calibration.resolve(),live_hashes)
        manifest=dict(trajectories=batch,live_source_hashes=live_hashes,mpc_spec=f.meta['spec'],mpc_parity_passed=not a.smoke,expert_stabilization_passed=stabilized,policy_hidden_layers=[256,256,256],rollout=report)
        manifest['stage_times_s']=f.meta['stage_times'][:4]
        if sink is None:(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        else:sink.manifest=manifest
    return report


def write_shard(directory,samples):
    index=len(list(directory.glob('shard_*.npz')))
    np.savez(directory/f'shard_{index:06d}.npz',inputs=np.concatenate([s[0] for s in samples]),targets=np.concatenate([s[1] for s in samples]),trajectory_id=np.concatenate([s[2] for s in samples]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--factors',type=Path,required=True);p.add_argument('--observer',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=60000);p.add_argument('--seconds',type=float,default=30.);p.add_argument('--seed',type=int,default=2026);p.add_argument('--multiplier',type=float,default=1.5);p.add_argument('--arm-multiplier',type=float);p.add_argument('--shard-rows',type=int,default=1000000);p.add_argument('--policy',type=Path);p.add_argument('--collect',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--diagnostic',action='store_true');p.add_argument('--native-expert',action='store_true');p.add_argument('--native-workers',type=int,default=1);p.add_argument('--native-compiled',action='store_true');p.add_argument('--expert-qualification',type=Path);a=p.parse_args();run(a)
