"""GPU truth-plus-measured-noise collection: no observer or source-audit gate.

The full world batch is solved together. Only seven applied accelerations are
recorded; optimizer z/dual stay alive solely for the next MPC warm start.
"""
import bootstrap
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import warp as wp
from factors import Factors
from mpc import MPC
from warmstart import WarmStart
from execution import Execution, TorqueDynamics
from initialization import FixtureIK
from projector import GPUProjectors
from empirical import EmpiricalBatch
from gpu_empirical import GPUEmpirical
from empirical_newton import EmpiricalNewton
from sampling import drop_positions
from train_policy import forward
from uncertainty_settings import settings


def loss_function(c):
    """Original ball-only guards, applied on GPU to the available noisy state."""
    gravity=jnp.asarray([c['gravity_x'],c['gravity_y'],c['gravity_z']],jnp.float32)
    @jax.jit
    def flags(x,reference,fixture):
        plane=c['radius']+reference[:,2]
        def position(t):
            a=-jnp.expm1(-c['drag']*t)/c['drag'];b=(t-a)/c['drag']
            return x[:,:3]+a[:,None]*x[:,3:6]+b[:,None]*gravity
        hi=jax.lax.fori_loop(0,6,lambda _,hi:jnp.where(position(hi)[:,2]>plane,2*hi,hi),jnp.ones(x.shape[0],x.dtype))
        def bisect(_,bounds):
            lo,hi=bounds;mid=(lo+hi)*.5;above=position(mid)[:,2]>plane
            return jnp.where(above,mid,lo),jnp.where(above,hi,mid)
        lo,hi=jax.lax.fori_loop(0,40,bisect,(jnp.zeros_like(hi),hi));hit=position((lo+hi)*.5)[:,:2]
        def inside(p,front,side,rear):return (p[:,0]/jnp.where(p[:,0]>=0,front,rear))**2+(p[:,1]/side)**2<=1
        reach=inside(hit,c['landing_forward'],c['landing_side'],.35)
        if c['fixture_landing_forward']>0:reach&=inside(hit-fixture[:,:2],c['fixture_landing_forward'],c['fixture_landing_side'],.12)
        return jnp.stack((~jnp.isfinite(x[:,np.r_[0:6,20:23]]).all(axis=1),(x[:,2]<-.15)|(x[:,2]>1.5),(x[:,2]>plane)&~reach),axis=1)
    return flags


def run(a,sink=None):
    start=time.monotonic();root=Path(__file__).parent
    a.output.mkdir(parents=True,exist_ok=True)
    model=json.loads(a.calibration.read_text());f=Factors(a.factors);c=f.cfg;batch=a.batch
    dt=.0005;steps=round(a.seconds/dt);control_ticks=round(c['control_dt']/dt)
    multipliers=settings(a.multiplier,getattr(a,'arm_multiplier',None))
    draws=EmpiricalBatch(a.calibration,batch,a.seed,a.multiplier,getattr(a,'arm_multiplier',None))
    origin=np.asarray(f.arrays['origin']);nominal=model['nominal_reference'];plane=nominal['nominal_face_base_m'][2]
    initial_plane=origin[2]+nominal['initial_offset_m'][2]
    initial=FixtureIK(f);rng=np.random.default_rng(a.seed);positions=[];postures=[];rejected=0
    while sum(map(len,positions))<batch:
        count=batch-sum(map(len,positions))
        ball,bounds=drop_positions(rng,count,origin,c,[r['ball_base_m'] for r in model['initial_conditions']],nominal_strike_z=plane)
        target=ball.copy();target[:,2]=initial_plane
        q,valid=initial.solve(jnp.asarray(target,jnp.float32));valid=np.asarray(valid);q=np.asarray(q)
        positions.append(ball[valid]);postures.append(q[valid]);rejected+=int((~valid).sum())
        if rejected>20*batch:raise RuntimeError('Cannot sample reachable fixture poses')
    ball=np.concatenate(positions);q0=np.concatenate(postures)
    fixture=ball-origin;fixture[:,2]=initial_plane-origin[2]
    parameters=draws.parameter_draws()
    plant=EmpiricalNewton(a.calibration,root/'calibration/newton_contact_fit.json',batch,a.output/'scene',f.meta['spec'],q0,ball,parameters)
    plant.enable_graph();noise=GPUEmpirical(draws)
    warm=TorqueDynamics(f) if a.policy else WarmStart(f)
    execution=Execution(f,warm,observer_geometry=False)
    teacher=None if a.policy else MPC(f,action_only=True)
    projectors=GPUProjectors(f) if teacher else None
    action=jax.jit(execution.applied_action) if teacher else None
    if a.policy:
        weights=np.load(a.policy);layers=[(jnp.asarray(weights[f'weight_{i}']),jnp.asarray(weights[f'bias_{i}'])) for i in range(4)]
        mean=jnp.asarray(weights['input_mean']);scale=jnp.asarray(weights['input_scale'])
        output_mean=jnp.asarray(weights['output_mean']);output_scale=jnp.asarray(weights['output_scale'])
        baseline=jnp.asarray(weights['output_baseline']) if np.any(weights['output_baseline']) else None
        @jax.jit
        def infer(inputs):
            result=forward(layers,(inputs-mean)/scale)*output_scale+output_mean
            return result if baseline is None else result+jnp.matmul(inputs,baseline,precision=jax.lax.Precision.HIGHEST)
    q0_gpu=jnp.asarray(q0,jnp.float32);fixture_gpu=jnp.asarray(fixture,jnp.float32)
    nominal_face=jnp.asarray(np.asarray(nominal['nominal_face_base_m'])-origin,jnp.float32)
    nominal_q=jnp.asarray(nominal['nominal_posture'],jnp.float32)
    lost=jnp.zeros(batch,bool);first_loss=jnp.full(batch,jnp.inf);contact_before_loss=jnp.zeros(batch,jnp.int32)
    loss_totals=jnp.zeros(3,jnp.int32);filter_totals=jnp.zeros(4,jnp.int32)
    ball_flags=loss_function(c)
    @jax.jit
    def prepare(state,t,first_contact,contacts,lost,first_loss,contact_before_loss,loss_totals,command_acc):
        u=jnp.clip((t-first_contact)/c['return_to_nominal_s'],0,1);fraction=u**3*(10+u*(-15+6*u))
        reference=(1-fraction[:,None])*fixture_gpu+fraction[:,None]*nominal_face
        posture=(1-fraction[:,None])*q0_gpu+fraction[:,None]*nominal_q
        flags=ball_flags(state,reference,fixture_gpu);event=jnp.any(flags,axis=1);first=event&~lost
        loss_totals+=jnp.sum(flags&first[:,None],axis=0,dtype=jnp.int32)
        contact_before_loss=jnp.where(lost,contact_before_loss,contacts)
        first_loss=jnp.where(first,t,first_loss);lost|=event
        safe=jnp.nan_to_num(state)
        safe=safe.at[:,:6].set(jnp.where(lost[:,None],jnp.array([0.,0.,.6,0.,0.,0.]),safe[:,:6]))
        safe=safe.at[:,6:13].set(jnp.where(lost[:,None],q0_gpu,safe[:,6:13]))
        safe=safe.at[:,13:].set(jnp.where(lost[:,None],0.,safe[:,13:]))
        tau=warm.torque(safe[:,6:13],safe[:,13:20],command_acc).reshape(batch,7)
        inputs=jnp.concatenate((safe,tau,reference,posture),axis=1)
        return inputs,posture,lost,first_loss,contact_before_loss,loss_totals
    @jax.jit
    def record_filter(inputs,outputs,raw,lost,totals):
        finite=jnp.isfinite(inputs).all(axis=1)&jnp.isfinite(outputs).all(axis=1)&jnp.isfinite(raw)
        keep=~lost&finite&(raw<=1.25)
        totals+=jnp.stack((keep.sum(),(~lost&~finite).sum(),(~lost&finite&(raw>1.25)).sum(),lost.sum())).astype(jnp.int32)
        return keep,totals
    lag=jnp.asarray(parameters['servo_lag_s'],jnp.float32)
    dead=np.broadcast_to(np.asarray(parameters['servo_dead_time_s']).reshape(batch,-1),(batch,7))
    delays=jnp.asarray(np.ceil(dead/dt-1e-9).astype(np.int32));maxdelay=int(np.max(delays))
    ring=jnp.zeros((maxdelay+1,batch,7));command=jnp.zeros((batch,7));command_acc=jnp.zeros_like(command)
    @jax.jit
    def actuator(state,ring,tick,command,command_acc,ramp,age):
        q=state[:,6:13];dq=state[:,13:20]
        def callback(carry):
            command,command_acc=carry
            velocity,acceleration,_=execution.callback(execution.reference(ramp,age),q,dq,command,command_acc)
            return velocity,acceleration
        command,command_acc=jax.lax.cond(tick%2==0,callback,lambda carry:carry,(command,command_acc))
        ring=ring.at[tick%(maxdelay+1)].set(command)
        target=ring[(tick-delays)%(maxdelay+1),jnp.arange(batch)[:,None],jnp.arange(7)[None,:]]
        motor_acc=(target-dq)*(-jnp.expm1(-dt/lag))/dt
        torque=warm.torque(q,dq,motor_acc).reshape(batch,7)
        return ring,command,command_acc,torque
    @jax.jit
    def make_ramp(outputs,state,command_acc):
        q=state[:,6:13];dq=state[:,13:20]
        return q,dq,command_acc,(outputs-command_acc)/execution.h
    z=dual=None;ramp=None;last_solve=0.;state_wp=plant.observe()
    timing=dict(mpc_wall_s=0.,plant_dispatch_wall_s=0.);collection_start=time.monotonic()
    for tick in range(steps+1):
        t=tick*dt
        noise.update(t,plant.counts)
        state=wp.to_jax(state_wp)
        if tick%control_ticks==0:
            observed=wp.to_jax(noise.truth_observation(state_wp,t))
            inputs,posture,lost,first_loss,contact_before_loss,loss_totals=prepare(observed,t,wp.to_jax(noise.first_contact_time),wp.to_jax(plant.counts),lost,first_loss,contact_before_loss,loss_totals,command_acc)
            if tick==steps:break
            solve_start=time.monotonic()
            if teacher:
                if z is None:z,dual=warm.initialize(inputs[:,:23],inputs[:,23:30])
                else:z,dual,_=warm.shift(z,dual,inputs[:,:23],inputs[:,23:30])
                z,dual,knot=teacher.solve(z,dual,inputs,projectors(posture),warm.event_parameters)
                outputs,raw=action(knot)
            else:
                outputs=infer(inputs);raw=jnp.max(jnp.abs(outputs)/execution.acc_limit,axis=1)
                outputs=jnp.clip(outputs,-.95*execution.acc_limit,.95*execution.acc_limit)
            ramp=make_ramp(outputs,state,command_acc)
            if sink is not None:
                keep,filter_totals=record_filter(inputs,outputs,raw,lost,filter_totals)
                sink.append(inputs,outputs,keep)
            # One MPC synchronization per update; no CPU copies at physics rate.
            outputs.block_until_ready();timing['mpc_wall_s']+=time.monotonic()-solve_start
            last_solve=t
            if tick%2000==0:
                print(json.dumps(dict(time_s=t,alive=int(np.asarray((~lost).sum())),wall_s=time.monotonic()-start,**timing)),flush=True)
        plant_start=time.monotonic()
        ring,command,command_acc,torque=actuator(state,ring,tick,command,command_acc,ramp,t-last_solve)
        state_wp=plant.step(wp.from_jax(torque),noise.acceleration,noise.rebound)
        timing['plant_dispatch_wall_s']+=time.monotonic()-plant_start
    wp.synchronize()
    uptime=np.minimum(np.asarray(first_loss),a.seconds)
    contacts=np.asarray(contact_before_loss);counts=plant.counts.numpy()
    filter_values=np.asarray(filter_totals);loss_values=np.asarray(loss_totals)
    report=dict(seed=a.seed,seeds=[f'{a.seed}:{i}' for i in range(batch)],rollout_seconds=a.seconds,
        uptime_s=uptime.tolist(),mean_uptime_s=float(uptime.mean()),survival_fraction=float(np.mean(uptime>=a.seconds)),
        expert_stabilization_passed=bool(teacher and uptime.mean()>=min(3,a.seconds)),
        calibration_sha256=hashlib.sha256(a.calibration.read_bytes()).hexdigest(),noise_disturbance_multiplier=a.multiplier,
        uncertainty_multipliers=multipliers,policy_kind='warp_mpc' if teacher else 'network',
        observation_model='simulator truth plus empirical position and clock/delivery residuals; no observer',
        mpc_feedback='instantaneous paused simulation',wall_s=time.monotonic()-start,collection_wall_s=time.monotonic()-collection_start,
        contacts=counts.tolist(),contacts_before_loss=contacts.tolist(),drop_bounds=bounds,rejected_ik_samples=rejected,
        filtering=dict(zip(('kept','nonfinite','projection','after_ball_loss'),map(int,filter_values))),
        projection_filter='unclipped acceleration at the applied first future knot only',
        loss_causes=dict(zip(('nonfinite','height','unreachable'),map(int,loss_values))),timing=timing,
        solver_numerics=teacher.qp.numerics if teacher else None,newton_cuda_graph=True,
        mpc_spec=f.meta['spec'],factor_metadata_sha256=hashlib.sha256((a.factors/'metadata.json').read_bytes()).hexdigest(),
        mpc_dimensions=dict(variables=f.n,nonlinear_constraints=f.m,qp_rows=f.n+f.m),
        first_event_center='observed-state and shifted-plan lifecycle; frozen within SQP',
        contact_bias='zero: collection has no fitted observer or adaptive bias estimator')
    if a.policy:report['checkpoint_sha256']=hashlib.sha256(a.policy.read_bytes()).hexdigest()
    (a.output/'rollout.json').write_text(json.dumps(report,indent=2)+'\n')
    if sink is not None:
        sink.manifest=dict(trajectories=batch,live_source_hashes={},mpc_spec=f.meta['spec'],mpc_parity_passed=True,
            expert_stabilization_passed=report['expert_stabilization_passed'],policy_hidden_layers=[256,256,256],rollout=report)
    return report
