"""Offline Monte Carlo experiments with an independently calibrated ball/arm plant.

There are no sockets, robot SDK calls, external executables, or realtime sleeps.
Optional parallel trials use explicitly spawned, isolated Python workers.
The actual native controller, SessionControl, JointVelocityTracking, projection,
raw-marker association, release gate, and BallObserver are reused. Physical time
advances in 0.5 ms steps, including sampled measured computation/delivery delay;
wall time spent solving never advances the plant a second time.

A frozen plant library/factor directory may differ from each candidate controller
so controller geometry changes cannot silently change the physical paddle.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import ctypes as ct
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from cpp.python_controller import NativeController, DOUBLE
from estimation.ball import ObserverConfig
from .native_estimator import BallObserver, RawBall, ReleaseGate
from .reach import within_landing_reach, nominal_return_fraction
from . import session as live
from .fixture import predicted_landing
from .raw_ball import MAX_BALL_GAP
from .rehearsal import executable_trajectory
from optimization.kinematic_execution import torque_view


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_source_hashes():
    """Hash modules actually imported in this candidate process, not a checkout guess."""
    names={obj.__module__ for obj in [BallObserver,RawBall,ReleaseGate,live.model,live.Panda,NativeController,executable_trajectory,torque_view]}
    names.update(name for name in ['arm.ball_flight','arm.spin_contact','tracking.streams','tracking.contact_bias'] if name in sys.modules)
    return {name:digest(Path(sys.modules[name].__file__).resolve()) for name in sorted(names)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def stats(values):
    a=np.asarray(values, float)
    if not len(a): return dict(n=0)
    return dict(n=len(a), mean=np.mean(a,axis=0).tolist(), std=np.std(a,axis=0).tolist(),
                p05=np.quantile(a,.05,axis=0).tolist(), median=np.median(a,axis=0).tolist(),
                p95=np.quantile(a,.95,axis=0).tolist(), maximum=np.max(a,axis=0).tolist())


class NativePlant:
    """Version-1 35-double parameter API for the offline empirical extension.

    gravity(3), linear/quadratic drag, restitution, friction, radius, mass,
    inertia ratio, normal rotation vector radians(3), local center offset(3),
    elliptical half axes(2), velocity lag(7), dead time(7), acceleration filter tau,
    restitution edge loss, restitution radial normalization in metres.
    """
    dt=.0005
    def __init__(self,library,factors,q,ball_base,release,parameters,config,delivery_budget,execution_metadata=None):
        # A structurally different MPC has its own runtime schema. Load the
        # frozen plant's library with its own profile; NativeController resets
        # this environment variable before loading the candidate separately.
        runtime=Path(factors)/'runtime_values.txt'
        if runtime.exists():os.environ['JUGGLING_RUNTIME_VALUES']=str(runtime.resolve())
        self.lib=ct.CDLL(str(Path(library).resolve())); L=self.lib
        self.error=ct.create_string_buffer(4096); ptr=ct.c_void_p; err=[ct.c_char_p,ct.c_size_t]
        if not hasattr(L,'fixture_plant_empirical_version') or L.fixture_plant_empirical_version()!=1:
            raise ValueError('Build the empirical offline plant extension first')
        L.fixture_plant_create.argtypes=[ct.c_char_p,DOUBLE,DOUBLE,ct.c_double,*err];L.fixture_plant_create.restype=ptr
        L.fixture_plant_destroy.argtypes=[ptr]
        L.fixture_plant_configure.argtypes=[ptr,DOUBLE,ct.c_int,*err]
        L.fixture_plant_disturbance.argtypes=[ptr,DOUBLE,DOUBLE,ct.c_int,*err]
        L.fixture_plant_gravity.argtypes=[ptr,DOUBLE,*err]
        L.fixture_plant_step.argtypes=[ptr,DOUBLE,DOUBLE,ct.POINTER(ct.c_int),*err]
        L.fixture_plant_plan.argtypes=[ptr,ct.c_double,DOUBLE,*err]
        L.fixture_plant_delivery_limit.argtypes=[ptr,ct.c_double,*err]
        L.fixture_plant_last_contact.argtypes=[ptr,DOUBLE]
        L.fixture_plant_motion_ratios.argtypes=[ptr,DOUBLE]
        if not hasattr(L,'fixture_plant_configure_execution'):
            raise ValueError('Rebuild empirical plant to separate candidate execution mesh from frozen physical model')
        L.fixture_plant_configure_execution.argtypes=[ptr,DOUBLE,DOUBLE,*err]
        meta=json.loads((Path(factors)/'metadata.json').read_text()); self.origin=np.asarray(meta['arrays']['origin'])
        ball=np.asarray(ball_base)-self.origin;self.x=np.r_[ball,np.zeros(3),q,np.zeros(10)].astype(float)
        self.acc=np.zeros(7);self.impacts=ct.c_int();self.contact=np.zeros(36);self.crossings=0
        self.handle=L.fixture_plant_create(str(Path(factors).resolve()).encode(),np.ascontiguousarray(q),np.ascontiguousarray(ball),release,self.error,len(self.error))
        if not self.handle: raise RuntimeError(self.error.value.decode())
        p=parameters;c=meta['config']  # Frozen plant defaults, never candidate MPC defaults.
        vec=np.r_[p['gravity_m_s2'],p.get('drag_s_inv',c['drag']),p.get('quadratic_drag_m_inv',0.),
            p.get('restitution',c['restitution']),p.get('friction',c['friction']),p.get('ball_radius_m',c['radius']),
            p.get('ball_mass_kg',c['mass']),p.get('inertia_ratio',c['inertia_ratio']),
            np.deg2rad(p.get('normal_delta_rotvec_deg',[0.,0.,0.])),p.get('center_offset_local_m',[0.,0.,0.]),
            p.get('paddle_axes_m',[c['paddle_radius']]*2),np.broadcast_to(p.get('servo_lag_s',.005),(7,)),
            np.broadcast_to(p.get('servo_dead_time_s',0.),(7,)),p.get('flight_acceleration_time_constant_s',0.),
            p.get('restitution_edge_loss',0.),p.get('restitution_reference_radius_m',.075)]
        self.call('fixture_plant_configure',np.ascontiguousarray(vec),len(vec))
        execution_metadata=execution_metadata or meta;execution=execution_metadata['spec'].get('execution',{})
        self.call('fixture_plant_configure_execution',np.ascontiguousarray(execution_metadata['stage_steps'][:3],dtype=float),
            np.array([execution.get('position_gain',100.),execution.get('velocity_gain',20.),execution.get('jerk_fraction',.2)]))
        self.call('fixture_plant_delivery_limit',delivery_budget)
    def call(self,name,*args):
        if getattr(self.lib,name)(self.handle,*args,self.error,len(self.error)):
            raise RuntimeError(self.error.value.decode())
    def disturbance(self,acceleration,impact,immediate=False):
        self.call('fixture_plant_disturbance',np.ascontiguousarray(acceleration,dtype=float),np.ascontiguousarray(impact,dtype=float),int(immediate))
    def gravity(self,value):self.call('fixture_plant_gravity',np.ascontiguousarray(value,dtype=float))
    def step(self):
        self.call('fixture_plant_step',self.x,self.acc,ct.byref(self.impacts))
        count=self.lib.fixture_plant_last_contact(self.handle,self.contact)
        if count!=self.crossings:
            self.crossings=count;return self.contact.copy()
    def plan(self,source,trajectory):self.call('fixture_plant_plan',float(source),np.ascontiguousarray(trajectory.ravel()))
    def ratios(self):
        result=np.zeros(4);self.lib.fixture_plant_motion_ratios(self.handle,result);return result
    def close(self):
        if self.handle:self.lib.fixture_plant_destroy(self.handle);self.handle=None


class EmpiricalDraws:
    """Row bootstrap with separate run/flight components and optional block order.

    Multiplicative expansion must be fitted/documented in the supplied artifact.
    No independent default wind or unmeasured disturbance is injected.
    """
    def __init__(self,spec,seed):
        self.spec=spec;self.positions={}
        names=('initial','measurement','delivery','flight','contact')
        self.rngs={name:np.random.default_rng(child) for name,child in zip(names,np.random.SeedSequence(seed).spawn(len(names)))}
    def rng_for(self,key):
        if key.startswith(('impact_','run_impact_')):return self.rngs['contact']
        if 'acceleration_' in key:return self.rngs['flight']
        if key.startswith('plan_delay'):return self.rngs['delivery']
        return self.rngs['measurement']
    def row(self,key,default):
        rows=self.spec.get(key)
        if rows is None or len(rows)==0:return np.asarray(default,dtype=float)
        rows=np.asarray(rows,dtype=float)
        rng=self.rng_for(key)
        block=int(self.spec.get('bootstrap_block_frames',1))
        if block>1 and key in ('capture_error_receive_delay_samples_s','vicon_delay_samples_s','plan_delay_samples_s'):
            index,left=self.positions.get(key,(0,0))
            if left<=0:index=int(rng.integers(len(rows)));left=block
            value=rows[index%len(rows)];self.positions[key]=(index+1,left-1)
        else:value=rows[rng.integers(len(rows))]
        return np.asarray(value,dtype=float)
    def residual(self,key):
        mean=np.asarray(self.spec.get(key.replace('_samples','_mean'),[0.,0.,0.]),float)
        return mean+float(self.spec.get('residual_expansion_factor',1.))*self.row(key,[0.,0.,0.])



def run_trial(args,empirical,robot,meta,seed,output):
    os.environ['JUGGLING_RUNTIME_VALUES']=str((args.factors/'runtime_values.txt').resolve())
    output.mkdir(parents=True,exist_ok=False);clock=perf_counter();draws=EmpiricalDraws(empirical.get('uncertainty',{}),seed)
    conditions=empirical['initial_conditions'];initial=conditions[seed%len(conditions)]
    q=np.asarray(initial['q'],float);ball_base=np.asarray(initial['ball_base_m'],float)
    face=np.asarray(initial.get('home_face_offset_m',robot.numeric_fk(q)[0]-robot.origin),float)
    cfg=meta['config'];gravity=np.array([cfg.get('gravity_'+a,d) for a,d in zip('xyz',[0.,0.,-9.81])])
    plant_cfg=dict(empirical['plant']);run_acc=draws.residual('run_acceleration_samples_m_s2')
    run_impact=draws.residual('run_impact_velocity_samples_local_m_s')
    parameter_samples=draws.spec.get('plant_parameter_samples',[])
    if parameter_samples:plant_cfg.update(parameter_samples[int(draws.rngs['initial'].integers(len(parameter_samples)))])
    nominal_gravity=np.array(plant_cfg['gravity_m_s2'],float)
    initial_gravity=np.array(plant_cfg.get('gravity_initial_m_s2',nominal_gravity),float)
    plant_cfg['gravity_m_s2']=initial_gravity.tolist()
    release=float(empirical.get('release_time_s',1.));budget=float(empirical.get('plan_delivery_budget_s',.035))
    reference_preview=float(empirical.get('controller_options',{}).get('reference_preview_s',0.))
    if not 0<=reference_preview<=.02:raise ValueError('Reference preview must be within 0..20 ms')
    # Diagnostic output from the optional experimental warmstart. The physical
    # plant never reads this file or uses the optimizer's contact impulses.
    if empirical.get('controller_options',{}).get('virtual_seed_trace',False):
        os.environ['JUGGLING_SEED_TRACE']=str(output/'virtual_seed.jsonl')
    else:os.environ.pop('JUGGLING_SEED_TRACE',None)
    if empirical.get('controller_options',{}).get('group_rho_trace',False):
        os.environ['JUGGLING_GROUP_RHO_TRACE']=str(output/'group_rho.jsonl')
    else:os.environ.pop('JUGGLING_GROUP_RHO_TRACE',None)
    if empirical.get('controller_options',{}).get('tail_endpoint_trace',False):
        os.environ['JUGGLING_TAIL_ENDPOINT_TRACE']=str(output/'tail_endpoint.jsonl')
    else:os.environ.pop('JUGGLING_TAIL_ENDPOINT_TRACE',None)
    if empirical.get('controller_options',{}).get('first_event_center_trace',False):
        os.environ['JUGGLING_FIRST_EVENT_CENTER_TRACE']=str(output/'first_event_center.jsonl')
    else:os.environ.pop('JUGGLING_FIRST_EVENT_CENTER_TRACE',None)
    if empirical.get('controller_options',{}).get('multistart_trace',False):
        os.environ['JUGGLING_MULTISTART_TRACE']=str(output/'multistart.jsonl')
    else:os.environ.pop('JUGGLING_MULTISTART_TRACE',None)
    os.environ.pop('JUGGLING_MULTISTART_VERIFY',None)
    plant=NativePlant(args.plant,args.plant_factors,q,ball_base,release,plant_cfg,cfg,budget,execution_metadata=meta)
    controller=NativeController(args.library,args.factors);controller.set_reference(face,q)
    contact_bias=None;bias_history=[]
    adaptation=empirical.get('controller_options',{}).get('contact_bias_adaptation',False)
    if adaptation is not False:
        if not isinstance(adaptation,dict):raise ValueError('Contact bias adaptation requires an options dictionary')
        if not meta['spec'].get('legacy_contact_bias'):raise ValueError('Controller lacks optional contact bias factors')
        from tracking.contact_bias import ContactBiasEstimator
        contact_bias=ContactBiasEstimator(robot,cfg,edge_loss=meta['spec'].get('prediction_edge_loss',0.),**adaptation)
    nominal=ObserverConfig(**{k:cfg[k] for k in ['radius','restitution','friction','paddle_radius','inertia_ratio']},
        drag=cfg['drag'],gravity_x=gravity[0],gravity_y=gravity[1],gravity_z=gravity[2],ball_mass=cfg['mass'],
        max_measurement_age=MAX_BALL_GAP+.02,contact_measurement_window=.015,initial_velocity_std=.08)
    if cfg.get('quadratic_drag',0.):
        if not hasattr(nominal,'quadratic_drag'):raise ValueError('Candidate observer lacks mixed-drag support')
        nominal.quadratic_drag=cfg['quadratic_drag']
    for key,value in empirical.get('observer_overrides',{}).items():
        if not hasattr(nominal,key):raise ValueError('Unknown observer override '+key)
        setattr(nominal,key,value)
    observer=BallObserver(robot,nominal);gate=ReleaseGate(gravity=gravity);raw=RawBall(position=ball_base)
    raw_context=SimpleNamespace(observer=None,raw_ball=raw,ball_frame=(np.eye(4),robot.origin),args=SimpleNamespace(extra_latency=0.))
    dt=plant.dt;arms=deque(maxlen=400);pending=deque();plans=deque();last_ball=None;last_arrival=-np.inf
    record=[];estimates=[];estimate_covariances=[];metrics=[];contacts=[];apices=[];measurements=[];disturbances=[];tail_plans=deque(maxlen=100)
    previous=None;next_solve=0.;busy_until=0.;started=None;release_event=None;last_vz=0.;dropout=0;dropouts=0
    max_observer_error=0.;failure=None;failure_stage=None;t=0.;last_arm=None;current_impact=None;last_guard=None
    reference=empirical['nominal_reference']
    nominal_face=np.array(reference['nominal_face_base_m'])-robot.origin
    nominal_posture=np.array(reference['nominal_posture']);return_start=-np.inf
    reference_face=face.copy()
    flight_base=np.zeros(3);flight_profile=None;flight_start=release
    def choose_disturbance(immediate=False):
        nonlocal current_impact,flight_base,flight_profile,flight_start
        key=('initial_flight_acceleration_samples_m_s2' if immediate and 'initial_flight_acceleration_samples_m_s2' in draws.spec
             else 'flight_acceleration_samples_m_s2')
        flight_base=run_acc+draws.residual(key);flight_start=release if immediate else t
        current_impact=run_impact+draws.residual('impact_velocity_samples_local_m_s')
        profiles=draws.spec.get('within_flight_acceleration_profiles_m_s2',[]);profile_index=None;flight_profile=None
        if profiles:
            profile_index=int(draws.rngs['flight'].integers(len(profiles)));flight_profile=np.asarray(profiles[profile_index],float)
            if flight_profile.ndim!=2 or flight_profile.shape[1]!=3 or not len(flight_profile):raise ValueError('Acceleration profile requires N by 3 rows')
        acceleration=flight_base+(0. if flight_profile is None else flight_profile[0])
        plant.disturbance(acceleration,current_impact,immediate)
        disturbances.append(dict(time_s=t,flight_start_s=flight_start,profile_index=profile_index,
            acceleration_base_m_s2=flight_base.tolist(),acceleration_target_m_s2=acceleration.tolist(),next_impact_local_m_s=current_impact.tolist()))
    choose_disturbance(True)
    try:
        for tick in range(int(np.ceil((release+args.seconds+.15)/dt))+1):
            t=tick*dt
            if tick:
                failure_stage='native plant';event=plant.step()
                if event is not None:
                    contacts.append(dict(time_s=float(event[0]),local_contact_m=event[1:4].tolist(),
                        pre_velocity_base_m_s=event[4:7].tolist(),post_velocity_base_m_s=event[7:10].tolist(),
                        normal_base=event[10:13].tolist(),face_base_m=event[13:16].tolist(),face_velocity_base_m_s=event[16:19].tolist(),
                        q=event[19:26].tolist(),normal_impulse_Ns=float(event[26]),hit=bool(event[27]),
                        paddle_axes_m=event[28:30].tolist(),sampled_residual_local_m_s=current_impact.tolist()))
                    if event[27]:
                        plant.gravity(nominal_gravity);choose_disturbance()
                if t>=release and last_vz>0>=plant.x[5]:apices.append(dict(time_s=t,ball_bottom_base_m=float(plant.x[2]+plant.origin[2]-cfg['radius'])))
            # The native plant is now at t. Delivering before this step would
            # install a future command up to one 0.5 ms plant tick too early.
            while plans and plans[0][0]<=t+1e-10:
                _,source,trajectory=plans.popleft();failure_stage='plan delivery';plant.plan(source,trajectory)
            last_vz=plant.x[5]
            if tick%20==0 and t>=flight_start and flight_profile is not None:
                # A complete measured residual segment is sampled per flight.
                # Its temporal structure is preserved; no 0.5 ms white forcing.
                spacing=float(draws.spec.get('within_flight_profile_dt_s',.01))
                if spacing<=0:raise ValueError('Acceleration profile spacing must be positive')
                times=np.arange(len(flight_profile))*spacing
                residual=np.array([np.interp(t-flight_start,times,flight_profile[:,j]) for j in range(3)])
                plant.disturbance(flight_base+residual,current_impact)
            if tick%4==0:
                last_arm=(t,plant.x[6:13].copy(),plant.x[13:20].copy(),plant.acc.copy())
                arms.append(last_arm)
                if observer.initialized:observer.record_arm(*last_arm[:3])
                if contact_bias is not None:contact_bias.record_arm(*last_arm[:3])
            rate=float(empirical.get('vicon_rate_hz',100.));period=max(1,round(1/(rate*dt)))
            if tick%period==0:
                if 'capture_error_receive_delay_samples_s' in draws.spec:
                    error,delay=draws.row('capture_error_receive_delay_samples_s',[0.,.005]);capture=t+error
                else:
                    delay=float(draws.row('vicon_delay_samples_s',empirical.get('vicon_delay_s',.005)))
                    capture=t+float(draws.row('capture_timestamp_error_samples_s',0.))
                if delay<0 or not np.isfinite(delay):raise ValueError('Negative/nonfinite empirical Vicon delay')
                arrival=max(t+delay,last_arrival+1e-9);last_arrival=arrival
                if dropout<=0 and draws.rngs['measurement'].random()<float(draws.spec.get('dropout_start_probability',0.)):
                    dropout=int(draws.row('dropout_burst_lengths_frames',0))
                missing=dropout>0
                if missing:dropout-=1;dropouts+=1
                noise=draws.row('noise_position_samples_m',[0.,0.,0.])
                point=plant.x[:3]+plant.origin+noise
                frame=dict(frame=tick//period+1,receive_time=arrival,capture_time_s=capture,effective_latency_s=arrival-capture,
                    latency_s=arrival-capture,unlabeled_markers=[] if missing else [dict(id=1,position_m=point.tolist())])
                pending.append(frame)
            if tick%40==0:record.append(np.r_[t,plant.x,plant.impacts.value])
            if t+1e-10<busy_until:continue
            failure_stage='measurement association'
            while pending and max(pending[0]['receive_time'],pending[0]['capture_time_s'])<=last_arm[0]+1e-10:
                frame=pending.popleft();capture=frame['capture_time_s'];arrival=frame['receive_time']
                if raw.last_frame is None:
                    # The first synthetic timestamp can be slightly negative
                    # after clock-residual sampling; anchor identity at that
                    # first observed time instead of predicting backwards.
                    raw.filter_time=capture;raw.seen_time=capture
                if observer.initialized and capture<=observer.t:continue
                p=live.Session.raw_ball_position(raw_context,frame)
                measurements.append([capture,arrival,*([np.nan]*3 if p is None else p),int(p is not None)])
                if p is None:continue
                last_ball=arrival
                if contact_bias is not None:contact_bias.record_position(capture,p-robot.origin,arrival)
                if not observer.initialized:
                    release_event=gate.update(capture,p,frame['frame'])
                    if gate.anchor is not None and not gate.armed:
                        gate.arm();raw.allow_motion=True
                    if release_event:
                        t0=release_event['capture_time'];past=[a for a in arms if a[0]<=t0];future=[a for a in arms if a[0]>=t0]
                        if not past or not future:raise RuntimeError('Missing arm history bracketing release')
                        a,b=past[-1],future[0];weight=(t0-a[0])/max(b[0]-a[0],1e-12)
                        q0=(1-weight)*a[1]+weight*b[1];v0=(1-weight)*a[2]+weight*b[2]
                        observer.record_arm(t0,q0,v0);observer.initialize(np.asarray(release_event['position'])-robot.origin,t0,velocity=release_event['velocity'])
                        for arm in arms:
                            if arm[0]>t0:observer.record_arm(*arm[:3])
                        raw_context.observer=observer;started=t
                elif capture>release_event['capture_time']:
                    observer.update_position(capture,p-robot.origin,arrival)
            if not observer.initialized:
                if t>.1:gate.missing(t)
                if t>release+.3:raise RuntimeError('Release not detected within 300 ms')
                continue
            if t-started>=args.seconds:break
            failure_stage='observer or workspace guard'
            if last_ball is None or t-last_ball>MAX_BALL_GAP:raise RuntimeError('Ball tracking lost during MPC')
            arm_t,aq,av,acc=last_arm
            if arm_t<next_solve-1e-10:continue
            x,covariance,info=observer.mpc_state(arm_t,aq,av)
            guard_info=dict(measurement_age=float(info['measurement_age']),stale=bool(info['stale']),
                last_impact=float(info['last_impact']) if np.isfinite(info['last_impact']) else None)
            last_guard=dict(time_s=arm_t,state=x.tolist(),observer=guard_info,
                reference_offset_m=reference_face.tolist())
            if info['stale']:raise RuntimeError('Ball observer measurements are stale or rejected')
            if not -.15<=x[2]<=1.5:raise RuntimeError('Ball left MPC workspace')
            if contact_bias is not None:
                contact_bias.observe(arm_t,x,info['last_impact'])
                learned_bias=contact_bias.bias_per_j8_local
                controller.set_contact_bias(learned_bias)
                bias_history.append(np.r_[arm_t,learned_bias])
            if not np.isfinite(return_start) and np.isfinite(info['last_impact']):return_start=info['last_impact']
            fraction=nominal_return_fraction(arm_t,return_start,cfg.get('return_to_nominal_s',0.))
            reference_face=(1-fraction)*face+fraction*nominal_face
            controller.set_reference(reference_face,(1-fraction)*q+fraction*nominal_posture)
            strike_height=cfg['radius']+reference_face[2]
            if x[2]>strike_height:
                if cfg.get('quadratic_drag',0.):
                    landing=predicted_landing(x,cfg['drag'],strike_height,gravity,cfg['quadratic_drag'])
                else:landing=predicted_landing(x,cfg['drag'],strike_height,gravity)
                last_guard.update(landing_m=landing.tolist(),landing_distance_from_origin_m=float(np.linalg.norm(landing)),
                    landing_distance_from_fixture_m=float(np.linalg.norm(landing-face[:2])))
                if not within_landing_reach(landing,face[:2],cfg):
                    raise RuntimeError('Predicted ball landing is out of reach')
            true_ball=plant.x[:3]+plant.origin-robot.origin
            max_observer_error=max(max_observer_error,float(np.linalg.norm(x[:3]-true_ball)))
            estimates.append(np.r_[arm_t,x])
            estimate_covariances.append(covariance[:9,:9].copy())
            tau=np.asarray(robot.rnea(aq,av,acc)).ravel();failure_stage='native MPC solve'
            if previous is None:
                controller.initialize(x,tau);primal,dual=controller.warmstart();controller.set_warmstart(primal,np.zeros_like(dual))
            Y,_,metric=controller.step(x,tau,0. if previous is None else arm_t-previous)
            diagnostic=dict(time=arm_t,state=x.copy(),tau=tau.copy(),acc=acc.copy(),plan=Y.copy(),
                trajectory=np.full((4,28),np.nan),metrics=metric.copy(),delay=np.nan)
            tail_plans.append(diagnostic)
            # The short-prefix research profile stores accelerations in later
            # control slots. Decode a copy for the unchanged ramp projector.
            failure_stage='executable reference projection';trajectory,ratio,projection=executable_trajectory(robot,meta,torque_view(robot,meta,Y),aq,av,acc)
            diagnostic['trajectory']=trajectory.copy()
            delay=float(draws.row('plan_delay_samples_s',empirical.get('plan_delay_s',0.)))
            diagnostic['delay']=delay
            if not np.isfinite(delay) or delay<0:raise ValueError('Negative/nonfinite empirical plan delay')
            # These total delays are calibrated before a batch, including that
            # candidate's separately benchmarked native computation. Observed
            # wall time here is diagnostic only: CPU contention in parallel
            # offline rollouts must not change the simulated delivery schedule.
            if delay>budget:raise RuntimeError(f'Measured state exceeded {1000*budget:g} ms delivery budget')
            deliver=max(t,arm_t+delay)
            # Controller-side preview of the executed reference can compensate
            # physical servo deadtime. It never changes plant/sensing latency or
            # makes a future measurement available. Zero preserves the baseline.
            source=arm_t-reference_preview if reference_preview else arm_t
            if deliver<=t+1e-10:plant.plan(source,trajectory)
            else:plans.append((deliver,source,trajectory.copy()))
            busy_until=deliver
            metrics.append(np.r_[arm_t,metric,delay,ratio,projection]);previous=arm_t;next_solve=arm_t+cfg['control_dt']-.001
    except (RuntimeError,ValueError,np.linalg.LinAlgError) as error:
        failure=str(error)
    finally:
        ratios=plant.ratios();controller.close();plant.close();observer.close();raw.close();gate.close()
    contact_xy=[c['local_contact_m'][:2] for c in contacts if c['hit']]
    hit_z=[c['face_base_m'][2]-(robot.origin[2]+nominal_face[2]) for c in contacts if c['hit']]
    apices_z=[a['ball_bottom_base_m']-(robot.origin[2]+nominal_face[2]) for a in apices]
    survived=max(0.,min(args.seconds,t-(started if started is not None else release)))
    summary=dict(seed=seed,completed=failure is None and started is not None,survival_s=survived,
        failure=failure,failure_stage=None if failure is None else failure_stage,impacts=plant.impacts.value,last_guard=last_guard,
        simulated_seconds=t,requested_seconds=args.seconds,wall_seconds=perf_counter()-clock,
        initial_condition=initial,nominal_reference=reference,reference_transition_start_s=float(return_start) if np.isfinite(return_start) else None,release_detection=release_event,release_delay_s=None if started is None else started-release,
        physical_robot_used=False,plant_dt_s=dt,velocity_callback_s=.001,readback_s=.002,reference_preview_s=reference_preview,
        sampled_delay_advances_physical_time=True,wall_computation_advances_physical_time=False,
        external_arbitrary_wind=False,plant_parameters=plant_cfg,observer_config=asdict(nominal),
        contact_xy_local_m=stats(contact_xy),strike_height_error_m=stats(hit_z),apex_above_nominal_plane_m=stats(apices_z),
        raw_rejections=dict(raw.rejections),bootstrap_acquisition_rejections=getattr(raw,'bootstrap_rejections',{}),dropout_frames=dropouts,observer_updates=len(observer.updates),
        observer_rejected=sum(not row['accepted'] for row in observer.updates),max_observer_position_error_m=max_observer_error,
        native_mpc_ms=stats([row[1] for row in metrics]),diagnostic_motion_limit_ratios=ratios.tolist(),
        observer_diagnostics=dict(covariance_state_order=['bx','by','bz','vx','vy','vz','wx','wy','wz'],
            covariance_times='estimate[:,0]',
            innovation_columns=['capture_time_s','arrival_time_s','accepted','near_impact','nis',
                                'innovation_x_m','innovation_y_m','innovation_z_m']),
        torque_diagnostic_limitation='Surrogate inverse dynamics; physical torque sensor/reflex is not reproduced.')
    write(output/'summary.json',summary);write(output/'contacts.json',contacts);write(output/'disturbances.json',disturbances)
    write(output/'apices.json',apices)
    if contact_bias is not None:
        write(output/'contact_bias_updates.json',contact_bias.records)
        write(output/'contact_bias_summary.json',dict(bias_per_j8_local_m_s=contact_bias.bias_per_j8_local.tolist(),
              attempted_updates=len(contact_bias.records),accepted_updates=contact_bias.accepted_samples,
              rejection_counts=dict(Counter(r['reason'] for r in contact_bias.records if not r['accepted'])),
              inputs='Delivered associated positions, measured joints, observer impact events/spin only; no plant state or disturbance truth.'))
    np.savez_compressed(output/'trajectory.npz',state=np.asarray(record),estimate=np.asarray(estimates),
        observer_covariance=np.asarray(estimate_covariances).reshape(-1,9,9),
        observer_innovations=np.array([[r['capture_time'],r['arrival_time'],r['accepted'],r['near_impact'],
            r['nis'],*r['innovation']] for r in observer.updates],dtype=float).reshape(-1,8),
        metrics=np.asarray(metrics),measurements=np.asarray(measurements),
        tail_plan_times=np.array([r['time'] for r in tail_plans]),tail_states=np.array([r['state'] for r in tail_plans]),
        tail_torques=np.array([r['tau'] for r in tail_plans]),tail_accelerations=np.array([r['acc'] for r in tail_plans]),
        tail_plans=np.array([r['plan'] for r in tail_plans]),tail_trajectories=np.array([r['trajectory'] for r in tail_plans]),
        tail_metrics=np.array([r['metrics'] for r in tail_plans]),tail_delay_s=np.array([r['delay'] for r in tail_plans]),
        contact_bias_history=np.asarray(bias_history))
    return summary


_worker_context=None


def _initialize_trial_worker(args,empirical):
    """Initialize numerical libraries/model independently in each spawn child."""
    global _worker_context
    robot,meta=live.model(args.factors)
    _worker_context=(args,empirical,robot,meta)


def _trial_worker(seed):
    args,empirical,robot,meta=_worker_context
    return run_trial(args,empirical,robot,meta,seed,args.output/f'seed_{seed:05d}')


def _spawn_results(worker,seeds,jobs,initializer=None,initargs=()):
    """Independent processes, deterministic result order, bounded pool lifetime."""
    with ProcessPoolExecutor(max_workers=jobs,mp_context=mp.get_context('spawn'),
                             initializer=initializer,initargs=initargs) as pool:
        yield from pool.map(worker,seeds,chunksize=1)


def run(args):
    batch_start=perf_counter()
    jobs=int(getattr(args,'jobs',1))
    if not 1<=jobs<=100:raise ValueError('Need 1..100 worker processes')
    jobs=min(jobs,args.trials)
    if not args.plant_factors:args.plant_factors=args.factors
    args.output.mkdir(parents=True,exist_ok=False)
    empirical=json.loads(args.uncertainty.read_text())
    if jobs==1:robot,meta=live.model(args.factors)
    else:meta=json.loads((args.factors/'metadata.json').read_text())
    provenance=dict(uncertainty_path=str(args.uncertainty.resolve()),uncertainty_sha256=digest(args.uncertainty),
        controller_library_sha256=digest(args.library),controller_metadata_sha256=digest(args.factors/'metadata.json'),
        estimator_library_sha256=digest(os.environ['JUGGLING_OFFLINE_ESTIMATOR']),runtime_values_sha256=digest(args.factors/'runtime_values.txt'),
        plant_library_sha256=digest(args.plant),plant_metadata_sha256=digest(args.plant_factors/'metadata.json'),
        physical_robot_used=False,controller_config=meta['config'],seed_start=args.seed,trials=args.trials,
        worker_processes=jobs,process_start_method='spawn' if jobs>1 else 'in_process',
        native_wall_timing_is_diagnostic_only=True,
        runner_sha256=digest(__file__),execution_mesh_from_candidate=True,
        runtime_source_sha256=runtime_source_hashes(),
        plan_delivery_order='advance plant to delivery tick, then install command')
    plant_meta=json.loads((args.plant_factors/'metadata.json').read_text())
    provenance['diagnostic_motion_denominators']=dict(
        **{name:plant_meta['arrays'][name] for name in ('velocity_limit','acceleration_limit','torque_limit')},
        surrogate_filtered_torque_rate_nm_s=800.)
    if 'simulation_execution_limits' in meta:
        provenance['simulation_execution_limits']=meta['simulation_execution_limits']
    if empirical.get('controller_options',{}).get('contact_bias_adaptation',False) is not False:
        # Worker imports happen after parent provenance is written.
        provenance['runtime_source_sha256']['tracking.contact_bias']=digest(Path(__file__).with_name('contact_bias.py'))
    write(args.output/'provenance.json',provenance);write(args.output/'empirical_model_used.json',empirical)
    reports=[];seeds=range(args.seed,args.seed+args.trials)
    results=(_spawn_results(_trial_worker,seeds,jobs,_initialize_trial_worker,(args,empirical)) if jobs>1 else
             (run_trial(args,empirical,robot,meta,seed,args.output/f'seed_{seed:05d}') for seed in seeds))
    for report in results:
        seed=report['seed'];reports.append(report)
        worker_wall=sum(r['wall_seconds'] for r in reports)
        aggregate=dict(trials=len(reports),completed=sum(r['completed'] for r in reports),
            survival_s=[r['survival_s'] for r in reports],survival_statistics=stats([r['survival_s'] for r in reports]),
            failure_counts=dict(Counter(r['failure'] or 'completed' for r in reports)),
            impacts=[r['impacts'] for r in reports],wall_seconds=worker_wall,
            sum_worker_wall_seconds=worker_wall,batch_elapsed_wall_seconds=perf_counter()-batch_start,
            wall_seconds_definition='Sum of individual trial worker wall times; not elapsed batch time.',
            worker_processes=jobs,physical_robot_used=False)
        write(args.output/'summary.json',aggregate)
        print(json.dumps(dict(seed=seed,survival_s=report['survival_s'],impacts=report['impacts'],failure=report['failure'],wall_seconds=report['wall_seconds'])),flush=True)
    try:
        plot(args.output,reports)
    except ModuleNotFoundError as error:
        if error.name!='matplotlib':raise
        print('Plotting dependency unavailable here; render recorded batch locally.',flush=True)
    aggregate['batch_elapsed_wall_seconds']=perf_counter()-batch_start
    write(args.output/'summary.json',aggregate)
    return aggregate


def plot(output,reports):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    survival_times=[0.];survival_probability=[1.];probability=1.;censored=[]
    for stamp in sorted({r['survival_s'] for r in reports}):
        at_risk=sum(r['survival_s']>=stamp for r in reports)
        failures=sum(r['survival_s']==stamp and not r['completed'] for r in reports)
        if at_risk:probability*=1-failures/at_risk
        survival_times.append(stamp);survival_probability.append(probability)
        if any(r['survival_s']==stamp and r['completed'] for r in reports):censored.append((stamp,probability))
    axes[0,0].step(survival_times,survival_probability,where='post',label='Survival estimate')
    if censored:
        censor_points=np.asarray(censored);axes[0,0].plot(censor_points[:,0],censor_points[:,1],'+',ms=8,label='Completed horizon (censored)')
        axes[0,0].legend(fontsize=8)
    axes[0,0].set(xlabel='Survival time [s]',ylabel='Fraction surviving',ylim=(0,1.03))
    for report in reports:
        folder=output/f"seed_{report['seed']:05d}";contacts=json.loads((folder/'contacts.json').read_text());hits=[c for c in contacts if c['hit']]
        if hits:
            xy=np.array([c['local_contact_m'][:2] for c in hits]);axes[0,1].scatter(xy[:,0]*1000,xy[:,1]*1000,s=8,alpha=.6)
            heights=np.array([c['face_base_m'][2] for c in hits]);axes[1,0].plot([c['time_s'] for c in hits],heights,'.-',ms=3,alpha=.6)
        data=np.load(folder/'trajectory.npz');metric=data['metrics']
        if len(metric):axes[1,1].plot(metric[:,0],metric[:,1],alpha=.4,lw=.5)
    axes[0,1].set(xlabel='Contact paddle-local X [mm]',ylabel='Contact paddle-local Y [mm]',aspect='equal')
    axes[1,0].set(xlabel='Time [s]',ylabel='Paddle center base Z at impact [m]')
    axes[1,1].set(xlabel='Time [s]',ylabel='Native MPC wall time [ms]')
    for axis in axes.flat:axis.grid(alpha=.25)
    fig.suptitle(f'Offline empirical uncertainty: {len(reports)} trials, {sum(r["completed"] for r in reports)} completed\nActual controller/observer/reference; independent calibrated plant; no hardware')
    fig.savefig(output/'report.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('uncertainty','factors','library','plant','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plant-factors',type=Path);parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--trials',type=int,default=10);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--jobs',type=int,default=1,help='Independent spawned offline trial workers; calibrated delay controls physics, not wall time')
    arguments=parser.parse_args()
    if arguments.seconds<=0 or not 1<=arguments.trials<=100:parser.error('Need positive duration and 1..100 trials')
    if not 1<=arguments.jobs<=100:parser.error('Need 1..100 worker processes')
    run(arguments)
