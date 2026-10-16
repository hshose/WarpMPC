"""Held-ball -> measured release -> observer -> C++ MPC offline integration.

Simulation PAUSES while computing. 0.5 ms plant; 1 ms native velocity adapter;
100 Hz position-only Vicon surrogate, 5 ms delay; 20 ms MPC, 1 SQP / 50 OSQP.
The arm is a 5 ms servo-lag surrogate and ball has impulse contact, not Newton.
"""
import argparse
from collections import deque
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from cpp.python_controller import NativeController, DOUBLE
from estimation.ball import BallObserver, ObserverConfig
from .fixture import fixture_home, ReleaseGate
from .session import model, DEFAULT_FACTORS
from .calibration import write_new


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);robot,meta=model(args.factors)
    anchor=robot.origin+np.array([args.dx,args.dy,args.height]);home=fixture_home(robot,anchor);q=np.array(home['q_home'])
    lib=ct.CDLL(str(args.plant.resolve()));err=ct.create_string_buffer(4096);ptr=ct.c_void_p;errors=[ct.c_char_p,ct.c_size_t]
    if not hasattr(lib,'fixture_plant_protocol') or lib.fixture_plant_protocol()!=2:raise ValueError('Rebuild offline plant for session protocol 2')
    lib.fixture_plant_create.argtypes=[ct.c_char_p,DOUBLE,DOUBLE,ct.c_double,*errors];lib.fixture_plant_create.restype=ptr
    lib.fixture_plant_step.argtypes=[ptr,DOUBLE,DOUBLE,ct.POINTER(ct.c_int),*errors]
    lib.fixture_plant_plan.argtypes=[ptr,ct.c_double,DOUBLE,*errors];lib.fixture_plant_destroy.argtypes=[ptr]
    release=2.;dt=.0005;ball=anchor-robot.origin
    plant=lib.fixture_plant_create(str(args.factors.resolve()).encode(),q,ball,release,err,len(err))
    if not plant:raise RuntimeError(err.value.decode())
    ctl=NativeController(args.library,args.factors);ctl.set_reference(home['face_offset_m'],q)
    gravity=[meta['config'].get('gravity_'+axis,default) for axis,default in zip('xyz',(0.,0.,-9.81))]
    gate=ReleaseGate(gravity=gravity);observer=BallObserver(robot,ObserverConfig(gravity_x=gravity[0],gravity_y=gravity[1],gravity_z=gravity[2],position_std=args.noise,contact_delay=args.contact_delay,max_measurement_age=.06,contact_measurement_window=.015,
        ball_mass=meta['config']['mass'],restitution=meta['config']['restitution'],drag=meta['config']['drag']))
    rng=np.random.default_rng(args.seed);pending=deque();measurements=[];arm_history=deque(maxlen=60)
    states=[];estimates=[];times=[];solve_times=[];events=[];impact_count=ct.c_int();acc=np.zeros(7)
    x=np.r_[ball,np.zeros(3),q,np.zeros(10)];previous=None;next_solve=0;started=None;first_plan=None;max_update=0;failure=None;max_pos_error=0.
    try:
        for k in range(round((release+args.seconds)/dt)+1):
            t=k*dt
            if k:
                if lib.fixture_plant_step(plant,x,acc,ct.byref(impact_count),err,len(err)):raise RuntimeError(err.value.decode())
            if k%20==0:
                arm_history.append((t,x[6:13].copy(),x[13:20].copy()))
                if observer.initialized:observer.record_arm(t,x[6:13],x[13:20])
                pending.append((t+.005,t,x[:3]+robot.origin+rng.normal(0,args.noise,3),k//20))
            while (pending and pending[0][0]<=t+1e-10
                   and (not observer.initialized or pending[0][0]<=observer.history[-1].t+1e-10)):
                arrival,capture,p,frame=pending.popleft();measurements.append([capture,arrival,*p])
                if not observer.initialized:
                    event=gate.update(capture,p,frame)
                    if capture>=1. and not gate.armed:gate.arm()
                    if event:
                        t0=event['capture_time'];arms=[a for a in arm_history if a[0]>=t0-1e-10]
                        assert arms and abs(arms[0][0]-t0)<1e-8
                        observer.record_arm(t0,arms[0][1],arms[0][2]);observer.initialize(np.array(event['position'])-robot.origin,t0)
                        observer.x[3:6]=event['velocity'];observer.P[3:6,3:6]=np.eye(3)*.08**2
                        for at,aq,av in arms[1:]:observer.record_arm(at,aq,av)
                        started=t;event['arrival_time']=arrival;events.append(event)
                else:
                    # Deliver only when there is causal, bracketing 100 Hz robot
                    # history, as in the live supervisor.
                    if capture>observer.t+1e-10:
                        observer.update_position(capture,p-robot.origin,arrival)
            if observer.initialized and k%20==0:
                tic=perf_counter();est,P,info=observer.mpc_state(t,x[6:13],x[13:20]);max_update=max(max_update,(perf_counter()-tic)*1000)
                if info['stale']:raise RuntimeError('Observer became stale')
                max_pos_error=max(max_pos_error,np.linalg.norm(est[:3]-x[:3]));estimates.append(np.r_[t,est])
                if t>=next_solve-1e-10:
                    tau=np.asarray(robot.rnea(x[6:13],x[13:20],acc)).ravel()
                    Y,_,metrics=ctl.step(est,tau,0 if previous is None else t-previous)
                    endpoint_acceleration=np.array([np.linalg.solve(np.asarray(robot.mass(y[6:13])),
                        y[23:30]-np.asarray(robot.bias(y[6:13],y[13:20])).ravel()) for y in Y[:4]])
                    trajectory=np.c_[Y[:4,6:20],Y[:4,23:30],endpoint_acceleration]
                    trajectory[0]=np.r_[x[6:20],tau,acc]
                    trajectory=np.ascontiguousarray(trajectory.ravel())
                    if lib.fixture_plant_plan(plant,t,trajectory,err,len(err)):
                        write_new(args.output/'rejected_plan.json',dict(time=t,plan=Y[:4].tolist(),metrics=metrics.tolist()))
                        raise RuntimeError(err.value.decode())
                    if first_plan is None:first_plan=t
                    solve_times.append(metrics);previous=t;next_solve=t+.02
            if k%20==0:times.append(t);states.append(x.copy())
    except Exception as e:failure=str(e)
    finally:ctl.close();lib.fixture_plant_destroy(plant)
    states=np.array(states);times=np.array(times);ms=np.array(solve_times);apices=[]
    for i in range(1,len(states)):
        if times[i]>release+.4 and states[i-1,5]>0>=states[i,5]:apices.append(float(max(states[i-1,2],states[i,2])-.02))
    passed=failure is None and impact_count.value>=int(args.seconds) and started is not None
    report=dict(passed=passed,failure=failure,physical_robot_used=False,simulator='impulse ball, illustrative 5 ms arm velocity-servo lag; not a hardware or Newton result',
                simulation_pauses_for_computation=True,dt_s=dt,mpc_dt_s=.02,sqp_iterations=1,osqp_iterations=sorted(set(map(int,ms[:,12]))) if len(ms) else [],
                vicon_rate_hz=100,vicon_latency_s=.005,vicon_noise_m=args.noise,observer_contact_delay_s=args.contact_delay,contact_measurement_window_s=.015,home=home,release=events,
                release_detection_delay_s=None if started is None else started-release,impacts=impact_count.value,
                first_mpc_command_delay_s=None if first_plan is None else first_plan-release,
                apices_m=apices,max_observer_position_error_m=float(max_pos_error),max_observer_prediction_ms=max_update,
                solve_ms=dict(median=float(np.median(ms[:,0])),mean=float(np.mean(ms[:,0])),p95=float(np.percentile(ms[:,0],95)),max=float(ms[:,0].max())) if len(ms) else {})
    write_new(args.output/'summary.json',report)
    np.savez_compressed(args.output/'trajectory.npz',t=times,x=states,estimates=np.array(estimates),measurements=np.array(measurements),solve=ms)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    axes[0,0].plot(times,states[:,2]-.02);axes[0,0].axhline(.75,c='k',ls='--');axes[0,0].set(ylabel='Ball bottom above strike plane [m]',xlabel='Time [s]')
    axes[0,1].plot(times,1000*(states[:,:2]-[args.dx,args.dy]));axes[0,1].set(ylabel='Ball lateral error [mm]',xlabel='Time [s]')
    axes[1,0].plot(times,states[:,6:13]-q);axes[1,0].set(ylabel='Joints relative to fixture HOME [rad]',xlabel='Time [s]')
    if len(ms):axes[1,1].plot(ms[:,0]);axes[1,1].set(ylabel='Native MPC computation [ms]',xlabel='MPC solve')
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.suptitle('OFFLINE fixture release: position-only feedback / 1 SQP, 50 OSQP\nImpulse ball + servo-lag surrogate; no physical robot; pass='+str(passed))
    fig.savefig(args.output/'report.pdf');plt.close(fig)
    print(json.dumps(report,indent=2));return passed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--dx',type=float,default=.04);p.add_argument('--dy',type=float,default=-.03);p.add_argument('--height',type=float,default=.67)
    p.add_argument('--noise',type=float,default=.0008);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--contact-delay',type=float,default=.001)
    p.add_argument('--factors',type=Path,default=DEFAULT_FACTORS);p.add_argument('--library',type=Path,default=DEFAULT_FACTORS.parent/'controller/libjuggling_controller.so')
    p.add_argument('--plant',type=Path,default=DEFAULT_FACTORS.parent/'controller/libjuggling_fixture_plant.so')
    args=p.parse_args();raise SystemExit(0 if run(args) else 1)
