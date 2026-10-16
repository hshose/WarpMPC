"""Real-arm MPC rehearsal with a virtual ball; the physical ball stays held."""
from dataclasses import asdict
from collections import deque
from copy import copy
import json
import numpy as np
from estimation.ball import BallObserver,ObserverConfig
from . import calibration as cal
from . import session as live
from .fixture import predicted_landing
from .reach import within_landing_reach, nominal_return_fraction

def executable_trajectory(robot,meta,Y,q,dq,acc0):
    """Bound small SQP acceleration defects; reconstruct the executed ramp."""
    limits=np.array(meta['arrays']['acceleration_limit'])
    acceleration=np.array([np.linalg.solve(np.asarray(robot.mass(y[6:13])),
        y[23:30]-np.asarray(robot.bias(y[6:13],y[13:20])).ravel()) for y in Y[:4]])
    ratio=float(np.max(abs(acceleration[1:3])/limits))
    if ratio>1.25:raise RuntimeError('MPC acceleration defect exceeds reference projection budget')
    projected=np.clip(acceleration,-.95*limits,.95*limits);projected[0]=acc0
    trajectory=np.zeros((4,28));trajectory[0,:14]=np.r_[q,dq]
    for k in range(4):
        trajectory[k,21:28]=projected[k]
        trajectory[k,14:21]=np.asarray(robot.rnea(trajectory[k,:7],trajectory[k,7:14],projected[k])).ravel()
        if k<3:
            h=meta['stage_steps'][k];a=projected[k];b=projected[k+1]
            trajectory[k+1,:7]=trajectory[k,:7]+h*trajectory[k,7:14]+h*h*(2*a+b)/6
            trajectory[k+1,7:14]=trajectory[k,7:14]+.5*h*(a+b)
    return trajectory,ratio,float(np.max(abs(projected[1:3]-acceleration[1:3])/limits))

class VirtualBall(BallObserver):
    """Propagate the contact/flight model against measured, interpolated joints.

    No ball measurement is consumed. Substeps are at most0.5 ms. This uses the
    observer model's mean dynamics without covariance or event Jacobians.
    """
    def _geometry(self,x,a,b,t):
        # Geometry only needs the interpolated face, not interpolated q/dq.
        f=(t-a.t)/(b.t-a.t)
        p=a.p+f*(b.p-a.p);n=a.n+f*(b.n-a.n);n=n/np.linalg.norm(n)
        rel=x[:3]-p;distance=n@rel
        return float(distance-self.cfg.radius),np.linalg.norm(rel-n*distance),float(n@(x[3:6]-(b.p-a.p)/(b.t-a.t)))

    def record_arm(self,t,q,dq):
        if not self.history:return super().record_arm(t,q,dq)
        a=self.history[-1];n=max(1,int(np.ceil((t-a.t)/.0005)))
        for k in range(1,n+1):
            w=k/n;super().record_arm(a.t+w*(t-a.t),(1-w)*a.q+w*np.asarray(q),(1-w)*a.dq+w*np.asarray(dq))

    def advance(self,t):
        if t<self.t or t>self.history[-1].t:raise ValueError('Virtual ball requires causal arm history')
        if self.t<self.history[0].t-1e-10:raise ValueError('Virtual ball history expired')
        x=self.x.copy();start=self.t;last=self.last_impact;c=self.cfg
        for a,b in zip(list(self.history)[:-1],list(self.history)[1:]):
            if b.t<=start+1e-10:continue
            if start>=t-1e-10:break
            end=min(t,b.t);dt=end-start;y=self._free(x,dt)
            ga,ra,va=self._geometry(x,a,b,start);gb,rb,vb=self._geometry(y,a,b,end)
            ea=ga+max(-va,0.)*c.contact_delay;eb=gb+max(-vb,0.)*c.contact_delay
            if start-last>c.impact_refractory and eb<=0 and max(ra,rb)<c.paddle_radius and vb<-.2 and ga>-.02:
                hit=dt*np.clip(ea/max(ea-eb,1e-12),0,1)
                y,hit,_=self._hit_map(x,dt,a,b,start,hit);last=start+hit;self.events.append(last)
            x=y;start=end
        self.x=x;self.t=t;self.last_impact=last
        return x.copy()

    def forecast(self,dt,q,dq,acc,acceleration_at=None):
        """Predict along the active acceleration ramp without altering history."""
        q=np.asarray(q);dq=np.asarray(dq);acc=np.asarray(acc)
        future=copy(self);future.history=deque(self.history);future.events=list(self.events)
        future.x=self.x.copy()
        qf=q.copy();vf=dq.copy();steps=max(1,int(np.ceil(dt/.0005)));h=dt/steps
        for k in range(steps):
            a=acc if acceleration_at is None else acceleration_at((k+.5)*h)
            qf+=h*vf+.5*h*h*a;vf+=h*a
            future.record_arm(self.t+(k+1)*h,qf,vf)
        return future.advance(self.t+dt),qf,vf


def run(session,args,robot,meta,controller,home,anchor,held_gate):
    cfg=ObserverConfig(**{k:meta['config'][k] for k in ['radius','restitution','friction','paddle_radius','inertia_ratio']},
        ball_mass=meta['config']['mass'],drag=meta['config']['drag'],contact_delay=0.,
        **{k:meta['config'][k] for k in ['gravity_x','gravity_y','gravity_z'] if k in meta['config']})
    # Initialize while holding the fixture pose, before starting the virtual drop.
    arm=session.arm
    seed=np.r_[anchor-robot.origin,np.zeros(3),arm['q'],arm['dq'],np.zeros(3)]
    tau=np.asarray(robot.rnea(arm['q'],arm['dq'],arm['ddq_d'])).ravel()
    controller.initialize(seed,tau);primal,dual=controller.warmstart()
    controller.set_warmstart(primal,np.zeros_like(dual));session.pump()
    virtual=VirtualBall(robot,cfg);arm=session.arm;t0=arm['receive_time']
    virtual.record_arm(t0,arm['q'],arm['dq']);virtual.initialize(anchor-robot.origin,t0)
    session.virtual_ball=virtual
    cal.write_new(args.output/'rehearsal.json',dict(ball_source='simulated',physical_ball_must_stay_held=True,
        duration_s=args.seconds,virtual_substep_max_s=.0005,initial_ball_base_m=anchor.tolist(),
        model=asdict(cfg),start_robot_time=t0,external_disturbance=False,measurement_noise=False))
    print(f'MPC REHEARSAL: DO NOT RELEASE THE BALL. {args.seconds:g}-second robot motion with a simulated ball.',flush=True)
    return_start=float("inf")
    previous=None;last_trajectory=None;next_solve=t0+.001  # Wait for fresh readback before the first solve.
    with (args.output/'mpc.jsonl').open('x') as log:
        while live.time.monotonic()-t0<args.seconds:
            session.pump();held_gate.missing(live.time.monotonic()-args.extra_latency)
            arm=session.arm;t=arm['receive_time']
            if t<next_solve:continue
            source_to_compute=live.time.monotonic()-t
            ball=virtual.advance(t);x=np.r_[ball[:6],arm['q'],arm['dq'],ball[6:9]]
            if not -.15<x[2]<1.5:raise RuntimeError('Virtual ball left MPC workspace')
            if not np.isfinite(return_start) and np.isfinite(virtual.last_impact): return_start=virtual.last_impact
            return_fraction=nominal_return_fraction(t,return_start,meta['config'].get('return_to_nominal_s',0.))
            reference_offset=(1-return_fraction)*np.array(home['face_offset_m'])
            if meta['config'].get('return_to_nominal_s',0.)>0:
                controller.set_reference(reference_offset,(1-return_fraction)*np.array(home['q_home'])+return_fraction*live.HOME)
            strike_height=cfg.radius+reference_offset[2]
            if x[2]>strike_height:
                landing=predicted_landing(x,cfg.drag,strike_height,[cfg.gravity_x,cfg.gravity_y,cfg.gravity_z])
                if not within_landing_reach(landing, home['face_offset_m'][:2], meta['config']):
                    cal.write_new(args.output/'workspace_abort.json',dict(time=t,state=x.tolist(),
                        landing_m=landing.tolist(),fixture_offset_m=home['face_offset_m'],
                        nominal_radius_m=float(np.linalg.norm(landing)),fixture_radius_m=float(np.linalg.norm(landing-np.array(home['face_offset_m'])[:2]))))
                    raise RuntimeError('Virtual ball landing is out of reach')
            lead=float(getattr(args,'rehearsal_prediction',0.))
            plan_time=t+lead;q=np.asarray(arm['q']);dq=np.asarray(arm['dq'])
            if lead:
                acceleration_at=None
                if last_trajectory is not None:
                    times=np.array(meta['stage_times'][:4]);accs=last_trajectory[:,21:28]
                    def planned_acc(dt):
                        age=t+dt-previous;k=int(np.clip(np.searchsorted(times,age,side='right')-1,0,2))
                        w=np.clip((age-times[k])/(times[k+1]-times[k]),0.,1.)
                        return (1-w)*accs[k]+w*accs[k+1]
                    base_acc=planned_acc(0.);limits=np.array(meta['arrays']['acceleration_limit'])
                    acceleration_at=lambda dt:np.clip(np.array(arm['ddq_d'])+planned_acc(dt)-base_acc,-limits,limits)
                predicted,q,dq=virtual.forecast(lead,q,dq,arm['ddq_d'],acceleration_at)
                x=np.r_[predicted[:6],q,dq,predicted[6:9]]
            tau=np.asarray(robot.rnea(q,dq,arm['ddq_d'])).ravel()
            Y,_,stats=controller.step(x,tau,0 if previous is None else plan_time-previous)
            if live.time.monotonic()-t>.05:raise RuntimeError('Computed rehearsal plan is too old to execute')
            if getattr(args,'rehearsal_project_reference',False):
                trajectory,raw_acceleration_ratio,projection_ratio=executable_trajectory(robot,meta,Y,q,dq,arm['ddq_d'])
            else:
                acceleration=np.array([np.linalg.solve(np.asarray(robot.mass(y[6:13])),y[23:30]-np.asarray(robot.bias(y[6:13],y[13:20])).ravel()) for y in Y[:4]])
                trajectory=np.c_[Y[:4,6:20],Y[:4,23:30],acceleration];trajectory[0]=np.r_[q,dq,tau,arm['ddq_d']]
                raw_acceleration_ratio=None;projection_ratio=0.
            # The offline harness models solve delay as queued transport delay.
            earliest_send=plan_time-float(getattr(args,'plan_delay',0.))-.001
            while live.time.monotonic()<earliest_send:session.pump()
            budget=float(getattr(args,'rehearsal_delivery_budget',.015))
            if live.time.monotonic()-t>budget:
                cal.write_new(args.output/'plan_timing_abort.json',dict(measurement_time=t,
                    age_s=live.time.monotonic()-t,source_to_compute_s=source_to_compute,
                    native_metrics=stats.tolist(),prediction_s=lead,first_plan=previous is None))
                raise RuntimeError(f'Rehearsal measured state exceeded {1000*budget:g} ms delivery budget')
            session.robot.send('plan '+str(plan_time)+' '+' '.join(map(str,trajectory.ravel())))
            log.write(json.dumps(dict(time=t,state=x.tolist(),plan=Y[:4].tolist(),metrics=stats.tolist(),
                ball_source='simulated',measurement_age=None,source_to_compute_s=source_to_compute,
                source_to_send_s=live.time.monotonic()-t,prediction_s=lead,plan_time=plan_time,
                measured_ball=ball.tolist(),raw_acceleration_ratio=raw_acceleration_ratio,
                acceleration_projection_ratio=projection_ratio,executed_trajectory=trajectory.tolist()))+'\n');log.flush()
            previous=plan_time;last_trajectory=trajectory.copy();next_solve=t+meta['config']['control_dt']-.001
    cal.write_new(args.output/'rehearsal_complete.json',dict(duration_s=live.time.monotonic()-t0,
        simulated_impacts=virtual.events,physical_ball_released=False))
    print('MPC REHEARSAL COMPLETE: stopping robot. Keep the physical ball held.',flush=True)
