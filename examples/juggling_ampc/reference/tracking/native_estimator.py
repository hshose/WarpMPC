"""Offline ctypes bridge to the exact C++ ball observer/association/release gate.

No SDK or hardware I/O. Arm geometry is evaluated from the caller's Panda model,
then passed in the same format as ball_experiment.cpp::geometry_sample.
"""
from collections import deque
from types import SimpleNamespace
import ctypes as ct
import os
import numpy as np

ARRAY=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
FIELDS=('position_std drag quadratic_drag gravity_x gravity_y gravity_z '
        'acceleration_time_constant acceleration_noise velocity_noise spin_noise '
        'initial_velocity_std initial_spin_std initial_acceleration_std restitution '
        'friction ball_mass radius paddle_radius inertia_ratio contact_delay '
        'impact_time_std impact_velocity_std impact_tangent_std impact_spin_std '
        'contact_measurement_std contact_measurement_window impact_refractory '
        'innovation_gate max_measurement_age history_seconds').split()

def array(x):return np.ascontiguousarray(x,dtype=np.float64)

INDEPENDENT_TANGENT_FEATURE=1

def observer_config_values(cfg,features=0):
    """Keep ABI1's 32-value packet and negotiate its optional final slot."""
    independent=float(getattr(cfg,'impact_independent_tangent_std',0.))
    if not np.isfinite(independent) or independent<0:
        raise ValueError('Independent tangential impact uncertainty must be finite and nonnegative')
    if independent and not (features&INDEPENDENT_TANGENT_FEATURE):
        raise RuntimeError('Estimator library does not support impact_independent_tangent_std; rebuild it or leave this option zero')
    return array([getattr(cfg,key,0.) if key=='quadratic_drag' else getattr(cfg,key) for key in FIELDS]+[1.,independent])

class Library:
    def __init__(self,path=None):
        self.lib=ct.CDLL(str(path or os.environ['JUGGLING_OFFLINE_ESTIMATOR']))
        self.error=ct.create_string_buffer(4096)
        L=self.lib;H=ct.c_void_p;D=ct.c_double;I=ct.c_int;U=ct.c_uint64;E=[ct.c_char_p,I]
        L.offline_estimator_abi.restype=I
        if L.offline_estimator_abi()!=1:raise RuntimeError('Offline estimator ABI mismatch')
        self.observer_features=0
        if hasattr(L,'offline_observer_features'):
            L.offline_observer_features.argtypes=[];L.offline_observer_features.restype=ct.c_uint
            self.observer_features=int(L.offline_observer_features())
        definitions={
            'observer_create':([ARRAY,I,*E],H),'observer_destroy':([H],None),
            'observer_arm':([H,D,ARRAY,*E],I),'observer_initialize':([H,D,ARRAY,ARRAY,*E],I),
            'observer_estimate':([H,D,ARRAY,ARRAY,ARRAY,*E],I),
            'observer_update':([H,D,ARRAY,D,ARRAY,ct.c_char_p,I,*E],I),
            'observer_pending':([H,D,ct.POINTER(I),*E],I),
            'raw_create':(E,H),'raw_destroy':([H],None),
            'raw_update':([H,U,D,H,H,I,H,H,I,ARRAY,*E],I),
            'gate_create':([ARRAY,*E],H),'gate_destroy':([H],None),
            'gate_update':([H,D,ARRAY,U,ARRAY,*E],I),
            'gate_arm':([H,*E],I),'gate_missing':([H,D,*E],I)}
        for name,(args,result) in definitions.items():
            f=getattr(L,'offline_'+name);f.argtypes=args;f.restype=result
    def create(self,name,*args):
        handle=getattr(self.lib,'offline_'+name+'_create')(*args,self.error,len(self.error))
        if not handle:raise RuntimeError(self.error.value.decode())
        return handle
    def call(self,name,*args):
        if getattr(self.lib,'offline_'+name)(*args,self.error,len(self.error)):
            raise RuntimeError(self.error.value.decode())

class BallObserver:
    def __init__(self,robot,cfg,library=None):
        self.robot=robot;self.cfg=cfg;self.api=Library(library)
        values=observer_config_values(cfg,self.api.observer_features)
        self.handle=self.api.create('observer',values,len(values));self.initialized=False
        self.history=deque();self.t=0.;self.updates=[]
    def close(self):
        if getattr(self,'handle',None):self.api.lib.offline_observer_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
    def record_arm(self,t,q,dq):
        q=array(q);dq=array(dq);p,n,J,W,_=self.robot.contact_fk(q)
        J=np.asarray(J);W=np.asarray(W);spatial=np.vstack((J,W))
        inertia=spatial@np.linalg.solve(np.asarray(self.robot.mass(q)),spatial.T)
        value=array(np.r_[np.asarray(p).ravel()-self.robot.origin,np.asarray(n).ravel(),J@dq,W@dq,inertia.ravel()])
        self.api.call('observer_arm',self.handle,float(t),value)
        self.history.append(SimpleNamespace(t=float(t)))
        while len(self.history)>2 and self.history[1].t<t-self.cfg.history_seconds:self.history.popleft()
        if self.initialized:self.t=max(self.t,float(t)-self.cfg.history_seconds)
    def initialize(self,position,t=0.,velocity=None):
        self.api.call('observer_initialize',self.handle,float(t),array(position),array(np.zeros(3) if velocity is None else velocity))
        self.initialized=True;self.t=float(t)
    def estimate(self,t):
        x=np.empty(12);P=np.empty((12,12));info=np.empty(5)
        self.api.call('observer_estimate',self.handle,float(t),x,P,info)
        self.t=info[3]
        return x,P,dict(last_impact=info[0],measurement_age=info[1],stale=bool(info[2]))
    def update_position(self,capture,position,arrival):
        info=np.empty(7);reason=ct.create_string_buffer(256)
        self.api.call('observer_update',self.handle,float(capture),array(position),float(arrival),info,reason,len(reason))
        self.t=info[6]
        result=dict(capture_time=capture,arrival_time=arrival,accepted=bool(info[0]),near_impact=bool(info[1]),nis=info[2],innovation=info[3:6].tolist(),reason=reason.value.decode())
        self.updates.append(result);return result
    def mpc_state(self,t,q,dq):
        x,P,info=self.estimate(t);return np.r_[x[:6],q,dq,x[6:9]],P,info
    def contact_pending(self,t):
        result=ct.c_int();self.api.call('observer_pending',self.handle,float(t),ct.byref(result));return bool(result.value)

class RawBall:
    """Native raw tracker, optionally seeded from an already-acquired fixture.

    Seeding reproduces a noiseless 3.1s acquisition before the first synthetic
    frame. Use only for trials intentionally starting from recorded settled
    fixture state, never as evidence for marker-acquisition robustness.
    Coordinates supplied to this adapter are robot-base axes, not Vicon axes.
    """
    def __init__(self,position=None,library=None):
        self.api=Library(library);self.handle=self.api.create('raw')
        self.seed_position=None if position is None else array(position)
        self.allow_motion=False;self.last_frame=None;self.selection=None;self.seen_time=None;self.rejections={}
    def close(self):
        if getattr(self,'handle',None):self.api.lib.offline_raw_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
    def _update(self,frame,t,points,face=None,prediction=None):
        xyz=array([p['position_m'] for p in points]).reshape(-1,3)
        ids=np.ascontiguousarray([p.get('id',k) for k,p in enumerate(points)],dtype=np.uint64)
        f=None if face is None else array(face)
        p=None if prediction is None else array(np.r_[prediction[0],np.asarray(prediction[1]).ravel(),prediction[2]])
        out=np.zeros(6)
        self.api.call('raw_update',self.handle,int(frame),float(t),xyz.ctypes.data,ids.ctypes.data,len(points),None if f is None else f.ctypes.data,None if p is None else p.ctypes.data,int(self.allow_motion),out)
        self.last_frame=int(frame);self.seen_time=out[5]
        if out[4] and self.selection is None:self.selection=dict(method='native acquired stationary marker')
        if not out[0]:self.rejections['not_associated']=self.rejections.get('not_associated',0)+1
        return out[1:4].copy() if out[0] else None
    def update(self,frame,prediction=None):
        t=float(frame['capture_time_s'])
        if self.seed_position is not None:
            # Fake acquisition ends before the real first frame; subsequent
            # frame numbers retain ordering, duplicates and restart behavior.
            p=self.seed_position;face=p-np.array([0.,0.,.8])
            for k in range(311):self._update(k,t-3.11+.01*k,[dict(id=1,position_m=p)],face)
            self.seed_position=None
            self.bootstrap_rejections=dict(self.rejections);self.rejections.clear()
        return self._update(1000+int(frame['frame']),t,frame['unlabeled_markers'],prediction=prediction)

class ReleaseGate:
    def __init__(self,gravity=(0.,0.,-9.81),library=None):
        self.api=Library(library);self.handle=self.api.create('gate',array(gravity));self.anchor=None;self.armed=False
    def close(self):
        if getattr(self,'handle',None):self.api.lib.offline_gate_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
    def arm(self):self.api.call('gate_arm',self.handle);self.armed=True
    def missing(self,t):self.api.call('gate_missing',self.handle,float(t))
    def update(self,t,p,frame):
        out=np.zeros(14);self.api.call('gate_update',self.handle,float(t),array(p),int(frame),out)
        if out[0]:self.anchor=out[1:4].copy()
        if out[4]:return dict(capture_time=out[5],position=out[6:9].tolist(),velocity=out[9:12].tolist(),observed_drop_m=out[12],frame=int(out[13]))
        return None
