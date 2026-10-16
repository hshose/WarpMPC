"""Thin ctypes bridge: the full RTI step and its timer execute in C++."""
import ctypes as ct
import os
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from arm import spin_formulation

ROOT=Path(__file__).resolve().parents[1]
DOUBLE=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
INTEGER=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')


class NativeController:
    def __init__(self,library=None,factors=None,compiled=True):
        self.directory=Path(factors or ROOT/'build/cpp_factors').resolve()
        runtime=self.directory/'runtime_values.txt'
        if runtime.exists():os.environ['JUGGLING_RUNTIME_VALUES']=str(runtime)
        self.lib=ct.CDLL(str(library or ROOT/'build/controller/libjuggling_controller.so'))
        self.error=ct.create_string_buffer(4096);L=self.lib;ptr=ct.c_void_p;err=[ct.c_char_p,ct.c_size_t]
        L.juggling_create.argtypes=[ct.c_char_p,ct.c_int,*err];L.juggling_create.restype=ptr
        L.juggling_destroy.argtypes=[ptr]
        L.juggling_step.argtypes=[ptr,DOUBLE,DOUBLE,ct.c_double,DOUBLE,DOUBLE,DOUBLE,*err]
        L.juggling_dimensions.argtypes=[ptr,INTEGER]
        L.juggling_pattern.argtypes=[ptr,INTEGER,INTEGER,INTEGER,INTEGER]
        L.juggling_linearize.argtypes=[ptr,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,*err]
        L.juggling_set_warmstart.argtypes=[ptr,DOUBLE,DOUBLE,*err]
        L.juggling_get_warmstart.argtypes=[ptr,DOUBLE,DOUBLE]
        L.juggling_initialize.argtypes=[ptr,DOUBLE,DOUBLE,*err]
        L.juggling_shift.argtypes=[ptr,ct.c_double,DOUBLE,DOUBLE,*err]
        self.handle=L.juggling_create(str(self.directory).encode(),int(compiled),self.error,len(self.error))
        if not self.handle:raise RuntimeError(self.error.value.decode())
        sizes=np.empty(4,np.int32);L.juggling_dimensions(self.handle,sizes);self.n,self.m,self.nh,self.na=map(int,sizes)
        self.hp=np.empty(self.n+1,np.int32);self.hi=np.empty(self.nh,np.int32);self.ap=np.empty(self.n+1,np.int32);self.ai=np.empty(self.na,np.int32)
        L.juggling_pattern(self.handle,self.hp,self.hi,self.ap,self.ai)
        self.metadata=json.loads((self.directory/'metadata.json').read_text())

    def close(self):
        if self.handle:self.lib.juggling_destroy(self.handle);self.handle=None

    def __del__(self):
        if getattr(self,'handle',None):self.close()

    def _call(self,name,*args):
        if getattr(self.lib,name)(self.handle,*args,self.error,len(self.error)):
            raise RuntimeError(self.error.value.decode())

    @staticmethod
    def _array(x):return np.ascontiguousarray(x,dtype=float)

    def initialize(self,x,tau):self._call('juggling_initialize',self._array(x),self._array(tau))
    def set_reference(self,face_offset,posture):
        if not hasattr(self.lib,'juggling_set_reference'):
            raise RuntimeError('Rebuild the optimized native variant for fixture references')
        if np.shape(face_offset)!=(3,) or np.shape(posture)!=(7,):
            raise ValueError('Expected three face coordinates and seven posture angles')
        self.lib.juggling_set_reference.argtypes=[ct.c_void_p,DOUBLE,DOUBLE,ct.c_char_p,ct.c_size_t]
        self._call('juggling_set_reference',self._array(face_offset),self._array(posture))
    def shift(self,dt,x,tau):self._call('juggling_shift',dt,self._array(x),self._array(tau))
    def set_contact_bias(self,bias_per_j8_local):
        if not hasattr(self.lib,'juggling_set_contact_bias'):
            raise RuntimeError('Contact bias needs the isolated experimental controller build')
        bias=self._array(bias_per_j8_local)
        if bias.shape!=(3,) or not np.isfinite(bias).all():raise ValueError('Expected finite contact bias vector')
        self.lib.juggling_set_contact_bias.argtypes=[ct.c_void_p,DOUBLE,ct.c_char_p,ct.c_size_t]
        self._call('juggling_set_contact_bias',bias)
    def set_warmstart(self,z,dual):self._call('juggling_set_warmstart',self._array(z),self._array(dual))
    def warmstart(self):
        z=np.empty(self.n);dual=np.empty(self.n+self.m);self.lib.juggling_get_warmstart(self.handle,z,dual);return z,dual

    def linearize(self,z,p):
        h=np.empty(self.nh);a=np.empty(self.na);gradient=np.empty(self.n);c=np.empty(self.m);cost=np.empty(1)
        self._call('juggling_linearize',self._array(z),self._array(p),h,gradient,a,c,cost)
        return sp.csc_matrix((h,self.hi,self.hp),shape=(self.n,self.n)),gradient,sp.csc_matrix((a,self.ai,self.ap),shape=(self.n+self.m,self.n)),c,float(cost[0])

    def step(self,x,tau,dt):
        plan=np.empty(self.n);target=np.empty(7);stats=np.empty(18)
        self._call('juggling_step',self._array(x),self._array(tau),dt,plan,target,stats)
        return plan.reshape(-1,34),target,stats


class CppSolver:
    """Runner-compatible wrapper; shift/solve both happen inside the timed step."""
    formulation=spin_formulation
    def __init__(self,cfg,robot,library=None,factors=None):
        self.cfg=cfg;self.robot=robot;self.native=NativeController(library,factors)
        for key,value in asdict(cfg).items():
            if key in ['codegen','codegen_optimization']:continue
            if key in ('ball_velocity_weight','paddle_lateral_weight') and key not in self.native.metadata['config'] and value==1.:continue
            if key in ('contact_edge_margin','contact_center_weight') and key not in self.native.metadata['config'] and value==(.02 if key=='contact_edge_margin' else 0.):continue
            if key=='jerk_margin' and key not in self.native.metadata['config'] and value==1.:continue
            if key=='warmstart_feedback_time' and key not in self.native.metadata['config'] and value==0.:continue
            if key=='initial_acceleration_margin' and key not in self.native.metadata['config'] and value==0.:continue
            if key in ('gravity_x','gravity_y','gravity_z') and key not in self.native.metadata['config'] and value==(0. if key!='gravity_z' else -9.81):continue
            if self.native.metadata['config'][key]!=value:raise ValueError(f'C++ factors have different {key}; re-export for this configuration.')
        self.pending_dt=0.;self.guard_ball_rollout=True;self.warmstart_fallback=None

    def shift(self,dt,x,tau):self.pending_dt=dt

    def solve(self,x,tau):
        if not self.guard_ball_rollout:raise ValueError('C++ controller currently uses the validated guarded warmstart.')
        Y,target,m=self.native.step(x,tau,self.pending_dt);self.pending_dt=0.;self.target=target
        self.warmstart_fallback=dict(reason='Guarded physical ball rollout',stage=int(m[15])) if m[15]>=0 else None
        info=dict(total_ms=m[0],cpp_total_ms=m[0],cpp_shift_ms=m[1],data_ms=m[2],qp_ms=m[3],cpp_merit_ms=m[4],
                  cpp_evaluator_threads=int(m[16]),cpp_sqp_iterations=int(m[17]),cost=m[5],violation=m[6],bound_violation=m[7],warm_dual_nonzeros=int(m[14]),
                  iterations=[dict(iter=int(m[12]),status=f'OSQP status {int(m[13])}',alpha=m[8],prim_res=m[9],dual_res=m[10])])
        return Y,info
