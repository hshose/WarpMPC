"""Native CPU EKF batching; GPU geometry is passed in the hardware ArmSample layout."""
import ctypes as ct
import numpy as np

class ObserverBatch:
    def __init__(self,library,cfg,position,velocity,geometry):
        self.lib=ct.CDLL(str(library));ptr=ct.c_void_p;array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.juggling_observers_create.argtypes=[ct.c_int,array,array,array,array];self.lib.juggling_observers_create.restype=ptr
        self.lib.juggling_observers_destroy.argtypes=[ptr];self.lib.juggling_observers_step.argtypes=[ptr,ct.c_double,array,array,array,ct.c_int,array]
        self.batch=len(position);self.out=np.zeros((self.batch,12),float)
        c=np.array([cfg['gravity_x'],cfg['gravity_y'],cfg['gravity_z'],cfg['drag'],cfg['mass'],cfg['radius'],cfg['restitution'],cfg['friction'],cfg['inertia_ratio'],cfg['paddle_radius']])
        self.handle=self.lib.juggling_observers_create(self.batch,c,self.array(position,(self.batch,3)),self.array(velocity,(self.batch,3)),self.array(geometry,(self.batch,48)))
        if not self.handle:raise RuntimeError('Could not initialize the native observer batch')
    @staticmethod
    def array(value,shape):
        a=np.ascontiguousarray(value,dtype=float)
        if a.shape!=shape:raise ValueError(f'Expected {shape}, got {a.shape}')
        return a
    def step(self,t,geometry,capture,position,predict=True):
        self.lib.juggling_observers_step(self.handle,float(t),self.array(geometry,(self.batch,48)),self.array(capture,(self.batch,)),self.array(position,(self.batch,3)),int(predict),self.out)
        return self.out.copy() if predict else None
    def close(self):
        if self.handle:self.lib.juggling_observers_destroy(self.handle);self.handle=None
    def __del__(self):
        if getattr(self,'handle',None):self.close()
