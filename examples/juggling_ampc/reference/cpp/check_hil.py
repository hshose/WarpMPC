"""Exercise callback input faults and plan expiry without starting any workers."""
import argparse
import ctypes as ct
import json
import tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from arm.model import HOME,Panda,VELOCITY_LIMIT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',type=Path,default=Path('build/controller/libjuggling_controller.so'))
    p.add_argument('--factors',type=Path,default=Path('build/cpp_factors'))
    p.add_argument('--output',type=Path,default=Path('docs/cpp_hil_checks.json'))
    args=p.parse_args()
    lib=ct.CDLL(str(args.library.resolve()))
    ptr=ct.c_void_p;array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.juggling_hil_create.argtypes=[ct.c_char_p,array,array,array,ct.c_double,ct.c_char_p,ct.c_size_t]
    lib.juggling_hil_create.restype=ptr
    lib.juggling_hil_create_acceleration.argtypes=lib.juggling_hil_create.argtypes
    lib.juggling_hil_create_acceleration.restype=ptr
    lib.juggling_hil_tick.argtypes=[ptr,ct.c_double,array,array,array,array,array,ct.POINTER(ct.c_double)]
    lib.juggling_hil_tick_acceleration.argtypes=lib.juggling_hil_tick.argtypes
    lib.juggling_hil_destroy.argtypes=[ptr]
    lib.juggling_hil_save.argtypes=[ptr,ct.c_char_p,ct.c_char_p,ct.c_size_t]
    metadata=json.loads((args.factors/'metadata.json').read_text());spec=metadata.get('spec',{})
    robot=Panda(assembly=spec.get('assembly'),**spec.get('geometry',{}));q=HOME.copy();dq=np.zeros(7);tau=np.asarray(robot.bias(q,dq)).ravel()
    error=ct.create_string_buffer(4096);results={}
    for name in ['stale_plan','clock_gap','nonfinite_time','nonfinite_joint','velocity_bound','first_fault_latched','nonfinite_reference_acceleration']:
        handle=lib.juggling_hil_create(str(args.factors.resolve()).encode(),q,dq,tau,.06,error,len(error))
        assert handle,error.value.decode()
        target=np.empty(7);reference=np.empty(21);age=ct.c_double();position=q.copy();velocity=dq.copy()
        try:
            if name=='nonfinite_reference_acceleration':
                acceleration=np.zeros(7);acceleration[3]=float('nan')
                status=lib.juggling_hil_tick_acceleration(handle,0.,position,velocity,acceleration,target,reference,ct.byref(age))
                assert status==2 and np.all(np.isfinite(reference)) and np.all(np.isfinite(target))
                results[name]=dict(fault=status,passed=True)
                continue
            elif name=='first_fault_latched':
                velocity[0]=VELOCITY_LIMIT[0]+.01
                assert lib.juggling_hil_tick(handle,0.,position,velocity,tau,target,reference,ct.byref(age))==2
                velocity[:]=0.
                # The later plan timeout and invalid clock must not replace the
                # original trigger in callback returns or the saved summary.
                for i in range(62):
                    assert lib.juggling_hil_tick(handle,i*.001,position,velocity,tau,target,reference,ct.byref(age))==2
                t=float('nan');expected=2
            elif name=='stale_plan':
                for i in range(61):
                    status=lib.juggling_hil_tick(handle,i*.001,position,velocity,tau,target,reference,ct.byref(age))
                    assert status==0,(i,status)
                t=.061;expected=4
            elif name=='clock_gap':t=.006;expected=1
            elif name=='nonfinite_time':t=float('nan');expected=1
            elif name=='nonfinite_joint':t=0.;position[2]=float('nan');expected=2
            else:t=0.;velocity[0]=VELOCITY_LIMIT[0]+.01;expected=2
            status=lib.juggling_hil_tick(handle,t,position,velocity,tau,target,reference,ct.byref(age))
            assert status==expected and np.all(np.isfinite(reference)) and np.all(np.isfinite(target)),(name,status)
            if name=='first_fault_latched':
                with tempfile.TemporaryDirectory() as directory:
                    assert lib.juggling_hil_save(handle,directory.encode(),error,len(error))==0,error.value.decode()
                    assert json.loads((Path(directory)/'hil_runtime.json').read_text())['fault']==expected
            results[name]=dict(fault=status,passed=True)
        finally:lib.juggling_hil_destroy(handle)
    acceleration=np.array([.1,-.2,.1,.2,-.1,.2,-.1])
    equivalent=np.asarray(robot.rnea(q,dq,acceleration)).ravel()
    a=lib.juggling_hil_create_acceleration(str(args.factors.resolve()).encode(),q,dq,acceleration,.06,error,len(error))
    b=lib.juggling_hil_create(str(args.factors.resolve()).encode(),q,dq,equivalent,.06,error,len(error))
    assert a and b,error.value.decode()
    try:
        ta=np.empty(7);tb=np.empty(7);ra=np.empty(21);rb=np.empty(21);age=ct.c_double()
        assert lib.juggling_hil_tick_acceleration(a,0.,q,dq,acceleration,ta,ra,ct.byref(age))==0
        assert lib.juggling_hil_tick(b,0.,q,dq,equivalent,tb,rb,ct.byref(age))==0
        delta=float(max(abs(ta-tb).max(),abs(ra-rb).max()));assert delta<1e-8
        results['reference_acceleration_boundary']=dict(passed=True,maximum_reference_difference=delta)
    finally:lib.juggling_hil_destroy(a);lib.juggling_hil_destroy(b)
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),passed=True,physical_robot_used=False,checks=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
