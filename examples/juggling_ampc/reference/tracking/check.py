"""Offline evidence for hand-eye, HOME relocation, release gate, and MPC costs."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

from arm.model import HOME
from cpp.python_controller import NativeController
from . import calibration as cal
from .fixture import ReleaseGate, fixture_home
from .session import model, calibration_waypoints, DEFAULT_FACTORS
from .streams import ball_position


def fail_expected(call):
    try: call()
    except (ValueError, RuntimeError): return
    raise AssertionError('Expected rejection')


def run(output, factors, library):
    output.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(739)
    robot, meta=model(factors); transforms=[]
    for q in calibration_waypoints():
        pin.framesForwardKinematics(robot.pin,robot.data,q)
        transforms.append(robot.data.oMf[robot.pin.getFrameId('panda_link8')].homogeneous.copy())
    X=cal.pose([.007,-.155,.045],R.from_euler('xyz',[.13,-.07,1.3]).as_quat())
    Y=cal.pose([1.2,-.4,.07],R.from_euler('z',.6).as_quat())
    samples=[]
    for A in transforms:
        C=Y@A@X
        C[:3,3]+=rng.normal(0,.0002,3)
        C[:3,:3]=C[:3,:3]@R.from_rotvec(rng.normal(0,np.deg2rad(.025),3)).as_matrix()
        samples.append(dict(B_T_F=A.tolist(),V_T_P=C.tolist()))
    result=cal.solve(samples); error=cal.errors(X,np.array(result['F_T_P']))
    result['source_kind']='synthetic'
    assert error[0]<.002 and error[1]<np.deg2rad(.3),error
    cal.write_new(output/'synthetic_calibration.json',result)
    fail_expected(lambda:cal.load(output/'synthetic_calibration.json'))
    fixed=cal.load(output/'synthetic_calibration.json',require_live=False)
    relocations=[]
    for yaw,xyz in [(-.8,[.4,.8,.03]),(2.5,[-.6,.1,.2])]:
        moved=cal.pose(xyz,R.from_euler('z',yaw).as_quat()); A=transforms[0]; paired=[]
        for _ in range(80):
            C=moved@A@X; C[:3,3]+=rng.normal(0,.0002,3)
            paired.append(dict(B_T_F=A,V_T_P=C))
        alignment=cal.align_home(fixed,paired); p=np.array([.42,.12,1.22])
        observed=moved[:3,:3]@p+moved[:3,3]; inv=np.array(alignment['B_T_V'])
        err=float(np.linalg.norm(inv[:3,:3]@observed+inv[:3,3]-p));assert err<.003
        relocations.append(err)
    fail_expected(lambda:cal.solve([samples[0]]*12))
    outlier=[dict(s) for s in samples];outlier[7]['V_T_P']=np.array(outlier[7]['V_T_P']).copy();outlier[7]['V_T_P'][0,3]+=.08
    fail_expected(lambda:cal.solve(outlier))
    # SDK EE transform need not be identity, and storage is column-major.
    F_T_EE=cal.pose([.03,-.02,.10],R.from_euler('xyz',[.3,-.2,.1]).as_quat())
    state=dict(O_T_EE=(transforms[0]@F_T_EE).ravel(order='F').tolist(),F_T_EE=F_T_EE.ravel(order='F').tolist())
    assert np.allclose(cal.flange_from_state(state),transforms[0])
    homes=[]
    for dx,dy in [(0,0),(.04,-.03),(-.04,.04),(.08,0)]:
        ball=robot.origin+np.array([dx,dy,.67]);home=fixture_home(robot,ball)
        p,n,*_=robot.numeric_fk(np.array(home['q_home']));assert np.linalg.norm(p[:2]-ball[:2])<.001
        homes.append(home)
    fail_expected(lambda:fixture_home(robot,robot.origin+[.2,0,.67]))
    fail_expected(lambda:fixture_home(robot,robot.origin+[0,0,.1]))
    fail_expected(lambda:fixture_home(robot,robot.origin+[0,0,.321]))
    releases=[]
    for seed in range(12):
        rand=np.random.default_rng(seed);gate=ReleaseGate();anchor=np.array([.4,.1,1.2]);event=None
        for k in range(320):
            t=k*.01;p=anchor.copy()
            if t>=2.8:p[2]-=.5*9.81*(t-2.8)**2
            p+=rand.normal(0,.0008,3)
            e=gate.update(t,p,k)
            if t>=1. and not gate.armed:gate.arm()
            if t<2.8:assert e is None
            if e is not None:event=e;break
        assert event is not None
        delay=event['capture_time']-2.8;assert .04<=delay<=.10,(seed,delay)
        v_error=abs(event['velocity'][2]+9.81*delay);assert v_error<.13,(seed,v_error)
        releases.append(dict(delay_s=delay,vertical_velocity_error=v_error))
    # Lost/duplicate observations cannot turn elapsed wall time into a release.
    gate=ReleaseGate();gate.anchor=np.array([.4,.1,1.2]);gate.arm();gate.update(0,gate.anchor,1)
    assert gate.update(.02,gate.anchor-[0,0,.03],1) is None
    fail_expected(lambda:gate.missing(.05))
    early=ReleaseGate();early.anchor=np.array([.4,.1,1.2]);fail_expected(lambda:early.update(0,early.anchor-[0,0,.015],1))
    marker=dict(subject='ball_subject',name='hsh_pingpong_ball',position_m=[1,2,3])
    assert np.array_equal(ball_position(dict(markers=[marker])),[1,2,3])
    fail_expected(lambda:ball_position(dict(markers=[marker,dict(marker,subject='other')])))
    assert ball_position(dict(markers=[dict(marker,position_m=None)])) is None
    # Exact change in cost/gradient from constant residual offsets, no changes
    # to constraint values/Jacobians/Hessians. Test compiled stage and terminal.
    base=NativeController(library,factors);shifted=NativeController(library,factors)
    home=homes[1];delta=np.array(home['face_offset_m']);qh=np.array(home['q_home'])
    shifted.set_reference(delta,qh);x=np.r_[np.array(home['ball_fixture_base_m'])-robot.origin,[0,0,0],qh,np.zeros(10)]
    tau=np.asarray(robot.bias(qh,np.zeros(7))).ravel();base.initialize(x,tau);z,dual=base.warmstart();p=np.r_[x,tau]
    h0,g0,a0,c0,f0=base.linearize(z,p);h1,g1,a1,c1,f1=shifted.linearize(z,p)
    assert np.max(abs(h0-h1))==0 and np.max(abs(a0-a1))==0 and np.array_equal(c0,c1)
    # Independent residual evaluation from serialized symbolic factors.
    import casadi as ca
    expected=0.;off=np.array(meta['arrays']['offset']);scale=np.array(meta['arrays']['scale']);cfg=meta['config']
    for k in range(len(meta['stage_steps'])+1):
        terminal=k==len(meta['stage_steps']);name='last_value' if terminal else meta['stage_names'][k]+'_value'
        fun=ca.Function.load(str(factors/(name+'.casadi')))
        r=np.array(fun(z[k*34:(k+1 if terminal else k+2)*34],p)[0]).ravel();s=np.sqrt(cfg['objective_scale']);dr=np.zeros_like(r)
        dr[0]=-s*cfg['energy_weight']*delta[2]*(2 if terminal else 1)
        dr[1:3]=-s*cfg['lateral_weight']*delta[:2]/.2*(2 if terminal else 1)
        if not terminal:
            dr[5:8]=-s*cfg['pose_weight']*delta/[.3,.3,.15];dr[37:44]=-s*cfg['posture_weight']*(qh-HOME)
            w=np.ones(54);w[:51]=np.sqrt(meta['stage_steps'][k]/.03);w[30:37]*=.03/meta['stage_steps'][k];r*=w;dr*=w
        expected+=r@dr+.5*(dr@dr)
    assert abs(f1-f0-expected)<1e-9,(f1-f0,expected)
    grad_errors=[]
    for i in [0,1,6,34+5,34+8,len(z)-34]:
        eps=1e-6;zz=z.copy();zz[i]+=eps;fp=shifted.linearize(zz,p)[-1];zz[i]-=2*eps;fm=shifted.linearize(zz,p)[-1]
        grad_errors.append(abs((fp-fm)/(2*eps)-g1[i]))
    assert max(grad_errors)<1e-5,grad_errors
    shifted.initialize(x,tau);fail_expected(lambda:shifted.set_reference(delta,qh))
    base.close();shifted.close()
    report=dict(passed=True,physical_robot_used=False,live_calibration=False,
                hand_eye_translation_error_m=error[0],hand_eye_rotation_error_deg=float(np.rad2deg(error[1])),
                calibration=result,relocated_world_ball_errors_m=relocations,fixture_homes=homes,
                release_trials=releases,degenerate_calibration_rejected=True,calibration_outlier_rejected=True,
                missing_ball_and_early_release_rejected=True,ambiguous_ball_rejected=True,
                reference_cost_difference=f1-f0,independent_cost_difference=expected,max_gradient_error=max(grad_errors))
    cal.write_new(output/'checks.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--factors',type=Path,default=DEFAULT_FACTORS)
    parser.add_argument('--library',type=Path,default=DEFAULT_FACTORS.parent/'controller/libjuggling_controller.so')
    args=parser.parse_args();run(args.output,args.factors,args.library)
