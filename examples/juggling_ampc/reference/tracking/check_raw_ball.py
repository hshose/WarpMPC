"""Offline raw-marker identity checks and actual preparation-only supervisor."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from .raw_ball import RawBall
from . import calibration as cal
from .session import model, DEFAULT_FACTORS
from .check_supervisor import exercise


def frame(k, points):
    return dict(frame=k, receive_time=k*.01+.005, latency_s=.005,
                paddle=dict(position_m=[0,0,1], quaternion_xyzw=[0,0,0,1]),
                unlabeled_markers=[dict(id=i,position_m=list(p)) for i,p in points])


def reject(call, text):
    try: call()
    except RuntimeError as e:
        assert text in str(e), str(e)
        return
    raise AssertionError('Expected '+text)


def run(output):
    output.mkdir(parents=True, exist_ok=False); rng=np.random.default_rng(834)
    tracker=RawBall(np.eye(4)); anchor=np.array([.04,-.03,1.67]); chosen=None
    for k in range(420):
        # More intermittent ceiling reflection; out-of-region clutter; rotating
        # marker IDs and shuffled array order must not determine identity.
        points=[(k%9,anchor+rng.normal(0,.0005,3)), (987,[.6,0,2.])]
        if k%4: points.append((444,[.15,.1,1.8]))
        rng.shuffle(points); p=tracker.update(frame(k,points))
        if p is not None and chosen is None: chosen=k*.01
    assert 3<=chosen<=3.05 and np.linalg.norm(tracker.position-anchor)<.003
    tracker.allow_motion=True
    for k in range(420,445):
        t=(k-420)*.01; p=anchor+[0,0,-4.905*t*t]
        observed=tracker.update(frame(k,[(42,p)]));assert np.allclose(observed,p)
    # Ambiguous observations become prediction-only, not an identity switch.
    predicted=tracker.estimate(4.45)[0][:3]
    assert tracker.update(frame(445,[(1,predicted),(2,predicted+[.001,0,0])])) is None
    assert tracker.status=='predicted' and tracker.rejections['ambiguous']==1
    empty=RawBall(np.eye(4))
    assert all(empty.update(frame(k,[])) is None for k in range(400))
    nearest=RawBall(np.eye(4))
    for k in range(400): nearest.update(frame(k,[(1,anchor),(2,anchor+[.1,0,0])]))
    assert np.allclose(nearest.position,anchor)
    missing=RawBall(np.eye(4))
    for k in range(350): missing.update(frame(k,[(1,anchor)]))
    for k in range(350,355): assert missing.update(frame(k,[])) is None
    assert missing.update(frame(355,[(99,anchor)])) is not None
    reject(lambda:missing.update(frame(365,[])), 'tracking lost')
    # Drive the real supervisor all the way to PREPARED with no ball release.
    robot,meta=model(DEFAULT_FACTORS)
    X=cal.pose([.007,-.155,.045],Rotation.from_euler('xyz',[.13,-.07,1.3]).as_quat())
    Y=cal.pose([.4,.8,.03],Rotation.from_euler('z',-.8).as_quat())
    # Reuse a prior offline fit, adjusting its CAD-face transform consistently.
    fixed=cal.load('runs/20260930_vicon_followup/supervisor_01/synthetic_hand_eye.json',require_live=False)
    fixed['F_T_P']=X.tolist()
    import pinocchio as pin
    from arm.model import HOME
    pin.framesForwardKinematics(robot.pin,robot.data,HOME)
    A=robot.data.oMf[robot.pin.getFrameId('panda_link8')].homogeneous.copy()
    p,n,J,N,R=robot.fk(HOME);face=np.eye(4);face[:3,:3]=np.array(R);face[:3,3]=np.array(p).ravel()
    fixed['P_T_CAD_face']=(cal.inverse(X)@cal.inverse(A)@face).tolist()
    calibration=output/'synthetic_calibration.json';cal.write_new(calibration,fixed)
    args=SimpleNamespace(output=output,calibration=calibration,factors=DEFAULT_FACTORS,
        plant=DEFAULT_FACTORS.parent/'controller/libjuggling_fixture_plant.so',
        library=DEFAULT_FACTORS.parent/'controller/libjuggling_controller.so',seconds=10.,ball='auto')
    prepared=exercise(args,robot,meta,X,Y,'prepare','prepared')
    assert prepared['error'] is None and prepared['commands'].get('plan',0)==0 and prepared['release_time_s'] is None,prepared
    # Full offline flight confirms the raw source feeds the existing observer.
    flight=exercise(args,robot,meta,X,Y,'fixture','raw_flight')
    assert flight['error'] is None and flight['impacts']>=10,flight
    stress=exercise(args,robot,meta,X,Y,'fixture','short_gaps_outliers',fault='short_gaps_outliers')
    assert stress['error'] is None and stress['impacts']>=10,stress
    assert stress['injected_frames']['dropout']>=40 and stress['raw_rejections']['ambiguous']>=8,stress
    assert stress['max_measurement_age_s']>=.05 and stress['max_position_error_m']<.03,stress
    result=dict(passed=True,physical_robot_used=False,selection_time_s=chosen,selection=tracker.selection,
        missing_ambiguous_and_empty_stream_checks=True,changing_ids_and_array_order=True,preparation=prepared,offline_flight=flight,
        dropout_outlier_flight=stress)
    cal.write_new(output/'checks.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
