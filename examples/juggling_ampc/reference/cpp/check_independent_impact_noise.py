"""One isolated impact covariance/ABI check; no controller, plant or hardware.

Run --python-only for algebra/packet checks, or --library PATH after building the
existing juggling_offline_estimator library. --old-library PATH additionally
checks that the default packet still works with a library lacking the feature.
"""
import argparse
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np

from estimation.ball import BallObserver as PythonObserver, ObserverConfig
from tracking.native_estimator import (
    FIELDS, INDEPENDENT_TANGENT_FEATURE, Library, array, observer_config_values,
)


class StaticPlane:
    origin=np.zeros(3)
    def __init__(self):
        self.n=np.array([.2,-.1,1.]);self.n/=np.linalg.norm(self.n)
        tangent=np.cross(self.n,[1.,0.,0.]);tangent/=np.linalg.norm(tangent)
        self.R=np.column_stack((tangent,np.cross(self.n,tangent),self.n))
    def numeric_fk(self,q):return np.zeros(3),self.n,np.zeros((3,7)),np.zeros((3,7))
    def contact_fk(self,q):return (*self.numeric_fk(q),self.R)
    def mass(self,q):return np.eye(7)


def fixture():
    cfg=ObserverConfig(position_std=1e-5,drag=.34,ball_mass=.0039,
        initial_velocity_std=0.,initial_spin_std=0.,initial_acceleration_std=0.,
        acceleration_noise=0.,velocity_noise=0.,spin_noise=0.,contact_delay=0.,
        impact_time_std=0.,impact_velocity_std=0.,impact_tangent_std=0.,impact_spin_std=0.)
    robot=StaticPlane();position=robot.n*(cfg.radius+.005);velocity=-robot.n
    return cfg,robot,position,velocity,.02


def python_estimate(cfg,robot,p,v,end):
    observer=PythonObserver(robot,cfg);q=np.zeros(7)
    observer.record_arm(0.,q,q);observer.initialize(p,0.)
    # Explicit known prior velocity for this algebra fixture, not a policy input.
    observer.x[3:6]=v
    observer.record_arm(end,q,q)
    x,P,info=observer.estimate(end)
    return x,P,info['last_impact']


def native_estimate(api,cfg,robot,p,v,end,packet=None):
    values=observer_config_values(cfg,api.observer_features) if packet is None else packet
    handle=api.create('observer',values,len(values))
    try:
        arm=array(np.r_[np.zeros(3),robot.n,np.zeros(6),np.zeros(36)])
        api.call('observer_arm',handle,0.,arm)
        api.call('observer_arm',handle,end,arm)
        api.call('observer_initialize',handle,0.,array(p),array(v))
        x=np.empty(12);P=np.empty((12,12));info=np.empty(5)
        api.call('observer_estimate',handle,end,x,P,info)
        return x,P,float(info[0])
    finally:api.lib.offline_observer_destroy(handle)


def check_delta(off,on,cfg,normal,end,sigma):
    x0,P0,t0=off;x1,P1,t1=on
    assert 0.<t1<end and abs(t1-t0)<1e-12, 'Expected one interior impact'
    remaining=end-t1;decay=np.exp(-cfg.drag*remaining)
    velocity_integral=-np.expm1(-cfg.drag*remaining)/cfg.drag
    G=np.zeros((12,3));G[:3]=velocity_integral*np.eye(3);G[3:6]=decay*np.eye(3)
    tangent=np.eye(3)-np.outer(normal,normal)
    expected=sigma**2*G@tangent@G.T;delta=P1-P0
    error=float(np.max(abs(delta-expected)));mean_error=float(np.max(abs(x1-x0)))
    spin_error=float(max(np.max(abs(delta[6:9,:])),np.max(abs(delta[:,6:9]))))
    assert error<2e-10,(error,delta,expected)
    assert mean_error==0. and spin_error<1e-12,(mean_error,spin_error)
    assert np.linalg.eigvalsh(delta).min()>-1e-12,'Noise increment must be PSD'
    assert np.linalg.norm(delta[:3,3:6])>0.,'Post-impact flight must create p-v covariance'
    return dict(impact_time_s=t1,remaining_flight_s=remaining,max_covariance_error=error,
                max_mean_error=mean_error,max_spin_covariance_increment=spin_error)


def check_packets(cfg):
    legacy=array([getattr(cfg,k,0.) if k=='quadratic_drag' else getattr(cfg,k) for k in FIELDS]+[1.,0.])
    assert legacy.shape==(32,)
    for features in [0,INDEPENDENT_TANGENT_FEATURE]:
        assert np.array_equal(observer_config_values(cfg,features),legacy)
    old_fields=asdict(cfg);old_fields.pop('impact_independent_tangent_std')
    assert np.array_equal(observer_config_values(SimpleNamespace(**old_fields)),legacy)
    enabled=replace(cfg,impact_independent_tangent_std=.06)
    assert observer_config_values(enabled,INDEPENDENT_TANGENT_FEATURE)[31]==.06
    try:observer_config_values(enabled,0)
    except RuntimeError:pass
    else:raise AssertionError('Old-library feature rejection missing')
    for value in [-.01,np.nan,np.inf]:
        try:observer_config_values(replace(cfg,impact_independent_tangent_std=value),INDEPENDENT_TANGENT_FEATURE)
        except ValueError:pass
        else:raise AssertionError('Invalid covariance parameter accepted')
    return legacy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-only',action='store_true');parser.add_argument('--library')
    parser.add_argument('--old-library');args=parser.parse_args()
    if not args.python_only and not args.library:parser.error('Use --python-only or --library')
    cfg,robot,p,v,end=fixture();sigma=.06;enabled=replace(cfg,impact_independent_tangent_std=sigma)
    legacy=check_packets(cfg)
    pyoff=python_estimate(cfg,robot,p,v,end);pyon=python_estimate(enabled,robot,p,v,end)
    result=dict(packet_compatibility=True,python=check_delta(pyoff,pyon,cfg,robot.n,end,sigma))
    if args.library:
        api=Library(args.library)
        assert api.observer_features&INDEPENDENT_TANGENT_FEATURE,'New feature query is missing'
        off=native_estimate(api,cfg,robot,p,v,end,packet=legacy)
        on=native_estimate(api,enabled,robot,p,v,end)
        result['native']=check_delta(off,on,cfg,robot.n,end,sigma)
        result['python_native_noise_increment_max_error']=float(np.max(abs((on[1]-off[1])-(pyon[1]-pyoff[1]))))
        assert result['python_native_noise_increment_max_error']<2e-10
        # Direct ABI callers also receive native validation of the new slot.
        for value in [-.01,np.nan,np.inf]:
            packet=legacy.copy();packet[31]=value
            try:
                handle=api.create('observer',packet,len(packet))
            except RuntimeError:continue
            api.lib.offline_observer_destroy(handle)
            raise AssertionError('Native observer accepted invalid independent noise')
    if args.old_library:
        old=Library(args.old_library)
        assert not (old.observer_features&INDEPENDENT_TANGENT_FEATURE),'Expected an old library'
        old_off=native_estimate(old,cfg,robot,p,v,end)
        result['old_library_default_works']=bool(np.isfinite(old_off[0]).all())
        if args.library:
            result['old_new_default_mean_identical']=bool(np.array_equal(old_off[0],off[0]))
            result['old_new_default_covariance_identical']=bool(np.array_equal(old_off[1],off[1]))
            assert result['old_new_default_mean_identical'], 'Default observer mean changed'
            assert result['old_new_default_covariance_identical'], 'Default observer covariance changed'
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
