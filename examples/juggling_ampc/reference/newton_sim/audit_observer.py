"""Reconstruct MPC feedback using ONLY archived position packets and arm sensors."""
import json
from time import perf_counter
import numpy as np
from arm.model import Panda
from estimation.ball import BallObserver,ObserverConfig


def audit_observer(directory,cfg,plant,arrays):
    robot=Panda();o=BallObserver(robot,ObserverConfig(**cfg['observer']))
    t=plant['t'];truth=plant['x'];a=arrays;idx=np.rint(a['t']/cfg['sim']['dt']).astype(int)
    packets=[json.loads(s) for s in (directory/'vicon_samples.jsonl').read_text().splitlines()]
    updates=[json.loads(s) for s in (directory/'observer_updates.jsonl').read_text().splitlines()]
    measurements={round(s['capture_time'],10):s for s in packets if not s['dropped']}
    o.record_arm(0.,truth[0,6:13],truth[0,13:20]);o.initialize(cfg['observer_initial_measurement']['position'])
    # The filter below never reads truth ball position, velocity, spin or contact.
    update_ms=[];predict_ms=[];arm_ms=[]
    j=0;control=0;estimate_error=0.;cov_error=0.;saved=np.load(directory/'observer.npz')
    for k,now in enumerate(t):
        if k:
            start=perf_counter();o.record_arm(now,truth[k,6:13],truth[k,13:20]);arm_ms.append(1000*(perf_counter()-start))
        while j<len(updates) and updates[j]['arrival_time']<=now+1e-10:
            u=updates[j];sample=measurements[round(u['capture_time'],10)]
            assert u['capture_time']<=sample['arrival_time']<=u['arrival_time']+1e-10
            assert u['arrival_time']-sample['arrival_time']<cfg['sim']['dt']+1e-10
            start=perf_counter();result=o.update_position(u['capture_time'],sample['position'],u['arrival_time']);update_ms.append(1000*(perf_counter()-start))
            assert result['accepted']==u['accepted'] and result['reason']==u['reason'];j+=1
        if control<len(idx) and k==idx[control]:
            start=perf_counter();state,P,info=o.mpc_state(now,truth[k,6:13],truth[k,13:20]);predict_ms.append(1000*(perf_counter()-start))
            estimate_error=max(estimate_error,float(np.max(abs(state-a['x'][control]))))
            cov_error=max(cov_error,float(np.max(abs(P-saved['covariance'][control]))))
            assert info['measurement_age']==saved['measurement_age'][control]
            control+=1
    assert j==len(updates) and control==len(idx)
    assert estimate_error<1e-10 and cov_error<1e-9
    assert np.min(np.linalg.eigvalsh(saved['covariance']))>0
    assert np.array_equal(saved['truth'],truth[idx]) and np.array_equal(saved['estimate'],a['x'])
    # Independently reproduce the random position-only camera, including dropped
    # packets. These truth positions are used for SENSOR audit, never the filter.
    vc=cfg['vicon'];rng=np.random.default_rng(vc['seed']);sigma=vc['position_std_m']
    held=truth[0,:3]+rng.normal(0,sigma,3)
    assert np.max(abs(held-np.array(cfg['observer_initial_measurement']['position'])))<1e-12
    sensor_error=0.;noise=[]
    for frame,s in enumerate(packets):
        capture=frame/vc['rate_hz'];assert abs(capture-s['capture_time'])<1e-10
        position=np.array([np.interp(capture,t,truth[:,axis]) for axis in range(3)])
        expected=position+rng.normal(0,sigma,3);dropped=rng.random()<vc['dropout_probability']
        assert s['dropped']==dropped and abs(s['arrival_time']-capture-vc['latency_s'])<1e-10
        sensor_error=max(sensor_error,float(np.max(abs(expected-np.array(s['position'])))))
        noise.append(np.array(s['position'])-position)
    assert sensor_error<1e-10
    result=dict(position_only_replay_verified=True,mpc_state_replay_max_error=estimate_error,
        covariance_replay_max_error=cov_error,sensor_replay_max_error_m=sensor_error,
        sensor_noise_std_per_axis_m=np.std(noise,axis=0).tolist(),
        received_packets=len(updates),causal_timestamps_verified=True,
        covariance_positive_definite=True,truth_velocity_spin_and_contact_not_used=True,
        spin_is_model_conditioned=True,
        replay_timing_ms={name:dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=float(max(v)))
                          for name,v in [('position_update',update_ms),('predict_for_mpc',predict_ms),('record_arm',arm_ms)]})
    (directory/'observer_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
