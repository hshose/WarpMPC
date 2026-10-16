"""Focused estimator checks: causal latency, flight consistency and observability."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .ball import BallObserver,ObserverConfig


class StaticPaddle:
    origin=np.zeros(3)
    def numeric_fk(self,q):return np.zeros(3),np.array([0.,0.,1.]),np.zeros((3,7)),np.zeros((3,7))
    def contact_fk(self,q):return (*self.numeric_fk(q),np.eye(3))
    def mass(self,q):return np.eye(7)


def main():
    robot=StaticPaddle();cfg=ObserverConfig();q=np.zeros(7)
    immediate=BallObserver(robot,cfg);delayed=BallObserver(robot,cfg)
    for o in [immediate,delayed]:o.record_arm(0.,q,q);o.initialize([0,0,1.5])
    rng=np.random.default_rng(314);F,b,_=immediate._flight(.005)
    x=rng.normal(size=12)
    free_error=float(np.max(abs(F@x+b-immediate._free(x,.005))))
    assert free_error<1e-12
    H=np.c_[np.eye(3),np.zeros((3,9))]
    O=np.vstack([H@np.linalg.matrix_power(F,i) for i in range(12)])
    rank=int(np.linalg.matrix_rank(O,tol=1e-12));assert rank==9
    assert np.max(abs(O[:,6:9]))==0
    actual=np.r_[[0.,0.,1.5],[.2,-.1,0.],[7.,-4.,15.],[.1,-.1,0.]]
    posteriors={};pending=[];errors=[];lag_error=0.;min_eig=np.inf
    for k in range(1,601):
        t=k*.0005
        actual=immediate._free(actual,.0005)
        for o in [immediate,delayed]:o.record_arm(t,q,q)
        if k%10==0:
            z=actual[:3]+rng.normal(0,.001,3)
            result=immediate.update_position(t,z,t);assert result['accepted']
            posteriors[k]=(immediate.x.copy(),immediate.P.copy());pending.append((k+10,k,z))
            if t>.1:errors.append(immediate.x[3:6]-actual[3:6])
        if pending and pending[0][0]<=k:
            _,capture,z=pending.pop(0);r=delayed.update_position(capture*.0005,z,t);assert r['accepted']
            target,P=posteriors[capture]
            lag_error=max(lag_error,float(np.max(abs(target-delayed.x))),float(np.max(abs(P-delayed.P))))
            min_eig=min(min_eig,float(np.linalg.eigvalsh(delayed.P)[0]))
    assert lag_error<1e-12 and min_eig>0
    velocity_rmse=float(np.sqrt(np.mean(np.sum(np.array(errors)**2,axis=1))));assert velocity_rmse<.04
    assert np.max(abs(delayed.x[6:9]))==0
    assert np.min(np.diag(delayed.P)[6:9])>=cfg.initial_spin_std**2
    # Future timestamps and duplicate captures do not alter the posterior.
    saved=delayed.x.copy()
    assert not delayed.update_position(.4,actual[:3],.3)['accepted']
    assert not delayed.update_position(delayed.last_capture,actual[:3],.3)['accepted']
    assert np.array_equal(saved,delayed.x)
    immediate.record_arm(.3005,q,q)
    outlier=immediate.update_position(.3005,actual[:3]+1.,.3005)
    assert not outlier['accepted'] and outlier['reason']=='innovation gate'
    # A model-predicted bounce changes spin through tangential contact, while
    # uncertainty remains finite and positive through the event-time Jacobian.
    bounce=BallObserver(robot,cfg);bounce.record_arm(0.,q,q);bounce.initialize([.01,0,.2]);bounce.x[3:6]=[.2,0,-1.]
    for k in range(1,301):bounce.record_arm(k*.0005,q,q)
    y,P,last,events=bounce._advance(.15)
    assert len(events)==1 and y[5]>0 and np.linalg.norm(y[6:9])>0
    assert np.linalg.eigvalsh(P)[0]>0
    assert not delayed.update_position(float('nan'),actual[:3],.3)['accepted']
    for k in range(601,1201):delayed.record_arm(k*.0005,q,q)
    _,_,health=delayed.estimate(.6)
    assert health['stale'] and len(delayed.history)<=502
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),passed=True,flight_mean_error=free_error,
        flight_observability_rank=rank,flight_spin_columns_zero=True,delayed_vs_capture_time_posterior_error=lag_error,
        velocity_rmse_m_s=velocity_rmse,min_covariance_eigenvalue=min_eig,spin_not_measured=True,
        outlier_rejected=True,future_and_duplicate_samples_rejected=True,stale_flag_and_bounded_history=True,impact_time_s=last)
    Path('docs/observer_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
