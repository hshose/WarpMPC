"""Check causal forecasting across flight/contact without changing live state."""
from pathlib import Path
import numpy as np
from tracking.session import model
from tracking.rehearsal import VirtualBall
from estimation.ball import ObserverConfig
from arm.model import HOME

def main():
 robot,meta=model(Path('build/variants/fixture39_a50l12e28v2/factors'))
 cfg=ObserverConfig(ball_mass=.0039,drag=.34,restitution=.54,contact_delay=0.)
 q=HOME.copy();dq=np.zeros(7);acc=np.zeros(7)
 for height,vz,contact in [(.5,-1.,False),(.025,-2.,True)]:
  obj=VirtualBall(robot,cfg);obj.record_arm(0.,q,dq)
  face=np.asarray(robot.contact_fk(q)[0]).ravel()-robot.origin
  obj.initialize(face+np.array([0.,0.,height]),0.);obj.x[5]=vz
  old=obj.x.copy();old_history=list(obj.history);old_events=list(obj.events)
  predicted,qf,vf=obj.forecast(.012,q,dq,acc)
  assert obj.t==0. and np.array_equal(obj.x,old) and list(obj.history)==old_history and obj.events==old_events
  obj.record_arm(.012,q,dq);actual=obj.advance(.012)
  assert np.max(abs(actual-predicted))<1e-10
  assert bool(obj.events)==contact
 print('PASS: free-flight and impact forecasts match propagation; live history/state unchanged')
if __name__=='__main__':main()
