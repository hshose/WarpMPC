import sys,json,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from loss import ball_lost

class BallOnlyGuards(unittest.TestCase):
    def test_robot_state_is_not_a_ball_loss_event(self):
        cfg=dict(gravity_x=0.,gravity_y=0.,gravity_z=-9.81,drag=.34,radius=.02,landing_forward=.5,landing_side=.5,fixture_landing_forward=0.,fixture_landing_side=.5)
        x=np.zeros((3,23));x[:,2]=.6;x[1,6:20]=1e6;x[2,6:20]=np.nan
        lost=ball_lost(x,np.zeros(3,bool),np.zeros((3,3)),np.zeros((3,3)),cfg)
        np.testing.assert_array_equal(lost,[False,False,False])
        x[1,3]=10.;x[2,2]=-.2
        lost=ball_lost(x,np.array([True,False,False]),np.zeros((3,3)),np.zeros((3,3)),cfg)
        np.testing.assert_array_equal(lost,[True,True,True])

if __name__=='__main__':unittest.main()
