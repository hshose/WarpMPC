import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from empirical import EmpiricalBatch

MODEL=Path(__file__).resolve().parents[1]/'calibration/model.json'

class EmpiricalTests(unittest.TestCase):
    def test_inflation_preserves_draws_and_profile_time_structure(self):
        a=EmpiricalBatch(MODEL,32,seed=123,multiplier=1.)
        b=EmpiricalBatch(MODEL,32,seed=123,multiplier=1.5)
        for t in [0.,.015,.431,2.]:
            np.testing.assert_allclose(b.acceleration(t),1.5*a.acceleration(t),atol=1e-14)
        np.testing.assert_allclose(b.impact,1.5*a.impact)
        ca,ra,pa=a.measurement(1.,np.zeros((32,3)))
        cb,rb,pb=b.measurement(1.,np.zeros((32,3)))
        np.testing.assert_allclose(cb-1,1.5*(ca-1),atol=1e-15)
        np.testing.assert_allclose(rb-1,1.5*(ra-1),atol=1e-15)
        np.testing.assert_allclose(pb,1.5*pa)
        np.testing.assert_allclose(b.plan_delay(),1.5*a.plan_delay())

    def test_arm_only_inflation_leaves_ball_contact_and_sensors_calibrated(self):
        nominal=EmpiricalBatch(MODEL,128,seed=731,multiplier=1.,arm_multiplier=1.)
        inflated=EmpiricalBatch(MODEL,128,seed=731,multiplier=1.,arm_multiplier=1.5)
        for t in (0.,.011,.6,2.):
            np.testing.assert_array_equal(nominal.acceleration(t),inflated.acceleration(t))
        np.testing.assert_array_equal(nominal.impact,inflated.impact)
        for a,b in zip(nominal.measurement(.8,np.zeros((128,3))),inflated.measurement(.8,np.zeros((128,3)))):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(nominal.plan_delay(),inflated.plan_delay())
        a,b=nominal.parameter_draws(),inflated.parameter_draws()
        np.testing.assert_array_equal(a['center_offset_local_m'],b['center_offset_local_m'])
        for key in ('servo_lag_s','servo_dead_time_s'):
            mean=np.asarray(nominal.model['plant'][key])
            np.testing.assert_allclose(b[key]-mean,1.5*(a[key]-mean),atol=1e-15)
        self.assertEqual(inflated.multipliers['arm_following'],1.5)

    def test_explicit_calibrated_validation_matches_legacy_unit_multiplier(self):
        a=EmpiricalBatch(MODEL,64,seed=911,multiplier=1.)
        b=EmpiricalBatch(MODEL,64,seed=911,multiplier=1.,arm_multiplier=1.)
        np.testing.assert_array_equal(a.acceleration(.432),b.acceleration(.432))
        np.testing.assert_array_equal(a.impact,b.impact)
        av,bv=a.parameter_draws(),b.parameter_draws()
        for key in av:np.testing.assert_array_equal(av[key],bv[key])

    def test_partial_contact_does_not_reset_other_worlds(self):
        a=EmpiricalBatch(MODEL,8)
        before=a.acceleration(.1).copy()
        a.new_flight([2,5],.1)
        np.testing.assert_array_equal(a.acceleration(.1)[[0,1,3,4,6,7]],before[[0,1,3,4,6,7]])
        self.assertFalse(np.array_equal(a.acceleration(.1)[[2,5]],before[[2,5]]))

    def test_world_noise_does_not_depend_on_other_world_contacts(self):
        a=EmpiricalBatch(MODEL,8,seed=123)
        b=EmpiricalBatch(MODEL,8,seed=123)
        a.new_flight([0],.1)
        a.new_flight([2],.2);b.new_flight([2],.2)
        np.testing.assert_array_equal(a.acceleration(.25)[2],b.acceleration(.25)[2])
        np.testing.assert_array_equal(a.impact[2],b.impact[2])
        for x,y in zip(a.measurement(.3,np.zeros((8,3))),b.measurement(.3,np.zeros((8,3)))):
            np.testing.assert_array_equal(x,y)

    def test_timing_bootstrap_rows_remain_paired_and_consecutive(self):
        a=EmpiricalBatch(MODEL,12)
        pool=np.asarray(a.spec['capture_error_receive_delay_samples_s'])
        first=a.row('capture_error_receive_delay_samples_s',blocked=True)
        index=a.blocks['capture_error_receive_delay_samples_s'][0].copy()
        np.testing.assert_array_equal(first,pool[(index-1)%len(pool)])
        second=a.row('capture_error_receive_delay_samples_s',blocked=True)
        np.testing.assert_array_equal(second,pool[index%len(pool)])
        self.assertTrue(np.all(a.parameter_draws()['servo_lag_s']>0))

if __name__=='__main__':unittest.main()
