import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_policy_export import inference_error


class InferenceErrorTests(unittest.TestCase):
    def test_small_residual_scale_allows_only_physical_roundoff(self):
        weights=dict(output_mean=np.zeros(1,np.float32),output_scale=np.full(1,1e-5,np.float32),output_baseline=np.ones((1,1),np.float32))
        expected=np.ones((1,1),np.float32);inputs=expected.copy()
        rounded=np.nextafter(expected,np.float32(2))
        self.assertTrue(inference_error(rounded,expected,inputs,weights)['passed'])
        self.assertFalse(inference_error(expected+1e-4,expected,inputs,weights)['passed'])

    def test_roundoff_allowance_does_not_hide_near_zero_error(self):
        weights=dict(output_mean=np.zeros(1,np.float32),output_scale=np.full(1,1e-5,np.float32),output_baseline=np.zeros((1,1),np.float32))
        expected=np.zeros((1,1),np.float32)
        self.assertFalse(inference_error(expected+1e-6,expected,expected,weights)['passed'])
        self.assertFalse(inference_error(expected+np.nan,expected,expected,weights)['passed'])


if __name__=='__main__':unittest.main()
