import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax.numpy as jnp
import warp as wp
from gpu_dataset import GPUDataset

class ReservoirTests(unittest.TestCase):
    def test_censoring_and_uniform_within_trajectory(self):
        store=GPUDataset(2000,5,device='cpu')
        valid=jnp.ones(2000,dtype=bool).at[-1].set(False)
        for tick in range(50):
            store.append(jnp.full((2000,40),tick,dtype=jnp.float32),jnp.full((2000,7),-tick,dtype=jnp.float32),valid)
        counts=store.seen.numpy();self.assertTrue(np.all(counts[:-1]==50));self.assertEqual(counts[-1],0)
        rows=store.data.numpy()[:1999*5];self.assertTrue(np.all(rows[:,:40]==rows[:,0,None]));self.assertTrue(np.all(rows[:,40:]==-rows[:,0,None]))
        self.assertLess(abs(rows[:,0].mean()-24.5),.6)
        self.assertEqual(len(store.indices(np.arange(2000))),1999*5)
        self.assertEqual(store.summary()['trajectories_with_data'],1999)

if __name__=='__main__':unittest.main()
