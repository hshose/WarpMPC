import sys,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax.numpy as jnp
from gpu_dataset import GPUDataset
from train_gpu import train

class GPUTrainingTests(unittest.TestCase):
    def test_training_without_sample_files(self):
        store=GPUDataset(20,4,device='cpu');rng=np.random.default_rng(3)
        for k in range(4):
            x=rng.normal(size=(20,40)).astype(np.float32)
            store.append(jnp.asarray(x),jnp.asarray(x[:,:7]*.1),jnp.ones(20,bool))
        store.manifest=dict(mpc_parity_passed=True,expert_stabilization_passed=True,live_source_hashes={},mpc_spec={})
        args=SimpleNamespace(seed=4,train_batch=16,source=Path('/tmp'),calibration=Path('/tmp'),authoritative_audit=None,epochs=2,learning_rate=.001)
        with tempfile.TemporaryDirectory() as tmp:
            path=train(store,args,Path(tmp))
            weights=np.load(path);self.assertEqual(weights['weight_3'].shape,(256,7))
            self.assertTrue(all(weights[k].dtype==np.float32 for k in weights.files))
            self.assertEqual({p.name for p in Path(tmp).iterdir()},{'policy.npz','training.json'})

if __name__=='__main__':unittest.main()
