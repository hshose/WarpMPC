import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import BallLossMask,split_trajectories,normalization
from snapshot import fingerprint,assert_current
from training_data import prepare,output_baseline

class DataContract(unittest.TestCase):
    def test_loss_never_recovers(self):
        mask=BallLossMask(3)
        np.testing.assert_array_equal(mask.update(1.,[False,True,False]),[True,False,True])
        np.testing.assert_array_equal(mask.update(2.,[False,False,True]),[True,False,False])
        np.testing.assert_array_equal(mask.update(3.,[False,False,False]),[True,False,False])
        np.testing.assert_array_equal(mask.first_loss,[np.inf,1.,2.])
    def test_split_is_by_trajectory(self):
        parts=split_trajectories(60000)
        self.assertEqual(len(set(np.concatenate(list(parts.values())))),60000)
        self.assertEqual([len(parts[k]) for k in ('train','validation','test')],[48000,6000,6000])
    def test_training_normalization_constant_feature(self):
        stats=normalization(np.array([[1.,2.],[1.,4.]]),np.ones((2,1)))
        self.assertTrue(np.all(stats['input_scale']>0));np.testing.assert_equal(stats['input_mean'],[1.,3.])
    def test_refit_invalidates_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);(source/'config').mkdir();(source/'newton_sim').mkdir()
            cal=source/'calibration.json';cal.write_text('{}');(source/'config/cpp_ball_spec.json').write_text('{}')
            sim=source/'newton_sim/scene.py';sim.write_text('version=1')
            manifest=fingerprint(source,cal);assert_current(source,cal,manifest)
            sim.write_text('version=2')
            with self.assertRaises(RuntimeError):assert_current(source,cal,manifest)
    def test_disk_preparation_uses_training_trajectories_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(7)
            ids=np.repeat(np.arange(10),3);x=rng.normal(size=(30,40));y=rng.normal(size=(30,84))
            parts=split_trajectories(10,5);train=np.isin(ids,parts['train'])
            x[~train]+=1000;y[~train]-=1000
            for i,ix in enumerate(np.array_split(np.arange(30),4)):
                np.savez(root/f'shard_{i:06d}.npz',inputs=x[ix],targets=y[ix],trajectory_id=ids[ix])
            nx,ny,masks,stats,actual=prepare(root,root/'cache',10,5)
            self.assertIsInstance(nx,np.memmap)
            np.testing.assert_allclose(stats['input_mean'],x[train].mean(axis=0),atol=1e-14)
            np.testing.assert_allclose(stats['output_scale'],y[train].std(axis=0),atol=1e-14)
            np.testing.assert_array_equal(masks['train'],train)
            np.testing.assert_allclose(nx,(x-stats['input_mean'])/stats['input_scale'],rtol=1e-6)
            np.testing.assert_allclose(ny,(y-stats['output_mean'])/stats['output_scale'],rtol=1e-6)
    def test_narrow_features_use_deployed_float32_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(13)
            x=rng.normal(size=(30,40)).astype(np.float32)
            x[:,0]=np.float32(1.123456)+np.arange(30,dtype=np.float32)*np.float32(1e-6)
            ids=np.repeat(np.arange(10),3);y=rng.normal(size=(30,84)).astype(np.float32)
            np.savez(root/'shard_000000.npz',inputs=x,targets=y,trajectory_id=ids)
            nx,_,_,norm,_=prepare(root,root/'cache',10,5)
            for value in norm.values():self.assertEqual(value.dtype,np.float32)
            np.testing.assert_array_equal(nx,(x-norm['input_mean'])/norm['input_scale'])

    def test_microbatch_offsets_keep_trajectories_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entries=[];ids=[]
            for part in range(2):
                folder=root/str(part);folder.mkdir()
                local=np.repeat(np.arange(5),2);global_ids=local+5*part;ids.extend(global_ids)
                np.savez(folder/'shard.npz',inputs=np.tile(global_ids[:,None],(1,40)),targets=np.tile(global_ids[:,None],(1,84)),trajectory_id=local)
                entries.append(dict(path=f'{part}/shard.npz',trajectory_offset=5*part))
            (root/'manifest.json').write_text(json.dumps(dict(shards=entries)))
            _,_,masks,_,splits=prepare(root,root/'cache',10,123)
            for key in masks:np.testing.assert_array_equal(masks[key],np.isin(ids,splits[key]))
    def test_residual_normalization_reconstructs_native_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(9)
            x=rng.normal(size=(30,40));baseline=output_baseline([0.,.03,.062,.097])
            y=x@baseline+rng.normal(0,.001,size=(30,84));ids=np.repeat(np.arange(10),3)
            np.savez(root/'shard_000000.npz',inputs=x,targets=y,trajectory_id=ids)
            nx,ny,_,norm,_=prepare(root,root/'cache',10,1,baseline)
            reconstructed=ny*norm['output_scale']+norm['output_mean']+x@norm['output_baseline']
            np.testing.assert_allclose(reconstructed,y,atol=2e-10)
            self.assertLess(float(np.max(norm['output_scale'])),.002)
if __name__=='__main__':unittest.main()
