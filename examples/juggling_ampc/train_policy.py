"""Train from censored rollout shards, with whole-trajectory held-out validation."""
import bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from training_data import prepare,output_baseline
from snapshot import assert_current


def forward(params,x):
    for w,b in params[:-1]:x=jax.nn.tanh(jnp.matmul(x,w,precision=jax.lax.Precision.HIGHEST)+b)
    w,b=params[-1];return jnp.matmul(x,w,precision=jax.lax.Precision.HIGHEST)+b

def train(a):
    if a.epochs<1 or a.batch<1 or a.learning_rate<=0:raise ValueError('Positive epochs, batch and learning rate required')
    if [int(v) for v in a.hidden.split(',')] != [256,256,256]:raise ValueError('Franka policy requires three hidden layers of 256 neurons')
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if 'mpc_spec' not in manifest:raise ValueError('Dataset must record its exact MPC specification')
    assert_current(a.source.resolve(),a.calibration.resolve(),manifest['live_source_hashes'])
    if not manifest.get('expert_stabilization_passed'):raise RuntimeError('Expert stabilization gate has not passed')
    if not manifest.get('mpc_parity_passed'):raise RuntimeError('Full MPC parity gate has not passed')
    baseline=output_baseline(manifest['stage_times_s'])
    x,y,masks,norm,splits=prepare(a.dataset,a.output/'data_cache',manifest['trajectories'],a.seed,baseline)
    rng=np.random.default_rng(a.seed);sizes=[x.shape[1],*[int(v) for v in a.hidden.split(',')],y.shape[1]]
    params=[(jnp.asarray(rng.normal(0,np.sqrt(2/(ni+no)),(ni,no)),dtype=jnp.float32),jnp.zeros(no,dtype=jnp.float32)) for ni,no in zip(sizes[:-1],sizes[1:])]
    momentum=jax.tree.map(jnp.zeros_like,params);variance=jax.tree.map(jnp.zeros_like,params)
    @jax.jit
    def step(params,m,v,t,xb,yb):
        loss,g=jax.value_and_grad(lambda p:jnp.mean((forward(p,xb)-yb)**2))(params)
        m=jax.tree.map(lambda old,grad:.9*old+.1*grad,m,g);v=jax.tree.map(lambda old,grad:.999*old+.001*grad**2,v,g)
        params=jax.tree.map(lambda p,mi,vi:p-a.learning_rate*(mi/(1-.9**t))/(jnp.sqrt(vi/(1-.999**t))+1e-8),params,m,v)
        return params,m,v,loss
    predict=jax.jit(forward)
    def evaluate(mask,detailed=False):
        indices=np.flatnonzero(mask);total=0.;physical=np.zeros(y.shape[1])
        for start in range(0,len(indices),a.batch):
            ix=indices[start:start+a.batch];e=np.asarray(predict(params,jnp.asarray(x[ix])))-y[ix];square=e.astype(float)**2;total+=float(np.sum(square))
            if detailed:physical+=np.sum(square,axis=0)*norm['output_scale']**2
        mse=total/(len(indices)*y.shape[1])
        if not detailed:return mse
        physical=physical.reshape(4,21)/len(indices)
        return dict(normalized_mse=mse,joint_position_rmse_rad=float(np.sqrt(physical[:,:7].mean())),joint_velocity_rmse_rad_s=float(np.sqrt(physical[:,7:14].mean())),joint_torque_rmse_nm=float(np.sqrt(physical[:,14:21].mean())))
    a.output.mkdir(parents=True,exist_ok=True);best=np.inf;history=[];t=0;start=time.monotonic();best_params=None
    # Preprocessing can be lengthy for 60k worlds; repeat the live refit check
    # immediately before the very first optimizer update, not only at startup.
    assert_current(a.source.resolve(),a.calibration.resolve(),manifest['live_source_hashes'])
    authoritative_audit=None
    if a.authoritative_audit:
        from pretraining_audit import wait_for_authoritative_audit
        authoritative_audit=wait_for_authoritative_audit(a.authoritative_audit,manifest['live_source_hashes'])
        assert_current(a.source.resolve(),a.calibration.resolve(),manifest['live_source_hashes'])
    for epoch in range(a.epochs):
        indices=rng.permutation(np.flatnonzero(masks['train']));training_loss=0.
        for k in range(0,len(indices),a.batch):
            ix=indices[k:k+a.batch];t+=1;params,momentum,variance,loss=step(params,momentum,variance,t,x[ix],y[ix])
            training_loss+=float(loss)*len(ix)
        val=evaluate(masks['validation'])
        if not np.isfinite(val) or not np.isfinite(training_loss):raise RuntimeError('Nonfinite policy training or validation loss')
        history.append(dict(epoch=epoch+1,training_mse=training_loss/len(indices),validation_mse=val));print(json.dumps(history[-1]),flush=True)
        if val<best:
            best=val;best_params=jax.tree.map(lambda a:np.asarray(a).copy(),params)
    params=best_params
    validation_metrics=evaluate(masks['validation'],True);test_metrics=evaluate(masks['test'],True)
    checkpoint=dict(norm,**{f'weight_{i}':w for i,(w,b) in enumerate(params)},**{f'bias_{i}':b for i,(w,b) in enumerate(params)})
    np.savez(a.output/'policy.npz',**checkpoint)
    report=dict(history=history,validation_mse=best,test_mse=test_metrics['normalized_mse'],validation_metrics=validation_metrics,test_metrics=test_metrics,elapsed_s=time.monotonic()-start,
        parameter_dtype='float32',split_counts={k:int(v.sum()) for k,v in masks.items()},trajectory_splits={k:v.tolist() for k,v in splits.items()},sizes=sizes,activation='tanh',seed=a.seed,
        authoritative_pretraining_audit=authoritative_audit,live_source_hashes=manifest['live_source_hashes'],mpc_spec=manifest['mpc_spec'],output_representation='normalized residual around fixed state/torque affine baseline',closed_loop_gate_passed=False)
    (a.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--authoritative-audit',type=Path);p.add_argument('--hidden',default='256,256,256');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=8192);p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--seed',type=int,default=2026);train(p.parse_args())
