"""Preallocated, GPU-resident per-trajectory reservoir; never writes sample files.

Each valid datum has equal retention probability within its trajectory. Loss
censoring is supplied by the rollout before insertion. Every nonempty trajectory
retains samples, including short unsuccessful runs.
"""
import numpy as np
import warp as wp

@wp.kernel
def insert(x:wp.array2d(dtype=float), y:wp.array2d(dtype=float), valid:wp.array(dtype=wp.int32),
           data:wp.array2d(dtype=float), seen:wp.array(dtype=wp.int32), slots:int, seed:int):
    w=wp.tid()
    if valid[w]==0:return
    n=seen[w];seen[w]=n+1
    slot=n
    if n>=slots:
        state=wp.rand_init(seed,w)
        slot=wp.randi(state,0,n+1)
    if slot<slots:
        row=w*slots+slot
        for j in range(40):data[row,j]=x[w,j]
        for j in range(7):data[row,40+j]=y[w,j]

class GPUDataset:
    def __init__(self,batch,slots,seed=2026,device='cuda:0'):
        if batch<1 or slots<1:raise ValueError('Positive batch and reservoir slots required')
        self.batch=batch;self.slots=slots;self.seed=seed;self.calls=0;self.device=device
        self.data=wp.empty((batch*slots,47),dtype=float,device=device)
        self.seen=wp.zeros(batch,dtype=wp.int32,device=device)
        self.manifest=None
    def append(self,x,y,valid):
        import jax.numpy as jnp
        valid=jnp.asarray(valid,dtype=jnp.int32)
        wp.launch(insert,self.batch,inputs=[wp.from_jax(x),wp.from_jax(y),wp.from_jax(valid),self.data,self.seen,self.slots,self.seed+self.calls],device=self.device)
        self.calls+=1
    def indices(self,worlds):
        counts=np.minimum(self.seen.numpy(),self.slots)
        return np.concatenate([w*self.slots+np.arange(counts[w],dtype=np.int32) for w in worlds])
    def summary(self):
        counts=self.seen.numpy();retained=np.minimum(counts,self.slots)
        return dict(storage='preallocated GPU reservoir; no dataset files',batch=self.batch,slots_per_trajectory=self.slots,
                    allocated_bytes=self.batch*self.slots*47*4,eligible_rows=int(counts.sum()),retained_rows=int(retained.sum()),
                    trajectories_with_data=int(np.count_nonzero(counts)),discard='uniform reservoir within each trajectory',inputs=40,outputs=7)
