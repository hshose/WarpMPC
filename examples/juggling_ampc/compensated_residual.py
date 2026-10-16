"""Accurate sparse KKT residuals using only float32 operations.

Dekker products and TwoSum accumulation retain low-order terms lost in the
ill-conditioned arm KKT system. No float64 GPU arithmetic is used.
"""
import numpy as np
import jax.numpy as jnp
import warp as wp


@wp.kernel(module_options={'fuse_fp':False,'enable_backward':False},module='unique')
def _residual(values:wp.array2d(dtype=wp.float32),x:wp.array2d(dtype=wp.float32),
              rhs:wp.array2d(dtype=wp.float32),starts:wp.array(dtype=wp.int32),
              columns:wp.array(dtype=wp.int32),indices:wp.array(dtype=wp.int32),
              result:wp.array2d(dtype=wp.float32)):
    row,batch=wp.tid()
    total=rhs[row,batch]
    correction=wp.float32(0.)
    for entry in range(starts[row],starts[row+1]):
        a=values[indices[entry],batch]
        b=x[columns[entry],batch]
        product=a*b
        split_a=wp.float32(4097.)*a
        high_a=split_a-(split_a-a)
        low_a=a-high_a
        split_b=wp.float32(4097.)*b
        high_b=split_b-(split_b-b)
        low_b=b-high_b
        error=((high_a*high_b-product)+high_a*low_b+low_a*high_b)+low_a*low_b
        term=-product
        updated=total+term
        virtual=updated-total
        addition_error=(total-(updated-virtual))+(term-virtual)
        correction=correction+addition_error-error
        total=updated
    result[row,batch]=total+correction


_call=wp.jax_kernel(_residual,num_outputs=1,vmap_method='broadcast_all')


def compile_residual(upper,symmetric=True):
    rows=upper.indices;cols=np.repeat(np.arange(upper.shape[1]),np.diff(upper.indptr))
    entries=[[] for _ in range(upper.shape[0])]
    for i,(r,c) in enumerate(zip(rows,cols)):
        entries[r].append((c,i))
        if symmetric and r!=c:entries[c].append((r,i))
    starts=np.r_[0,np.cumsum([len(row) for row in entries])].astype(np.int32)
    pairs=np.array([pair for row in entries for pair in sorted(row)],dtype=np.int32)
    starts,columns,indices=map(jnp.asarray,(starts,pairs[:,0],pairs[:,1]))
    def residual(values,x,rhs):
        shape=(rhs.shape[1],rhs.shape[0])
        return _call(values.T,x.T,rhs.T,starts,columns,indices,
                     launch_dims=shape,output_dims={'result':shape})[0].T
    return residual
