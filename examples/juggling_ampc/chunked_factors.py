"""Partition large straight-line Warp factors at live-variable boundaries.

No arithmetic is approximated or reordered. Runtime-optimized kernels pass only
live scalars between chunks, avoiding pathological compilation of 400k-operation
single CUDA functions. WarpMPC's linear algebra optimizations are unaffected.
"""
import re
import jax.numpy as jnp
from warpmpc.numpysadi import generate_source, load_function


def convert_chunked(function, chunk_size=4000, batched=False):
    if chunk_size < 1:raise ValueError('Positive chunk size required')
    source=generate_source(function,backend='warp')
    header=source.split('@wp.kernel\n')[0]
    body=source.split('    _sample = wp.tid()\n',1)[1].split('\n\n',1)[0]
    lines=[line.strip() for line in body.splitlines() if line.strip()]
    assignments=[line for line in lines if not line.startswith('out') or '_flat[' not in line.split(' = ')[0]]
    writes=[line for line in lines if line.startswith('out') and '_flat[' in line.split(' = ')[0]]
    # Generated statements are scalar assignments; inputs are immutable arrays.
    names={line.split(' = ',1)[0] for line in assignments}
    token=re.compile(r'\b[A-Za-z_]\w*\b')
    dependencies=[set(token.findall(line.split(' = ',1)[1]))&names for line in assignments]
    output_names=[]
    for line in writes:
        matches=set(token.findall(line.split(' = ',1)[1]))&names
        if len(matches)!=1:raise ValueError('Expected a named generated output scalar')
        output_names.append(next(iter(matches)))
    bounds=list(range(0,len(assignments),chunk_size))+[len(assignments)]
    boundary_set=set(bounds);live=set(output_names);at={len(assignments):sorted(live)}
    for i in range(len(assignments)-1,-1,-1):
        live.discard(assignments[i].split(' = ',1)[0]);live.update(dependencies[i])
        if i in boundary_set:at[i]=sorted(live)
    if live:raise ValueError('Generated factor reads uninitialized temporary variables')
    nin=function.n_in();inputs=[f'arg{i}_flat' for i in range(nin)];chunks=[]
    for k,(start,end) in enumerate(zip(bounds[:-1],bounds[1:])):
        name=f'{function.name()}_'+('transposed_' if batched else '')+f'chunk_{k}';before=at[start];after=at[end]
        assigned={line.split(' = ',1)[0] for line in assignments[start:end]}
        changed=[v for v in after if v in assigned]
        needed=set().union(*dependencies[start:end])&set(before)
        before_index={v:i for i,v in enumerate(before)}
        merged_index={v:max(1,len(before))+i for i,v in enumerate(changed)}
        mapping=[merged_index.get(v,before_index.get(v,-1)) for v in after]
        args=[f'{s}: wp.array(dtype=wp.float32)' for s in inputs+['carry_in','carry_out']]
        code=header+'\n@wp.kernel\ndef _kernel('+', '.join(args)+'):\n    _sample = wp.tid()\n'
        if batched:
            code+=f'    _batch = carry_in.shape[0] // {max(1,len(before))}\n'
            code+=''.join(f'    {v} = carry_in[{i} * _batch + _sample]\n' for i,v in enumerate(before) if v in needed)
            code+=''.join('    '+re.sub(r'(arg\d+_flat)\[_sample \* \d+ \+ (\d+)\]',r'\1[\2 * _batch + _sample]',line)+'\n' for line in assignments[start:end])
            code+=''.join(f'    carry_out[{i} * _batch + _sample] = wp.float32({v})\n' for i,v in enumerate(changed))
        else:
            code+=''.join(f'    {v} = carry_in[_sample * {max(1,len(before))} + {i}]\n' for i,v in enumerate(before) if v in needed)
            code+=''.join('    '+line+'\n' for line in assignments[start:end])
            code+=''.join(f'    carry_out[_sample * {len(changed)} + {i}] = wp.float32({v})\n' for i,v in enumerate(changed))
        code+=f'\n_call = wp.jax_kernel(_kernel, num_outputs=1, launch_dims=1, output_dims={{"carry_out": {len(changed)}}}, vmap_method="broadcast_all")\n'
        code+=f'\ndef {name}('+', '.join(inputs+['carry'])+'):\n'
        if batched:
            code+='    batch = carry.shape[1]\n'
            code+='    changed = _call('+', '.join(inputs+['carry.reshape(-1)'])+f', launch_dims=(batch,), output_dims={{"carry_out": ({len(changed)} * batch,)}})[0].reshape({len(changed)},batch)\n'
            code+=f'    return jnp.concatenate([carry,changed],axis=0)[jnp.array({mapping!r}),:]\n'
        else:
            code+='    changed = _call('+', '.join(inputs+['carry'])+')[0]\n'
            code+=f'    return jnp.concatenate([carry,changed])[jnp.array({mapping!r})]\n'
        chunks.append(load_function(code,name))
    final_index={v:i for i,v in enumerate(at[len(assignments)])}
    sizes=[function.size1_out(i)*function.size2_out(i) for i in range(function.n_out())]
    def evaluate(*values):
        if batched:
            batch=values[0].shape[0]
            flat=[jnp.asarray(v,dtype=jnp.float32).reshape(batch,function.size1_in(i),function.size2_in(i)).swapaxes(1,2).reshape(batch,-1).T.reshape(-1) for i,v in enumerate(values)]
            carry=jnp.zeros((1,batch),dtype=jnp.float32)
            for chunk in chunks:carry=chunk(*flat,carry)
            packed=carry[jnp.array([final_index[n] for n in output_names]),:]
            result=[];offset=0
            for i,size in enumerate(sizes):
                result.append(packed[offset:offset+size,:].T.reshape(batch,function.size2_out(i),function.size1_out(i)).swapaxes(1,2));offset+=size
            return result[0] if len(result)==1 else tuple(result)
        flat=[jnp.ravel(jnp.asarray(v,dtype=jnp.float32).T) for v in values]
        carry=jnp.zeros((1,),dtype=jnp.float32)
        for chunk in chunks:carry=chunk(*flat,carry)
        packed=carry[jnp.array([final_index[n] for n in output_names])]
        result=[];offset=0
        for i,size in enumerate(sizes):
            result.append(packed[offset:offset+size].reshape(function.size2_out(i),function.size1_out(i)).T);offset+=size
        return result[0] if len(result)==1 else tuple(result)
    evaluate.chunks=chunks
    evaluate.max_live=max(map(len,at.values()))
    return evaluate
