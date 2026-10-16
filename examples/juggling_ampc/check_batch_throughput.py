"""Measure the full optimized MPC at a requested batch size, without a plant."""
import bootstrap
import argparse, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from mpc import MPC
from plan_parity import metrics


def main(a):
    fixtures = np.load(a.directory / 'native_fixtures.npz')
    index = np.arange(a.batch) % len(fixtures['z'])
    values = [np.asarray(fixtures[k][index], np.float32) for k in ('z', 'dual', 'p', 'projector')]
    mpc = MPC(a.directory / 'gpu_factors')
    times = []
    for _ in range(a.repeat + 1):
        start = time.monotonic()
        plan = mpc.solve(*values)[2]
        plan.block_until_ready()
        times.append(time.monotonic() - start)
        print(json.dumps(dict(batch=a.batch, elapsed_s=times[-1])), flush=True)
    result = metrics(mpc.factors, np.asarray(plan), fixtures['plan'][index])
    result.update(batch=a.batch, initial_s=times[0], repeat_s=times[1:])
    if a.profile:
        z,dual,p,projector=map(jnp.asarray,values)
        components={}
        for name,call in [('assemble',lambda:mpc.assemble(z,p,projector)),('merit',lambda:mpc.merit_factors(z,p,projector))]:
            jax.block_until_ready(call());measure=[]
            for _ in range(3):
                start=time.monotonic();jax.block_until_ready(call());measure.append(time.monotonic()-start)
            components[name]=measure
        h,g,matrix,c,cost=mpc.assemble(z,p,projector)
        lower=jnp.broadcast_to(jnp.r_[jnp.asarray(mpc.factors.arrays['constraint_lower']),jnp.asarray(mpc.factors.arrays['all_lower'])],dual.shape)
        upper=jnp.broadcast_to(jnp.r_[jnp.asarray(mpc.factors.arrays['constraint_upper']),jnp.asarray(mpc.factors.arrays['all_upper'])],dual.shape)
        # Representative same-pattern QP for timing, not an extra parity fixture.
        call=lambda:mpc.qp.solve(h,matrix,g,lower,upper,z,dual)
        jax.block_until_ready(call());measure=[]
        for _ in range(3):
            start=time.monotonic();jax.block_until_ready(call());measure.append(time.monotonic()-start)
        components['qp']=measure;result['components_s']=components
        print(json.dumps(components),flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise RuntimeError('Batched native MPC parity failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
