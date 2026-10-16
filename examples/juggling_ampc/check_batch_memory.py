"""Compile the requested batch shape and report XLA memory without allocating it."""
import bootstrap
import argparse,json,time
from pathlib import Path
import jax
import numpy as np
from mpc import MPC
p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('--batch',type=int,default=60000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
t=time.monotonic();m=MPC(a.factors);f=m.factors
shapes=[(a.batch,f.n),(a.batch,f.m+f.n),(a.batch,40),(a.batch,49)]
compiled=m.solve.lower(*[jax.ShapeDtypeStruct(s,np.float32) for s in shapes]).compile()
stats=compiled.memory_analysis();fields=['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']
r={k:int(getattr(stats,k)) for k in fields};r['mpc_bytes']=r['argument_size_in_bytes']+r['output_size_in_bytes']+r['temp_size_in_bytes']-r['alias_size_in_bytes'];r.update(batch=a.batch,elapsed_s=time.monotonic()-t,device=str(jax.devices()[0]),excludes='Newton plant, observer host memory, compilation cache and allocator overhead')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
