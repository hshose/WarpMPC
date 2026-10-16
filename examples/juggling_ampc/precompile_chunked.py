"""Populate the optimized Warp cache one bounded factor at a time."""
import bootstrap
import argparse,time
from pathlib import Path
from factors import Factors
from chunked_factors import convert_chunked
import warp as wp
p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('name',choices=['stage','reduced','tail']);a=p.parse_args()
f=Factors(a.factors);t=time.monotonic();fn=convert_chunked(f.functions[a.name].expand())
print('Generated',a.name,'chunks',len(fn.chunks),'max_live',fn.max_live,'seconds',time.monotonic()-t,flush=True)
wp.init()
for i,chunk in enumerate(fn.chunks):
    chunk.__globals__['_kernel'].module.load('cuda:0')
    print('Ready',a.name,i,'seconds',time.monotonic()-t,flush=True)
