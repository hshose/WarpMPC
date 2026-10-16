"""Compile one large factor independently to populate Warp's shared kernel cache."""
import bootstrap
import argparse,time
from pathlib import Path
from factors import Factors
from warpmpc.numpysadi import convert
import warp as wp
p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('name',choices=['stage','reduced','tail']);a=p.parse_args()
f=Factors(a.factors);start=time.monotonic();function=convert(f.functions[a.name].expand(),backend='warp');print('Generated',a.name,time.monotonic()-start,flush=True)
wp.init();function.__globals__['_juggling_'+a.name+'_warp_kernel'].module.load('cuda:0');print('Compiled',a.name,time.monotonic()-start,flush=True)
