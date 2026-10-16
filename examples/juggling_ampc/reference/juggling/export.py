"""Export independent stage factors and assembled sparse SQP data as C99."""
import argparse
from dataclasses import asdict
import json
import os
import math
from pathlib import Path
import casadi as ca
from .formulation import Config,SCALE,NX,NZ,build_problem


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True,help='Experiment config.json')
    parser.add_argument('--output',type=Path,default=Path('build/export'))
    args=parser.parse_args()
    config=json.loads(args.config.read_text());cfg=Config(**config.get('mpc',config))
    problem=build_problem(cfg)
    directory=args.output.resolve();directory.mkdir(parents=True,exist_ok=True)
    previous=Path.cwd()
    try:
        os.chdir(directory)
        generator=ca.CodeGenerator('juggling_problem.c',{'with_header':True})
        for name in ['first','stage','last','diagnostics','data','value']:
            generator.add(problem[name])
        generator.generate()
    finally:os.chdir(previous)
    manifest=dict(config=asdict(cfg),nx=NX,nz=NZ,scale=SCALE.tolist(),
        constraint_lower=[float(v) if math.isfinite(v) else None for v in problem['cl']],
        constraint_upper=[float(v) if math.isfinite(v) else None for v in problem['ch']],
        unbounded_encoding='null lower/upper bounds mean negative/positive infinity',
        note='CasADi C API outputs sparse CSC data; sparsity functions are included in the C header.')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(directory/'juggling_problem.c')

if __name__=='__main__':main()
