"""Isolated Newton runner for nonuniform-grid conservative pruning."""
import argparse,json,shutil,sys
from pathlib import Path
from optimization.mesh.formulation import grid
from optimization.mesh.reporting import report
from .combined_formulation import Config
from .combined_solver import CombinedSolver


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',type=int,default=5)
    parser.add_argument('--accel-weight',type=float,default=.4);parser.add_argument('--rate-weight',type=float,default=.15)
    options,args=parser.parse_known_args()
    if args and args[0]=='--':args=args[1:]
    if '--horizon' not in args:args+=['--horizon','24']
    run_name=args[args.index('--name')+1] if '--name' in args else 'newton50'
    before=set(Path('runs').glob('*'));config=None
    import newton_sim.run_experiment as runner
    def configure(**kw):
        nonlocal config
        config=Config(**kw,pruning_prefix=options.prefix,accel_weight=options.accel_weight,rate_weight=options.rate_weight)
        return config
    runner.SpinConfig=configure;runner.SpinSolver=CombinedSolver
    runner.report=report
    original_open=Path.open
    def redirected_open(self,*a,**kw):
        if self==Path('EXPERIMENTS.md'):self=Path('optimization/pruning/EXPERIMENTS.md')
        return original_open(self,*a,**kw)
    Path.open=redirected_open;sys.argv=['optimization.pruning.combined_run',*args]
    try:runner.main()
    finally:
        Path.open=original_open
        for directory in set(Path('runs').glob('*'))-before:
            if not directory.name.endswith('_'+run_name):continue
            for source in ['pruning','mesh']:
                shutil.copytree('optimization/'+source,directory/'source/optimization'/source,
                    ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
            steps,times,growth=grid(config)
            (directory/'pruning.json').write_text(json.dumps(dict(variant='mesh_dedup',prefix_nodes=options.prefix,
                prefix_seconds=float(times[options.prefix]),dropped_tail_groups=config.pruning_groups,
                deduplicate_endpoints=True,accel_weight=options.accel_weight,rate_weight=options.rate_weight,
                mesh_steps=steps.tolist(),mesh_times=times.tolist(),mesh_growth=float(growth),
                note='All node endpoint accelerations, contact/dynamics, simple bounds, interval acceleration and torque rate retained.'),indent=2)+'\n')


if __name__=='__main__':main()
