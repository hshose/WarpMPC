"""Run pruning candidates without changing the baseline Newton/controller files.

Example: OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.run
    --variant inactive --prefix 5 -- --name prun_inactive5 --seconds 5
"""
import argparse,json,shutil,sys
from pathlib import Path
from .formulation import Config
from .solver import PruningSolver


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['inactive','dedup','drop_endpoints'],default='inactive')
    parser.add_argument('--prefix',type=int,default=5)
    parser.add_argument('--accel-weight',type=float,default=.4);parser.add_argument('--rate-weight',type=float,default=.15)
    options,args=parser.parse_known_args()
    if args and args[0]=='--':args=args[1:]
    groups=('workspace','tilt','upward','paddle_velocity','jerk')
    if options.variant=='drop_endpoints':groups+=('endpoint_acceleration',)
    run_name=args[args.index('--name')+1] if '--name' in args else 'newton50'
    before=set(Path('runs').glob('*'))
    import newton_sim.run_experiment as runner
    runner.SpinConfig=lambda **kw:Config(**kw,pruning_prefix=options.prefix,pruning_groups=groups,
        deduplicate_endpoints=options.variant=='dedup',accel_weight=options.accel_weight,rate_weight=options.rate_weight)
    runner.SpinSolver=PruningSolver
    # The baseline runner appends a global experiment index; keep this agent's
    # evidence local rather than interleave writes to a shared document.
    original_open=Path.open
    def redirected_open(self,*a,**kw):
        if self==Path('EXPERIMENTS.md'):self=Path('optimization/pruning/EXPERIMENTS.md')
        return original_open(self,*a,**kw)
    Path.open=redirected_open
    sys.argv=['optimization.pruning.run',*args]
    try:runner.main()
    finally:
        Path.open=original_open
        for directory in set(Path('runs').glob('*'))-before:
            if not directory.name.endswith('_'+run_name):continue
            shutil.copytree('optimization/pruning',directory/'source/optimization/pruning',
                ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
            (directory/'pruning.json').write_text(json.dumps(dict(variant=options.variant,prefix_nodes=options.prefix,
                prefix_seconds=options.prefix*.03,dropped_tail_groups=groups,deduplicate_endpoints=options.variant=='dedup',
                accel_weight=options.accel_weight,rate_weight=options.rate_weight,
                note='All contact/dynamics constraints, simple bounds, interval acceleration and torque rate retained.'),indent=2)+'\n')


if __name__=='__main__':main()
