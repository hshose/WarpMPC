"""Run the shared Newton experiment with an isolated compiled candidate."""
import argparse,json,os,sys
from pathlib import Path
from arm.spin_formulation import Config as BaseConfig
from cpp.python_controller import CppSolver


def config(**kwargs):
    metadata=json.loads((Path(os.environ['JUGGLING_VARIANT_FACTORS'])/'metadata.json').read_text())
    return BaseConfig(**metadata['config'])


class NativeVariantSolver(CppSolver):
    def __init__(self,cfg,robot):
        factors=os.environ['JUGGLING_VARIANT_FACTORS'];library=os.environ['JUGGLING_VARIANT_LIBRARY']
        metadata=json.loads((Path(factors)/'metadata.json').read_text())
        # Model coefficients come from the compiled candidate, independently of
        # optional physical simulator mismatch settings.
        for key,value in metadata['config'].items():setattr(cfg,key,value)
        super().__init__(cfg,robot,library=library,factors=factors)
        from optimization import native_export
        self.formulation=native_export


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--factors',required=True);p.add_argument('--library',required=True)
    args,rest=p.parse_known_args();factors=Path(args.factors).resolve();library=Path(args.library).resolve()
    os.environ['JUGGLING_VARIANT_FACTORS']=str(factors);os.environ['JUGGLING_VARIANT_LIBRARY']=str(library)
    meta=json.loads((factors/'metadata.json').read_text())
    sys.argv=[sys.argv[0],*rest,'--controller','cpp','--horizon',str(meta['config']['horizon']),'--h',str(meta['config']['h'])]
    path=Path('newton_sim/run_experiment.py');source=path.read_text()
    source=source.replace('scene=Scene(sc,directory)', f"sc.assembly={meta['spec'].get('assembly')!r};sc.geometry={meta['spec'].get('geometry',{})!r};scene=Scene(sc,directory)")
    source=source.replace('from arm.spin_formulation import Config as SpinConfig','from optimization.run_native import config as SpinConfig')
    source=source.replace('from cpp.python_controller import CppSolver','from optimization.run_native import NativeVariantSolver as CppSolver')
    source=source.replace('from arm.reporting import report','from optimization.reporting import report')
    source=source.replace("['arm','newton_sim','juggling','estimation','cpp']","['arm','newton_sim','juggling','estimation','cpp','optimization']")
    source=source.replace("paths=[Path('build/controller/libjuggling_controller.so'),Path('build/cpp_factors/libjuggling_factors.so')]",f'paths=[Path({str(library)!r}),*sorted(Path({str(factors)!r}).glob("*.so"))]')
    source=source.replace("shutil.copyfile('build/cpp_factors/metadata.json',directory/'native/factor_metadata.json')",f"shutil.copyfile({str(factors/'metadata.json')!r},directory/'native/factor_metadata.json')\n        config['native_variant']=json.loads(Path({str(factors/'metadata.json')!r}).read_text())")
    env={'__name__':'optimization._native_runner','__package__':'newton_sim','__file__':str(path)}
    exec(compile(source,str(path),'exec'),env);env['main']()


if __name__=='__main__':main()
