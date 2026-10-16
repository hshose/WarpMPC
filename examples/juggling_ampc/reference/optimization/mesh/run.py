"""Run the unchanged Newton loop with an isolated mesh controller.

Only controller imports, source-archiving list and experiment-log destination
are substituted. Newton, observer, torque interpolation and scoring are shared.
"""
from pathlib import Path

def main():
    path=Path('newton_sim/run_experiment.py')
    source=path.read_text().replace('from arm.solver_spin import SpinSolver','from optimization.mesh.solver import MeshSolver as SpinSolver')
    source=source.replace('from arm.reporting import report','from optimization.mesh.reporting import report')
    source=source.replace('from arm.spin_formulation import Config as SpinConfig','from optimization.mesh.formulation import Config as SpinConfig')
    source=source.replace("['arm','newton_sim','juggling','estimation','cpp']", "['arm','newton_sim','juggling','estimation','cpp','optimization']")
    source=source.replace("with Path('EXPERIMENTS.md').open('a')", "with Path('optimization/mesh/EXPERIMENTS.md').open('a')")
    env={'__name__':'optimization.mesh._runner','__package__':'newton_sim','__file__':str(path)}
    exec(compile(source,str(path),'exec'),env)
    env['main']()

if __name__=='__main__':main()
