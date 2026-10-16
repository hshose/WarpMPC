"""Recorded event-remapping regression, including guarded-rollout fallback.

No hardware connection. A rising ball deliberately leaves the optimizer's
workspace on the first rollout stage, exposing the saved shifted seed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from arm import spin_formulation as formulation
from arm.model import Panda, HOME
from cpp.python_controller import NativeController
from optimization.mesh.formulation import Config, grid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('release',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--native-only',action='store_true')
    mode.add_argument('--python-only',action='store_true')
    args=parser.parse_args()
    factors=args.release/'factors'
    meta=json.loads((args.metadata or factors/'metadata.json').read_text());spec=meta['spec']
    cfg=Config(**meta['config'],mesh_horizon=spec.get('mesh_horizon',1.2))
    robot=Panda(assembly=spec['assembly'],**spec['geometry'])
    native=None if args.python_only else NativeController(args.release/'build/controller/libjuggling_controller.so',factors)
    if not args.native_only:
        from arm.solver_state_shift import StateShiftSolver
        from arm.spin_simulator import SpinSimulator
        from arm.simulator import SimConfig
        from optimization.mesh.solver import MeshSolver
    x=np.r_[[0.,0.,1.49],[0.,0.,10.],HOME,np.zeros(10)]
    tau=np.asarray(robot.bias(HOME,np.zeros(7))).ravel()
    steps,times,growth=grid(cfg)
    original=np.tile(np.r_[x,tau,0.,.5,cfg.h,0.],(cfg.horizon+1,1))
    original[:,32]=np.r_[steps,steps[-1]];original[:,33]=times
    dual=np.zeros(native.n+native.m) if native is not None else None
    # Source event parameters from moving_mpc_20261005_194112, line1947.
    dt=.019976553972810507
    fractions=(.6838929363961674,.5406645964351932)
    cases=[(6.592706501549179,-.00006761002427670745),
           (6.592706501549179,0.),(.00001,6.5),(0.,0.)]
    results=[]
    try:
        for impulses in cases:
            old=original.copy();old[:,30]=0.;old[:,31]=.5
            old[:2,30]=impulses;old[:2,31]=fractions
            z=((old-formulation.OFFSET)/formulation.SCALE).ravel()
            dominant=int(np.argmax(np.abs(impulses)))
            expected=(times[dominant]+fractions[dominant]*steps[dominant]-dt)/steps[0] if any(impulses) else .5
            expected_uniform=(dominant*cfg.h+fractions[dominant]*cfg.h-dt)/cfg.h if any(impulses) else .5
            checks=[]
            if native is not None:
                native.set_warmstart(z,dual);native.shift(dt,x,tau)
                native_y=native.warmstart()[0].reshape(-1,formulation.NZ)*formulation.SCALE+formulation.OFFSET
                checks.append(('native',native_y,expected))
            if args.native_only:
                for name,y,want in checks:
                    assert abs(y[0,30]-sum(impulses))<1e-12,(name,y[0,30],impulses)
                    assert abs(y[0,31]-want)<1e-12,(name,y[0,31],want)
                results.append(dict(source_impulses=impulses,expected_fraction=expected,native_fraction=float(native_y[0,31])))
                continue
            python=MeshSolver.__new__(MeshSolver)
            python.cfg=cfg;python.robot=robot;python.steps=steps;python.times=times;python.growth=growth
            python.z=z.copy();python.dual=None;python.guard_ball_rollout=True
            python.nominal=SpinSimulator(robot,SimConfig(restitution=cfg.restitution,drag=cfg.drag,
                ball_mass=cfg.mass,radius=cfg.radius,paddle_radius=cfg.paddle_radius,max_interval=.003),
                friction=cfg.friction,inertia_ratio=cfg.inertia_ratio)
            python.shift(dt,x,tau)
            assert python.warmstart_fallback['stage']==0,python.warmstart_fallback
            python_y=python.z.reshape(-1,formulation.NZ)*formulation.SCALE+formulation.OFFSET
            uniform=StateShiftSolver.__new__(StateShiftSolver)
            uniform.cfg=cfg;uniform.robot=robot;uniform.formulation=formulation;uniform.z=z.copy();uniform.dual=None
            uniform.shift(dt,x,tau)
            uniform_y=uniform.z.reshape(-1,formulation.NZ)*formulation.SCALE+formulation.OFFSET
            checks.extend([('mesh',python_y,expected),('uniform',uniform_y,expected_uniform)])
            for name,y,want in checks:
                assert abs(y[0,30]-sum(impulses))<1e-12,(name,y[0,30],impulses)
                assert abs(y[0,31]-want)<1e-12,(name,y[0,31],want)
            results.append(dict(source_impulses=impulses,expected_fraction=expected,
                native_fraction=float(native_y[0,31]) if native is not None else None,mesh_fraction=float(python_y[0,31]),
                uniform_fraction=float(uniform_y[0,31]),fallback=python.warmstart_fallback))
    finally:
        if native is not None:native.close()
    args.output.write_text(json.dumps(dict(passed=True,cases=results),indent=2)+'\n')
    print(args.output.read_text())


if __name__=='__main__':main()
