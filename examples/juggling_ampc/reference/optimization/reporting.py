"""Use the shared diagnostics with the actual nonuniform-grid caption."""
from pathlib import Path
import numpy as np


def report(directory,a,p,config,metrics,robot):
    variant=config.get('native_variant',{})
    steps=variant.get('stage_steps')
    if steps is None:
        from arm.reporting import report as original
        return original(directory,a,p,config,metrics,robot)
    source=Path('arm/reporting.py').read_text().replace('from .model import','from arm.model import')
    old='f"MPC {1000*config[\'mpc\'][\'control_dt\']:g} ms / nodes {1000*config[\'mpc\'][\'h\']:g} ms\\n"'
    caption=f'MPC {1000*config["mpc"]["control_dt"]:g} ms / {len(steps)} intervals, {1000*steps[0]:.1f}–{1000*steps[-1]:.1f} ms / horizon {sum(steps):.3f} s\n'
    if old not in source:raise RuntimeError('Shared report caption changed; update variant caption adapter')
    source=source.replace(old,repr(caption));env={};exec(compile(source,'arm/reporting.py','exec'),env)
    env['report'](directory,a,p,config,metrics,robot)
