"""Check combined pruning against the full correctly weighted mesh problem."""
import json
from pathlib import Path
import numpy as np
from arm.model import Panda
from optimization.mesh import formulation as full
from . import combined_formulation as pruned


def main():
    robot=Panda();cfg=pruned.Config(horizon=24,actuator_margin=.98,codegen=False)
    reference=full.build_problem(full.Config(horizon=24,actuator_margin=.98,codegen=False),robot)
    actual=pruned.build_problem(cfg,robot)
    path=next(Path('runs').glob('*mesh24_wind20'))
    Y=np.load(path/'predictions.npz')['Y'];indices=list(range(full.NP+1))
    for k,rows in enumerate(actual['kept_rows']):indices.extend(full.NP+1+k*reference['ncon_stage']+rows)
    indices.append(len(reference['cl'])-1);indices=np.array(indices);errors=np.zeros(5)
    for y in Y[::20]:
        z=((y-full.OFFSET)/full.SCALE).ravel();p=y[0,:full.NP]
        before=reference['data'](z,p);after=actual['data'](z,p)
        expected=[before[0],before[1],before[2][indices,:],before[3][indices],before[4]]
        errors=np.maximum(errors,[float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(after,expected)])
    assert max(errors)<1e-10
    assert np.array_equal(actual['cl'],reference['cl'][indices])
    assert np.array_equal(actual['ch'],reference['ch'][indices])
    covered=set();begin,end=reference['groups']['endpoint_acceleration']
    for k,rows in enumerate(actual['kept_rows']):covered.update(k+(int(r)-begin)//7 for r in rows if begin<=r<end)
    assert covered==set(range(cfg.horizon+1))
    result=dict(pass_=True,nonlinear_rows=len(actual['cl']),full_mesh_rows=len(reference['cl']),
        endpoint_nodes_covered=sorted(covered),max_errors=dict(zip(['H','gradient','A','constraints','cost'],errors.tolist())),
        mesh_steps=actual['mesh_steps'].tolist(),mesh_times=actual['mesh_times'].tolist(),
        mesh_growth=actual['mesh_growth'],source_predictions=str(path/'predictions.npz'))
    Path('optimization/pruning/combined_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
