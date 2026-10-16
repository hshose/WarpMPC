"""Check sparse-row removal against the unchanged physical formulation."""
import json
from pathlib import Path
import numpy as np
from arm.model import Panda
from arm import spin_formulation as baseline
from . import formulation as pruned


def main():
    robot=Panda();base=baseline.build_problem(baseline.Config(actuator_margin=.98,codegen=False),robot)
    trajectory=np.load('runs/20260924T175306.001124Z_cpp_final_wind/predictions.npz')['Y']
    cases=[]
    for dedup,endpoints in [(False,False),(True,False),(False,True)]:
        cfg=pruned.Config(actuator_margin=.98,codegen=False,deduplicate_endpoints=dedup)
        if endpoints:cfg.pruning_groups+=('endpoint_acceleration',);cfg.pruning_prefix=8
        trial=pruned.build_problem(cfg,robot);indices=list(range(baseline.NP+1))
        for k,rows in enumerate(trial['kept_rows']):indices.extend(baseline.NP+1+k*base['ncon_stage']+rows)
        indices.append(len(base['cl'])-1);indices=np.array(indices)
        errors=np.zeros(5)
        for Y in trajectory[::20]:
            z=((Y-baseline.OFFSET)/baseline.SCALE).ravel();p=Y[0,:baseline.NP]
            reference=base['data'](z,p);actual=trial['data'](z,p)
            expected=[reference[0],reference[1],reference[2][indices,:],reference[3][indices],reference[4]]
            errors=np.maximum(errors,[float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(actual,expected)])
        assert max(errors)<1e-10,errors
        assert np.array_equal(trial['cl'],base['cl'][indices])
        assert np.array_equal(trial['ch'],base['ch'][indices])
        covered=set()
        for k,rows in enumerate(trial['kept_rows']):
            begin,end=base['groups']['endpoint_acceleration']
            covered.update(k+(int(r)-begin)//7 for r in rows if begin<=r<end)
        if not endpoints:assert covered==set(range(cfg.horizon+1)),covered
        cases.append(dict(deduplicate_endpoints=dedup,drop_endpoints=endpoints,
            nonlinear_rows=len(trial['cl']),baseline_rows=len(base['cl']),
            max_errors=dict(zip(['H','gradient','A','constraints','cost'],errors.tolist())),
            endpoint_nodes_covered=sorted(covered),pass_=True))
    Path('optimization/pruning/checks.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
