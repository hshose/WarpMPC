"""Generate the Juggling row using the paper's symbolic counting conventions.

No GPU initialization or numeric factorization is required. The SQP instruction
count is the mean of expanded sparse stage assembly graphs, including the cap
and first-event cost. The separately evaluated first/terminal factors are not
part of that per-interval mean. This is a graph count, not executed GPU FLOPs.
"""
import bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
from factors import Factors
from warpmpc.jax_osqp import build_osqp_plan, OSQPSettings
from warpmpc.jax_qdldl.core import _make_solve_levels


def report(directory):
    f=Factors(directory)
    plan=build_osqp_plan(f.P,f.A,settings=OSQPSettings(rho=f.cfg['rho'],scaling=0))
    q=plan.qdldl_plan
    counts={name:fn.expand().n_instructions() for name,fn in f.functions.items()}
    stages=[counts[name]+counts.get('cap_'+name,0) for name in f.meta['stage_names']]
    # In this profile every interval has a restitution cap; retain generality.
    for k in range(f.cap_nodes,len(stages)):
        stages[k]-=counts.get('cap_'+f.meta['stage_names'][k],0)
    result=dict(variables=f.n,nonlinear_rows=f.m,qp_rows=f.m+f.n,horizon=f.cfg['horizon'],
        interval_seconds=f.meta['stage_steps'],functions=counts,
        stage_instructions_mean=float(np.mean(stages)),stage_instructions_by_interval=stages,
        kkt_nnz=2*f.P.nnz-f.n+2*f.A.nnz+f.A.shape[0],
        factor_instructions=int(np.sum(q.process_mask)),
        independent_levels=len(_make_solve_levels(q.col_rows,q.col_mask)),max_levels=int(q.n),
        sqp_iterations=f.cfg['sqp_iters'],qp_iterations=f.cfg['osqp_iters'],segment_budget=256,
        factor_buffer_gib_60000=q.nnz_l*60000*4/2**30)
    fields=[result[k] for k in ('horizon','variables','qp_rows','kkt_nnz','factor_instructions')]
    fields += [round(result['stage_instructions_mean']),result['sqp_iterations'],result['qp_iterations'],256]
    result['latex_row']='Juggling & '+' & '.join(r'\num{'+str(v)+'}' for v in fields)+f" & {result['independent_levels']}/{result['max_levels']} "+r'\\'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('factors',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=report(a.factors);text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
