#!/usr/bin/env python3
"""CPU sparse-LU reproduction of the adjoint's regularized refinement equations."""
import argparse
import csv
import os
from pathlib import Path
import sys
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from benchmarks.problems import make_linear_mpc
from warpmpc.jax_osqp import OSQPSettings,build_osqp_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-root',type=Path,required=True);args=parser.parse_args()
    root=args.results_root;source=root/'u0-jacobian';out=root/'adjoint_refinement_diagnostics';out.mkdir(exist_ok=True)
    with np.load(source/'inputs.npz') as z:inp={k:z[k] for k in z.files}
    with np.load(source/'reference.npz') as z:ref={k:z[k] for k in z.files}
    with np.load(root/'active_kkt_diagnostics/active_kkt.npz') as z:exact=z['jacobian'];changed=z['changed']
    with np.load(root/'precision_diagnostics/optimized64_converged_forward.npz') as z:ad=z['jacobian']
    scales=abs(ref['fd_half']).max(axis=0);err=(abs(ad-exact)/scales).max(axis=(1,2))
    ids=np.unique(np.r_[np.flatnonzero(changed)[np.argsort(err[changed])[-4:]],np.flatnonzero(~changed)[np.argsort(err[~changed])[-4:]],337,845,958])
    problem=make_linear_mpc(40);plan=build_osqp_plan(problem.p_matrix,problem.a_matrix,problem.l,problem.u,OSQPSettings(rho_is_vec=True),derivatives=True);d=plan.derivative_plan
    results=[]
    for b in ids:
        upper=d.adjoint_upper.copy();v=upper.data;x=ref['x'][b];y=ref['y'][b]
        g=inp['a'][0,d.g_a_value_map].astype(float)*d.g_sign
        aeq=inp['a'][0,d.aeq_a_value_map].astype(float)
        gx=np.zeros(d.n_ineq);np.add.at(gx,d.g_rows,g*x[d.g_cols])
        h=np.r_[-inp['l'][b,d.lower_indices],inp['u'][b,d.upper_indices]]
        lam=np.r_[np.maximum(-y[d.lower_indices],0),np.maximum(y[d.upper_indices],0)]
        v[d.diag_first_pos]=1;v[d.diag_second_pos]=0
        v[d.p_block_pos]=inp['p'][0,d.p_full_value_map]
        v[d.g_block_pos]=g;v[d.aeq_block_pos]=aeq;v[d.glambda_block_pos]=lam[d.g_rows]*g
        v[d.slack_diag_pos]=gx-h;v[d.aeq_t_block_pos]=aeq
        matrix=(upper+upper.T-sp.diags(upper.diagonal())).tocsc()
        rhs=np.zeros((d.adjoint_dim,4));action_start=41*12
        rhs[action_start:action_start+4]=-np.eye(4)
        for eps,steps in [(1e-6,4000),(1e-8,400),(1e-10,400)]:
            regularized=matrix+sp.diags(np.r_[np.full(d.half_dim,eps),np.full(d.half_dim,-eps)])
            lu=splu(regularized.tocsc());sol=lu.solve(rhs)
            for k in range(1,steps+1):
                sol-=lu.solve(matrix@sol-rhs)
                if k not in (400,4000):continue
                start=d.half_dim+plan.n+d.n_ineq
                jac=sol[start:start+12].T
                results.append(dict(sample=int(b),active_set_changed=bool(changed[b]),eps=eps,refinements=k,
                                    error_vs_active_kkt_max=float((abs(jac-exact[b])/scales).max()),
                                    error_vs_fd_max=float((abs(jac-ref['fd_half'][b])/scales).max()),
                                    difference_from_gpu_refine400_max=float((abs(jac-ad[b])/scales).max()),
                                    adjoint_residual_max=float(abs(matrix@sol-rhs).max())))
        print('sample',b,results[-1],flush=True)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)


if __name__=='__main__':main()
