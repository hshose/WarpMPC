#!/usr/bin/env python3
"""Matched-input precision, convergence, and backend controls for sensitivity tails."""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from benchmarks.problems import make_linear_mpc
from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales,entry_statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases',default='optimized32,optimized64,baseline32')
    parser.add_argument('--iterations',default='2000,20000')
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--include-long-baseline',action='store_true',help='Also repeat the unoptimized control at longer iteration counts')
    parser.add_argument('--selected-only',action='store_true',help='Use only the worst 20 samples plus 12 evenly spread controls')
    args=parser.parse_args()
    source=args.results_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'metadata.json').read_text())
    with np.load(source/'inputs.npz') as f: inputs={k:f[k] for k in f.files}
    with np.load(source/'reference.npz') as f: ref={k:f[k] for k in f.files}
    with np.load(source/'grid_f2000_r400.npz') as f: old=f['jacobian']
    scales=fd_entry_scales(ref['fd_half'])
    error=abs(old-ref['fd_half'])/scales
    # Worst 20 by error, plus 12 uniformly spread controls. Same selection in all controls.
    selected=np.unique(np.r_[np.argsort(error.max(axis=(1,2)))[-20:],np.linspace(0,len(old)-1,12,dtype=int)])
    problem=make_linear_mpc(meta['horizon'])
    pmat=problem.p_matrix.copy();pmat.data=inputs['p'][0].astype(float)
    amat=problem.a_matrix.copy();amat.data=inputs['a'][0].astype(float)
    iterations=[int(v) for v in args.iterations.split(',')]
    details=dict(original_metadata=meta,matched_inputs='Original float32 data promoted exactly to float64; original full-batch FD normalization scales',selected_cpu_samples=selected.tolist(),cases=args.cases,iterations=iterations)
    (out/'metadata.json').write_text(json.dumps(details,indent=2)+'\n')
    if args.selected_only and not args.cpu_only:
        for key in ('q','l','u'):
            inputs[key] = inputs[key][selected]
        for key in ('x','y','fd_half'):
            ref[key] = ref[key][selected]
        details['evaluated_samples'] = selected.tolist()
        (out/'metadata.json').write_text(json.dumps(details,indent=2)+'\n')
    if args.cpu_only:
        import osqp
        records=[]
        for iters in iterations:
            xall=[];yall=[]
            for index in selected:
                solver=osqp.OSQP()
                solver.setup(pmat,inputs['q'][index].astype(float),amat,inputs['l'][index].astype(float),inputs['u'][index].astype(float),
                             verbose=False,scaling=0,adaptive_rho=False,rho_is_vec=True,check_termination=0,
                             max_iter=iters,polishing=False,warm_starting=False,rho=.1,sigma=1e-6,alpha=1.6)
                result=solver.solve(raise_error=False)
                xall.append(result.x);yall.append(result.y)
                records.append(dict(iterations=iters,sample=int(index),status=result.info.status,
                                    primal_residual=result.info.prim_res,dual_residual=result.info.dual_res,
                                    x_error_inf=float(np.max(abs(result.x-ref['x'][index])))))
            np.savez_compressed(out/f'cpu_f{iters}.npz',samples=selected,x=np.array(xall),y=np.array(yall))
            print(f'CPU {iters}: max primal residual {max(r["primal_residual"] for r in records if r["iterations"]==iters):.5g}',flush=True)
        with (out/'cpu_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        return
    import jax
    import jax.numpy as jnp
    from benchmarks.jax_cache import configure_jax_compilation_cache
    from warpmpc.jax_osqp import (OSQPSettings,build_osqp_plan,compile_osqp,
                         rhs_for_compiled_qdldl,batch_from_compiled_qdldl)
    configure_jax_compilation_cache()
    jax.config.update('jax_enable_x64',True)
    if args.require_gpu and not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU required')
    details.update(devices=[str(d) for d in jax.devices()],jax_version=jax.__version__)
    (out/'metadata.json').write_text(json.dumps(details,indent=2)+'\n')
    rows=[]
    for case in args.cases.split(','):
        dtype=np.float64 if case.endswith('64') else np.float32
        opts={} if case.startswith('baseline') else dict(transpose_work=True,segmented=True,segment_budget=16,segment_strategy='optimal',level_scheduled_solve=True)
        for iters in iterations:
            if case.startswith('baseline') and iters != iterations[0] and not args.include_long_baseline:
                continue
            started=time.time(); print(f'BEGIN {case} forward={iters}',flush=True)
            settings=OSQPSettings(max_iter=iters,scaling=0,adaptive_rho=False,rho_is_vec=True,check_termination=0,warm_starting=False,polishing=False)
            plan=build_osqp_plan(problem.p_matrix,problem.a_matrix,problem.l,problem.u,settings,derivatives=True)
            if case.startswith('free_rho'):
                # A fully unbounded row has zero multiplier at the optimum.
                # Its ADMM penalty can change without changing the QP.
                rho=plan.rho_vec.copy();rho[plan.constr_type<0]=.1
                upper=plan.kkt_upper.copy();upper.data[plan.rho_to_kkt]=-1/rho
                plan=replace(plan,rho_vec=rho,rho_inv_vec=1/rho,kkt_upper=upper)
            solver=compile_osqp(plan,dtype=dtype,derivatives=True,derivative_refinement_iters=400,**opts)
            values=tuple(jnp.asarray(inputs[k],dtype=dtype) for k in ('p','a','q','l','u'))
            action=slice(*meta['action_slice'])
            @jax.jit
            def gradients(p,a,l,u,x,y):
                def one(seed):
                    dx=jnp.zeros_like(x).at[:,action].set(jnp.broadcast_to(seed,(x.shape[0],4)))
                    state=solver.adjoint_derivative_compute(p,a,l,u,x,y,dx,jnp.zeros_like(y))
                    _,dl,du=solver.adjoint_derivative_get_vec(state)
                    return -(dl[:,:12]+du[:,:12])
                return jnp.moveaxis(jax.lax.map(one,jnp.eye(4,dtype=x.dtype)),0,1)
            x,z,y,pr,dr,obj=solver.solve(*values)
            jac=gradients(values[0],values[1],values[3],values[4],x,y)
            x,z,y,pr,dr,jac=[np.asarray(v,dtype=float) for v in (x,z,y,pr,dr,jac)]
            actual_ax=(amat@x.T).T
            violation=np.maximum(np.maximum(inputs['l']-actual_ax,actual_ax-inputs['u']),0).max(axis=1)
            errors=abs(jac-ref['fd_half'])/scales
            np.savez_compressed(out/f'{case}_f{iters}.npz',x=x,z=z,y=y,jacobian=jac,normalized_errors=errors,
                                constraint_violation=violation,primal_residual=pr,dual_residual=dr)
            rows.append(dict(case=case,iterations=iters,samples=len(x),**entry_statistics(errors),violation_max=float(violation.max()),
                             violation_p99=float(np.quantile(violation,.99)),violation_over_1e_3=int((violation>1e-3).sum()),
                             x_reference_difference_max=float(np.max(abs(x-ref['x']))),seconds=time.time()-started))
            # Direct solve residual on the initial ADMM RHS isolates KKT factor/backsolve accuracy.
            lx,dinv=solver.factor(values[0],values[1]);rhs=np.c_[-inputs['q'].astype(float),np.zeros(inputs['l'].shape)]
            solution=solver.qdldl.solve(lx,dinv,rhs_for_compiled_qdldl(solver.qdldl,jnp.asarray(rhs,dtype=dtype)))
            solution=np.asarray(batch_from_compiled_qdldl(solver.qdldl,solution),dtype=float)
            upper=plan.kkt_upper.copy()
            upper.data[plan.p_to_kkt]=inputs['p'][0]*plan.p_scale+plan.p_diag_sigma
            upper.data[plan.a_to_kkt]=inputs['a'][0]*plan.a_scale
            import scipy.sparse as sp
            kkt=upper+upper.T-sp.diags(upper.diagonal())
            kkt_residual=(kkt@solution.T).T-rhs
            rows[-1]['kkt_initial_residual_max']=float(np.max(abs(kkt_residual)))
            from scipy.sparse.linalg import splu, LinearOperator, onenormest
            lu = splu(kkt.tocsc())
            inverse = LinearOperator(kkt.shape, matvec=lu.solve, matmat=lu.solve,
                                     rmatvec=lambda v: lu.solve(v, trans='T'),
                                     rmatmat=lambda v: lu.solve(v, trans='T'))
            rows[-1]['kkt_condition1_estimate'] = float(abs(kkt).sum(axis=0).max()) * float(onenormest(inverse))
            if iters==iterations[0]:
                # Same backward arithmetic at the converged reference, removing forward error.
                exact_jac=np.asarray(gradients(values[0],values[1],values[3],values[4],jnp.asarray(ref['x'],dtype=dtype),jnp.asarray(ref['y'],dtype=dtype)),dtype=float)
                np.savez_compressed(out/f'{case}_converged_forward.npz',jacobian=exact_jac,normalized_errors=abs(exact_jac-ref['fd_half'])/scales)
            with (out/'summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            print(rows[-1],flush=True)
            del solver,gradients
            jax.clear_caches()
    details['status']='complete';(out/'metadata.json').write_text(json.dumps(details,indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
