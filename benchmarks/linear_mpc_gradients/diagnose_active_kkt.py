#!/usr/bin/env python3
"""Independent active-set KKT sensitivities and small-step FD diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
import os
from pathlib import Path
import sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from benchmarks.problems import make_linear_mpc
from benchmarks.linear_mpc_gradients.benchmark_sensitivity_accuracy import reference_one
from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales,entry_statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-dir',type=Path,required=True);parser.add_argument('--precision-dir',type=Path,required=True);args=parser.parse_args()
    source=args.results_dir;out=args.precision_dir.parent/'active_kkt_diagnostics';out.mkdir(exist_ok=True)
    meta=json.loads((source/'metadata.json').read_text())
    with np.load(source/'inputs.npz') as f:inp={k:f[k] for k in f.files}
    with np.load(source/'reference.npz') as f:ref={k:f[k] for k in f.files}
    with np.load(source/'active_set_analysis/active_sets.npz') as f:changed=f['changed']
    with np.load(args.precision_dir/'optimized64_converged_forward.npz') as f:ad=f['jacobian']
    p=make_linear_mpc(meta['horizon']);p.p_matrix.data=inp['p'][0].astype(float);p.a_matrix.data=inp['a'][0].astype(float)
    n=p.p_matrix.shape[0];scales=fd_entry_scales(ref['fd_half']);jac=[];residuals=[]
    for b in range(len(ref['x'])):
        ax=p.a_matrix@ref['x'][b];l=inp['l'][b];u=inp['u'][b]
        active=(l==u)|(abs(ax-l)<=1e-7)|(abs(ax-u)<=1e-7)
        indices=np.flatnonzero(active);a=p.a_matrix[active].tocsc()
        k=sp.bmat([[p.p_matrix,a.T],[a,None]],format='csc');rhs=np.zeros((k.shape[0],12))
        for i in range(12):rhs[n+np.flatnonzero(indices==i)[0],i]=-1
        sol=splu(k).solve(rhs);jac.append(sol[slice(*meta['action_slice'])]);residuals.append(np.max(abs(k@sol-rhs)))
    jac=np.array(jac);np.savez_compressed(out/'active_kkt.npz',jacobian=jac,residuals=residuals,changed=changed)
    report={}
    for name,v in [('vs_original_fd',abs(jac-ref['fd_half'])/scales),('adjoint400_vs_active_kkt',abs(ad-jac)/scales)]:
        report[name]={group:entry_statistics(v[mask]) for group,mask in [('all',np.ones(len(changed),bool)),('stable',~changed),('changed',changed)]}
    err=abs(ad-ref['fd_half'])/scales
    stable_ids=np.flatnonzero(~changed);changed_ids=np.flatnonzero(changed)
    selected=np.unique(np.r_[stable_ids[np.argsort(err[~changed].max(axis=(1,2)))[-6:]],changed_ids[np.argsort(err[changed].max(axis=(1,2)))[-6:]]])
    tasks=[]
    for step in [2e-5,2e-6]:
        for b in selected:
            tasks.append((int(b),meta['horizon'],meta['dtype'],inp['q'][b].astype(float),inp['l'][b].astype(float),inp['u'][b].astype(float),
                          inp['dx'].astype(float),inp['dy'].astype(float),step,1e-10,100000))
    records=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for task,result in zip(tasks,pool.map(reference_one,tasks)):
            b=result[0];small_fd=result[4];step=task[-3]/2
            records.append(dict(sample=int(b),step=step,original_active_set_changed=bool(changed[b]),
                                active_kkt_vs_fd_max=float((abs(jac[b]-small_fd)/scales).max()),
                                adjoint400_vs_fd_max=float((abs(ad[b]-small_fd)/scales).max()),
                                original_fd_vs_small_fd_max=float((abs(ref['fd_half'][b]-small_fd)/scales).max())))
    with (out/'small_step_fd.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
