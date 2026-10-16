#!/usr/bin/env python3
"""Re-solve finite-difference stencils and audit changes in active inequalities."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmarks.problems import make_linear_mpc
from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales, entry_statistics

TOLERANCES = np.array([1e-6, 1e-7, 1e-8])


def active_sets(ax, lower, upper, y):
    inequality = lower != upper
    # Keep lower and upper sides distinct; exclude invariant equality rows.
    lower_slack = np.abs(ax - lower)
    upper_slack = np.abs(upper - ax)
    geometric = np.stack([np.where(inequality & (lower_slack <= tol), -1,
                          np.where(inequality & (upper_slack <= tol), 1, 0))
                          for tol in TOLERANCES]).astype(np.int8)
    dual = np.stack([np.where(inequality & (y < -tol), -1,
                     np.where(inequality & (y > tol), 1, 0))
                     for tol in TOLERANCES]).astype(np.int8)
    return geometric, dual


def worker(task):
    import osqp
    index, meta, q, lower, upper = task
    p = make_linear_mpc(meta['horizon'])
    p.p_matrix.data = p.p_matrix.data.astype(meta['dtype']).astype(float)
    p.a_matrix.data = p.a_matrix.data.astype(meta['dtype']).astype(float)
    solver = osqp.OSQP()
    solver.setup(p.p_matrix, q, p.a_matrix, lower, upper, verbose=False,
                 scaling=0, adaptive_rho=False, rho_is_vec=True, check_termination=25,
                 max_iter=meta['reference_max_iter'], polishing=False, warm_starting=False,
                 rho=.1, sigma=1e-6, alpha=1.6,
                 eps_abs=meta['reference_tolerance'], eps_rel=meta['reference_tolerance'])
    residuals = []
    def solve(l, u):
        solver.update(l=l, u=u)
        result = solver.solve(raise_error=False)
        if result.info.status_val != 1:
            raise ValueError(f'Sample {index}: {result.info.status}')
        residuals.append([result.info.prim_res, result.info.dual_res])
        return result.x, *active_sets(p.a_matrix @ result.x, l, u, result.y)
    x, base, dual_base = solve(lower, upper)
    counts = np.zeros((2, 12, 2, 3), dtype=int)
    dual_counts = np.zeros_like(counts)
    fd = np.zeros((2, 4, 12))
    action = slice(*meta['action_slice'])
    for step_index, step in enumerate([meta['fd_step'], meta['fd_step']/2]):
        for component in range(12):
            solutions = []
            for sign_index, sign in enumerate([1, -1]):
                l, u = lower.copy(), upper.copy()
                l[component] -= sign * step
                u[component] -= sign * step
                xp, active, dual_active = solve(l, u)
                solutions.append(xp[action].copy())
                counts[step_index, component, sign_index] = np.count_nonzero(active != base, axis=1)
                dual_counts[step_index, component, sign_index] = np.count_nonzero(dual_active != dual_base, axis=1)
            fd[step_index, :, component] = (solutions[0]-solutions[1])/(2*step)
    return index, counts, dual_counts, base, dual_base, fd, np.max(residuals, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--reference-only', action='store_true', help='Audit the FD stencil without requiring GPU grid artifacts')
    parser.add_argument('--reuse-audit', action='store_true', help='Recompute diagnostics from an existing active_sets.npz without reference re-solves')
    args = parser.parse_args()
    p = args.results_dir
    meta = json.loads((p/'metadata.json').read_text())
    if meta['status'] != 'complete' or meta['objective'] != 'u0-jacobian':
        raise ValueError('Expected complete action Jacobian study')
    out = p/'active_set_analysis'
    out.mkdir(exist_ok=True)
    with np.load(p/'inputs.npz') as archive:
        inputs = {key: archive[key] for key in ('q', 'l', 'u', 'p', 'a')}
    with np.load(p/'reference.npz') as archive:
        reference = {key: archive[key] for key in ('fd_half', 'x')}
    n = meta['batch_size']
    tasks = [(i, meta, inputs['q'][i].astype(float), inputs['l'][i].astype(float), inputs['u'][i].astype(float)) for i in range(n)]
    start = time.time()
    if args.reuse_audit:
        audit = np.load(out/'active_sets.npz')
        if not np.array_equal(audit['reference_fd_half'], reference['fd_half']):
            raise ValueError('Cached audit does not match reference')
        counts, dual_counts, base, dual_base, fd, residuals = [audit[key] for key in
            ('changed_constraint_counts', 'dual_changed_constraint_counts', 'base_active', 'base_dual_active', 'fd', 'residuals')]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for count, result in enumerate(pool.map(worker, tasks), 1):
                results.append(result)
                if count % 50 == 0:
                    print(f'Active-set audit {count}/{n}: {time.time()-start:.1f}s', flush=True)
        counts, dual_counts, base, dual_base, fd, residuals = [np.stack([r[k] for r in results]) for k in range(1,7)]
    # Main stencil is h/2, the exact reference used in the plotted errors.
    # Keep the same sample population across all GPU iteration settings.
    changed = np.any(counts[:,1,:,:,1] != 0, axis=(1,2))
    np.savez_compressed(out/'active_sets.npz', changed=changed, changed_constraint_counts=counts,
                        dual_changed_constraint_counts=dual_counts, base_active=base,
                        base_dual_active=dual_base, tolerances=TOLERANCES, fd=fd, residuals=residuals,
                        reference_fd_half=reference['fd_half'])
    scales = fd_entry_scales(reference['fd_half'])
    report = dict(definition='Exclude a whole QP if any inequality lower/upper activity differs from the base solution at any of the 24 +/- h/2 coordinate perturbations',
                  activity='absolute primal distance to a finite inequality bound <= 1e-7; equality constraints excluded',
                  total_samples=n, excluded_samples=int(changed.sum()), excluded_percent=float(100*changed.mean()),
                  normalization='original full-batch FD entry maxima, unchanged after filtering',
                  fd_reproduction_max_abs=float(np.max(abs(fd[:,1]-reference['fd_half']))),
                  elapsed_seconds=time.time()-start, tolerance_audit=[], tail_association=[])
    for k, tol in enumerate(TOLERANCES):
        for step_index, step in enumerate([meta['fd_step'],meta['fd_step']/2]):
            geometric = np.any(counts[:,step_index,:,:,k] != 0,axis=(1,2))
            dual = np.any(dual_counts[:,step_index,:,:,k] != 0,axis=(1,2))
            report['tolerance_audit'].append(dict(tolerance=float(tol), fd_step=step,
                        geometric_changed=int(geometric.sum()), dual_changed=int(dual.sum()),
                        classification_disagreement=int(np.count_nonzero(geometric != dual))))
    if args.reference_only:
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        return
    rows=[]
    sample_rows=[]
    problem=make_linear_mpc(meta['horizon'])
    problem.a_matrix.data=inputs['a'][0].astype(float)
    problem.p_matrix.data=inputs['p'][0].astype(float)
    for forward in meta['forward_iterations']:
        for refinement in meta['refinements']:
            with np.load(p/f'grid_f{forward}_r{refinement}.npz') as archive:
                data = {key: archive[key] for key in ('jacobian', 'x', 'y')}
            errors=abs(data['jacobian']-reference['fd_half'])/scales
            for name, mask in [('all',np.ones(n,dtype=bool)), ('active_set_stable',~changed), ('active_set_changed',changed)]:
                if not mask.any():
                    continue
                rows.append(dict(forward_iterations=forward,refinement_iterations=refinement,group=name,
                                 samples=int(mask.sum()),**entry_statistics(errors[mask])))
            worst_sample = int(np.argmax(errors.max(axis=(1,2))))
            for threshold in (0.01, 0.1, 1.0):
                tail = errors > threshold
                tail_samples = np.any(tail,axis=(1,2))
                report['tail_association'].append(dict(forward_iterations=forward, refinement_iterations=refinement,
                    threshold=threshold, tail_entries=int(tail.sum()), changed_tail_entries=int(tail[changed].sum()),
                    tail_samples=int(tail_samples.sum()), changed_tail_samples=int(np.count_nonzero(tail_samples & changed)),
                    worst_sample=worst_sample, worst_sample_changes_active_set=bool(changed[worst_sample])))
            for i in range(n):
                geom, dual=active_sets(problem.a_matrix@data['x'][i],inputs['l'][i],inputs['u'][i],data['y'][i])
                ax = problem.a_matrix@data['x'][i]
                primal_residual = max(0., float(np.max(inputs['l'][i]-ax)),float(np.max(ax-inputs['u'][i])))
                dual_residual = np.max(abs(problem.p_matrix@data['x'][i]+inputs['q'][i]+problem.a_matrix.T@data['y'][i]))
                sample_rows.append(dict(forward_iterations=forward,refinement_iterations=refinement,sample=i,
                    fd_active_set_changed=bool(changed[i]), primal_residual_inf=primal_residual, dual_residual_inf=float(dual_residual),
                    primal_reference_difference_inf=float(np.max(abs(data['x'][i]-reference['x'][i]))), normalized_max=float(errors[i].max()),
                    normalized_mean=float(errors[i].mean()),forward_active_mismatch=int(np.count_nonzero(geom[1]!=base[i,1])),
                    forward_dual_support_mismatch=int(np.count_nonzero(dual[1]!=dual_base[i,1]))))
    for filename, records in [('group_statistics.csv',rows),('sample_diagnostics.csv',sample_rows)]:
        with (out/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
