#!/usr/bin/env python3
"""Matched-input 2x2 precision/free-row-penalty study using an existing FD reference.

Forward solutions are shared over refinement counts; adjoint executables are
shared over forward counts and penalties. Neither changes the solver arithmetic.
Explicitly set both penalties so results do not depend on the library default.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmarks.problems import make_linear_mpc
from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales, entry_statistics


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.vram_cases import vram_case

def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--refinements', type=int, nargs='+', help='Override the reference study refinement grid')
    penalty = parser.add_mutually_exclusive_group()
    penalty.add_argument('--fixed-only', action='store_true', default=True, help='Use the default free-row fix for both precisions (default)')
    penalty.add_argument('--include-legacy', dest='fixed_only', action='store_false', help='Also run explicit legacy rho_free=1e-6 controls')
    args = parser.parse_args()
    if args.refinements and (min(args.refinements) < 1 or len(set(args.refinements)) != len(args.refinements)):
        parser.error('Refinement counts must be positive and distinct')
    source, out = args.reference_dir, args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty to avoid mixing studies')
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / 'metadata.json').read_text())
    if args.refinements:
        meta['refinements'] = sorted(args.refinements)
    penalties = (True,) if args.fixed_only else (False, True)
    points_per_case = len(meta['forward_iterations']) * len(meta['refinements'])
    total_points = 2 * len(penalties) * points_per_case
    with np.load(source / 'inputs.npz') as f:
        inputs = {k: f[k] for k in f.files}
    with np.load(source / 'reference.npz') as f:
        reference = f['fd_half']
    scales = fd_entry_scales(reference)
    assert reference.shape == (meta['batch_size'], 4, 12)
    assert inputs['q'].dtype == np.float32, 'Promote the original float32 QPs, not regenerated data'
    problem = make_linear_mpc(meta['horizon'])
    amat = problem.a_matrix.copy()
    amat.data = inputs['a'][0].astype(float)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ('warpmpc/jax_osqp', 'warpmpc/jax_qdldl') for p in (ROOT / folder).glob('*.py')}
    source_path = Path(__file__).resolve()
    source_hashes[str(source_path.relative_to(ROOT))] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ('inputs.npz', 'reference.npz')}
    audit = source / 'active_set_analysis' / 'active_sets.npz'
    if audit.exists():
        shutil.copy2(audit, out / 'active_sets.npz')
    import jax
    import jax.numpy as jnp
    from benchmarks.jax_cache import configure_jax_compilation_cache
    from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan, compile_osqp
    jax.config.update('jax_enable_x64', True)
    configure_jax_compilation_cache()
    if args.require_gpu and not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('GPU required')
    settings = OSQPSettings(max_iter=25, scaling=0, adaptive_rho=False, rho_is_vec=True,
                            check_termination=0, warm_starting=False, polishing=False)
    base_plan = build_osqp_plan(problem.p_matrix, problem.a_matrix, problem.l, problem.u,
                                settings, derivatives=True)
    opts = dict(transpose_work=True, segmented=True, segment_budget=16,
                segment_strategy='optimal', level_scheduled_solve=True)
    action = slice(*meta['action_slice'])
    started = time.time()
    study = dict(status='running', completed_points=0, total_points=total_points, reference_sha256=hashes,
                 source_sha256=source_hashes, jax_version=jax.__version__,
                 devices=[str(d) for d in jax.devices()], started=time.ctime(),
                 matched_inputs='Identical original float32 inputs, promoted exactly for float64',
                 normalization='Per-entry max absolute FD over the original full batch',
                 forward_iterations=meta['forward_iterations'], refinements=meta['refinements'],
                 fixed_only=args.fixed_only)
    write_json(out / 'metadata.json', study)
    for dtype in (np.float32, np.float64):
        values = tuple(jnp.asarray(inputs[k], dtype=dtype) for k in ('p', 'a', 'q', 'l', 'u'))
        cases = []
        for fixed in penalties:
            name = np.dtype(dtype).name + ('_free_rho' if fixed else '')
            directory = out / name
            directory.mkdir()
            for file in ('inputs.npz', 'reference.npz'):
                shutil.copy2(source / file, directory / file)
            label = f'{np.dtype(dtype).itemsize * 8}-bit, ' + ('free-row penalty fix' if fixed else 'original free-row penalty')
            case_meta = dict(meta, status='running', completed_points=0, total_points=points_per_case,
                             dtype=np.dtype(dtype).name, configuration_label=label,
                             free_row_rho=.1 if fixed else 1e-6, source_sha256=source_hashes,
                             reference_sha256=hashes, output_dir=str(directory),
                             jax_version=jax.__version__, started=time.ctime(),
                             primary_error=study['normalization'], study='matched_precision_grid')
            write_json(directory / 'metadata.json', case_meta)
            case_settings = replace(settings, rho_free=None if fixed else 1e-6)
            plan = build_osqp_plan(problem.p_matrix, problem.a_matrix, problem.l, problem.u,
                                   case_settings, derivatives=True)
            case = dict(name=name, directory=directory, metadata=case_meta, states={}, rows=[])
            cases.append(case)
            for forward in meta['forward_iterations']:
                with vram_case(batch_size=meta['batch_size'], metadata=dict(dtype=np.dtype(dtype).name, name=name, forward=forward, mode='forward')):
                    tick = time.time()
                    print(f'BEGIN {name} forward={forward}', flush=True)
                    fplan = replace(plan, settings=replace(case_settings, max_iter=forward))
                    solver = compile_osqp(fplan, dtype=dtype, **opts)
                    x, z, y, pr, dr, obj = solver.solve(*values)
                    x, z, y, pr, dr = [np.asarray(v) for v in (x, z, y, pr, dr)]
                    actual_ax = (amat @ x.astype(float).T).T
                    violation = np.maximum(np.maximum(inputs['l']-actual_ax, actual_ax-inputs['u']), 0).max(axis=1)
                    state = dict(x=x, z=z, y=y, primal_residual=pr, dual_residual=dr,
                                 constraint_violation=violation)
                    np.savez_compressed(directory / f'forward_f{forward}.npz', **state)
                    case['states'][forward] = state
                    print(f'FORWARD {name} f={forward} violation_max={violation.max():.8g} seconds={time.time()-tick:.1f}', flush=True)
                    del solver
                    jax.clear_caches()
        # Adjoint construction depends on P,A,bounds,x,y, not the forward rho.
        for refinement in meta['refinements']:
            solver = compile_osqp(base_plan, dtype=dtype, derivatives=True,
                                  derivative_refinement_iters=refinement, **opts)
            @jax.jit
            def gradients(p, a, l, u, x, y):
                def one(seed):
                    dx = jnp.zeros_like(x).at[:, action].set(jnp.broadcast_to(seed, (x.shape[0], 4)))
                    state = solver.adjoint_derivative_compute(p, a, l, u, x, y, dx, jnp.zeros_like(y))
                    _, dl, du = solver.adjoint_derivative_get_vec(state)
                    return -(dl[:, :12] + du[:, :12])
                return jnp.moveaxis(jax.lax.map(one, jnp.eye(4, dtype=x.dtype)), 0, 1)
            for case in cases:
                for forward, state in case['states'].items():
                    with vram_case(batch_size=meta['batch_size'], metadata=dict(dtype=np.dtype(dtype).name, name=case['name'], forward=forward, refinement=refinement, mode='adjoint')):
                        tick = time.time()
                        jac = np.asarray(gradients(values[0], values[1], values[3], values[4],
                                                   jnp.asarray(state['x']), jnp.asarray(state['y'])))
                        if jac.shape != reference.shape or not np.isfinite(jac).all():
                            raise ValueError(f'Invalid Jacobian: {case["name"]}, {forward}, {refinement}')
                        errors = abs(jac.astype(float)-reference) / scales
                        np.savez_compressed(case['directory'] / f'grid_f{forward}_r{refinement}.npz',
                                            gradient=jac, jacobian=jac, **state)
                        row = dict(sweep='grid', forward_iterations=forward, refinement_iterations=refinement,
                                   nonfinite_count=0, **entry_statistics(errors),
                                   violation_max=float(state['constraint_violation'].max()), seconds=time.time()-tick)
                        case['rows'].append(row)
                        with (case['directory'] / 'summary.csv').open('w', newline='') as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader(); writer.writerows(case['rows'])
                        case['metadata']['completed_points'] = len(case['rows'])
                        if len(case['rows']) == points_per_case:
                            case['metadata']['status'] = 'complete'
                        write_json(case['directory'] / 'metadata.json', case['metadata'])
                        study['completed_points'] += 1
                        write_json(out / 'metadata.json', study)
                        print(f'POINT {study["completed_points"]}/{total_points} {case["name"]} f={forward} r={refinement} p99={row["p99"]:.8g} max={row["max"]:.8g}', flush=True)
            del solver, gradients
            jax.clear_caches()
    study.update(status='complete', elapsed_seconds=time.time()-started)
    write_json(out / 'metadata.json', study)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
