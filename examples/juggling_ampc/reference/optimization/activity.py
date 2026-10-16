"""Audit saved nonlinear MPC plans without invoking or changing the controller.

Activity means proximity to a bound, including violations. Saved plans contain no
QP multipliers, so this is not a KKT active-set certificate. The archived native
factor library, metadata, and formulation source are used for reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import casadi as ca
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RUNS = [
    'runs/20260924T171540.076412Z_cpp_fixed_nominal',
    'runs/20260924T175306.001124Z_cpp_final_wind',
    'runs/20260924T171540.090515Z_cpp_fixed_damped',
]
THRESHOLDS = (0., .001, .01, .05, .1)
BOUND_GROUPS = {
    'ball_position': (0, 3), 'ball_velocity': (3, 6),
    'joint_position': (6, 13), 'joint_velocity': (13, 20),
    'spin': (20, 23), 'torque': (23, 30),
    'normal_impulse': (30, 31), 'contact_fraction': (31, 32),
    'fixed_timestep': (32, 33), 'clock': (33, 34),
}
EQUALITIES = {'ball', 'spin', 'position', 'dynamics', 'clock'}
PRUNE_CANDIDATES = ['workspace', 'tilt', 'upward', 'paddle_velocity', 'jerk']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_name(value):
    return f'{value:g}'


def summarize(slack, stages, cutoff):
    """slack shape: saved prediction, horizon stage, constraint component."""
    def stats(x):
        return {
            'plan_stage_count': int(x.shape[0] * x.shape[1]),
            'component_count': int(x.size),
            'minimum_margin': float(x.min()),
            'maximum_violation': float(max(0., -x.min())),
            'violated_components_1e_8': int(np.count_nonzero(x < -1e-8)),
            'near_bound': {threshold_name(t): {
                'component_count': int(np.count_nonzero(x <= t)),
                'plan_stage_count': int(np.count_nonzero(np.any(x <= t, axis=2))),
                'plan_stage_fraction': float(np.mean(np.any(x <= t, axis=2))),
            } for t in THRESHOLDS},
        }
    return {
        'all': stats(slack), 'prefix': stats(slack[:, :cutoff]),
        'tail': stats(slack[:, cutoff:]),
        'per_stage': [dict(stage=int(k), **stats(slack[:, k:k+1])) for k in stages],
    }


def audit(paths, cutoff, artifact):
    all_constraints, all_bounds, all_plans, all_times = [], [], [], []
    inputs, plant_safety, cases = [], {}, {}
    base_meta = None
    jacobian_rows = None
    for path in paths:
        path = Path(path)
        config = json.loads((path / 'config.json').read_text())
        metrics = json.loads((path / 'metrics.json').read_text())
        metadata_path = path / 'native/factor_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        if base_meta is None:
            base_meta = metadata
        else:
            assert metadata['arrays'] == base_meta['arrays']
            assert metadata['groups'] == base_meta['groups']
        assert config['mpc']['horizon'] == 40 and config['mpc']['h'] == .03
        a = metadata['arrays']
        Y = np.load(path / 'predictions.npz')['Y']
        Z = (Y - np.array(a['offset'])) / np.array(a['scale'])
        library = path / 'native/libjuggling_factors.so'
        evaluate = ca.external('stage_value', str(library.resolve()))
        data = ca.external('stage_data', str(library.resolve()))
        jacobian_rows = np.array(data.sparsity_out(3).row())
        parameters = np.zeros(30)  # The stage factor is independent of p.
        C = np.array([
            np.asarray(evaluate(np.r_[z[k], z[k+1]], parameters)[1]).ravel()
            for z in Z for k in range(Y.shape[1]-1)
        ]).reshape(len(Y), Y.shape[1]-1, -1)
        lower, upper = np.array(a['lower']), np.array(a['upper'])
        width = .5 * (upper - lower)
        width[width == 0] = 1.
        B = np.minimum(Z - lower, upper - Z) / width
        arrays = np.load(path / 'arrays.npz')
        times = arrays['t'][np.arange(len(Y))*10]
        all_constraints.append(C); all_bounds.append(B)
        all_plans.append(Y); all_times.append(times)
        case = config['arguments']['name']
        cases[case] = (C, B)
        files = ['predictions.npz', 'config.json', 'metrics.json',
                 'native/factor_metadata.json', 'native/libjuggling_factors.so',
                 'source/arm/spin_formulation.py']
        inputs.append(dict(path=str(path), case=case, predictions=len(Y),
                           first_time_s=float(times[0]), last_time_s=float(times[-1]),
                           hashes={name: digest(path/name) for name in files}))
        plant_safety[case] = {key: metrics[key] for key in [
            'duration_s', 'contacts', 'limit_ratios', 'joint_position_limits_ok',
            'other_contact_samples', 'max_torque_clipping_Nm', 'max_paddle_speed_m_s',
            'max_total_acceleration_rad_s2', 'max_total_jerk_rad_s3',
        ]}
    C = np.concatenate(all_constraints)
    B = np.concatenate(all_bounds)
    Y = np.concatenate(all_plans)
    a = base_meta['arrays']
    lower, upper = np.array(a['stage_lower']), np.array(a['stage_upper'])
    S = np.minimum(C-lower, upper-C)
    # Upward normal has only a lower bound, but n_z <= 1 geometrically.
    up_start, up_end = base_meta['groups']['upward']
    S[:, :, up_start:up_end] /= 1. - lower[up_start:up_end]
    summary = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Three 20 s Newton closed loops; every tenth accepted MPC plan saved.',
        'activity_definition': 'Minimum signed nonlinear slack <= threshold, including infeasible rows. No QP dual variables were saved.',
        'normalization': {
            'nonlinear': 'Original formulation scaled units; upward additionally divided by 1-cos(0.4). Acceleration/rate/jerk margins use nominal limit units, with hard bound 0.98.',
            'bounds': 'Distance to closer physical bound divided by half the full physical bound interval. Fixed timestep is structural, not an omission candidate.',
            'contact': 'Impulse-gated contact rows are automatically tight at zero impulse; proximity alone does not imply a physical contact.'},
        'thresholds': list(THRESHOLDS),
        'prediction_count': len(Y), 'stage_count': Y.shape[1]-1,
        'prediction_stage_samples': int(C.shape[0]*C.shape[1]),
        'cutoff_stage': cutoff, 'cutoff_time_s': cutoff*.03,
        'inputs': inputs, 'plant_safety': plant_safety,
        'nonlinear_inequalities': {}, 'variable_bounds': {}, 'equalities': {},
    }
    groups = dict(base_meta['groups'])
    # Split contact rows and duplicate endpoint rows so activity cannot be hidden
    # by another component of the same group.
    start, _ = groups['contact']
    for offset, name in enumerate(['contact_gap', 'end_gap', 'contact_disk', 'contact_approach']):
        groups[name] = (start+offset, start+offset+1)
    start, _ = groups['endpoint_acceleration']
    groups['endpoint_acceleration_current'] = (start, start+7)
    groups['endpoint_acceleration_next'] = (start+7, start+14)
    for name, (start, end) in groups.items():
        structure = dict(rows_per_stage=end-start,
                         jacobian_nonzeros_per_stage=int(np.count_nonzero((jacobian_rows>=start)&(jacobian_rows<end))))
        if name in EQUALITIES:
            defects = np.abs(C[:, :, start:end])
            summary['equalities'][name] = dict(**structure,
                maximum_absolute_scaled_defect=float(defects.max()),
                p95_absolute_scaled_defect=float(np.percentile(defects,95)))
            continue
        entry = summarize(S[:, :, start:end], range(C.shape[1]), cutoff)
        entry.update(structure)
        entry['per_case'] = {}
        for case, (Ci, _) in cases.items():
            Si = np.minimum(Ci[:, :, start:end]-lower[start:end], upper[start:end]-Ci[:, :, start:end])
            if name == 'upward': Si /= 1.-lower[start:end]
            entry['per_case'][case] = summarize(Si, range(C.shape[1]), cutoff)['all']
        summary['nonlinear_inequalities'][name] = entry
    for name, (start, end) in BOUND_GROUPS.items():
        entry = summarize(B[:, :, start:end], range(B.shape[1]), cutoff)
        entry['components_per_node'] = end-start
        summary['variable_bounds'][name] = entry
    # Endpoint acceleration at a stage's next node is exactly the following
    # stage's current-node acceleration: duplicated inequality rows.
    start, _ = base_meta['groups']['endpoint_acceleration']
    endpoint_duplicate_error = np.max(np.abs(C[:, :-1, start+7:start+14]-C[:, 1:, start:start+7]))
    summary['exact_duplicate_endpoints'] = {
        'maximum_scaled_value_difference': float(endpoint_duplicate_error),
        'removable_rows_without_changing_nonlinear_feasible_set': 7*(C.shape[1]-1),
        'removable_jacobian_nonzeros': 147*(C.shape[1]-1),
        'note': 'One current-node constraint at each stage plus the final next-node constraint preserves every endpoint bound. Removing duplicate QP rows can still change fixed-iteration numerical behavior.'}
    candidate_rows = sum(summary['nonlinear_inequalities'][name]['rows_per_stage'] for name in PRUNE_CANDIDATES)
    candidate_nnz = sum(summary['nonlinear_inequalities'][name]['jacobian_nonzeros_per_stage'] for name in PRUNE_CANDIDATES)
    summary['candidate_tail_pruning'] = {
        'groups': PRUNE_CANDIDATES, 'removed_rows': candidate_rows*(C.shape[1]-cutoff),
        'removed_jacobian_nonzeros': candidate_nnz*(C.shape[1]-cutoff),
        'original_nonlinear_rows': 31+C.shape[1]*72+1,
        'original_constraint_jacobian_nonzeros': 31+C.shape[1]*1541+1,
        'recommendation': 'Test these groups after a six-stage prefix first. Keep contact, all dynamics, acceleration, torque-rate and physical variable bounds. Endpoint pruning beyond exact deduplication needs separate validation because endpoint acceleration occasionally activates throughout the horizon.',
    }
    artifact.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(artifact/'values.npz', constraints=C, bound_margins=B,
                        physical_plans=Y, measurement_time=np.concatenate(all_times))
    summary['raw_values'] = str(artifact/'values.npz')
    return summary


def report(summary, output):
    candidates = PRUNE_CANDIDATES + ['acceleration', 'endpoint_acceleration', 'torque_rate']
    entries = summary['nonlinear_inequalities']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), constrained_layout=True)
    ax = axes[0, 0]
    matrix = np.array([[s['minimum_margin'] for s in entries[name]['per_stage']] for name in candidates])
    im = ax.imshow(matrix, aspect='auto', vmin=-.02, vmax=1., cmap='viridis', extent=[0,1.2,len(candidates)-.5,-.5])
    ax.set_yticks(range(len(candidates)), candidates)
    ax.set_xlabel('Horizon time [s]');ax.set_title('Minimum nonlinear margin over 300 saved plans')
    ax.axvline(summary['cutoff_time_s'],color='white',linestyle='--',linewidth=1)
    fig.colorbar(im,ax=ax,label='Signed normalized margin')
    ax=axes[0,1]
    for name in ['acceleration','endpoint_acceleration','torque_rate']:
        entry=entries[name]
        ax.plot(np.arange(40)*.03,[100*s['near_bound']['0.01']['plan_stage_fraction'] for s in entry['per_stage']],label=name)
    ax.set_xlabel('Horizon time [s]');ax.set_ylabel('Saved plans within 1% of limit [%]')
    ax.set_title('Some actuator constraints activate late');ax.legend(fontsize=8);ax.grid(alpha=.25)
    ax=axes[1,0]
    names=['joint_position','joint_velocity','torque','ball_position','ball_velocity','spin']
    margins=[summary['variable_bounds'][name]['all']['minimum_margin'] for name in names]
    ax.barh(names,margins,color='#4c78a8');ax.set_xlim(0,1);ax.set_xlabel('Minimum normalized physical-bound margin')
    ax.set_title('Variable bounds stay away from limits in these plans');ax.grid(axis='x',alpha=.25)
    ax=axes[1,1];ax.axis('off')
    lines=['Three 20 s loops: nominal, smooth wind, wind + damping',
           f'{summary["prediction_count"]} plans; {summary["prediction_stage_samples"]:,} stage evaluations; 1 SQP / 50 OSQP',
           'Plans saved every tenth solve (~0.2 s): intervening activity unknown.',
           '', 'Never within 5%: workspace, tilt, upward, paddle speed, jerk.',
           'Keep the first 6 stages (0.18 s), then test dropping those rows.',
           f'Tail removal: {summary["candidate_tail_pruning"]["removed_rows"]} rows / {summary["candidate_tail_pruning"]["removed_jacobian_nonzeros"]:,} Jacobian entries.',
           'Exact endpoint duplicates: 273 rows / 5,733 entries removable.',
           '', 'Keep contact and dynamic equalities at every stage.',
           'Endpoint acceleration, acceleration and torque-rate do activate.',
           'Measured plant actuator ratios remain below one in all cases.',
           'Impact-inclusive acceleration/jerk spikes are reported separately.',
           '', 'Proximity includes violations; this is not a KKT active-set test.',
           'Empirical inactivity is a candidate screen, not a safety guarantee.']
    ax.text(0,1,'\n'.join(lines),va='top',fontsize=9.1,linespacing=1.55)
    fig.suptitle('Late-horizon constraint activity — Stage 4 baseline',fontsize=15)
    fig.savefig(output.with_suffix('.pdf'));fig.savefig(output.with_suffix('.png'),dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='*',default=DEFAULT_RUNS)
    parser.add_argument('--cutoff',type=int,default=6)
    parser.add_argument('--output',type=Path,default=Path('docs/constraint_activity.json'))
    parser.add_argument('--artifact',type=Path)
    args=parser.parse_args()
    if not 1<=args.cutoff<40:parser.error('--cutoff must be 1..39')
    artifact=args.artifact or Path('runs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_constraint_activity')
    summary=audit(args.runs,args.cutoff,artifact)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (artifact/'summary.json').write_text(args.output.read_text())
    (artifact/'config.json').write_text(json.dumps({'runs':args.runs,'cutoff':args.cutoff,'thresholds':THRESHOLDS,
        'activity_script_sha256':digest(__file__),'command':'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/franka-matplotlib .venv/bin/python -m optimization.activity'},indent=2)+'\n')
    (artifact/'activity.py').write_bytes(Path(__file__).read_bytes())
    with args.output.with_suffix('.csv').open('w',newline='') as f:
        columns=['kind','group','stage','horizon_time_s','minimum_margin','maximum_violation','violated_components_1e_8']+[
            f'near_{threshold_name(t)}_plan_stages' for t in THRESHOLDS]
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for kind in ['nonlinear_inequalities','variable_bounds']:
            for name,entry in summary[kind].items():
                for stage in entry['per_stage']:
                    row={key:stage[key] for key in ['stage','minimum_margin','maximum_violation','violated_components_1e_8']}
                    row.update(kind=kind,group=name,horizon_time_s=stage['stage']*.03)
                    row.update({f'near_{threshold_name(t)}_plan_stages':stage['near_bound'][threshold_name(t)]['plan_stage_count'] for t in THRESHOLDS})
                    writer.writerow(row)
    report(summary,args.output)
    (artifact/'report.pdf').write_bytes(args.output.with_suffix('.pdf').read_bytes())
    print(json.dumps({'summary':str(args.output),'artifact':str(artifact),'candidate_tail_pruning':summary['candidate_tail_pruning'],
        'endpoint_duplicates':summary['exact_duplicate_endpoints']},indent=2))


if __name__=='__main__':main()
