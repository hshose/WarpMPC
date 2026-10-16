"""Collect a logical 60k batch, optionally split to fit a smaller GPU.

Each part is a synchronous batched Newton rollout. Parts retain distinct seeds
and global trajectory IDs; splitting never mixes an episode across partitions.
"""
import argparse, json, subprocess, sys
from pathlib import Path


def collect(a, common):
    if a.batch < 1 or a.microbatch < 1:
        raise ValueError('Positive total and physical batch sizes required')
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports, shards = [], []
    identity = None
    for part, offset in enumerate(range(0, a.batch, a.microbatch)):
        count = min(a.microbatch, a.batch - offset)
        directory = output / 'parts' / f'{part:04d}'
        manifest_path = directory / 'manifest.json'
        if not manifest_path.exists():
            command = [sys.executable, str(Path(__file__).with_name('rollout.py')), *common,
                       '--batch', str(count), '--seed', str(a.seed + part), '--seconds', '30',
                       '--collect', '--output', str(directory)]
            subprocess.run(command, check=True)
        manifest = json.loads(manifest_path.read_text())
        report = manifest['rollout']
        if manifest['trajectories'] != count or report['seed'] != a.seed + part or report['rollout_seconds'] != 30:
            raise ValueError(f'Existing part has a different collection specification: {directory}')
        if not manifest.get('expert_stabilization_passed'):
            raise RuntimeError(f'Collection part failed expert stabilization: {directory}')
        current = {k: manifest[k] for k in ('live_source_hashes', 'mpc_spec', 'policy_hidden_layers', 'stage_times_s')}
        current.update({k: report[k] for k in ('calibration_sha256', 'noise_disturbance_multiplier', 'drop_bounds', 'policy_kind', 'mpc_feedback', 'implementation_sha256', 'solver_numerics')})
        if identity is not None and identity != current:
            raise ValueError('Collection parts have incompatible models or controller settings')
        identity = current
        reports.append(report)
        for path in sorted(directory.glob('shard_*.npz')):
            shards.append(dict(path=str(path.relative_to(output)), trajectory_offset=offset))
        if not manifest['mpc_parity_passed']:
            raise ValueError('A collection part has no passed MPC parity check')
    combined = dict(reports[0])
    for field in ('seeds', 'uptime_s', 'robot_velocity_flag_ticks', 'projection_flag_ticks', 'contacts', 'contacts_before_loss'):
        combined[field] = [v for r in reports for v in r[field]]
    uptime = combined['uptime_s']
    combined.update(mean_uptime_s=sum(uptime) / len(uptime), survival_fraction=sum(t >= 30 for t in uptime) / len(uptime),
                    wall_s=sum(r['wall_s'] for r in reports), physical_batch_size=a.microbatch,
                    seed_protocol='part seed = base seed + part index; independent local world index',
                    expert_stabilization_passed=all(r['expert_stabilization_passed'] for r in reports))
    combined['loss_causes'] = {k: sum(r['loss_causes'][k] for r in reports) for k in reports[0]['loss_causes']}
    combined['timing'] = {k: sum(r['timing'][k] for r in reports) for k in reports[0]['timing']}
    combined['filtering'] = {k: sum(r['filtering'][k] for r in reports) for k in reports[0]['filtering']}
    manifest = {k: identity[k] for k in ('live_source_hashes', 'mpc_spec', 'policy_hidden_layers', 'stage_times_s')}
    manifest.update(trajectories=a.batch, shards=shards, rollout=combined, mpc_parity_passed=True,
                    expert_stabilization_passed=combined['expert_stabilization_passed'])
    (output / 'rollout.json').write_text(json.dumps(combined, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=60000)
    p.add_argument('--microbatch', type=int, default=60000)
    p.add_argument('--seed', type=int, default=2026)
    args, common = p.parse_known_args()
    collect(args, common)
