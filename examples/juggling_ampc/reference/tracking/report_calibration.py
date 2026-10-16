"""Report a completed measured hand-eye calibration from its saved evidence."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from arm.model import HOME
from . import calibration as cal


def json_lines(path):
    path = Path(path)
    if path.exists():
        with path.open() as stream: return [json.loads(line) for line in stream]
    with gzip.open(str(path)+'.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def report(run, calibration):
    fixed = cal.load(calibration)
    status = json.loads((run/'robot/session.json').read_text())
    if status['fault'] or not status['control_completed']:
        raise ValueError('Report requires a completed, fault-free robot session')
    X = np.array(fixed['F_T_P']); Y = np.array(fixed['calibration_V_T_B'])
    captures = [json.loads(path.read_text()) for path in sorted(run.glob('pose_*.json'))]
    mean_errors = []; scatter = []; time_error = []; counts = []
    for item in captures:
        mean = item['mean']; observations = item['observations']
        mean_errors.append(cal.errors(Y@np.array(mean['B_T_F'])@X, np.array(mean['V_T_P'])))
        scatter.append(np.percentile([cal.errors(np.array(mean['V_T_P']), np.array(o['V_T_P']))
                                     for o in observations], 95, axis=0))
        time_error.extend(abs(o['capture_time']-o['robot_time']) for o in observations)
        counts.append(len(observations))
    mean_errors = np.array(mean_errors); scatter = np.array(scatter)
    arm = json_lines(run/'robot/robot.jsonl'); vicon = json_lines(run/'vicon.jsonl')
    final = json.loads((run/'robot/robot_final.json').read_text())
    t = np.array([r['elapsed'] for r in arm]); q = np.array([r['q'] for r in arm]); qd = np.array([r['q_d'] for r in arm])
    latency = np.array([r['latency_s'] for r in vicon if r['latency_s'] is not None])
    result = dict(physical_robot_used=True, calibration=str(calibration), poses=len(captures),
        observations_per_pose=counts, total_observations=sum(counts), robot_session=status,
        translation_rms_m=fixed['translation_rms_m'], rotation_rms_deg=fixed['rotation_rms_deg'],
        max_translation_m=fixed['max_translation_m'], holdout_max_translation_m=fixed['holdout_max_translation_m'],
        holdout_max_rotation_deg=fixed['holdout_max_rotation_deg'],
        maximum_pair_time_difference_s=max(time_error), median_pair_time_difference_s=float(np.median(time_error)),
        vicon_frames=len(vicon), paddle_visible_frames=sum(r['paddle'] is not None for r in vicon),
        vicon_latency_median_s=float(np.median(latency)), vicon_latency_max_s=float(np.max(latency)),
        duration_s=float(t[-1]), final_robot_mode=final['robot_mode'], final_errors=final['current_errors'],
        final_max_home_error_rad=float(np.max(np.abs(np.array(final['q'])-HOME))),
        max_encoder_vs_sdk_command_rad=float(np.max(np.abs(q-qd))),
        F_T_P_translation_m=X[:3, 3].tolist(),
        P_T_CAD_face_translation_m=np.array(fixed['P_T_CAD_face'])[:3, 3].tolist(),
        base_z_vs_vicon_up_deg=float(np.rad2deg(np.arccos(np.clip(Y[2, 2], -1, 1)))),
        note='Fit residuals measure internal agreement, not independently measured absolute accuracy. SDK latency gives approximate capture timestamps.')
    cal.write_new(run/'analysis.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    poses = np.arange(1, len(captures)+1); withheld = np.arange(len(captures))%4 == 3
    for ax, column, scale, label in [(axes[0, 0], 0, 1000., 'Position [mm]'),
                                    (axes[0, 1], 1, 180/np.pi, 'Rotation [degrees]')]:
        ax.plot(poses, scale*mean_errors[:, column], 'o-', label='Pose-mean fit residual')
        ax.plot(poses, scale*scatter[:, column], 's--', label='Within-pose Vicon scatter (p95)')
        ax.scatter(poses[withheld], scale*mean_errors[withheld, column], marker='x', s=80, c='k', label='Withheld during validation')
        ax.set(xlabel='Calibration pose', ylabel=label, xticks=[1, 4, 8, 12, 16]); ax.legend(fontsize=8)
    axes[1, 0].plot(t, q-HOME)
    axes[1, 0].set(xlabel='Robot control time [s]', ylabel='Joints from HOME [rad]')
    axes[1, 1].plot(t, 1000*np.max(np.abs(q-qd), axis=1))
    axes[1, 1].set(xlabel='Robot control time [s]', ylabel='Max encoder vs SDK position command [mrad]')
    for ax in axes.flat: ax.grid(alpha=.25)
    fig.suptitle('Measured Franka / Vicon hand-eye calibration: '+run.name+'\n'
                 f"RMS {1000*fixed['translation_rms_m']:.3f} mm / {fixed['rotation_rms_deg']:.3f} degrees; "
                 f"held-out max {1000*fixed['holdout_max_translation_m']:.3f} mm / {fixed['holdout_max_rotation_deg']:.3f} degrees")
    fig.savefig(run/'report.pdf'); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--calibration', type=Path, default=Path('config/paddle_hand_eye.json'))
    args = p.parse_args(); report(args.run, args.calibration)
