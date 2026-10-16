"""Verify archived native evidence and report common/own-input timing tails."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cpp.reporting import benchmark_report, mock_report, timing
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    directory = parser.parse_args().directory
    hashes = json.loads((directory/'sha256.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name
    result = benchmark_report(directory)
    for file in sorted(directory.glob('*steps.csv')) + [directory/'own_mesh24_prune4.csv']:
        data = np.genfromtxt(file, delimiter=',', names=True)
        assert np.all(data['iterations'] == 50)
        name = file.stem.replace('_steps', '')
        steady = data['index'] > 0
        if name not in result:
            result[name] = dict(all_calls=timing(data['total_ms']),
                               steady_calls=timing(data['total_ms'][steady]),
                               first_calls=timing(data['total_ms'][~steady]),
                               exactly_50_osqp=True)
        for label, mask in [('all_calls', np.ones(len(data), dtype=bool)), ('steady_calls', steady)]:
            times = data['total_ms'][mask]
            result[name][label].update(p99_ms=float(np.percentile(times, 99)),
                                      calls_above_10ms=int(np.count_nonzero(times > 10)),
                                      fraction_above_10ms=float(np.mean(times > 10)))
    two = np.genfromtxt(directory/'mesh24_prune_steps.csv', delimiter=',', names=True)
    four = np.genfromtxt(directory/'mesh24_prune4_steps.csv', delimiter=',', names=True)
    error = {name: float(np.max(abs(two[name]-four[name])))
             for name in ['iterations', 'alpha', 'violation', 'clipping']}
    assert max(error.values()) == 0, error
    mock = {name: mock_report(directory/name, seconds=20)
            for name in ['hil_mesh24_prune', 'hil_mesh24_prune4']}
    assert all(value['task_pass'] for value in mock.values())
    own = np.genfromtxt(directory/'own_mesh24_prune4.csv', delimiter=',', names=True)
    steady = own['index'] > 0
    times = own['total_ms'][steady]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(own['total_ms'], linewidth=.7)
    axes[0, 0].axhline(10, color='black', linestyle='--')
    axes[0, 0].set(xlabel='Call index (includes first calls)', ylabel='C++ time [ms]', title='Every call; no outlier removal')
    axes[0, 1].plot(np.sort(times), np.linspace(0, 1, len(times)))
    axes[0, 1].axvline(10, color='black', linestyle='--')
    axes[0, 1].set(xlabel='C++ time after first call [ms]', ylabel='Cumulative fraction')
    stats = result['own_mesh24_prune4']['steady_calls']
    labels = ['median', 'mean', 'p95', 'p99', 'max']
    axes[1, 0].bar(labels, [stats[label+'_ms'] for label in labels])
    axes[1, 0].axhline(10, color='black', linestyle='--')
    axes[1, 0].set(ylabel='C++ time [ms]', title=f'{stats["calls_above_10ms"]} / {len(times)} subsequent calls exceed 10 ms')
    axes[1, 1].bar(['warmstart', 'data', 'QP', 'merit'],
                   [np.mean(own[key][steady]) for key in ['shift_ms', 'data_ms', 'qp_ms', 'merit_ms']])
    axes[1, 1].set(ylabel='Mean component time [ms]', title='Four evaluator workers; single-thread OSQP')
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.suptitle('24 geometric intervals + conservative pruning: own closed-loop wind inputs\n998 updates × 3 replays; one SQP / 50 OSQP; constructor/build excluded')
    fig.savefig(directory/'own_replay_report.pdf')
    plt.close(fig)
    audit = dict(hash_count=len(hashes), all_archived_hashes_verified=True,
                 timing=result, worker_count_output_diagnostics_max_error=error,
                 mock=mock, scope='C++ step wall time; no trimming. Common replay999 inputs, own replay998 inputs; three repetitions each.')
    (directory/'optimization_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
