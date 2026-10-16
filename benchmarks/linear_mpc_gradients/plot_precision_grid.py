#!/usr/bin/env python3
"""Plot completed precision/penalty grids, filtered controls, and both histogram bin styles."""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-precision-grid')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales, entry_statistics, active_set_filter

CASES = ('float32', 'float32_free_rho', 'float64', 'float64_free_rho')
LABELS = ('32-bit', '32-bit + free-row fix', '64-bit', '64-bit + free-row fix')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='Defaults to RESULTS_DIR/figures')
    parser.add_argument('--skip-histograms', action='store_true')
    parser.add_argument('--comparison-only', action='store_true', help='Only regenerate the configuration overlays and index')
    parser.add_argument('--workers', type=int, default=4, help='Independent plotting processes')
    parser.add_argument('--no-tex', action='store_true')
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.results_dir / 'figures'
    study = json.loads((args.results_dir / 'metadata.json').read_text())
    cases = tuple(c for c in CASES if not study.get('fixed_only') or c.endswith('_free_rho'))
    labels = tuple(LABELS[CASES.index(c)] for c in cases)
    for case in cases:
        metadata = json.loads((args.results_dir / case / 'metadata.json').read_text())
        if metadata['status'] != 'complete':
            raise ValueError(f'Incomplete configuration: {case}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    audit = args.results_dir / 'active_sets.npz'
    if not audit.exists():
        raise ValueError('The common FD active-set audit is required')
    def plot_case(item):
        case, population = item
        target = args.output_dir / case / population
        target.mkdir(parents=True, exist_ok=True)
        shared = ['--results-dir', str(args.results_dir / case), '--output-dir', str(target)]
        if population == 'stable':
            shared += ['--active-set-filter', str(audit)]
        command = [sys.executable, str(script_dir / 'plot_sensitivity_accuracy.py'), *shared, '--metric', 'fd_entry_max']
        if args.no_tex:
            command += ['--no-tex']
        with (target / 'plot.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            if not args.skip_histograms:
                subprocess.run([sys.executable, str(script_dir / 'plot_u0_sensitivity_histograms.py'),
                                *shared, '--normalization', 'fd-entry-max', '--bin-scale', 'both'],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f'PLOTS COMPLETE {case} {population}', flush=True)
    if not args.comparison_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(plot_case, [(case, population) for case in cases for population in ('all', 'stable')]))
    # Direct overlays at the paper's fixed counts make precision effects easy to compare.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmarks.linear_mpc_gradients import plot_style as style
    style._apply_paper_rc(plt)
    if args.no_tex:
        plt.rcParams.update({'text.usetex': False, 'font.family': 'serif'})
    with np.load(args.results_dir / cases[0] / 'reference.npz') as f:
        reference = f['fd_half']
    scale = fd_entry_scales(reference)
    statistics = []
    for population in ('all', 'stable'):
        keep, excluded = active_set_filter(audit if population == 'stable' else None, reference)
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.5), sharey='col')
        for case, label, color, marker in zip(cases, labels, ('#08519C', '#009E73', '#7E57C2', '#D55E00'), ('o', 's', '^', 'D')):
            meta = json.loads((args.results_dir / case / 'metadata.json').read_text())
            for row, (values, fixed) in enumerate(((meta['refinements'], meta['paper_forward_iterations']),
                                                   (meta['forward_iterations'], meta['paper_refinement_iterations']))):
                records = []
                for value in values:
                    forward, refinement = (fixed, value) if row == 0 else (value, fixed)
                    with np.load(args.results_dir / case / f'grid_f{forward}_r{refinement}.npz') as f:
                        errors = (abs(f['jacobian'].astype(float)-reference)/scale)[keep]
                    stats = entry_statistics(errors)
                    records.append(stats)
                    statistics.append(dict(case=case, population=population, forward_iterations=forward,
                                           refinement_iterations=refinement, **stats))
                for col, statistic in enumerate(('median', 'mean', 'max')):
                    ax = axes[row, col]
                    ax.plot(values, [max(r[statistic], 1e-16) for r in records], label=label, color=color, marker=marker,
                            linewidth=.75, markersize=2.7, markeredgewidth=.5)
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                    ax.set_xticks(values, [str(v) for v in values])
                    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                    ax.tick_params(axis='both', which='major', length=1.6, width=.35, pad=.7)
                    ax.tick_params(axis='both', which='minor', length=.8, width=.25)
                    for spine in ax.spines.values():
                        spine.set_linewidth(.35)
                    ax.set_title(f'{statistic.capitalize()} error; ' + (f'{fixed} OSQP iterations' if row == 0 else f'{fixed} refinement steps'))
                    ax.set_xlabel('Backward Refinement Iterations' if row == 0 else 'Forward OSQP Iterations')
                    ax.grid(True, which='major', color='0.86', linewidth=.35)
                    ax.axvline(meta['paper_refinement_iterations'] if row == 0 else meta['paper_forward_iterations'], color='.6', linestyle=':', linewidth=.6)
                    if col == 0:
                        ax.set_ylabel('Normalized Absolute Entry Error')
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', ncol=len(cases))
        if excluded:
            fig.text(.5, .01, excluded.replace('%', r'\%') if not args.no_tex else excluded, ha='center')
        fig.subplots_adjust(top=.86, bottom=.13, hspace=.55, wspace=.32)
        for ext in ('pdf', 'png'):
            fig.savefig(args.output_dir / f'precision_comparison_{population}.{ext}', bbox_inches='tight', dpi=300)
        plt.close(fig)
    with (args.output_dir / 'comparison_statistics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    index = ['# Precision and free-row penalty study', '',
             f'Identical {len(reference):,} QPs and finite-difference references; each of the 48 entries is normalized by its full-batch maximum absolute FD sensitivity.', '',
             ('Both precisions use the free-row penalty fix (rho = 0.1). Paper setting: 25 forward / 100 refinement iterations.' if study.get('fixed_only') else
              'Original free-row penalty: 1e-6. Fixed free-row penalty: 0.1. Paper setting: 25 forward / 100 refinement iterations.'), '',
             '[Configuration comparison, all samples](precision_comparison_all.pdf) | [Stable active sets](precision_comparison_stable.pdf)', '',
             'Each histogram PDF has one page per iteration pair, with all 48 Jacobian entries. Log bins use a linear core at zero; uniform bins use a linear x axis. Both use log counts. ' + active_set_filter(audit, reference)[1], '',
             '| Configuration / population | Refinement curves | OSQP curves | Log-bin histograms | Linear-bin histograms |',
             '|---|---|---|---|---|']
    for case, label in zip(cases, labels):
        for population in ('all', 'stable'):
            base = f'{case}/{population}'
            suffix = '_fd_entry_max' + ('_stable_active_set' if population == 'stable' else '')
            index.append(f'| {label}, {population} | [PDF]({base}/qp_u0_sensitivity_refinement_error{suffix}.pdf) | [PDF]({base}/qp_u0_sensitivity_forward_error{suffix}.pdf) | [PDF]({base}/u0_jacobian_normalized_absolute_error_histograms_log_bins.pdf) | [PDF]({base}/u0_jacobian_normalized_absolute_error_histograms_linear_bins.pdf) |')
    if (args.output_dir / 'RESULTS.md').exists():
        index += ['', '[Numerical summary](RESULTS.md)']
    (args.output_dir / 'INDEX.md').write_text('\n'.join(index) + '\n')
    print('COMPARISON PLOTS COMPLETE' if args.comparison_only else 'ALL PLOTS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
