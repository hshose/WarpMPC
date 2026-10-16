#!/usr/bin/env python3
"""Compare precision and conditioning controls using saved diagnosis artifacts."""
import argparse
import csv
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.linear_mpc_gradients import plot_style as style
from benchmarks.linear_mpc_gradients.sensitivity_normalization import entry_statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,help='Defaults to RESULTS_ROOT/figures');args=p.parse_args()
    args.output_dir = args.output_dir or args.results_root/'figures'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with np.load(args.results_root/'u0-jacobian/active_set_analysis/active_sets.npz') as z:stable=~z['changed']
    rows=[]
    for directory in ['precision_diagnostics','penalty_diagnostics']:
        base=args.results_root/directory
        with (base/'summary.csv').open() as stream:records=list(csv.DictReader(stream))
        for record in records:
            with np.load(base/f"{record['case']}_f{record['iterations']}.npz") as z:
                errors=z['normalized_errors'];violation=z['constraint_violation']
                for group,mask in [('all',np.ones(len(stable),bool)),('stable',stable)]:
                    rows.append(dict(case=record['case'],iterations=int(record['iterations']),group=group,samples=int(mask.sum()),
                                     violation_max=float(violation[mask].max()),**entry_statistics(errors[mask])))
    with (args.output_dir/'precision_conditioning_statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    style._apply_paper_rc(plt)
    styles=[('optimized32','float32',style.OURS_BLUE_DARK,'o'),('optimized64','float64','#009E73','s'),
            ('baseline32','float32, baseline','#7E57C2','x'),('free_rho32',r'float32, free-row $\rho=0.1$','#D55E00','^')]
    for group in ['all','stable']:
        fig,axes=plt.subplots(1,2,figsize=(3.4,1.5))
        for ax,key,title in zip(axes,['violation_max','p99'],['Forward Feasibility','Backward Sensitivity Error']):
            for case,label,color,marker in styles:
                selected=sorted([r for r in rows if r['case']==case and r['group']==group],key=lambda r:r['iterations'])
                ax.plot([r['iterations'] for r in selected],[r[key] for r in selected],color=color,marker=marker,
                        linewidth=.75,markersize=2.7,alpha=.75,label=label)
            ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([2000,20000],labels=['2000','20000'])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.set_xlabel('Forward OSQP Iterations',labelpad=.2)
            ax.set_title(style._paper_title(title),pad=2);ax.grid(alpha=.25,linewidth=.35)
            ax.tick_params(length=1.6,width=.35,pad=.7)
            for spine in ax.spines.values():spine.set_linewidth(.35)
        axes[0].set_ylabel('Maximum Constraint Violation',labelpad=.2)
        axes[1].set_ylabel('99th Percentile Normalized Error',labelpad=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.54,1.04),handlelength=1.2)
        fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.76,wspace=.43)
        if group=='stable':fig.text(.5,-.01,'Excluded 2.7\\% (27/1000); same FD normalization scales',ha='center',fontsize=4)
        for extension in ['pdf','png']:
            fig.savefig(args.output_dir/f'qp_sensitivity_precision_conditioning_{group}.{extension}',dpi=300,bbox_inches='tight',pad_inches=.005)
        plt.close(fig)


if __name__=='__main__':main()
