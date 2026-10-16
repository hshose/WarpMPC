#!/usr/bin/env python3
"""Relate normalized sensitivity errors to absolute and normalized FD magnitude."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import numpy as np
from benchmarks.linear_mpc_gradients.sensitivity_normalization import fd_entry_scales,entry_statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    source=args.results_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'metadata.json').read_text())
    with np.load(source/'reference.npz') as f:ref=f['fd_half']
    scales=fd_entry_scales(ref);mag=abs(ref)/scales
    bins=np.array([0,1e-8,1e-6,1e-4,1e-2,.1,.5,1.000000000001])
    records=[];tails=[];worst=[]
    for forward in meta['forward_iterations']:
        for refinement in meta['refinements']:
            with np.load(source/f'grid_f{forward}_r{refinement}.npz') as f:jac=f['jacobian']
            err=abs(jac-ref)/scales
            for lo,hi in zip(bins[:-1],bins[1:]):
                mask=(mag>=lo)&(mag<hi)
                if not mask.any():continue
                records.append(dict(forward=forward,refinement=refinement,magnitude_lower=lo,magnitude_upper=hi,
                    count=int(mask.sum()),tail_over_point1=int((err[mask]>.1).sum()),tail_over_1=int((err[mask]>1).sum()),
                    **entry_statistics(err[mask])))
            for threshold in [.01,.1,1.]:
                mask=err>threshold
                if not mask.any():continue
                tails.append(dict(forward=forward,refinement=refinement,error_threshold=threshold,count=int(mask.sum()),
                    fd_abs_min=float(abs(ref[mask]).min()),fd_abs_median=float(np.median(abs(ref[mask]))),fd_abs_max=float(abs(ref[mask]).max()),
                    fd_normalized_min=float(mag[mask].min()),fd_normalized_median=float(np.median(mag[mask])),fd_normalized_max=float(mag[mask].max()),
                    fd_normalized_below_1e_6=int((mag[mask]<1e-6).sum()),fd_normalized_above_half=int((mag[mask]>.5).sum())))
            if forward==2000 and refinement==400:
                for flat in np.argsort(err.ravel())[-30:][::-1]:
                    b,j,i=np.unravel_index(flat,err.shape)
                    worst.append(dict(sample=int(b),action=int(j),state=int(i),fd=float(ref[b,j,i]),backward=float(jac[b,j,i]),
                                      scale=float(scales[j,i]),normalized_fd=float(mag[b,j,i]),normalized_error=float(err[b,j,i])))
                final=err;final_jac=jac
    for filename,rows in [('magnitude_bins.csv',records),('tail_magnitudes.csv',tails),('worst_entries.csv',worst)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmarks.linear_mpc_gradients import plot_style as style
    style._apply_paper_rc(plt)
    fig,axes=plt.subplots(1,2,figsize=(3.4,1.6),sharey=True)
    for ax,values,title in zip(axes,[mag,abs(final_jac)/scales],['Finite Differences','Backward Sensitivities']):
        ax.scatter(values.ravel(),np.maximum(final.ravel(),1e-16),s=.6,alpha=.2,color=style.OURS_BLUE_DARK,rasterized=True)
        ax.set_xscale('symlog',linthresh=1e-8);ax.set_yscale('log')
        ax.set_xlabel('Entry Magnitude / FD Maximum');ax.set_title(style._paper_title(title))
        ax.grid(alpha=.25,linewidth=.35);ax.tick_params(length=1.6,width=.35)
        for spine in ax.spines.values():spine.set_linewidth(.35)
    axes[0].set_ylabel('Normalized Absolute Entry Error')
    fig.suptitle('2000 OSQP Iterations, 400 Refinement Steps',y=.995)
    fig.subplots_adjust(left=.13,right=.98,bottom=.24,top=.77,wspace=.2)
    fig.savefig(out/'error_vs_sensitivity_magnitude.pdf',bbox_inches='tight',pad_inches=.01)
    fig.savefig(out/'error_vs_sensitivity_magnitude.png',dpi=300,bbox_inches='tight',pad_inches=.01)
    plt.close(fig)
    print(json.dumps([r for r in tails if r['forward']==2000 and r['refinement']==400],indent=2))


if __name__=='__main__':main()
