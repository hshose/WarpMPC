"""Position-only observer diagnostics; ground truth is used ONLY after the run."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_observer_results(directory,feedback,arrays,plant,covariances,info,metrics):
    t=arrays['t'];idx=np.rint(t/.0005).astype(int)
    truth=plant['x'][idx];estimate=arrays['x'];P=np.asarray(covariances)
    error=np.c_[estimate[:,:6]-truth[:,:6],estimate[:,20:23]-truth[:,20:23]]
    age=np.array([i['measurement_age'] for i in info]);last=np.array([i['last_impact'] for i in info])
    # Report every update and the settled-flight subset separately. Contact-time
    # impulse/compliance disagreement must not disappear from the all-time score.
    mask=(t>.1)&(t-last>.02)
    def scores(mask):
        return {label:dict(rmse_norm=float(np.sqrt(np.mean(np.sum(error[mask,j:j+3]**2,axis=1)))),
                           p95_norm=float(np.percentile(np.linalg.norm(error[mask,j:j+3],axis=1),95)),
                           max_norm=float(np.max(np.linalg.norm(error[mask,j:j+3],axis=1))))
                for label,j in [('position_m',0),('velocity_m_s',3),('spin_rad_s',6)]} if np.any(mask) else {}
    updates=feedback.observer.updates
    result=dict(all_control_updates=scores(np.ones(len(t),bool)),settled_flight=scores(mask),
        received_measurements=len(updates),accepted_measurements=sum(i['accepted'] for i in updates),
        rejected_measurements=sum(not i['accepted'] for i in updates),estimated_impacts=len(feedback.observer.events),
        max_measurement_age_s=float(max(age)),spin_observable_in_flight=False,
        spin_note='Spin is a contact-model-conditioned estimate. Rotation about a fixed face normal is unobservable; no truth-spin measurement is used.')
    metrics['observer']=result
    np.savez_compressed(directory/'observer.npz',t=t,estimate=estimate,truth=truth,error=error,covariance=P,
                        measurement_age=age,last_impact=last)
    for name,records in [('vicon_samples',feedback.samples),('observer_updates',updates)]:
        (directory/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    (directory/'observer_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (directory/'observer_impacts.json').write_text(json.dumps(feedback.observer.events)+'\n')
    fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    colors=['C0','C1','C2']
    for row,(label,start,scale,units) in enumerate([('Position',0,1000,'mm'),('Velocity',3,1,'m/s'),('Spin',6,1,'rad/s')]):
        for j,c in enumerate(colors):
            statej=start+j if row<2 else 20+j
            axes[row,0].plot(t,truth[:,statej]*scale,color=c,lw=1,label='xyz'[j]+' truth')
            axes[row,0].plot(t,estimate[:,statej]*scale,color=c,lw=.7,ls='--')
            axes[row,1].plot(t,error[:,start+j]*scale,color=c,lw=.7,label='xyz'[j])
            sigma=np.sqrt(np.maximum(P[:,start+j,start+j],0))*scale
            axes[row,1].fill_between(t,-2*sigma,2*sigma,color=c,alpha=.07)
        axes[row,0].set_title(f'{label}: truth / estimate (dashed)');axes[row,1].set_title(f'{label} error; shaded ±2σ')
        for ax in axes[row]:ax.set_ylabel(units);ax.set_xlabel('Simulation time [s]');ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=3)
    c=feedback.sensor
    fig.suptitle(f'Position-only observer — {directory.name}\n{c.rate_hz:g} Hz, {c.latency_s*1000:g} ms latency, {c.position_std_m*1000:g} mm noise/axis; spin is model-conditioned',fontsize=11)
    fig.savefig(directory/'observer.pdf');plt.close(fig)
