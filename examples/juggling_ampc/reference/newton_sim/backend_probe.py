"""Compare Newton CPU and GPU execution for one articulated scene."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from .scene import Scene,SceneConfig


def main():
    directory=Path('runs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_newton_backend_probe');directory.mkdir()
    rows=[];traces={}
    for device in ['cpu','cuda:0']:
        cfg=SceneConfig(device=device,visuals=False);start=perf_counter();s=Scene(cfg,directory/device.replace(':','_'));build=perf_counter()-start
        x=s.observation();tau=np.asarray(s.robot.bias(x[6:13],x[13:20])).ravel();xs=[x];elapsed=[]
        for k in range(200):
            start=perf_counter();x=s.step(tau);elapsed.append(1000*(perf_counter()-start));xs.append(x)
        traces[device]=np.array(xs);traces[device+'_ms']=np.array(elapsed)
        rows.append(dict(device=device,config=asdict(cfg),build_s=build,first_step_ms=elapsed[0],
                         median_step_ms=float(np.median(elapsed[10:])),p95_step_ms=float(np.percentile(elapsed[10:],95))))
        print(rows[-1],flush=True)
    difference=np.max(abs(traces['cpu']-traces['cuda:0']),axis=0).tolist()
    (directory/'results.json').write_text(json.dumps(dict(backends=rows,max_absolute_state_difference=difference),indent=2)+'\n')
    np.savez_compressed(directory/'traces.npz',**traces)
    from arm.reporting import plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for device in ['cpu','cuda:0']:
        axes[0].plot(np.arange(201)*.0005,traces[device][:,2],label=device)
        axes[1].semilogy(np.arange(200)*.0005,traces[device+'_ms'],label=device)
    axes[2].semilogy(np.maximum(difference,1e-12),marker='.')
    axes[0].set(xlabel='Time [s]',ylabel='Ball height [m]');axes[0].legend()
    axes[1].set(xlabel='Time [s]',ylabel='Newton step including transfer [ms]');axes[1].legend()
    axes[2].set(xlabel='Observed-state component',ylabel='Max CPU/GPU difference [SI units]')
    fig.suptitle('Newton one-arm CPU/GPU probe · 0.5 ms steps · gravity hold and free flight\nGPU timings include host/device observation/control transfers; no MPC or contacts in this short check.')
    fig.savefig(directory/'report.pdf');plt.close(fig)
    with Path('EXPERIMENTS.md').open('a') as f:f.write(f'\n## {datetime.now(timezone.utc).isoformat()} — Newton CPU/GPU probe\n\n[PDF]({directory}/report.pdf), [results]({directory}/results.json). Single scene; cold JIT separated from warmed steps.\n')
    print(directory,flush=True)

if __name__=='__main__':main()
