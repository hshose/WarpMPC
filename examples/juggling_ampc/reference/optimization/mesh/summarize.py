"""Write a compact comparison of mesh closed-loop experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .formulation import Config,grid

def main():
    rows=[]
    for p in sorted(Path('runs').glob('*mesh*/metrics.json')):
        m=json.loads(p.read_text());c=json.loads((p.parent/'config.json').read_text())
        if c.get('controller_formulation')!='optimization.mesh.formulation':continue
        ap=np.array(m['measured_apices_m']);cfg=Config(**c['mpc']);h,t,g=grid(cfg)
        row=dict(run=str(p.parent),nodes=cfg.horizon,growth=g,first_dt_s=h[0],last_dt_s=h[-1],horizon_s=t[-1],
            wind=c['arguments']['wind'],damping=c['arguments']['damping'],duration_s=m['duration_s'],task_pass=m['task_pass'],
            contacts=m['contacts'],apex_min=float(min(ap)),apex_max=float(max(ap)),apex_mean=float(np.mean(ap)),
            apex_rmse=float(np.sqrt(np.mean((ap-.75)**2))),limit_ratios=m['limit_ratios'],
            max_lateral_distance_m=m['max_lateral_distance_m'],clipping_Nm=m['max_torque_clipping_Nm'],
            solve_median_ms=m['solve_median_ms'],qp_median_ms=m['qp_median_ms'])
        rows.append(row)
        from arm.model import Panda
        from .reporting import report
        report(p.parent,np.load(p.parent/'arrays.npz'),np.load(p.parent/'plant.npz'),c,m,Panda())
    out=Path('optimization/mesh');(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for n in sorted({r['nodes'] for r in rows}):
        h,t,g=grid(Config(horizon=n));axes[0,0].plot(t[:-1],1000*h,'.-',label=f'{n} nodes; growth={g:.5f}')
    axes[0,0].set(xlabel='Prediction time [s]',ylabel='Interval [ms]',title='Fixed 1.2-second prediction horizon');axes[0,0].legend()
    long=[r for r in rows if r['duration_s']>=19.99]
    for r in long:
        m=json.loads((Path(r['run'])/'metrics.json').read_text())
        axes[0,1].plot(m['measured_apex_times_s'],m['measured_apices_m'],'.-',label=f"{r['nodes']} nodes {r['wind']} d={r['damping']}")
    axes[0,1].axhline(.75,color='k',ls='--');axes[0,1].set(xlabel='Simulation time [s]',ylabel='Apex [m]',title='Measured bounce apex');axes[0,1].legend(fontsize=7)
    keys=['velocity','torque','acceleration','torque_rate','jerk']
    for r in long:axes[1,0].plot(keys,[r['limit_ratios'][k] for k in keys],'.-',label=f"{r['nodes']} {r['wind']} d={r['damping']}")
    axes[1,0].axhline(1,color='r',ls='--');axes[1,0].set(ylabel='Maximum / limit',title='Measured continuous-plant limits');axes[1,0].tick_params(axis='x',rotation=20)
    axes[1,1].axis('off');lines=['Nodes  Seconds  Pass  Bounces  Apex RMSE  QP median (Python)']
    for r in rows:lines.append(f"{r['nodes']:5d} {r['duration_s']:8.1f} {str(r['task_pass']):5s} {r['contacts']:8d} {r['apex_rmse']*1000:9.1f}mm {r['qp_median_ms']:8.2f}ms")
    axes[1,1].text(0,1,'\n'.join(lines),va='top',family='monospace',fontsize=8)
    fig.suptitle('Nonuniform-grid Newton screening; 1 SQP / 50 OSQP; position-only observer')
    fig.tight_layout();fig.savefig(out/'report.pdf');plt.close(fig)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
