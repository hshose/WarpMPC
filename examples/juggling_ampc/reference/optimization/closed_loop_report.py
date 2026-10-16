"""Compare the actual native Newton trials with their Stage 4 baselines."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAIRS = {
 'nominal': ('runs/20260924T171540.076412Z_cpp_fixed_nominal',
             'runs/20260925T102052.776111Z_native_mesh24_prune_nominal'),
 'wind': ('runs/20260924T175306.001124Z_cpp_final_wind',
          'runs/20260925T102022.721751Z_native_mesh24_prune_wind'),
 'wind + damping': ('runs/20260924T171540.090515Z_cpp_fixed_damped',
                    'runs/20260925T103132.682961Z_native_mesh24_prune_damped_paired'),
}
EXTRA_DAMPED='runs/20260925T102053.966390Z_native_mesh24_prune_damped'
STRESS='runs/20260925T102239.471296Z_native_mesh24_prune_sensorstress'
RATIOS=['velocity','torque','acceleration','torque_rate','jerk']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(path,case,controller):
    path=Path(path);m=json.loads((path/'metrics.json').read_text());c=json.loads((path/'config.json').read_text())
    assert c['arguments']['controller']=='cpp'
    assert m['sqp_iterations_observed']==[1] and m['qp_iterations_observed']==[50]
    assert c['sim']['dt']==.0005 and c['mpc']['control_dt']==.02
    apex=np.array(m['measured_apices_m']);target=c['mpc']['apex']
    # Independently recover peaks from native physics samples using the runner's
    # velocity sign crossing and larger adjacent height convention.
    plant=np.load(path/'plant.npz');X=plant['x']
    crossings=np.flatnonzero((X[:-1,5]>0)&(X[1:,5]<=0))
    peak_indices=crossings+(X[crossings+1,2]>X[crossings,2])
    reconstructed=X[peak_indices,2]-c['mpc']['radius']
    assert np.array_equal(reconstructed,apex)
    lateral=float(np.max(np.linalg.norm(X[:,:2],axis=1)))
    assert abs(lateral-m['max_lateral_distance_m'])<1e-14
    output=dict(case=case,controller=controller,run=str(path),task_pass=m['task_pass'],
        duration_s=m['duration_s'],impacts=m['contacts'],stop_reason=m['stop_reason'],
        apex_count=len(apex),apex_target_m=target,apex_rmse_m=float(np.sqrt(np.mean((apex-target)**2))),
        apex_min_m=float(apex.min()),apex_max_m=float(apex.max()),
        apex_mean_m=float(apex.mean()),apex_mean_error_m=float(np.mean(apex-target)),
        apex_max_absolute_error_m=float(np.max(abs(apex-target))),
        measured_apices_m=apex.tolist(),measured_apex_times_s=m['measured_apex_times_s'],
        max_lateral_distance_m=lateral,actuator_limit_ratios=m['limit_ratios'],
        joint_position_limits_ok=m['joint_position_limits_ok'],other_contact_samples=m['other_contact_samples'],
        max_torque_clipping_Nm=m['max_torque_clipping_Nm'],
        native_impact_inclusive_acceleration_maxima=m['max_total_acceleration_rad_s2'],
        native_impact_inclusive_jerk_maxima=m['max_total_jerk_rad_s3'],
        wind=c['arguments']['wind'],synthetic_joint_damping=c['arguments']['damping'],
        observer_sensor=c['vicon'],position_only_feedback=c['arguments']['feedback']=='vicon',
        mpc_stages=c['mpc']['horizon'],simulation_dt_s=c['sim']['dt'],control_dt_s=c['mpc']['control_dt'],
        computation_delay_modeled=c['computation_delay_modeled'],
        native_evaluator_threads=c['native_controller'].get('evaluator_threads',1),
        hashes={name:digest(path/name) for name in ['config.json','metrics.json','plant.npz']})
    if 'native_variant' in c:
        output['native_variant']=dict(spec=c['native_variant']['spec'],
            horizon_s=c['native_variant']['stage_times'][-1],growth=c['native_variant']['mesh_growth'],
            first_step_s=c['native_variant']['stage_steps'][0],last_step_s=c['native_variant']['stage_steps'][-1])
        audit=path/'native_prediction_audit.json'
        if audit.exists():output['full_original_constraint_audit']=str(audit)
    return output


def results():
    pairs={};rows=[]
    for case,paths in PAIRS.items():
        baseline=load_run(paths[0],case,'baseline 40');candidate=load_run(paths[1],case,'native 24 + pruning')
        assert baseline['duration_s']==candidate['duration_s']==20.
        for key in ['wind','synthetic_joint_damping','simulation_dt_s','control_dt_s']:
            assert baseline[key]==candidate[key],(case,key)
        for sensor_key in ['rate_hz','latency_s','position_std_m','dropout_probability','seed']:
            assert baseline['observer_sensor'][sensor_key]==candidate['observer_sensor'][sensor_key]
        differences={key:candidate[key]-baseline[key] for key in [
            'apex_rmse_m','apex_min_m','apex_max_m','apex_max_absolute_error_m','max_lateral_distance_m']}
        differences['actuator_limit_ratios']={key:candidate['actuator_limit_ratios'][key]-baseline['actuator_limit_ratios'][key] for key in RATIOS}
        pairs[case]=dict(baseline=baseline,candidate=candidate,candidate_minus_baseline=differences,
            sensor_noise_seed_matches=baseline['observer_sensor']['seed']==candidate['observer_sensor']['seed'],
            comparison_caveat='Physical settings and measurement-noise seed match; one run per controller.')
        rows.extend([baseline,candidate])
    stress=load_run(STRESS,'sensor stress (separate)','native 24 + pruning')
    assert stress['duration_s']==10.
    extra=load_run(EXTRA_DAMPED,'additional damped seed7','native 24 + pruning')
    changes=[f'{case}: apex RMSE delta {1000*pair["candidate_minus_baseline"]["apex_rmse_m"]:+.3f} mm; lateral maximum delta {1000*pair["candidate_minus_baseline"]["max_lateral_distance_m"]:+.3f} mm; worst absolute apex error delta {1000*pair["candidate_minus_baseline"]["apex_max_absolute_error_m"]:+.3f} mm.' for case,pair in pairs.items()]
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        scope='Actual compiled C++ controllers in Newton; no Python prototype results included.',
        definitions=dict(apex_rmse='Root mean square of all measured post-bounce apices minus 0.75 m; includes startup bounces.',
            apex_reference='Ball center height minus ball radius, above the nominal paddle strike plane.',
            lateral='Maximum Euclidean horizontal distance of the ball center from the target axis over all physics samples.',
            actuator_ratios='Actuated motion metrics from the runner; native collision-inclusive acceleration/jerk spikes are retained separately.',
            task_pass='Existing run task acceptance flag; passing does not imply identical tracking performance or exact nonlinear MPC feasibility.'),
        shared_settings=dict(newton_timestep_s=.0005,sqp_iterations=1,osqp_iterations=50,nominal_control_period_s=.02,
            simulation_clock='Newton pauses while MPC solves; these closed-loop results do not model computation latency.'),
        paired_20s_cases=pairs,separate_10s_sensor_stress=stress,additional_20s_damped_seed7=extra,
        observations=changes+[
            'All six paired native Newton runs passed 20 s with 26 impacts; the separate stress run passed 10 s with 13 impacts.',
            'Nominal apex range widened from 0.697-0.774 m to 0.684-0.795 m. Wind range widened from 0.704-0.782 m to 0.698-0.791 m.',
            'Actuator ratios and tracking differences are listed without an identical-performance claim.',
            'Physical settings and sensor seeds match in each pair: nominal/wind seed7, damped seed19. The earlier candidate damped seed7 trial is retained as an additional case.',
            'One deterministic run per paired case is evidence of retained juggling, with modest tracking differences; it is not a statistical equivalence or robustness guarantee.',
            'Sensor stress is separately scoped: 100 Hz positions, 10 ms latency, 2 mm per-axis noise, 10% dropout, seed11; no paired 20 s comparison is implied.'
        ],script_sha256=digest(__file__))


def plot(data,path):
    colors=['#497ba7','#ee8747'];labels=['Baseline: 40 stages','Native: 24 + pruning']
    pairs=data['paired_20s_cases'];cases=list(pairs);rows=[]
    for pair in pairs.values():rows.extend([pair['baseline'],pair['candidate']])
    stress=data['separate_10s_sensor_stress'];display=rows+[stress]
    fig=plt.figure(figsize=(13,10.5),layout='constrained')
    grid=fig.add_gridspec(4,3,height_ratios=[1.55,1.35,1.65,1.65])
    ax=fig.add_subplot(grid[0,:]);ax.axis('off')
    table_rows=[]
    for i,r in enumerate(display):
        short_case=r['case'].replace('wind + damping','damped').replace('sensor stress (separate)','sensor stress')
        method='40' if r['controller']=='baseline 40' else '24 + prune'
        table_rows.append([short_case+' / '+method,f'{r["duration_s"]:.0f}',str(r['impacts']),
            f'{1000*r["apex_rmse_m"]:.2f}',f'{r["apex_min_m"]:.3f}–{r["apex_max_m"]:.3f}',
            f'{1000*r["max_lateral_distance_m"]:.2f}','PASS' if r['task_pass'] else 'FAIL'])
    table=ax.table(cellText=table_rows,colLabels=['Case / native controller','Duration [s]','Impacts','Apex RMSE [mm]','Apex range [m]','Max lateral [mm]','Task'],
        cellLoc='center',loc='center',colWidths=[.23,.10,.08,.16,.16,.17,.10])
    table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.4)
    for (i,j),cell in table.get_celld().items():
        if i==0:cell.set_facecolor('#e6eaf0');cell.set_text_props(weight='bold')
        elif i==7:cell.set_facecolor('#e9f2e9')
        elif i%2==0:cell.set_facecolor('#fff3e9')
    x=np.arange(3);width=.35
    for col,(key,title) in enumerate([('apex_rmse_m','Apex RMSE [mm]'),('max_lateral_distance_m','Maximum lateral excursion [mm]'),('apex_max_absolute_error_m','Worst absolute apex error [mm]')]):
        ax=fig.add_subplot(grid[1,col])
        for j,which in enumerate(['baseline','candidate']):
            vals=[1000*pairs[case][which][key] for case in cases]
            bars=ax.bar(x+(j-.5)*width,vals,width,color=colors[j],label=labels[j])
            ax.bar_label(bars,labels=[f'{v:.1f}' for v in vals],padding=2,fontsize=7)
        ax.set_xticks(x,['Nominal','Wind','Damped']);ax.set_title(title,fontsize=10)
        ax.set_ylim(0,(1.5 if col==0 else 1.2)*max(p.get_height() for p in ax.patches));ax.grid(axis='y',alpha=.2)
        if col==0:ax.legend(fontsize=7,loc='upper left')
    ax=fig.add_subplot(grid[2,:2]);matrix=np.array([[r['actuator_limit_ratios'][k] for k in RATIOS] for r in display])
    ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='YlOrRd')
    ax.set_xticks(range(5),['Velocity','Torque','Acceleration','Torque-rate','Jerk'])
    ax.set_yticks(range(7),['Nominal / 40','Nominal / 24','Wind / 40','Wind / 24','Damped / 40','Damped / 24','Stress / 24'])
    for i in range(7):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.65 else 'black')
    ax.set_title('Measured actuator maxima / limits (1.0 = limit)',fontsize=10)
    ax=fig.add_subplot(grid[2,2]);ax.axis('off')
    notes=['All cases: no unintended contacts,','joint-position violations or torque clipping.','',
           'Tracking differences:',
           '• Lateral maxima increase in nominal/wind.',
           '• Nominal/wind apex ranges widen.',
           f'• Damped RMSE: {1000*pairs["wind + damping"]["baseline"]["apex_rmse_m"]:.2f} → {1000*pairs["wind + damping"]["candidate"]["apex_rmse_m"]:.2f} mm.',
           '', 'Stress is a separate 10 s trial:',
           '100 Hz, 10 ms latency, 2 mm noise,',
           '10% dropout, seed 11.']
    ax.text(0,1,'\n'.join(notes),va='top',fontsize=8.5,linespacing=1.45)
    ax=fig.add_subplot(grid[3,:]);ax.axis('off')
    notes=['Native controller evidence paths (metrics/config/plant hashes retained in closed_loop_results.json):']
    notes.extend(f'{r["case"]} / {"40" if r["controller"]=="baseline 40" else "24"}: {r["run"]}' for r in display)
    notes.append('Additional damped seed7 (20 s / 26 impacts): '+data['additional_20s_damped_seed7']['run'])
    notes.extend(['','All paired cases: 20 s, 0.5 ms Newton, position-only observer, 1 SQP / 50 OSQP; nominal 20 ms updates.',
                  'Prediction horizon remains 1.2 s: baseline 40 × 30 ms; candidate 24 stages, 30–76.95 ms, growth 1.041803.',
                  'Candidate retains full constraints for five stages (163 ms), then prunes inactive groups and duplicate endpoint rows.',
                  'Noise seeds match within pairs: nominal/wind7, damped19. Earlier candidate damped seed7 is retained separately.',
                  'Newton pauses for solves. One run per case; no statistical-equivalence claim. Timing is reported separately.',
                  'Apex RMSE includes every observed bounce; actuator acceleration/jerk exclude separately retained native impact spikes.'])
    ax.text(0,1,'\n'.join(notes),va='top',fontsize=7.8,family='DejaVu Sans',linespacing=1.38)
    fig.suptitle('Closed-loop native C++ comparison — 24-stage mesh and late constraint pruning',fontsize=14)
    fig.savefig(path);fig.savefig(path.with_suffix('.png'),dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-directory',type=Path,default=Path('optimization'));args=parser.parse_args()
    data=results();args.output_directory.mkdir(parents=True,exist_ok=True)
    (args.output_directory/'closed_loop_results.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    plot(data,args.output_directory/'closed_loop_report.pdf')
    for case,pair in data['paired_20s_cases'].items():
        delta=pair['candidate_minus_baseline'];print(case,json.dumps({k:v*1000 for k,v in delta.items() if k!='actuator_limit_ratios'}))
    print('stress apex RMSE [mm]',1000*data['separate_10s_sensor_stress']['apex_rmse_m'])


if __name__=='__main__':main()
