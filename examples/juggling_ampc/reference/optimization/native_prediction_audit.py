"""Evaluate every original nonlinear constraint on saved native-variant plans.

This audit uses the archived unpruned Stage 4 factor library and original bounds,
then independently applies the variant's prescribed timestep/clock bounds. It
reports violations rather than treating a fixed-budget RTI plan as feasible.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import casadi as ca
import numpy as np

BASELINE = Path('runs/20260924T175306.001124Z_cpp_final_wind')
BOUND_GROUPS = {'ball_position':(0,3),'ball_velocity':(3,6),'joint_position':(6,13),
                'joint_velocity':(13,20),'spin':(20,23),'torque':(23,30),
                'normal_impulse':(30,31),'contact_fraction':(31,32),
                'fixed_timestep':(32,33),'clock':(33,34)}
EQUALITIES = {'ball','spin','position','dynamics','clock'}
OMISSION_CANDIDATES = {'workspace','tilt','upward','paddle_velocity','jerk'}
THRESHOLDS = (0.,1e-3,.01,.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slack_summary(values):
    flat=np.asarray(values).reshape(-1)
    if not len(flat):return {'component_count':0}
    return dict(component_count=len(flat),minimum_signed_margin=float(flat.min()),
                maximum_violation=float(max(0.,-flat.min())),
                violated_components_1e_8=int(np.count_nonzero(flat < -1e-8)),
                near_bound_counts={f'{t:g}':int(np.count_nonzero(flat<=t)) for t in THRESHOLDS})


def audit(path,baseline=BASELINE):
    path=Path(path);baseline=Path(baseline)
    config=json.loads((path/'config.json').read_text())
    metadata=json.loads((path/'native/factor_metadata.json').read_text())
    assert config['native_variant']==metadata
    bm=json.loads((baseline/'native/factor_metadata.json').read_text())
    # Only discretization/runner fields may differ: material, physical and cost
    # parameters must match the archived numerical factor used below.
    irrelevant={'horizon','h','control_dt','sim_dt','sqp_iters','osqp_iters','codegen','codegen_optimization'}
    for key,value in bm['config'].items():
        if key not in irrelevant:assert metadata['config'][key]==value,(key,value,metadata['config'][key])
    ba=bm['arrays'];ma=metadata['arrays'];scale=np.array(ba['scale']);offset=np.array(ba['offset'])
    assert ma['scale']==ba['scale'] and ma['offset']==ba['offset']
    assert metadata['groups']==bm['groups']
    library=baseline/'native/libjuggling_factors.so'
    factor=ca.external('stage_value',str(library.resolve()))
    Y=np.load(path/'predictions.npz')['Y'];Z=(Y-offset)/scale
    count,nodes,dim=Y.shape;N=nodes-1
    assert N==metadata['config']['horizon'] and dim==34
    steps=np.array(metadata['stage_steps']);times=np.array(metadata['stage_times'])
    assert len(steps)==N and len(times)==N+1
    assert np.all(steps>0) and np.allclose(np.diff(times),steps,rtol=0,atol=1e-14)
    assert abs(times[-1]-1.2)<1e-12 and abs(steps[0]-.03)<1e-14
    assert np.allclose(steps,.03*metadata['mesh_growth']**np.arange(N),rtol=0,atol=1e-14)
    C=np.array([np.array(factor(np.r_[z[k],z[k+1]],np.zeros(30))[1]).ravel()
                for z in Z for k in range(N)]).reshape(count,N,-1)
    lower,upper=np.array(ba['stage_lower']),np.array(ba['stage_upper'])
    margin=np.minimum(C-lower,upper-C)
    groups=metadata['groups'];ids=np.array(metadata['constraint_ids']);offsets=metadata['stage_offsets']
    retained=np.zeros((N,len(lower)),dtype=bool)
    cl=np.array(ma['constraint_lower']);cu=np.array(ma['constraint_upper'])
    endpoint_nodes=set();endpoint=groups['endpoint_acceleration'];endpoint_representatives={}
    for k in range(N):
        rows=ids[offsets[k]:offsets[k+1]]
        assert np.all((rows>=0)&(rows<len(lower))) and len(np.unique(rows))==len(rows)
        retained[k,rows]=True
        assert np.array_equal(cl[offsets[k]:offsets[k+1]],lower[rows])
        assert np.array_equal(cu[offsets[k]:offsets[k+1]],upper[rows])
        endpoint_nodes.update(k+(int(r)-endpoint[0])//7 for r in rows if endpoint[0]<=r<endpoint[1])
        for row in rows:
            if endpoint[0]<=row<endpoint[1]:
                endpoint_representatives[(k+(int(row)-endpoint[0])//7,(int(row)-endpoint[0])%7)]=(k,int(row))
    assert endpoint_nodes==set(range(N+1)),endpoint_nodes
    assert set(endpoint_representatives)=={(k,j) for k in range(N+1) for j in range(7)}
    duplicate_error=0.
    for k in range(N):
        for row in range(*endpoint):
            if not retained[k,row]:
                ref_k,ref_row=endpoint_representatives[(k+(row-endpoint[0])//7,(row-endpoint[0])%7)]
                duplicate_error=max(duplicate_error,float(np.max(abs(C[:,k,row]-C[:,ref_k,ref_row]))))
    assert duplicate_error<1e-13,duplicate_error
    for name in EQUALITIES|{'contact','acceleration','torque_rate'}:
        start,end=groups[name];assert retained[:,start:end].all(),name
    prefix=metadata['spec']['prefix']
    for name in metadata['spec']['drop']:
        start,end=groups[name]
        assert retained[:prefix,start:end].all() and not retained[prefix:,start:end].any(),name
    lows=np.tile(ba['lower'],(N+1,1));highs=np.tile(ba['upper'],(N+1,1))
    lows[:,32]=highs[:,32]=np.r_[steps,steps[-1]]/scale[32];highs[:,33]=times[-1]/scale[33]
    assert np.allclose(lows,np.array(ma['all_lower']).reshape(N+1,34),rtol=0,atol=1e-13)
    assert np.allclose(highs,np.array(ma['all_upper']).reshape(N+1,34),rtol=0,atol=1e-13)
    B=np.minimum(Z-lows,highs-Z)
    normalized_width=.5*(highs-lows);normalized_width[normalized_width==0]=1
    BM=B/normalized_width
    all_groups={}
    for name,(start,end) in groups.items():
        mask=retained[:,start:end]
        entry=dict(kind='equality' if name in EQUALITIES else 'inequality',
                   rows_per_stage=end-start,retained_rows=int(mask.sum()),omitted_rows=int((~mask).sum()),
                   all=slack_summary(margin[:,:,start:end]),
                   retained=slack_summary(margin[:,:,start:end][:,mask]),
                   omitted=slack_summary(margin[:,:,start:end][:,~mask]),
                   per_stage=[dict(stage=k,time_s=float(times[k]),retained_components=int(mask[k].sum()),
                                    **slack_summary(margin[:,k,start:end])) for k in range(N)])
        if name in EQUALITIES:entry['maximum_absolute_scaled_defect']=float(np.max(abs(C[:,:,start:end])))
        all_groups[name]=entry
    original_bounds={name:dict(scaled=slack_summary(B[:,:,a:b]),half_width_normalized=slack_summary(BM[:,:,a:b]))
                     for name,(a,b) in BOUND_GROUPS.items()}
    plant=np.load(path/'plant.npz');X=plant['x'];q=X[:,6:13].T;dq=X[:,13:20].T
    fk=ca.external('panda_fk',str(library.resolve()))
    # Map the compiled function once over measured joint states; geometry is
    # evaluation only, never fed back into the controller or observer.
    qq=ca.MX.sym('q',7);vv=ca.MX.sym('v',7);pos,normal,J,_,_=fk(qq)
    observed=ca.Function('audit_geometry',[qq,vv],[normal,J@vv])
    normals,velocity=observed.map(q.shape[1])(q,dq)
    normals=np.array(normals).T;velocity=np.array(velocity).T
    workspace=plant['paddle_pos']/np.array([.4,.4,.18])
    measured=dict(workspace_maximum_component_ratio=float(np.max(abs(workspace))),
                  tilt_maximum_component_ratio=float(np.max(abs(normals[:,:2])/np.sin(.4))),
                  upward_minimum_normal_z=float(normals[:,2].min()),
                  upward_lower_bound=float(np.cos(.4)),
                  paddle_velocity_maximum_component_ratio=float(np.max(abs(velocity))/1.2),
                  paddle_speed_maximum_m_s=float(np.max(plant['paddle_speed'])),
                  velocity_limit_component_m_s=1.2)
    measured['all_geometry_constraints_within_original_bounds']=bool(
        measured['workspace_maximum_component_ratio']<=1+1e-8 and
        measured['tilt_maximum_component_ratio']<=1+1e-8 and
        measured['upward_minimum_normal_z']>=np.cos(.4)-1e-8 and
        measured['paddle_velocity_maximum_component_ratio']<=1+1e-8)
    metrics=json.loads((path/'metrics.json').read_text())
    measured['actuator_limit_ratios']=metrics['limit_ratios']
    measured['actuator_jerk_within_original_limit']=bool(metrics['limit_ratios']['jerk']<=1+1e-8)
    arrays=np.load(path/'arrays.npz');capture_indices=np.arange(count)*10
    capture_times=arrays['t'][capture_indices]
    omitted_safety_names=sorted(OMISSION_CANDIDATES & set(metadata['spec']['drop']))
    omitted_safety_pass=all(all_groups[name]['omitted']['maximum_violation']<=1e-8 for name in omitted_safety_names)
    bounds_physical_pass=all(original_bounds[name]['scaled']['maximum_violation']<=1e-8 for name in
                             ['ball_position','ball_velocity','joint_position','joint_velocity','spin','torque'])
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),run=str(path),baseline_reference=str(baseline),
        inputs_sha256={name:sha(path/name) for name in ['config.json','predictions.npz','plant.npz','metrics.json','native/factor_metadata.json']},
        original_factor_sha256=sha(library),audit_script_sha256=sha(__file__),
        definitions=dict(activity='Signed nonlinear margin includes violations; no KKT multipliers available.',
            constraint_units='Original Stage 4 scaled constraint units; no additional upward rescaling.',
            bound_units='Scaled variable units plus margin normalized by half the physical allowed interval.',
            sampling='Every tenth accepted control plan is saved, about 0.2 s apart. Unsaved solves and times between shooting nodes are not covered.'),
        saved_plans=count,total_controller_updates=int(metrics['controller_updates']),stage_samples=count*N,
        saved_plan_capture_time_range_s=[float(capture_times[0]),float(capture_times[-1])],
        mesh=dict(nodes=nodes,stages=N,growth=metadata['mesh_growth'],steps_s=steps.tolist(),times_s=times.tolist(),
            prescribed_horizon_s=float(times[-1]),prescribed_horizon_error_s=float(abs(times[-1]-1.2)),
            first_step_s=float(steps[0]),last_step_s=float(steps[-1]),full_constraint_prefix_stages=prefix,
            full_constraint_prefix_duration_s=float(times[prefix]),
            maximum_optimized_step_error_s=float(np.max(abs(Y[:,:,32]-np.r_[steps,steps[-1]]))),
            maximum_optimized_clock_error_s=float(np.max(abs(Y[:,:,33]-times))),
            maximum_optimized_sum_steps_error_s=float(np.max(abs(Y[:,:-1,32].sum(axis=1)-1.2)))),
        structure=dict(endpoint_nodes_covered=sorted(endpoint_nodes),all_endpoint_nodes_covered=True,
            distinct_node_joint_endpoint_bounds_covered=len(endpoint_representatives),
            omitted_endpoint_row_maximum_difference_from_retained_equivalent=duplicate_error,
            dynamics_contact_acceleration_rate_rows_preserved=True,retained_bounds_match_original=True,
            stage_variable_bounds_match_original_with_prescribed_mesh=True,
            retained_stage_constraint_rows=int(retained.sum()),omitted_stage_constraint_rows=int((~retained).sum())),
        original_constraint_groups=all_groups,original_variable_bounds=original_bounds,measured_plant=measured,
        verdict=dict(omitted_geometry_and_jerk_within_original_bounds_at_saved_nodes=omitted_safety_pass,
            physical_position_velocity_spin_torque_bounds_at_saved_nodes=bounds_physical_pass,
            measured_geometry_and_actuator_jerk_within_original_bounds=bool(measured['all_geometry_constraints_within_original_bounds'] and measured['actuator_jerk_within_original_limit']),
            exact_nonlinear_feasibility_claimed=False,
            note='The omission screen can pass while retained constraints, fixed h/clock, or contact-variable bounds have small RTI residuals. Values and defects are reported explicitly.'))
    output=path/'native_prediction_audit.json';output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+');p.add_argument('--baseline',type=Path,default=BASELINE);args=p.parse_args()
    for directory in args.runs:
        r=audit(directory,args.baseline)
        print(json.dumps(dict(run=r['run'],verdict=r['verdict'],mesh=r['mesh'],
            omitted_groups={n:{'minimum_signed_margin':r['original_constraint_groups'][n]['omitted']['minimum_signed_margin'],
                'maximum_violation':r['original_constraint_groups'][n]['omitted']['maximum_violation']} for n in OMISSION_CANDIDATES},
            other_inequality_violations={n:g['all']['maximum_violation'] for n,g in r['original_constraint_groups'].items() if g['kind']=='inequality'}),indent=2))


if __name__=='__main__':main()
