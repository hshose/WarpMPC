"""Validate saved closed-loop evidence without rerunning physical simulation."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from juggling.disturbances import wind_acceleration


def audit(directory):
    directory=Path(directory);cfg=json.loads((directory/'config.json').read_text());m=json.loads((directory/'metrics.json').read_text())
    p=np.load(directory/'plant.npz');a=np.load(directory/'arrays.npz');f=np.load(directory/'frames.npz')
    it=[json.loads(s) for s in (directory/'iterations.jsonl').read_text().splitlines()]
    dt=cfg['sim']['dt'];period=cfg['mpc']['control_dt'];t=p['t'];x=p['x'];errors={}
    assert dt==.0005 and cfg['mpc']['sqp_iters']==1 and cfg['mpc']['osqp_iters']==50
    errors['plant_clock_s']=float(np.max(abs(t-np.arange(len(t))*dt)))
    assert errors['plant_clock_s']<1e-10
    errors['engine_clock_s']=abs(m['engine_final_time_s']-t[-1]);assert errors['engine_clock_s']<1e-8
    assert len(it)==len(a['solve_ms'])==len(a['tau_target'])==len(a['t'])-1
    errors['control_clock_s']=float(np.max(abs(np.array([r['t'] for r in it])-a['t'][:-1])))
    assert errors['control_clock_s']<1e-10
    observer_result=None
    if 'observer' in cfg:
        from .audit_observer import audit_observer
        observer_result=audit_observer(directory,cfg,p,a)
        errors['measured_joint_state']=float(np.max(abs(a['x'][:,6:20]-x[np.rint(a['t']/dt).astype(int),6:20])))
        assert errors['measured_joint_state']==0
    else:
        errors['measured_control_state']=float(np.max(abs(a['x']-x[np.rint(a['t']/dt).astype(int)])))
        assert errors['measured_control_state']==0
    assert all(len(r['iterations'])==1 and r['iterations'][0]['iter']==50 for r in it)
    assert all(r['warm_dual_nonzeros']>0 for r in it[1:])
    expected_wind=np.array([wind_acceleration(v+.5*dt,cfg['arguments']['wind']) for v in t[:-1]])
    errors['wind_acceleration_m_s2']=float(np.max(abs(expected_wind-p['wind'])))
    assert errors['wind_acceleration_m_s2']<1e-12
    ci=p['control_index'];age=t[:-1]-a['t'][ci];targets=a['tau_target'];starts=np.vstack([p['tau'][0]-.5*dt*p['torque_rate'][0],targets[:-1]])
    slope=(targets-starts)/period
    expected_tau=starts[ci]+np.minimum((age+.5*dt)/period,1.)[:,None]*(targets-starts)[ci]
    expected_rate=slope[ci]*((age<period-dt/2)[:,None])
    errors['torque_ramp_Nm']=float(np.max(abs(expected_tau-p['tau'])))
    errors['torque_rate_Nm_s']=float(np.max(abs(expected_rate-p['torque_rate'])))
    assert errors['torque_ramp_Nm']<1e-9 and errors['torque_rate_Nm_s']<1e-9
    frame_idx=np.rint(f['t']/dt).astype(int);origin=f['qpos'][0,7:10]-x[0,:3]
    errors['rendered_joint_state_rad']=float(np.max(abs(f['qpos'][:,:7]-x[frame_idx,6:13])))
    errors['rendered_ball_position_m']=float(np.max(abs(f['qpos'][:,7:10]-origin-x[frame_idx,:3])))
    assert errors['rendered_joint_state_rad']<3e-7 and errors['rendered_ball_position_m']<3e-7
    assert np.max(abs(np.linalg.norm(f['qpos'][:,10:14],axis=1)-1))<1e-10
    # Integrate logged world spin to compare against independent native attitude.
    attitude=Rotation.from_quat(f['qpos'][0,10:14],scalar_first=True);previous=np.zeros(3);j=1;angle_error=0.
    for i,omega in enumerate(p['spin']):
        attitude=Rotation.from_rotvec(.5*dt*(previous+omega))*attitude;previous=omega
        if j<len(frame_idx) and i+1==frame_idx[j]:
            actual=Rotation.from_quat(f['qpos'][j,10:14],scalar_first=True)
            angle_error=max(angle_error,float((actual.inv()*attitude).magnitude()));j+=1
    errors['spin_integral_orientation_rad']=angle_error;assert angle_error<.03,angle_error
    if x.shape[1]==23:assert np.max(abs(x[1:,20:23]-p['spin']))==0
    peaks=np.flatnonzero((x[:-1,5]>0)&(x[1:,5]<=0));pi=peaks+(x[peaks+1,2]>x[peaks,2]);apices=x[pi,2]-.02
    assert np.max(abs(apices-np.array(m['measured_apices_m'])))<1e-12
    ratios=dict(velocity=float(np.max(abs(x[:,13:20])/cfg['limits']['velocity'])))
    for key,limit in [('tau','torque'),('acceleration','acceleration'),('torque_rate','torque_rate'),('jerk','jerk')]:
        ratios[limit]=float(np.max(abs(p[key])/cfg['limits'][limit]))
    assert max(abs(ratios[k]-m['limit_ratios'][k]) for k in ratios)<1e-12
    intervals=np.diff(a['t']);extended=int(np.count_nonzero(intervals>period+dt/2))
    assert extended==m['extended_update_intervals'];assert max(intervals)<=period+.005+1e-10
    other=json.loads((directory/'other_contacts.json').read_text());assert len(other)==m['other_contact_samples']
    task_pass=bool(m['stop_reason']=='duration completed' and t[-1]>=cfg['arguments']['seconds']-dt/2
        and len(apices)>=max(1,int(cfg['arguments']['seconds']*.8)) and np.all((apices>=.5)&(apices<=1.))
        and np.all(x[:,6:13]>=np.array(cfg['limits']['joint_lower'])-1e-6)
        and np.all(x[:,6:13]<=np.array(cfg['limits']['joint_upper'])+1e-6)
        and max(ratios.values())<=1.0001 and m['max_torque_clipping_Nm']<1e-6 and not other)
    assert task_pass==m['task_pass']
    for name in ['report.pdf','contact.pdf']+(['observer.pdf'] if observer_result else []):
        info=subprocess.check_output(['pdfinfo',str(directory/name)],text=True)
        assert any(line.startswith('Pages:') and line.split()[-1]=='1' for line in info.splitlines())
    cpp_result=None
    if 'native_controller' in cfg:
        native=cfg['native_controller'];v=a['cpp_total_ms']
        assert np.all(np.isfinite(v)) and np.all(v>0)
        assert np.max(abs(v-np.array([row['cpp_total_ms'] for row in it])))==0
        if 'evaluator_threads' in native:
            assert all(row['cpp_evaluator_threads']==native['evaluator_threads'] for row in it)
        for name,expected in native['binaries'].items():
            assert hashlib.sha256((directory/'native'/Path(name).name).read_bytes()).hexdigest()==expected
        for name,values in [('all_calls',v),('after_first_call',v[1:])]:
            stats=dict(count=len(values),median=float(np.median(values)),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))
            assert stats==m['cpp_timing_ms'][name]
        cpp_result=dict(timer_and_statistics_verified=True,binary_hashes_verified=True,evaluator_threads=native.get('evaluator_threads',1))
    video=None
    if (directory/'video.mp4').exists():
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,r_frame_rate,nb_frames','-of','json',str(directory/'video.mp4')],text=True))['streams'][0]
        assert int(video['nb_frames'])==len(f['t']) and video['r_frame_rate']=='50/1'
    result=dict(run=directory.name,verified=True,errors=errors,task_pass=task_pass,plant_steps=len(t)-1,
        controller_updates=len(it),nominal_update_s=period,extended_intervals=extended,maximum_update_s=float(max(intervals)),
        one_sqp_exactly_50_osqp=True,feedback_mode='vicon' if observer_result else 'exact',observer=observer_result,wind_verified=True,torque_ramp_verified=True,
        native_and_logged_simulation_clocks_agree=True,rendered_states_match_trace=True,cpp=cpp_result,video=video)
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('directories',nargs='+');args=p.parse_args()
    for directory in args.directories:audit(directory)


if __name__=='__main__':main()
