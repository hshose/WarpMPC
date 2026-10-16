"""Summarize a completed moving-MPC rehearsal; reads files only."""
import argparse,json
from pathlib import Path
import numpy as np
from arm.model import Panda

def analyze(path):
    read=lambda n:json.loads((path/n).read_text())
    plans=[json.loads(l) for l in (path/'mpc.jsonl').read_text().splitlines()]
    if not plans:raise ValueError('No MPC plans in recording')
    rows=[]
    with (path/'robot/robot.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['receive_time']>=plans[0]['time']:rows.append(r)
    active=[r for r in rows if r['mpc'] and r.get('reference_valid',False)]
    spec=read('model_metadata.json')['spec'];robot=Panda(assembly=spec['assembly'],**spec['geometry'])
    def stats(x):
        x=np.asarray(x);return dict(count=len(x),mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x))) if len(x) else None
    joint=[];velocity=[];paddle=[]
    for r in active:
        ref=np.array(r['reference_q_dq_ddq']);q=np.array(r['q']);dq=np.array(r['dq'])
        joint.append(np.linalg.norm(ref[:7]-q));velocity.append(np.linalg.norm(ref[7:14]-dq))
        paddle.append(np.linalg.norm(np.asarray(robot.contact_fk(ref[:7])[0])-np.asarray(robot.contact_fk(q)[0])))
    rates=np.array([r['guard_filtered_torque_rate'] for r in rows]);i,j=np.unravel_index(np.argmax(abs(rates)),rates.shape)
    commands=[json.loads(l) for l in (path/'robot_stream_commands.jsonl').read_text().splitlines()]
    send_ages=[1000*(r['send_time']-float(r['command'].split()[1])) for r in commands if r['command'].startswith('plan ')]
    result=dict(mode='simulated_ball_real_arm_rehearsal',physical_ball_should_remain_held=True,native_session=read('robot/session.json'),
        failure=read('failure.json') if (path/'failure.json').exists() else None,plans=len(plans),
        native_mpc_ms=stats([r['metrics'][0] for r in plans]),source_to_send_ms=stats(send_ages),
        callback_plan_age_ms=stats([1000*r['mpc_plan_age_s'] for r in active]),
        joint_position_error_norm_rad=stats(joint),joint_velocity_error_norm_rad_s=stats(velocity),paddle_position_error_m=stats(paddle),
        peak_filtered_torque_rate=dict(joint=int(j)+1,value_Nm_s=float(rates[i,j]),time_from_first_plan_s=rows[i]['receive_time']-plans[0]['time'],limit_Nm_s=read('native_description.json').get('measured_torque_rate_guard_Nm_s',300.) if (path/'native_description.json').exists() else 300.),
        physical_ball_impacts_expected=0)
    (path/'rehearsal_analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recording',type=Path);a=p.parse_args();print(json.dumps(analyze(a.recording),indent=2))
