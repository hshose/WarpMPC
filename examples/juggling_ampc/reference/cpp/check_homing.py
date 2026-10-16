"""Check the compiled homing reference before using it on hardware."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'runs/20260930_hardware_preflight/homing_reference'
    output.mkdir(parents=True,exist_ok=True)
    build=root/'build/homing_check';build.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((root/'build/variants/mesh24_prune/factors/metadata.json').read_text())['arrays']
    home=np.array(metadata['home']);lower=np.array(metadata['joint_lower']);upper=np.array(metadata['joint_upper'])
    state=json.loads((output.parent/'robot_state.json').read_text())
    compiler=shutil.which('clang++') or shutil.which('g++')
    source=build/'trace.cpp'
    source.write_text(r'''
#include "joint_homing.hpp"
#include "json_text.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){
 if(argc==2){std::cout<<juggling::json_text("failure\n\"quoted\"\\path\tend");return 0;}
 juggling::JointHoming::Joints start,goal,lo,hi;
 for(auto* v:{&start,&goal,&lo,&hi})for(double& x:*v)if(!(std::cin>>x))return 2;
 try{
  juggling::JointHoming path(start,goal,lo,hi);
  std::cout<<std::setprecision(17)<<path.duration()<<'\n';
  for(int i=0;i<=10000;++i){double t=path.duration()*i/10000.;auto r=path.at(t);std::cout<<t;
   for(const auto* v:{&r.q,&r.dq,&r.ddq,&r.jerk})for(double x:*v)std::cout<<','<<x;
   std::cout<<'\n';}
 }catch(const std::exception& error){std::cerr<<error.what();return 3;}
}
''')
    exe=build/'trace'
    subprocess.run([compiler,'-std=c++17','-O3','-DNDEBUG','-march=native','-I',str(root/'cpp/include'),str(source),'-o',str(exe)],check=True)
    cases={'observed_robot':np.array(state['q_d']),'already_home':home,'lower_joint_bounds':lower,'upper_joint_bounds':upper}
    summary={};traces={}
    for name,start in cases.items():
        text=' '.join(map(str,np.r_[start,home,lower,upper]))
        result=subprocess.run([str(exe)],input=text,text=True,capture_output=True,check=True).stdout
        duration=float(result.splitlines()[0]);data=np.array([np.fromstring(row,sep=',') for row in result.splitlines()[1:]])
        t=data[:,0];q=data[:,1:8];v=data[:,8:15];a=data[:,15:22];j=data[:,22:29]
        assert np.allclose(q[0],start,atol=1e-14,rtol=0) and np.array_equal(q[-1],home)
        assert np.max(abs(v[[0,-1]]))==0 and np.max(abs(a[[0,-1]]))==0
        assert np.all(q>=lower-1e-12) and np.all(q<=upper+1e-12)
        ratios=[float(np.max(abs(values))/limit) for values,limit in [(v,.15),(a,.25),(j,1.)]]
        assert max(ratios)<=1.+1e-10
        # Independent finite differences of the emitted position/velocity traces.
        derivative_error=float(np.max(abs(np.gradient(q,t,axis=0,edge_order=2)[2:-2]-v[2:-2])))
        acceleration_error=float(np.max(abs(np.gradient(v,t,axis=0,edge_order=2)[2:-2]-a[2:-2])))
        assert derivative_error<1e-7 and acceleration_error<1e-6
        summary[name]=dict(duration_s=duration,max_velocity_rad_s=float(abs(v).max()),max_acceleration_rad_s2=float(abs(a).max()),max_jerk_rad_s3=float(abs(j).max()),
                           normalized_maxima=ratios,velocity_derivative_error=derivative_error,acceleration_derivative_error=acceleration_error)
        traces[name]=data
    for name,start in [('outside_limits',lower-.01),('nonfinite',[float('nan')]*7)]:
        p=subprocess.run([str(exe)],input=' '.join(map(str,np.r_[start,home,lower,upper])),text=True,capture_output=True)
        assert p.returncode!=0,name
        summary[name]=dict(rejected=True)
    quoted=subprocess.check_output([str(exe),'json'],text=True)
    assert json.loads(quoted)=='failure\n"quoted"\\path\tend'
    np.savez_compressed(output/'trajectories.npz',**traces)
    artifact=dict(passed=True,physical_motion=False,kind='Compiled reference checks; not hardware tracking validation',
                  compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
                  profile_sha256=hashlib.sha256((root/'cpp/include/joint_homing.hpp').read_bytes()).hexdigest(),
                  json_failure_record_roundtrip=True,cases=summary)
    (output/'checks.json').write_text(json.dumps(artifact,indent=2)+'\n')
    data=traces['observed_robot'];t=data[:,0]
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for axis,offset,title,limit in zip(axes.flat,[1,8,15,22],['Joint position [rad]','Joint velocity [rad/s]','Joint acceleration [rad/s²]','Joint jerk [rad/s³]'],[None,.15,.25,1.]):
        axis.plot(t,data[:,offset:offset+7]);axis.set(xlabel='Time [s]',title=title);axis.grid(alpha=.25)
        if limit:
            axis.axhline(limit,color='black',ls='--');axis.axhline(-limit,color='black',ls='--')
    fig.suptitle('Planned homing from the read-only robot snapshot; no physical motion\nZero endpoint velocity/acceleration; all seven joints finish together')
    fig.savefig(output/'report.pdf');plt.close(fig)
    # Reuse the archived Newton/MuJoCo scene for a static swept-path check.
    # This cannot establish clearance of the actual workspace or CAD attachment.
    import mujoco
    scene=root/'runs/20260925T102022.721751Z_native_mesh24_prune_wind/engine_scene.xml'
    model=mujoco.MjModel.from_xml_path(str(scene));state=mujoco.MjData(model)
    assert model.nq==14 and model.nv==13
    penetrations=[]
    for row in data[::50]:
        state.qpos[:7]=row[1:8];state.qpos[7:10]=[0,0,3]
        mujoco.mj_forward(model,state)
        for contact in state.contact:
            names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or str(g) for g in contact.geom]
            if contact.dist<0 and not any('ball' in name for name in names):
                penetrations.append(dict(time_s=float(row[0]),geoms=names,depth_m=float(-contact.dist)))
    collision=dict(kind='Offline swept-path check against the existing Panda/racket/floor collision model',
                   scene=str(scene.relative_to(root)),scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                   samples=len(data[::50]),penetrating_contacts=penetrations,
                   physical_obstacles_or_actual_attachment_checked=False,passed=not penetrations)
    (output/'collision_check.json').write_text(json.dumps(collision,indent=2)+'\n')
    assert collision['passed'],collision
    print(json.dumps(artifact,indent=2))


if __name__=='__main__':main()
