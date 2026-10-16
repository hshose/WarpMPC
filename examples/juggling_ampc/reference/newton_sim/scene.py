"""Panda + rigid racket + hollow ball authored in Newton; native contact response."""
from dataclasses import dataclass,asdict
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
os.environ.setdefault('WARP_CACHE_PATH','/tmp/franka-warp')
import warp as wp
import newton
from newton.solvers import SolverMuJoCo
from arm.model import Panda,HOME,MODEL_DIR,FIXTURE_PARTS


@dataclass
class SceneConfig:
    dt:float=.0005
    contact_stiffness:float=2.0e6  # Native solref reference-acceleration gain [1/s^2].
    contact_damping:float=650.    # Native solref reference-acceleration gain [1/s].
    contact_impedance:float=.95
    friction:float=.3
    friction_damping:float=1500.  # Independent tangential solreffriction gain [1/s]; zero inherits normal damping.
    joint_damping:float=0.       # Synthetic coefficient; not hardware identification.
    drag:float=.34
    gravity_x:float=0.
    gravity_y:float=0.
    gravity_z:float=-9.81
    assembly:dict|None=None
    geometry:dict|None=None
    ball_mass:float=.0027
    radius:float=.02
    initial_height:float=.67
    visuals:bool=True
    self_collision:bool=True
    integrator:str='RK4'
    device:str='cpu'


def numbers(a):return ' '.join(f'{float(v):.16g}' for v in a)


def visual_meshes(k):
    """Preserve official DAE part transforms/colors while exporting MuJoCo OBJ."""
    import trimesh
    directory=MODEL_DIR/'meshes'/'converted';directory.mkdir(exist_ok=True)
    import json
    manifest=directory/f'link{k}.json'
    if manifest.exists():return json.loads(manifest.read_text())
    scene=trimesh.load(MODEL_DIR/'meshes'/'visual'/f'link{k}.dae',force='scene')
    rows=[]
    for i,node in enumerate(scene.graph.nodes_geometry):
        transform,geom=scene.graph[node];mesh=scene.geometry[geom].copy();mesh.apply_transform(transform)
        material=getattr(mesh.visual,'material',None)
        color=np.asarray(getattr(material,'diffuse',[210,215,220,255]),dtype=float)/255
        path=directory/f'link{k}_{i}.obj';mesh.export(path,include_color=False,include_texture=False)
        rows.append(dict(path=str(path.relative_to(MODEL_DIR)),color=color.tolist()))
    manifest.write_text(json.dumps(rows,indent=2)+'\n');return rows


def make_mjcf(cfg,robot,directory):
    root=ET.Element('mujoco',model='panda_juggling_newton')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true',inertiafromgeom='false',fusestatic='false')
    ET.SubElement(root,'option',timestep=str(cfg.dt),gravity='0 0 -9.81',integrator=cfg.integrator,
                  solver='Newton',iterations='100',tolerance='1e-10',cone='elliptic')
    asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',solref=numbers([-cfg.contact_stiffness,-cfg.contact_damping]),
        solimp=numbers([cfg.contact_impedance,cfg.contact_impedance,.001,.5,2]),
        friction=numbers([cfg.friction,.0001,.00001]),condim='3',margin='0',gap='0')
    ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .05',rgba='.17 .21 .25 1')
    urdf=ET.fromstring(robot.urdf_xml)
    links={l.get('name'):l for l in urdf.findall('link')}
    bodies={'panda_link0':ET.SubElement(world,'body',name='panda_link0')}
    for joint in urdf.findall('joint'):
        parent=joint.find('parent').get('link');name=joint.find('child').get('link');o=joint.find('origin')
        quat=Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_quat(scalar_first=True)
        body=ET.SubElement(bodies[parent],'body',name=name,pos=o.get('xyz'),quat=numbers(quat));bodies[name]=body
        ine=links[name].find('inertial');inertia=ine.find('inertia').attrib
        ET.SubElement(body,'inertial',pos=ine.find('origin').get('xyz'),mass=ine.find('mass').get('value'),
            fullinertia=numbers([float(inertia['i'+s]) for s in ['xx','yy','zz','xy','xz','yz']]))
        if joint.get('type')=='revolute':
            limit=joint.find('limit')
            ET.SubElement(body,'joint',name=joint.get('name'),type='hinge',axis='0 0 1',
                range=f"{limit.get('lower')} {limit.get('upper')}",damping='0',armature='0',frictionloss='0')
    for k in range(8):
        body=bodies[f'panda_link{k}'];name=f'collision_link{k}'
        ET.SubElement(asset,'mesh',name=name,file=str(MODEL_DIR/'meshes'/'collision'/f'link{k}.stl'))
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,group='3',rgba='.75 .78 .8 1')
        if cfg.visuals:
            for i,row in enumerate(visual_meshes(k)):
                name=f'visual_link{k}_{i}'
                ET.SubElement(asset,'mesh',name=name,file=str(MODEL_DIR/row['path']))
                ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,group='2',contype='0',conaffinity='0',rgba=numbers(row['color']))
    for i,(mass,center,size) in enumerate(FIXTURE_PARTS):
        ET.SubElement(bodies['panda_link8'],'geom',name=f'fixture_{i}',type='box',pos=numbers(robot.assembly_rotation@center),quat=numbers(Rotation.from_matrix(robot.assembly_rotation).as_quat(scalar_first=True)),size=numbers(size/2),rgba='.15 .17 .2 1')
    # Tilt the disk about its top-face center, matching contact_fk exactly.
    correction=robot.paddle_normal_rotation
    center=robot.paddle_surface_offset*(np.array([0.,0.,1.])-correction[:,2])
    ET.SubElement(bodies['paddle'],'geom',name='racket',type='cylinder',
        pos=numbers(center),quat=numbers(Rotation.from_matrix(correction).as_quat(scalar_first=True)),
        size=numbers([.075,robot.paddle_surface_offset]),rgba='.75 .06 .065 1')
    # A visual handle connects the fixture to the disk; its inertia is already
    # represented by the explicit Stage 2 racket/fixture payload assumptions.
    ET.SubElement(bodies['panda_link8'],'geom',name='handle_visual',type='capsule',fromto='.06 0 .065 .14 0 .048',size='.01',
                  contype='0',conaffinity='0',group='2',rgba='.45 .25 .12 1')
    ball=ET.SubElement(world,'body',name='ball',pos=numbers(robot.origin+[0,0,cfg.initial_height]))
    ET.SubElement(ball,'freejoint',name='ball_free')
    ET.SubElement(ball,'inertial',pos='0 0 0',mass=str(cfg.ball_mass),
                  diaginertia=numbers(np.full(3,2/3*cfg.ball_mass*cfg.radius**2)))
    ET.SubElement(ball,'geom',name='ball_shape',type='sphere',size=str(cfg.radius),rgba='1 .52 .055 1')
    contact=ET.SubElement(root,'contact')
    if cfg.friction_damping:
        ET.SubElement(contact,'pair',geom1='racket',geom2='ball_shape',condim='3',
            solref=numbers([-cfg.contact_stiffness,-cfg.contact_damping]),solreffriction=numbers([0,-cfg.friction_damping]),
            solimp=numbers([cfg.contact_impedance,cfg.contact_impedance,.001,.5,2]),
            friction=numbers([cfg.friction,cfg.friction,.0001,.00001,.00001]))
    # CAD collision envelopes overlap at adjacent joints and the bolted fixture.
    for joint in urdf.findall('joint'):
        ET.SubElement(contact,'exclude',body1=joint.find('parent').get('link'),body2=joint.find('child').get('link'))
    if not cfg.self_collision:
        names=list(bodies)
        for i,a in enumerate(names):
            for b in names[i+1:]:ET.SubElement(contact,'exclude',body1=a,body2=b)
    path=directory/'authored_scene.xml';ET.indent(root);ET.ElementTree(root).write(path,encoding='unicode')
    return path


class Scene:
    def __init__(self,cfg=None,directory=Path('build/newton_scene')):
        self.cfg=cfg or SceneConfig();cfg=self.cfg;self.robot=Panda(assembly=cfg.assembly,**(cfg.geometry or {}));directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        wp.init();builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.armature=0.;builder.default_joint_cfg.target_ke=0.;builder.default_joint_cfg.target_kd=0.
        path=make_mjcf(cfg,self.robot,directory)
        builder.add_mjcf(str(path),ignore_inertial_definitions=False,enable_self_collisions=cfg.self_collision,
                         parse_visuals_as_colliders=False,collapse_fixed_joints=False)
        builder.joint_q[:7]=HOME.tolist()
        self.model=builder.finalize(device=cfg.device)
        self.solver=SolverMuJoCo(self.model,use_mujoco_cpu=(cfg.device=='cpu'),skip_visual_only_geoms=False,
            integrator=cfg.integrator,solver='Newton',iterations=100,tolerance=1e-10,cone='elliptic',
            save_to_mjcf=str(directory/'engine_scene.xml'),update_data_interval=0)
        # The constructor exports before the first step sets dt. Make the saved
        # replay model state the actual clock explicitly as well.
        exported=ET.parse(directory/'engine_scene.xml');exported.getroot().find('option').set('timestep',str(cfg.dt))
        exported.write(directory/'engine_scene.xml',encoding='unicode')
        self.state=self.model.state();self.next=self.model.state();self.control=self.model.control()
        self.solver.reset(self.state)
        newton.eval_fk(self.model,self.state.joint_q,self.state.joint_qd,self.state)
        self.ball=next(i for i,s in enumerate(self.model.body_label) if s.endswith('/ball') or s=='ball')
        self.paddle=next(i for i,s in enumerate(self.model.body_label) if s.endswith('/paddle') or s=='paddle')
        self.force=np.zeros((self.model.body_count,6),dtype=np.float32)
        self.joint_force=np.zeros(self.model.joint_dof_count,dtype=np.float32)

    def observation(self):
        q=self.state.joint_q.numpy();dq=self.state.joint_qd.numpy()
        # For CPU, retain the engine's float64 coordinates between steps. Newton
        # state arrays remain the canonical interface for control and recording.
        body=self.state.body_q.numpy()[self.ball];vel=self.state.body_qd.numpy()[self.ball]
        return np.r_[body[:3]-self.robot.origin,vel[:3],q[:7],dq[:7]].astype(float)

    def step(self,tau,wind=None):
        x=self.observation();wind=np.zeros(3) if wind is None else wind
        self.joint_force[:7]=tau-self.cfg.joint_damping*x[13:20]
        self.control.joint_f.assign(self.joint_force)
        self.force.fill(0);self.force[self.ball,:3]=self.cfg.ball_mass*(wind-self.cfg.drag*x[3:6]+np.array([self.cfg.gravity_x,self.cfg.gravity_y,self.cfg.gravity_z+9.81]))
        self.state.clear_forces();self.state.body_f.assign(self.force)
        self.solver.step(self.state,self.next,self.control,None,self.cfg.dt)
        self.state,self.next=self.next,self.state
        return self.observation()

    def contact_info(self):
        """Native solver force samples; no collision decisions or impulses here."""
        import mujoco
        if self.cfg.device!='cpu':raise NotImplementedError('Detailed contact audit currently uses the CPU backend')
        m=self.solver.mj_model;d=self.solver.mj_data;force=np.zeros(6);normal=0.;depth=0.;other=[]
        for i in range(d.ncon):
            c=d.contact[i];mujoco.mj_contactForce(m,d,i,force)
            if np.linalg.norm(force[:3])<1e-8:continue
            names=[m.geom(int(g)).name.rsplit('/',1)[-1] for g in c.geom]
            if any('ball_shape' in n for n in names) and any('racket' in n for n in names):
                normal+=force[0];depth=max(depth,-c.dist)
            else:other.append(dict(geoms=names,force_N=float(np.linalg.norm(force[:3])),depth_m=float(-c.dist)))
        return dict(normal_force_N=normal,penetration_m=depth,other_contacts=other,
                    acceleration=d.qacc_smooth[:7].copy(),total_acceleration=d.qacc[:7].copy())
