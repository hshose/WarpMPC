"""Replicated native Newton/MuJoCo scene; no scripted ball impacts.

All worlds advance together. Forces and observation extraction stay on the GPU.
The scalar source scene authors geometry, inertias and native contact settings.
"""
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverMuJoCo
from newton_sim.scene import make_mjcf,SceneConfig
from arm.model import Panda,HOME

@wp.kernel
def forces(tau:wp.array2d(dtype=float),acceleration:wp.array2d(dtype=float),
           velocity:wp.array(dtype=wp.spatial_vector),body_force:wp.array(dtype=wp.spatial_vector),
           joint_force:wp.array(dtype=float),joint_velocity:wp.array(dtype=float),damping:float,body_stride:int,dof_stride:int,ball:int,mass:float,drag:float,quadratic_drag:float,gravity:wp.vec3):
    w=wp.tid()
    for j in range(7):joint_force[w*dof_stride+j]=tau[w,j]-damping*joint_velocity[w*dof_stride+j]
    idx=w*body_stride+ball;v=wp.spatial_top(velocity[idx])
    a=wp.vec3(acceleration[w,0],acceleration[w,1],acceleration[w,2])+gravity-wp.vec3(0.,0.,-9.81)-drag*v-quadratic_drag*wp.length(v)*v
    body_force[idx]=wp.spatial_vector(mass*a,wp.vec3(0.))

@wp.kernel
def observation(q:wp.array(dtype=float),dq:wp.array(dtype=float),body:wp.array(dtype=wp.transform),
                vel:wp.array(dtype=wp.spatial_vector),out:wp.array2d(dtype=float),qstride:int,dstride:int,bstride:int,ball:int,origin:wp.vec3):
    w=wp.tid();idx=w*bstride+ball;p=wp.transform_get_translation(body[idx])-origin;v=wp.spatial_top(vel[idx]);spin=wp.spatial_bottom(vel[idx])
    for j in range(3):
        out[w,j]=p[j];out[w,3+j]=v[j];out[w,20+j]=spin[j]
    for j in range(7):out[w,6+j]=q[w*qstride+j];out[w,13+j]=dq[w*dstride+j]

class NewtonBatch:
    def __init__(self,batch,cfg:SceneConfig,directory,initial_q=None,ball_base=None,quadratic_drag=0.,scene_transform=None,ball_velocity=None):
        self.batch=batch;self.cfg=cfg;self.quadratic_drag=quadratic_drag;self.robot=Panda(assembly=cfg.assembly,**(cfg.geometry or {}))
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        cfg.visuals=False
        template=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(template)
        template.default_joint_cfg.armature=0.;template.default_joint_cfg.target_ke=0.;template.default_joint_cfg.target_kd=0.
        path=make_mjcf(cfg,self.robot,directory)
        if scene_transform is not None:scene_transform(path,self.robot)
        template.add_mjcf(str(path),ignore_inertial_definitions=False,enable_self_collisions=cfg.self_collision,parse_visuals_as_colliders=False,collapse_fixed_joints=False)
        self.qstride=len(template.joint_q);self.dstride=len(template.joint_qd);self.bstride=len(template.body_label)
        self.ball=next(i for i,s in enumerate(template.body_label) if s.endswith('/ball') or s=='ball')
        self.paddle=next(i for i,s in enumerate(template.body_label) if s.endswith('/paddle') or s=='paddle')
        template.joint_q[:7]=HOME.tolist()
        builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder);builder.replicate(template,world_count=batch)
        q=np.asarray(builder.joint_q).reshape(batch,-1)
        if initial_q is not None:q[:,:7]=initial_q
        if ball_base is not None:q[:,7:10]=ball_base
        builder.joint_q=q.ravel().tolist()
        if ball_velocity is not None:
            dq=np.asarray(builder.joint_qd).reshape(batch,-1);dq[:,7:10]=ball_velocity;builder.joint_qd=dq.ravel().tolist()
        self.model=builder.finalize(device=cfg.device)
        self.solver=SolverMuJoCo(self.model,use_mujoco_cpu=cfg.device=='cpu',integrator=cfg.integrator,
            solver='Newton',iterations=100,tolerance=1e-10,cone='elliptic',update_data_interval=0)
        self.state=self.model.state();self.next=self.model.state();self.control=self.model.control();self.solver.reset(self.state)
        newton.eval_fk(self.model,self.state.joint_q,self.state.joint_qd,self.state)
        self.out=wp.zeros((batch,23),dtype=float,device=cfg.device)
    def observe(self):
        wp.launch(observation,self.batch,inputs=[self.state.joint_q,self.state.joint_qd,self.state.body_q,self.state.body_qd,self.out,self.qstride,self.dstride,self.bstride,self.ball,wp.vec3(*self.robot.origin)],device=self.cfg.device)
        return self.out
    def step(self,tau,acceleration):
        c=self.cfg;self.state.clear_forces()
        wp.launch(forces,self.batch,inputs=[tau,acceleration,self.state.body_qd,self.state.body_f,self.control.joint_f,self.state.joint_qd,c.joint_damping,self.bstride,self.dstride,self.ball,c.ball_mass,c.drag,self.quadratic_drag,wp.vec3(c.gravity_x,c.gravity_y,c.gravity_z)],device=c.device)
        self.solver.step(self.state,self.next,self.control,None,c.dt);self.state,self.next=self.next,self.state
        return self.observe()
