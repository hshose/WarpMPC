"""Measured plant parameters and rebound residuals around native Newton contacts.

The contact solver supplies the entire deterministic bounce. A measured residual
is applied as a one-step external force after separation; no scripted restitution
or ball repositioning replaces Newton's collision response.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from newton_batch import NewtonBatch
from newton_sim.scene import SceneConfig
from empirical_scene import author_plant_geometry


@wp.kernel
def contact_parameters(body:wp.array(dtype=wp.transform), velocity:wp.array(dtype=wp.spatial_vector),
        center:wp.array(dtype=wp.vec3), rotation:wp.quat, radius:float, axes:wp.vec2,
        body_stride:int, ball:int, paddle:int, stiffness:float,
        table_radius:wp.array(dtype=float), table_damping:wp.array(dtype=float),
        pairs:wp.array2d(dtype=wp.vec2), pair:int,
        touching:wp.array(dtype=int), counts:wp.array(dtype=int),
        residual:wp.array2d(dtype=float), acceleration:wp.array2d(dtype=float),
        combined:wp.array2d(dtype=float), dt:float):
    w=wp.tid();ib=w*body_stride+ball;ip=w*body_stride+paddle
    pose=body[ip];bp=wp.transform_get_translation(body[ib])
    local=wp.quat_rotate_inv(rotation,wp.transform_point(wp.transform_inverse(pose),bp)-center[w])
    radial=wp.sqrt(local[0]*local[0]+local[1]*local[1]);damping=table_damping[table_damping.shape[0]-1]
    for k in range(table_radius.shape[0]-1):
        if radial>=table_radius[k] and radial<table_radius[k+1]:
            u=(radial-table_radius[k])/(table_radius[k+1]-table_radius[k])
            damping=(1.0-u)*table_damping[k]+u*table_damping[k+1]
    pairs[w,pair]=wp.vec2(-stiffness,-damping)
    normal=wp.transform_vector(pose,wp.quat_rotate(rotation,wp.vec3(0.,0.,1.)))
    relative=wp.spatial_top(velocity[ib])-wp.spatial_top(velocity[ip])-wp.cross(wp.spatial_bottom(velocity[ip]),bp-wp.transform_get_translation(pose))
    gap=local[2]-radius
    inside=local[0]*local[0]/(axes[0]*axes[0])+local[1]*local[1]/(axes[1]*axes[1])<=1.0
    extra=wp.vec3(0.)
    if touching[w]==1 and gap>=0.0 and wp.dot(relative,normal)>0.0:
        extra=wp.transform_vector(pose,wp.quat_rotate(rotation,wp.vec3(residual[w,0],residual[w,1],residual[w,2])))/dt
        counts[w]+=1;touching[w]=0
    elif gap<0.0 and gap>-.015 and inside:
        touching[w]=1
    for j in range(3):combined[w,j]=acceleration[w,j]+extra[j]


class EmpiricalNewton:
    def __init__(self, model_path, contact_path, batch, directory, controller_spec,
                 initial_q, ball_base, parameters, device='cuda:0'):
        if device=='cpu':
            raise ValueError('Per-world empirical contacts require the GPU batched solver')
        model=json.loads(Path(model_path).read_text());plant=model['plant']
        fit=json.loads(Path(contact_path).read_text())
        if not fit['passed']:raise ValueError('Native contact calibration validation failed')
        if fit['model_sha256']!=hashlib.sha256(Path(model_path).read_bytes()).hexdigest():raise ValueError('Native contact fit is stale; recalibrate after the empirical refit')
        gravity=plant['gravity_m_s2'];cfg=SceneConfig(device=device,visuals=False,
            assembly=controller_spec.get('assembly'),geometry=controller_spec.get('geometry'),
            gravity_x=gravity[0],gravity_y=gravity[1],gravity_z=gravity[2],
            drag=plant['drag_s_inv'],ball_mass=plant['ball_mass_kg'],radius=plant['ball_radius_m'],
            friction=plant['friction'],contact_stiffness=fit['stiffness'],contact_damping=fit['rows'][0]['damping'])
        self.scene=NewtonBatch(batch,cfg,directory,initial_q=initial_q,ball_base=ball_base,
            quadratic_drag=plant['quadratic_drag_m_inv'],scene_transform=lambda path,robot:author_plant_geometry(path,robot,plant))
        s=self.scene;self.plant=plant;self.batch=batch;self.device=device
        import mujoco
        native=s.solver.mj_model
        racket=mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_GEOM,'racket')
        if racket<0:
            candidates=[i for i in range(native.ngeom) if 'racket' in (mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
            if len(candidates)!=1:raise ValueError('Cannot uniquely identify native racket geometry')
            racket=candidates[0]
        pair=np.flatnonzero((native.pair_geom1==racket)|(native.pair_geom2==racket))
        if len(pair)!=1:raise ValueError('Expected one explicit racket contact pair')
        self.pair=int(pair[0])
        nominal=s.robot.paddle_normal_rotation
        rotation=nominal@Rotation.from_rotvec(np.deg2rad(plant['normal_delta_rotvec_deg'])).as_matrix()
        self.rotation=wp.quat(*Rotation.from_matrix(rotation).as_quat())
        offsets=np.asarray(parameters['center_offset_local_m'])
        face=np.array([0.,0.,s.robot.paddle_surface_offset])+offsets@nominal.T
        self.center=wp.array(face,dtype=wp.vec3,device=device)
        physical=s.solver.mjw_model
        geom_pos=np.broadcast_to(physical.geom_pos.numpy(),(batch,native.ngeom,3)).copy()
        geom_pos[:,racket]+=((offsets-np.asarray(plant['center_offset_local_m']))@nominal.T)
        physical.geom_pos=wp.array(geom_pos,dtype=wp.vec3,device=device)
        pair_values=np.broadcast_to(physical.pair_solref.numpy(),(batch,native.npair,2)).copy()
        physical.pair_solref=wp.array(pair_values,dtype=wp.vec2,device=device)
        self.table_radius=wp.array([r['radius_m'] for r in fit['rows']],dtype=float,device=device)
        self.table_damping=wp.array([r['damping'] for r in fit['rows']],dtype=float,device=device)
        self.stiffness=fit['stiffness'];self.touching=wp.zeros(batch,dtype=int,device=device)
        self.counts=wp.zeros(batch,dtype=int,device=device);self.combined=wp.zeros((batch,3),device=device)

    def observe(self):return self.scene.observe()

    def _step_uncaptured(self, torque, acceleration, rebound):
        s=self.scene;p=self.plant
        wp.launch(contact_parameters,self.batch,inputs=[s.state.body_q,s.state.body_qd,self.center,self.rotation,
            p['ball_radius_m'],wp.vec2(*p['paddle_axes_m']),s.bstride,s.ball,s.paddle,self.stiffness,
            self.table_radius,self.table_damping,s.solver.mjw_model.pair_solref,self.pair,
            self.touching,self.counts,rebound,acceleration,self.combined,s.cfg.dt],device=self.device)
        return s.step(torque,self.combined)

    def enable_graph(self):
        """Capture both state-buffer directions; all physical clocks stay unchanged."""
        self.graph_inputs=[wp.zeros((self.batch,n),device=self.device) for n in (7,3,3)]
        self.graphs=[]
        for _ in range(2):
            with wp.ScopedCapture(device=self.device) as capture:
                self._step_uncaptured(*self.graph_inputs)
            self.graphs.append(capture.graph)
        self.graph_index=0

    def step(self,torque,acceleration,rebound):
        if not hasattr(self,'graphs'):
            return self._step_uncaptured(torque,acceleration,rebound)
        for dest,source in zip(self.graph_inputs,(torque,acceleration,rebound)):wp.copy(dest,source)
        wp.capture_launch(self.graphs[self.graph_index])
        self.graph_index=1-self.graph_index
        s=self.scene;s.state,s.next=s.next,s.state
        return s.out
