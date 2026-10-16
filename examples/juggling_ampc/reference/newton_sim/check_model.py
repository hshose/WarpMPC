"""Focused independent Newton/MuJoCo vs Pinocchio/CasADi model verification."""
from pathlib import Path
import json
import mujoco
import numpy as np
from .scene import Scene,SceneConfig
from arm.model import PADDLE_SURFACE_OFFSET


def main():
    s=Scene(SceneConfig(visuals=False),Path('build/newton_checks'));m=s.solver.mj_model;d=s.solver.mj_data
    rng=np.random.default_rng(26);errors=dict(mass=0.,bias=0.,paddle_position=0.,paddle_normal=0.)
    pad=next(i for i in range(m.nbody) if m.body(i).name.endswith('_paddle'))
    for _ in range(30):
        q=np.clip(s.model.joint_q.numpy()[:7]+rng.normal(0,.12,7),s.robot.q_min,s.robot.q_max);v=rng.normal(0,.15,7)
        d.qpos[:7]=q;d.qvel[:7]=v;mujoco.mj_forward(m,d)
        mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
        errors['mass']=max(errors['mass'],float(np.max(abs(mass[:7,:7]-np.asarray(s.robot.mass(q))))))
        errors['bias']=max(errors['bias'],float(np.max(abs(d.qfrc_bias[:7]-np.asarray(s.robot.bias(q,v)).ravel()))))
        p,n,*_=s.robot.numeric_fk(q)
        surface=d.xpos[pad]+PADDLE_SURFACE_OFFSET*d.xmat[pad].reshape(3,3)[:,2]
        errors['paddle_position']=max(errors['paddle_position'],float(np.max(abs(surface-p))))
        errors['paddle_normal']=max(errors['paddle_normal'],float(np.max(abs(d.xmat[pad].reshape(3,3)[:,2]-n))))
    assert max(errors.values())<1e-4,errors
    assert m.npair==1 and np.allclose(m.pair_solref[0],[-2e6,-115])
    assert np.allclose(m.pair_solreffriction[0],[0,-1500]) and np.allclose(m.pair_friction[0,:2],[.3,.3])
    result=dict(samples=30,max_errors=errors,armature=m.dof_armature[:7].tolist(),native_damping=m.dof_damping[:7].tolist(),
        ball_mass=float(m.body_mass[-1]),ball_inertia=m.body_inertia[-1].tolist(),newton='1.7.0.dev0',mujoco=mujoco.__version__,
        racket_ball_pair=dict(solref=m.pair_solref[0].tolist(),solreffriction=m.pair_solreffriction[0].tolist(),friction=m.pair_friction[0].tolist()),
        fixture='Stage 3 revised 70 g bracket, 20 g upright plus 50 g beam behind the hitting face')
    Path('docs/newton_model_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
