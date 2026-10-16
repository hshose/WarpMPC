"""Batched native fixture IK and its unchanged startup reach/pose checks."""
import casadi as ca
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from warpmpc.numpysadi import convert


class FixtureIK:
    def __init__(self,factors,backend='warp'):
        f=factors;arr=f.arrays;home=np.asarray(arr['home'])
        fk=ca.Function.load(str(f.directory/'panda_contact_fk.casadi'))
        parameters=ca.DM(arr['factor_parameters'])
        normal=Rotation.from_rotvec(np.deg2rad(f.meta['spec']['geometry']['paddle_normal_rotvec_deg'])).as_matrix()
        nominal=np.asarray(fk(home,parameters)[4])@normal.T
        q=ca.SX.sym('q',7);target=ca.SX.sym('target',3)
        p,n,J,W,R=fk(q,parameters)
        residual=ca.vertcat((p-target)/.001,ca.vec((R-ca.DM(nominal))/.01),.1*(q-ca.DM(home)))
        function=ca.Function('fixture_ik_residual',[q,target],[residual,ca.densify(ca.jacobian(residual,q))])
        evaluate=jax.vmap(convert(function,backend=backend))
        trial_residual=jax.vmap(convert(ca.Function('fixture_ik_trial',[q,target],[residual]),backend=backend))
        # Round strictly inward: nearest float32 can sit outside the double
        # precision native API's exact local joint-envelope comparison.
        low=jnp.asarray(np.nextafter(np.maximum(arr['joint_lower'],home-1).astype('float32'),np.float32(np.inf)))
        high=jnp.asarray(np.nextafter(np.minimum(arr['joint_upper'],home+1).astype('float32'),np.float32(-np.inf)))
        pose=jax.vmap(convert(ca.Function('fixture_path_pose',[q],[p,n]),backend=backend))
        @jax.jit
        def solve(target):
            batch=target.shape[0];initial=jnp.tile(jnp.asarray(home,dtype=target.dtype),(batch,1))
            def body(_,carry):
                q,damping,done=carry;r,J=evaluate(q,target);r=r.reshape(batch,19);J=J.reshape(batch,19,7)
                JT=jnp.swapaxes(J,1,2);system=JT@J+damping[:,None,None]*jnp.eye(7)
                step=-jnp.linalg.solve(system,(JT@r[:,:,None]))[:,:,0]
                trial=jnp.clip(q+step,low,high);rt=trial_residual(trial,target).reshape(batch,19)
                take=(jnp.sum(rt**2,axis=1)<jnp.sum(r**2,axis=1))&~done
                done=done|(jnp.linalg.norm(step,axis=1)<1e-10)
                next_damping=jnp.where(take,jnp.maximum(1e-8,damping*.3),jnp.minimum(1e20,damping*10))
                return jnp.where(take[:,None],trial,q),jnp.where(done,damping,next_damping),done
            q,_,_=jax.lax.fori_loop(0,100,body,(initial,jnp.full(batch,1e-3),jnp.zeros(batch,bool)))
            p,n=pose(q);p=p.reshape(batch,3);n=n.reshape(batch,3)
            valid=(jnp.linalg.norm(p-target,axis=1)<=.001)&(jnp.linalg.norm(n-jnp.array([0.,0.,1.]),axis=1)<=.01)&jnp.isfinite(q).all(axis=1)
            def check_path(i,valid):
                qp=(1-i/100.)*jnp.asarray(home,dtype=q.dtype)+(i/100.)*q
                pp,nn=pose(qp);pp=pp.reshape(batch,3)-jnp.asarray(arr['origin']);nn=nn.reshape(batch,3)
                return valid&(jnp.linalg.norm(pp[:,:2],axis=1)<=.35)&(jnp.abs(pp[:,2])<=.15)&(nn[:,2]>=np.cos(.35))
            valid=jax.lax.fori_loop(0,101,check_path,valid)
            return q,valid
        self.solve=solve
