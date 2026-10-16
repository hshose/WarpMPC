"""Native double-precision SVD projector, mapped across reference postures."""
import casadi as ca
import numpy as np

class Projectors:
    def __init__(self,factors,batch,threads=8):
        f=factors;fk=ca.Function.load(str(f.directory/'panda_contact_fk.casadi'));q=ca.MX.sym('q',7)
        p,n,J,W,R=fk(q,ca.DM(f.arrays['factor_parameters']))
        jac=ca.vertcat(J,-ca.skew(n)@W)
        self.mapped=ca.Function('reference_jacobian',[q],[ca.vec(jac)]).map(batch,'thread',threads)
        self.batch=batch
    def __call__(self,posture):
        jac=np.asarray(self.mapped(np.asarray(posture,dtype=float).T)).T.reshape(self.batch,7,6).transpose(0,2,1)
        _,s,v=np.linalg.svd(jac,full_matrices=True)
        rows=v[:,:6,:]*(s>1e-8)[:,:,None]
        return (np.eye(7)-rows.transpose(0,2,1)@rows).transpose(0,2,1).reshape(self.batch,49).astype('float32')


class GPUProjectors:
    """Only the reference Jacobian crosses the CasADi-to-Warp boundary."""
    def __init__(self, factors):
        import jax
        import jax.numpy as jnp
        from warpmpc.numpysadi import convert
        f=factors;fk=ca.Function.load(str(f.directory/'panda_contact_fk.casadi'));q=ca.MX.sym('q',7)
        _,n,J,W,_=fk(q,ca.DM(f.arrays['factor_parameters']))
        # The normal contributes only two independent rows. Export their
        # tangent basis directly: float32 roundoff cannot invent a sixth mode.
        axis=ca.if_else(ca.fabs(n[0])<.9,ca.DM([1.,0.,0.]),ca.DM([0.,1.,0.]))
        tangent=ca.cross(n,axis);tangent/=ca.norm_2(tangent)
        second=ca.cross(n,tangent)
        orientation=-ca.skew(n)@W
        jac=ca.vertcat(J,tangent.T@orientation,second.T@orientation)
        evaluate=jax.vmap(convert(ca.Function('reference_jacobian_gpu',[q],[ca.densify(jac)]).expand(),backend='warp'))
        @jax.jit
        def project(posture):
            jac=evaluate(posture).reshape(-1,5,7)
            # Retain the native rank cutoff on the five independent rows.
            # Only this small decomposition uses float64; MPC remains float32.
            with jax.enable_x64():
                _,s,v=jnp.linalg.svd(jac.astype(jnp.float64),full_matrices=False)
                rows=v*(s>1e-8)[:,:,None]
                result=jnp.eye(7)-rows.swapaxes(1,2)@rows
            return result.swapaxes(1,2).reshape(-1,49).astype(jnp.float32)
        self.__call_impl=project

    def __call__(self, posture):
        return self.__call_impl(posture)
