"""Batched native-OSQP semantics using WarpMPC's sparse QDLDL kernels.

Unlike the generic fixed-scaling wrapper, each QP gets ten Ruiz passes and
its own equality/free-row rho. These are OSQP defaults used by the Franka
controller, not tunable approximations to its fixed-budget solve. The float32
path translates ADMM coordinates only after this native scaling, using the
original SQP gradient/relative bounds to avoid subtracting large nearby terms.
The real-arithmetic forty-step iteration remains algebraically identical.
"""
import numpy as np
import os
from functools import partial
import jax
import jax.numpy as jnp
from warpmpc.jax_osqp import build_osqp_plan, compile_osqp, OSQPSettings
from warpmpc.jax_osqp.solver import values_for_compiled_qdldl, rhs_for_compiled_qdldl, batch_from_compiled_qdldl


class NativeQP:
    def __init__(self, P, A, rho=.001, iterations=40, backend='warp', dtype=np.float32):
        self.plan = plan = build_osqp_plan(P, A, settings=OSQPSettings(rho=rho, scaling=0))
        self.compiled = compile_osqp(plan, dtype=dtype, qdldl_backend=backend,
            qdldl_factor_backend=backend, qdldl_solve_backend=backend,
            transpose_work=True, segmented=True, segment_budget=256,
            segment_strategy='optimal', level_scheduled_solve=True,
            level_scheduled_solve_threshold=int(os.environ.get("JUGGLING_SOLVE_LEVEL_THRESHOLD","32")))
        self.dtype = dtype
        self.incremental=os.environ.get("JUGGLING_QP_INCREMENTAL","1" if np.dtype(dtype)==np.dtype(np.float32) else "0")=="1"
        centered=self.incremental
        compensated=None
        if os.environ.get("JUGGLING_QP_COMPENSATED_RESIDUAL","1" if np.dtype(dtype)==np.dtype(np.float32) else "0")=="1":
            if np.dtype(dtype)!=np.dtype(np.float32):raise ValueError("Compensated residual is float32 only")
            from compensated_residual import compile_residual
            compensated=compile_residual(plan.kkt_upper)
        refinement=int(os.environ.get("JUGGLING_QP_REFINEMENT","0"))
        if refinement not in (0,1,2):raise ValueError("QP refinement must be 0, 1, or 2")
        kr=jnp.asarray(plan.kkt_upper.indices);kc=jnp.asarray(np.repeat(np.arange(plan.n+plan.m),np.diff(plan.kkt_upper.indptr)))
        off=jnp.asarray(plan.kkt_upper.indices!=np.repeat(np.arange(plan.n+plan.m),np.diff(plan.kkt_upper.indptr)))
        n,m=plan.n,plan.m
        pr=jnp.asarray(plan.p_upper.indices);pc=jnp.asarray(np.repeat(np.arange(n),np.diff(plan.p_upper.indptr)))
        ar=jnp.asarray(plan.a_rows);ac=jnp.asarray(plan.a_cols)
        def ax(a,x):return jnp.zeros((x.shape[0],m),x.dtype).at[:,ar].add(a*x[:,ac])
        self.gradient_residual=None
        if os.environ.get("JUGGLING_QP_COMPENSATED_PRODUCTS","1" if np.dtype(dtype)==np.dtype(np.float32) else "0")=="1":
            if np.dtype(dtype)!=np.dtype(np.float32):raise ValueError("Compensated products are float32 only")
            from compensated_residual import compile_residual
            self.gradient_residual=compile_residual(plan.p_upper)
            a_residual=compile_residual(plan.a_matrix,symmetric=False)
            def ax(a,x):return -a_residual(a,x,jnp.zeros((x.shape[0],m),x.dtype))
        center_residual=None
        if centered and np.dtype(dtype)==np.dtype(np.float32):
            from compensated_residual import compile_residual
            center_residual=compile_residual(plan.p_upper)
        def px(p,x):
            value=jnp.zeros_like(x).at[:,pr].add(p*x[:,pc])
            return value.at[:,pc].add(jnp.where(pr!=pc,p*x[:,pr],0.))
        self.numerics=dict(dtype=np.dtype(dtype).name, backend=backend,
            incremental_coordinates=self.incremental, kkt_refinement_steps=refinement,
            compensated_kkt_residual=compensated is not None,
            compensated_products=self.gradient_residual is not None,
            ruiz_iterations=10, admm_iterations=iterations, rho=rho,
            sigma=1e-6, alpha=1.6, transpose_work=True, segmented=True,
            segment_budget=256, segment_strategy='optimal', level_scheduled_solve=True,
            level_threshold=int(os.environ.get("JUGGLING_SOLVE_LEVEL_THRESHOLD","32")))
        self.matvec=ax
        def pnorm(p):
            v=jnp.zeros((p.shape[0],n),p.dtype).at[:,pc].max(jnp.abs(p))
            return v
        def limit(v):return jnp.where(v<1e-4,1.,jnp.minimum(v,1e4))
        @jax.jit
        def scale(p,a,q):
            batch=q.shape[0];d=jnp.ones_like(q);e=jnp.ones((batch,m),q.dtype);c=jnp.ones((batch,1),q.dtype)
            def body(_,carry):
                p,a,q,d,e,c=carry
                an=jnp.zeros_like(q).at[:,ac].max(jnp.abs(a))
                en=jnp.zeros_like(e).at[:,ar].max(jnp.abs(a))
                dn=jax.lax.rsqrt(limit(jnp.maximum(pnorm(p),an)));en=jax.lax.rsqrt(limit(en))
                p=p*dn[:,pr]*dn[:,pc];a=a*en[:,ar]*dn[:,ac];q=q*dn
                cn=1./limit(jnp.maximum(jnp.mean(pnorm(p),axis=1,keepdims=True),limit(jnp.max(jnp.abs(q),axis=1,keepdims=True))))
                return p*cn,a,q*cn,d*dn,e*en,c*cn
            return jax.lax.fori_loop(0,10,body,(p,a,q,d,e,c))
        self.scale=scale
        @partial(jax.jit,static_argnames=('return_state',))
        def solve(p,a,q,l,u,x,y,gradient=None,relative_lower=None,relative_upper=None,
                  retry_state=None,retry_mask=None,return_state=False):
            original_primal=x
            p,a,q,d,e,c=scale(p,a,q)
            l=jnp.clip(l,-1e30,1e30)*e;u=jnp.clip(u,-1e30,1e30)*e
            rv=jnp.where(u-l<1e-4,1000.*rho,rho)
            rv=jnp.where((l < -1e26)&(u > 1e26),1e-6,rv)
            ri=1./rv
            k=jnp.broadcast_to(jnp.asarray(plan.kkt_upper.data,dtype=q.dtype),(q.shape[0],plan.kkt_upper.nnz))
            k=k.at[:,plan.p_to_kkt].set(p+jnp.asarray(plan.p_diag_sigma,dtype=q.dtype))
            k=k.at[:,plan.a_to_kkt].set(a)
            k=k.at[:,plan.rho_to_kkt].set(-ri)
            kernel=self.compiled.qdldl
            # Use the public three-output factorization API; only L and D^-1
            # are needed by ADMM, so the ordinary diagonal is not retained.
            lx,_,di=kernel.factor(values_for_compiled_qdldl(kernel,k))
            x=x/d;y=y*c/e
            if centered:
                # Translate only AFTER native Ruiz scaling and rho selection.
                # This is algebraically the same fixed-budget ADMM iteration,
                # with the supplied primal warm start as the coordinate origin.
                anchor=x
                if gradient is not None:
                    q=gradient*d*c
                    l=jnp.clip(relative_lower,-1e30,1e30)*e
                    u=jnp.clip(relative_upper,-1e30,1e30)*e
                else:
                    shift=ax(a,anchor)
                    q=-center_residual(p,anchor,-q) if center_residual is not None else q+px(p,anchor)
                    l=l-shift;u=u-shift
                x=jnp.zeros_like(x)
            z=ax(a,x)
            if retry_state is not None:
                x,z,y=tuple(jnp.where(retry_mask[:,None],old,new) for old,new in zip(retry_state,(x,z,y)))
            def body(_,state):
                x,z,y=state
                rz=z-ri*y
                rhs=jnp.concatenate((1e-6*x-q,rz),axis=1)
                sol=batch_from_compiled_qdldl(kernel,kernel.solve(lx,di,rhs_for_compiled_qdldl(kernel,rhs)))
                # Float32 iterative refinement improves the same KKT solve;
                # it does not change the native 40-step ADMM budget/settings.
                for _ in range(refinement):
                    if compensated is None:
                        product=jnp.zeros_like(rhs).at[:,kr].add(k*sol[:,kc])
                        product=product.at[:,kc].add(jnp.where(off,k*sol[:,kr],0.))
                        residual=rhs-product
                    else:residual=compensated(k,sol,rhs)
                    sol=sol+batch_from_compiled_qdldl(kernel,kernel.solve(lx,di,rhs_for_compiled_qdldl(kernel,residual)))
                xt=sol[:,:n];zt=rz+ri*sol[:,n:]
                xn=1.6*xt-.6*x;zr=1.6*zt-.6*z
                zn=jnp.clip(zr+ri*y,l,u);yn=y+rv*(zr-zn)
                return xn,zn,yn
            x,z,y=jax.lax.fori_loop(0,iterations,body,(x,z,y))
            result=((x*d+original_primal if centered else x*d),y*e/c)
            return (*result,(x,z,y)) if return_state else result
        self.solve=solve
