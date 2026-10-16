"""Batched physical warm-start integration matching the native controller.

This integrator only seeds the optimizer. The rollout plant is native Newton.
"""
import casadi as ca
import numpy as np
import jax
import jax.numpy as jnp
from warpmpc.numpysadi import convert

class WarmStart:
    def __init__(self,factors,backend='warp'):
        f=factors;c=f.cfg;arr=f.arrays;N=c['horizon'];directory=f.directory
        spec=f.meta['spec'];hybrid=spec.get('kinematic_tail',{})
        prefix=int(hybrid.get('dynamic_prefix',N));endpoint=hybrid.get('endpoint_accelerations',False)
        neglect=spec.get('dynamics_simplification',{}).get('neglect_ball_reaction',False)
        seed_options=spec.get('solver_options',{})
        parameter=ca.DM(arr['factor_parameters'])
        fk=ca.Function.load(str(directory/'panda_fk.casadi'));contact=ca.Function.load(str(directory/'panda_contact_fk.casadi'))
        mass=ca.Function.load(str(directory/'panda_mass.casadi'));bias=ca.Function.load(str(directory/'panda_bias.casadi'));rnea=ca.Function.load(str(directory/'panda_rnea.casadi'))
        prediction_fk=contact if spec.get('legacy_contact_plane') else fk
        x=ca.MX.sym('x',23);q=x[6:13];dq=x[13:20];pos,n,J,D,_=prediction_fk(q,parameter);rel=x[:3]-pos+ca.DM(arr['origin'])
        gap=ca.dot(n,rel)-c['radius'];tangent=rel-n*ca.dot(n,rel);speed=ca.dot(n,x[3:6])-(n.T@J-rel.T@D)@dq
        # Warp calls retain every translated output even when a JAX caller
        # discards it. Prune the CasADi graphs before translation: the 28-step
        # contact bisection needs only the gap, never a paddle Jacobian.
        def geometry_function(name,outputs):
            return jax.vmap(convert(ca.Function(name,[x],outputs).expand(),backend=backend))
        self.gap=geometry_function('seed_gap',[gap])
        self.crossing_geometry=geometry_function('seed_crossing',[ca.vertcat(gap,speed)])
        self.collision_geometry=geometry_function('seed_collision',[ca.vertcat(ca.norm_2(tangent),speed)])
        self.prediction_geometry=geometry_function('seed_prediction_geometry',[ca.vertcat(gap,ca.norm_2(tangent),speed)])
        support_position=x[:3]+n*(c['radius']-ca.dot(n,rel))
        self.support_state=geometry_function('seed_support',[ca.vertcat(support_position,J@dq)])
        pos,n,J,W,R=contact(q,parameter);rel=x[:3]-pos+ca.DM(arr['origin']);tangent=rel-n*ca.dot(n,rel)
        J=J-ca.skew(tangent)@W
        response=ca.MX.zeros(7,3) if neglect else ca.solve(mass(q),J.T,'symbolicqr')
        K=ca.MX.eye(3)+(ca.MX.eye(3)-n@n.T)/c['inertia_ratio']+c['mass']*J@response
        virtual=ca.MX.sym('virtual')
        restitution=c['restitution']
        if spec.get('legacy_contact_prediction'):
            edge=spec.get('prediction_edge_loss',0.)
            restitution=ca.if_else(virtual,c['restitution'],ca.fmin(1.5,ca.fmax(0.,c['restitution']-edge*(ca.sumsqr(tangent)/.075**2)**2)))
        relative=x[3:6]-ca.cross(x[20:23],c['radius']*n)-J@dq;T=R[:,:2];vn=ca.dot(n,relative)
        inv=ca.solve(T.T@K@T,ca.MX.eye(2),'symbolicqr')
        def impulse(j):
            desired=-inv@(T.T@(relative+K@n*j));cap=c['friction']*j
            return j*n+T@(desired*cap/ca.sqrt(cap**2+ca.sumsqr(desired)+1e-10))
        lo=0.;hi=ca.fmax(1.,-3*vn)
        for _ in range(50):
            mid=.5*(lo+hi);positive=ca.dot(n,K@impulse(mid))+(1+restitution)*vn>0
            hi=ca.if_else(positive,mid,hi);lo=ca.if_else(positive,lo,mid)
        j=.5*(lo+hi);u=impulse(j);out=ca.MX(x)
        out[:3]+=(1e-10-(ca.dot(n,rel)-c['radius']))*n;out[3:6]+=u;out[20:23]-=ca.cross(n,u)/(c['inertia_ratio']*c['radius']);out[13:20]-=c['mass']*response@u
        impact=jax.vmap(convert(ca.Function('seed_impact',[x,virtual],[out,j]).expand(),backend=backend))
        self.impact=lambda x,virtual=None:impact(x,jnp.zeros((x.shape[0],1),x.dtype) if virtual is None else virtual[:,None].astype(x.dtype))
        a=ca.MX.sym('a',7);torque=ca.MX.sym('torque',7);q=ca.MX.sym('q',7);dq=ca.MX.sym('dq',7)
        self.torque=jax.vmap(convert(ca.Function('seed_torque',[q,dq,a],[rnea(q,dq,a)]).expand(),backend=backend))
        self.acceleration=jax.vmap(convert(ca.Function('seed_acceleration',[q,dq,torque],[ca.solve(mass(q),torque-bias(q,dq),'symbolicqr')]).expand(),backend=backend))
        gravity=jnp.asarray([c['gravity_x'],c['gravity_y'],c['gravity_z']],dtype=jnp.float32)
        def free(x,t,a):
            t=jnp.asarray(t);t=jnp.broadcast_to(t,(x.shape[0],))[:,None];vd=-jnp.expm1(-c['drag']*t)/c['drag'];pd=(t-vd)/c['drag']
            position=x[:,:3]+(vd*x[:,3:6]+pd*gravity);velocity=jnp.exp(-c['drag']*t)*x[:,3:6]+vd*gravity
            q=x[:,6:13]+(t*x[:,13:20]+.5*t*t*a);dq=x[:,13:20]+t*a
            return jnp.concatenate((position,velocity,q,dq,x[:,20:23]),axis=1)
        self.free=free
        @jax.jit
        def rollout(initial,duration,a,max_step,stage_start=0.,virtual_count=None,allow_virtual=False):
            b=initial.shape[0];t=jnp.zeros(b,initial.dtype);count=jnp.zeros(b,jnp.int32);first_j=jnp.zeros(b,initial.dtype);first_t=jnp.zeros(b,initial.dtype)
            if virtual_count is None:virtual_count=jnp.zeros(b,jnp.int32)
            def condition(v):return jnp.any((v[1]<duration-1e-9)&(v[2]<=16))
            def body(v):
                state,t,count,fj,ft,virtual_count=v;active=(t<duration-1e-9)&(count<=16);dt=jnp.maximum(0,jnp.minimum(max_step,duration-t));end=free(state,dt,a)
                ga=self.crossing_geometry(state).reshape(b,2);gb=self.crossing_geometry(end).reshape(b,2);cross=(ga[:,0]>=-1e-10)&(gb[:,0]<=0)&((ga[:,1]<0)|(gb[:,1]<0))&active
                def resolve_collision(_):
                    def bisect(_,bounds):
                        lo,hi=bounds;mid=.5*(lo+hi);above=self.gap(free(state,mid,a)).reshape(b)>0
                        return jnp.where(above,mid,lo),jnp.where(above,hi,mid)
                    lo,hi=jax.lax.fori_loop(0,28,bisect,(jnp.zeros_like(dt),dt));hit=.5*(lo+hi);collision=free(state,hit,a);g=self.collision_geometry(collision).reshape(b,2)
                    lead=stage_start+t+hit
                    virtual=allow_virtual&(virtual_count<seed_options.get('warmstart_seed_max_contacts',2))&(lead>max(.3,seed_options.get('warmstart_seed_lead',.3)))&(g[:,0]>c['paddle_radius'])&(g[:,0]<=seed_options.get('warmstart_seed_radius',0.))&(g[:,1]<0)&cross
                    collide=cross&((g[:,0]<=c['paddle_radius'])|virtual)&(g[:,1]<0)
                    def impact(_):
                        after,j=self.impact(collision,virtual)
                        return after.reshape(b,23),j.reshape(b)
                    after,j=jax.lax.cond(jnp.any(collide),impact,lambda _:(collision,jnp.zeros_like(dt)),operand=None)
                    return jnp.where(collide[:,None],after,end),jnp.where(collide,hit,dt),collide,j,hit,virtual
                # No crossing means no root search; no accepted paddle contact
                # means no impact solve. Neither branch changes active worlds.
                next_state,elapsed,collide,j,hit,virtual=jax.lax.cond(jnp.any(cross),resolve_collision,
                    lambda _:(end,dt,jnp.zeros(b,bool),jnp.zeros_like(dt),jnp.zeros_like(dt),jnp.zeros(b,bool)),operand=None)
                first=collide&(count==0);fj=jnp.where(first,j,fj);ft=jnp.where(first,t+hit,ft)
                state=jnp.where(active[:,None],next_state,state)
                t=jnp.where(active,t+elapsed,t);count+=collide.astype(jnp.int32)
                return state,t,count,fj,ft,virtual_count+virtual.astype(jnp.int32)
            state,t,count,fj,ft,virtual_count=jax.lax.while_loop(condition,body,(initial,t,count,first_j,first_t,virtual_count))
            return state,fj,ft,count>16,virtual_count
        self.rollout=lambda *args:rollout(*args)[:4]
        steps=jnp.asarray(f.meta['stage_steps'],dtype=jnp.float32);times=jnp.asarray(f.meta['stage_times'],dtype=jnp.float32);scale=jnp.asarray(arr['scale'],dtype=jnp.float32);offset=jnp.asarray(arr['offset'],dtype=jnp.float32)
        @jax.jit
        def initialize(x,tau):
            b=x.shape[0];Y=jnp.zeros((b,N+1,34),x.dtype);state=x
            for k in range(N):
                tor=jnp.zeros((b,7),x.dtype) if k>prefix else self.torque(state[:,6:13],state[:,13:20],jnp.zeros((b,7),x.dtype)).reshape(b,7)
                Y=Y.at[:,k,:23].set(state).at[:,k,23:30].set(tor)
                out,j,hit,failed,_=rollout(state,steps[k],jnp.zeros((b,7),x.dtype),.003)
                def support_fallback(_):
                    supported=jnp.concatenate((self.support_state(state).reshape(b,6),state[:,6:]),axis=1)
                    return jnp.where(failed[:,None],supported,out)
                state=jax.lax.cond(jnp.any(failed),support_fallback,lambda _:out,operand=None);j=jnp.where(failed,-gravity[2]*steps[k],j);fraction=jnp.where(failed,.5,jnp.where(j!=0,hit/steps[k],.5))
                Y=Y.at[:,k,30].set(j).at[:,k,31].set(fraction)
            terminal_control=jnp.zeros((b,7),x.dtype) if hybrid else self.torque(state[:,6:13],state[:,13:20],jnp.zeros((b,7),x.dtype)).reshape(b,7)
            Y=Y.at[:,N,:23].set(state).at[:,N,23:30].set(terminal_control)
            Y=Y.at[:,0,23:30].set(tau).at[:,:,32].set(jnp.r_[steps,steps[-1]]).at[:,:,33].set(times).at[:,N,31].set(.5)
            return ((Y-offset)/scale).reshape(b,-1),jnp.zeros((b,f.n+f.m),x.dtype)
        from first_event import FirstEvent
        self.event_tracker=FirstEvent(f,free,self.prediction_geometry)
        def initialize_with_event(x,tau):
            z,dual=initialize(x,tau)
            self.event_state,self.event_parameters=self.event_tracker.update(z,x,self.event_tracker.empty(x),0.)
            return z,dual
        self.initialize=initialize_with_event
        dt=c['control_dt'];cpu_times=np.asarray(f.meta['stage_times']);cpu_steps=np.asarray(f.meta['stage_steps']);src=np.minimum(np.searchsorted(cpu_times,cpu_times+dt+1e-12,side='right')-1,N)
        dual_src=np.full(f.n+f.m,-1,dtype=int);ids=np.asarray(f.meta['constraint_ids']);begin,end=f.meta['groups']['contact']
        for k,i in enumerate(src):
            if i>=N:continue
            dual_src[f.m+k*34:f.m+(k+1)*34]=np.arange(f.m+i*34,f.m+(i+1)*34)
            if hybrid and (k>prefix)!=(i>prefix):dual_src[f.m+k*34+23:f.m+k*34+30]=-1
            if k<N and (not hybrid or (k!=prefix and i!=prefix and (k>=prefix)==(i>=prefix))):
                old_rows=range(f.meta['stage_offsets'][i],f.meta['stage_offsets'][i+1]);lookup={ids[r]:r for r in old_rows}
                for row in range(f.meta['stage_offsets'][k],f.meta['stage_offsets'][k+1]):
                    ident=ids[row]
                    if not begin<=ident<end:dual_src[row]=lookup.get(ident,-1)
        dual_src[f.m:f.m+30]=-1
        @jax.jit
        def shift(z,dual,x,tau):
            b=x.shape[0];old=z.reshape(b,N+1,34)*scale+offset;Y=old.at[:,:,30].set(0).at[:,:,31].set(.5)
            for k,i in enumerate(src):
                row=old[:,i];s=times[k]+dt-times[i];h=steps[min(i,N-1)];acc=row[:,23:30] if hybrid and i==N else (old[:,min(i+1,N),13:20]-row[:,13:20])/h
                shifted=free(row[:,:23],s,acc);Y=Y.at[:,k,:20].set(shifted[:,:20]).at[:,k,20:23].set(row[:,20:23])
                if i<N:
                    event=row[:,31]*h;end=free(row[:,:23],h,acc);imp=(old[:,i+1,3:6]-end[:,3:6])*jnp.exp(c['drag']*(h-event))[:,None]
                    hit=free(row[:,:23],event,acc);hit=hit.at[:,3:6].add(imp);after=free(hit,s-event,acc);passed=s>=event-1e-12
                    Y=Y.at[:,k,:6].set(jnp.where(passed[:,None],after[:,:6],Y[:,k,:6]));Y=Y.at[:,k,20:23].set(jnp.where(passed[:,None],old[:,i+1,20:23],row[:,20:23]))
                    Y=Y.at[:,k,23:30].set((1-s/h)*row[:,23:30]+s/h*old[:,i+1,23:30])
                else:Y=Y.at[:,k,23:30].set(row[:,23:30])
                if hybrid and endpoint and i>=prefix:
                    a0=self.acceleration(row[:,6:13],row[:,13:20],row[:,23:30]).reshape(b,7) if i==prefix else row[:,23:30]
                    endpoint_acc=a0 if i==N else (1-s/h)*a0+(s/h)*old[:,i+1,23:30]
                    control=endpoint_acc if k>prefix else self.torque(Y[:,k,6:13],Y[:,k,13:20],endpoint_acc).reshape(b,7)
                    Y=Y.at[:,k,23:30].set(control)
            strongest=jnp.zeros((b,N),x.dtype);batch_index=jnp.arange(b)
            for i in range(N):
                relative=times[i]+old[:,i,31]*steps[i]-dt;k=jnp.clip(jnp.searchsorted(times,relative+1e-12,side='right')-1,0,N)
                valid=(relative>=-1e-12)&(k<N);dest=jnp.minimum(k,N-1);impulse=old[:,i,30]
                Y=Y.at[batch_index,dest,30].add(jnp.where(valid,impulse,0))
                dominant=valid&(jnp.abs(impulse)>strongest[batch_index,dest]);frac=jnp.clip((relative-times[dest])/steps[dest],0,1)
                Y=Y.at[batch_index,dest,31].set(jnp.where(dominant,frac,Y[batch_index,dest,31]));strongest=strongest.at[batch_index,dest].set(jnp.where(dominant,jnp.abs(impulse),strongest[batch_index,dest]))
            feedback=c['warmstart_feedback_time']
            if feedback>0:
                off=x[:,6:13]-Y[:,0,6:13];delta=x[:,13:20]-Y[:,0,13:20]
                for k in range(N+1):
                    decay=jnp.exp(-times[k]/feedback);q=Y[:,k,6:13];v=Y[:,k,13:20]
                    a=Y[:,k,23:30] if hybrid and k>prefix else self.acceleration(q,v,Y[:,k,23:30]).reshape(b,7);qn=q+off;vn=v+decay*delta
                    tor=a-decay/feedback*delta if hybrid and k>prefix else self.torque(qn,vn,a-decay/feedback*delta).reshape(b,7)
                    Y=Y.at[:,k,6:13].set(qn).at[:,k,13:20].set(vn).at[:,k,23:30].set(tor)
                    if k<N:off+=.5*steps[k]*(decay+jnp.exp(-times[k+1]/feedback))*delta
            Y=Y.at[:,0,:23].set(x).at[:,0,23:30].set(tau).at[:,:,32].set(jnp.r_[steps,steps[-1]]).at[:,:,33].set(times)
            if hybrid and endpoint:
                a0=self.acceleration(Y[:,prefix,6:13],Y[:,prefix,13:20],Y[:,prefix,23:30]).reshape(b,7)
                for k in range(prefix,N):
                    a1=Y[:,k+1,23:30];average=.5*(a0+a1)
                    Y=Y.at[:,k+1,6:13].set(Y[:,k,6:13]+steps[k]*Y[:,k,13:20]+.5*steps[k]**2*average)
                    Y=Y.at[:,k+1,13:20].set(Y[:,k,13:20]+steps[k]*average);a0=a1
            shifted=Y;state=x;failed=jnp.zeros(b,bool);virtual_contacts=jnp.zeros(b,jnp.int32)
            for k in range(N):
                Y=Y.at[:,k,:6].set(state[:,:6]).at[:,k,20:23].set(state[:,20:23]).at[:,k,30].set(0).at[:,k,31].set(.5)
                acc=(Y[:,k+1,13:20]-Y[:,k,13:20])/steps[k]
                state,j,hit,resting,virtual_contacts=rollout(Y[:,k,:23],steps[k],acc,.01,times[k],virtual_contacts,True)
                failed|=resting|(state[:,2]<-.15)|(state[:,2]>1.5)
                Y=Y.at[:,k,30].set(j).at[:,k,31].set(jnp.where(j!=0,hit/steps[k],.5))
            Y=Y.at[:,N,:6].set(state[:,:6]).at[:,N,20:23].set(state[:,20:23]);Y=jnp.where(failed[:,None,None],shifted,Y)
            dual=jnp.where(jnp.asarray(dual_src)>=0,dual[:,np.maximum(dual_src,0)],0.)
            return ((Y-offset)/scale).reshape(b,-1),dual,failed,((shifted-offset)/scale).reshape(b,-1)
        def shift_with_event(z,dual,x,tau):
            z,dual,failed,shifted=shift(z,dual,x,tau)
            self.event_state,self.event_parameters=self.event_tracker.update(shifted,x,self.event_state,dt)
            return z,dual,failed
        self.shift=shift_with_event
