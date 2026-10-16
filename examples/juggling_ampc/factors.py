"""Native Gauss--Newton factors, reference shifts, and sparse global assembly."""
import json
import os
from pathlib import Path
import casadi as ca
import numpy as np
import scipy.sparse as sp


class Factors:
    def __init__(self,directory):
        self.directory=Path(directory);self.meta=json.loads((self.directory/'metadata.json').read_text())
        meta=self.meta;cfg=meta['config'];self.cfg=cfg;self.arrays=meta['arrays'];self.n=34*(cfg['horizon']+1)
        self.base_m=len(self.arrays['constraint_lower']);self.functions={};self.patterns={}
        self.cap_nodes=int(meta['spec'].get('prediction_constraints',{}).get('cap_nodes',0))
        self.arrays['constraint_lower']=list(self.arrays['constraint_lower'])+[-np.inf]*self.cap_nodes
        self.arrays['constraint_upper']=list(self.arrays['constraint_upper'])+[meta['spec'].get('prediction_constraints',{}).get('cap_value',0.)]*self.cap_nodes
        meta['constraint_ids']=list(meta['constraint_ids'])+[-2]*self.cap_nodes
        self.m=self.base_m+self.cap_nodes
        if not meta['spec'].get('runtime_parameters'):raise ValueError('Expected parametric native factors with actual cost weights')
        self.layouts=[('first',0,0,meta['stage_steps'][0])]+[(name,k*34,meta['stage_offsets'][k],meta['stage_steps'][k]) for k,name in enumerate(meta['stage_names'])]+[('last',cfg['horizon']*34,meta['stage_offsets'][-1],meta['stage_steps'][-1])]
        self.layouts += [('cap_'+meta['stage_names'][k],k*34,self.base_m+k,meta['stage_steps'][k]) for k in range(self.cap_nodes)]
        for name in set(x[0] for x in self.layouts):
            native_name=name.removeprefix('cap_')
            native=ca.Function.load(str(self.directory/(native_name+'_data.casadi')))
            nv=native.size1_in(0);z=ca.MX.sym('z',nv);p=ca.MX.sym('p',46);h=ca.MX.sym('h');projector=ca.MX.sym('projector',7,7)
            stage_index=ca.MX.sym('stage_index');stage_time=ca.MX.sym('stage_time')
            r,jr,c,jc=native(z,p[:30],ca.DM(self.arrays['factor_parameters']))
            if nv==68:
                spec=meta['spec']
                if spec.get('legacy_contact_prediction') and spec.get('prediction_edge_loss',0.):
                    edge=ca.Function.load(str(self.directory/'edge_prediction_value.casadi'))
                    r[52]+=edge(z,p[:30],ca.DM(self.arrays['factor_parameters']),spec['prediction_edge_loss'])
                if spec.get('prediction_plane_offset_m',0.)!=0.:
                    raise ValueError('The selected kinematic profile requires zero plane offset')
                j0=spec.get('radial_contact_scaling',{}).get('j0_m_s',0.)
                if j0:
                    row=next(row for kind,col,row,step in self.layouts if kind==name)
                    ids=meta['constraint_ids'][row:row+int(c.numel())]
                    # Cap layouts reuse only the restitution residual below.
                    if not name.startswith('cap_'):
                        radial=ids.index(meta['groups']['contact'][0]+2)
                        impulse=z[30]*self.arrays['scale'][30]+self.arrays['offset'][30]
                        c[radial]*=np.hypot(8.,j0)/ca.sqrt(impulse**2+j0**2)
                if name.startswith('cap_'):
                    c=r[52]/(np.sqrt(cfg['objective_scale'])*cfg['comp_weight'])
                    r=ca.MX.zeros(0,1)
                jr=ca.jacobian(r,z);jc=ca.jacobian(c,z)
            nr=int(r.numel());weights=np.ones(nr)
            if nr>=54:
                w=ca.vertcat(*[ca.sqrt(h/.03)*(.03/h if 30<=i<37 else 1) if i<51 and meta['quadrature'] else 1 for i in range(nr)])
            else:w=ca.DM(weights)
            shift=ca.MX.zeros(nr,1);offset=p[30:33];posture=p[33:40]-ca.DM(self.arrays['home']);s=np.sqrt(cfg['objective_scale'])
            if nr>=54:
                shift[0]=s*cfg['energy_weight']*offset[2];shift[1:3]=s*cfg['lateral_weight']*offset[:2]/.2
                shift[5:8]=s*cfg['pose_weight']*offset/ca.DM([.3/cfg['paddle_lateral_weight'],.3/cfg['paddle_lateral_weight'],.15])
                shift[37:44]=s*cfg['posture_weight']*posture
            elif nr==5:
                shift[0]=2*s*cfg['energy_weight']*offset[2];shift[1:3]=2*s*cfg['lateral_weight']*offset[:2]/.2
            residual=w*(r-shift);J=ca.diag(w)@jr
            H=J.T@J;gradient=J.T@residual;cost=.5*ca.dot(residual,residual)
            if name not in ('first','last') and not name.startswith('cap_'):
                reg=meta['spec'].get('nullspace_regularization',{})
                for start,scale,gain,delta in [(6,1.,reg.get('position_weight',0.),posture),(13,2.,reg.get('velocity_weight',0.),ca.DM.zeros(7))]:
                    weight=cfg['objective_scale']*(h/.03 if meta['quadrature'] else 1)*gain**2
                    diff=scale*z[start:start+7]-delta
                    H[start:start+7,start:start+7]+=weight*scale**2*(projector.T@projector)
                    gradient[start:start+7]+=weight*scale*(projector.T@projector)@diff
                    cost+=.5*weight*ca.sumsqr(projector@diff)
                event_weight=meta['spec'].get('first_event_center',{}).get('weight',0.)
                if event_weight:
                    # Freeze event identity/time/target outside both SQP iterations.
                    # p[40:46] = active, lead, stage, target_xyz.
                    tau=ca.fmin(h,ca.fmax(0.,p[41]-stage_time))
                    next_coefficient=.5*tau*tau/h
                    current_coefficient=tau-next_coefficient
                    physical=z[:34]*ca.DM(self.arrays['scale'])+ca.DM(self.arrays['offset'])
                    following=z[34:]*ca.DM(self.arrays['scale'])+ca.DM(self.arrays['offset'])
                    contact_q=physical[6:13]+current_coefficient*physical[13:20]+next_coefficient*following[13:20]
                    fk=ca.Function.load(str(self.directory/'panda_contact_fk.casadi'))
                    position,normal,_,_,rotation=fk(contact_q,ca.DM(self.arrays['factor_parameters']))
                    R=np.asarray(self.arrays['factor_parameters'][26:35]).reshape(3,3,order='F')
                    grip=R.T@np.array([-1.,0.,0.]);grip[2]=0.;grip/=np.linalg.norm(grip)
                    preferred=rotation@ca.DM(cfg['contact_center_grip_offset']*grip)
                    relative=p[43:46]-position+ca.DM(self.arrays['origin'])
                    event_residual=np.sqrt(cfg['objective_scale'])*event_weight/cfg['paddle_radius']*(relative-normal*ca.dot(normal,relative)-preferred)
                    event_residual*=p[40]*ca.if_else(p[42]==stage_index,1.,0.)
                    event_jacobian=ca.jacobian(event_residual,z)
                    H+=event_jacobian.T@event_jacobian
                    gradient+=event_jacobian.T@event_residual
                    cost+=.5*ca.sumsqr(event_residual)
            H=ca.triu(H)
            hp=H.sparsity().get_triplet();ap=jc.sparsity().get_triplet()
            self.patterns[name]=(np.array(hp,dtype=int),np.array(ap,dtype=int),nv,int(c.numel()))
            packed=ca.Function('juggling_'+name,[z,p,ca.vertcat(h,ca.vec(projector),stage_index,stage_time)],[ca.densify(ca.vertcat(*H.nonzeros())),ca.densify(gradient),ca.densify(ca.vertcat(*jc.nonzeros())),ca.densify(c),cost])
            self.functions[name]=packed
        hset={(i,i) for i in range(self.n)};aset={(self.m+i,i) for i in range(self.n)}
        for name,col,row,h in self.layouts:
            hp,ap,_,_=self.patterns[name];hset.update(zip(hp[0]+col,hp[1]+col));aset.update(zip(ap[0]+row,ap[1]+col))
        def pattern(entries,shape):
            r,c=zip(*entries);return sp.csc_matrix((np.ones(len(r)),(r,c)),shape=shape)
        self.P=pattern(hset,(self.n,self.n));self.A=pattern(aset,(self.m+self.n,self.n))
        def index(a):return {(int(a.indices[j]),i):j for i in range(a.shape[1]) for j in range(a.indptr[i],a.indptr[i+1])}
        hi=index(self.P);ai=index(self.A);self.maps=[]
        self.diag=np.array([hi[i,i] for i in range(self.n)]);self.identity=np.array([ai[self.m+i,i] for i in range(self.n)])
        for name,col,row,h in self.layouts:
            hp,ap,nv,nc=self.patterns[name]
            self.maps.append((np.array([hi[r+col,c+col] for r,c in hp.T],dtype=int),np.array([ai[r+row,c+col] for r,c in ap.T],dtype=int),nv,nc))

    def projector(self,reference):
        f=ca.Function.load(str(self.directory/'panda_contact_fk.casadi'))
        q=reference[3:];_,n,J,W,_=f(q,ca.DM(self.arrays['factor_parameters']))
        n=np.asarray(n).ravel();cross=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        _,sv,vh=np.linalg.svd(np.vstack([J,-cross@np.asarray(W)]),full_matrices=True)
        v=vh[:len(sv)][sv>1e-8];return np.eye(7)-v.T@v

    def numpy(self,z,p):
        if len(p)==40:p=np.r_[p,np.zeros(6)]
        projector=self.projector(p[30:40]);hv=np.zeros(self.P.nnz);av=np.zeros(self.A.nnz);g=np.zeros(self.n);con=np.zeros(self.m);cost=0.
        hv[self.diag]=self.cfg['proximal'];av[self.identity]=1.
        for layout,mapping in zip(self.layouts,self.maps):
            name,col,row,h=layout;hi,ai,nv,nc=mapping
            H,G,A,C,V=self.functions[name](z[col:col+nv],p,np.r_[h,projector.ravel(order='F'),col//34,self.meta['stage_times'][col//34]])
            hv[hi]+=np.asarray(H).ravel();av[ai]=np.asarray(A).ravel();g[col:col+nv]+=np.asarray(G).ravel();con[row:row+nc]=np.asarray(C).ravel();cost+=float(V)
        return sp.csc_matrix((hv,self.P.indices,self.P.indptr),shape=self.P.shape),g,sp.csc_matrix((av,self.A.indices,self.A.indptr),shape=self.A.shape),con,cost

    def compile(self,backend='warp'):
        import jax
        import jax.numpy as jnp
        from warpmpc.numpysadi import convert
        from chunked_factors import convert_chunked
        def compile_factor(function):
            expanded=function.expand()
            if backend=='warp' and int(os.environ.get('JUGGLING_FACTOR_CHUNK_SIZE','4000'))>0:
                transposed=os.environ.get('JUGGLING_FACTOR_TRANSPOSE','1')=='1'
                call=convert_chunked(expanded,int(os.environ.get('JUGGLING_FACTOR_CHUNK_SIZE','4000')),batched=transposed)
                return call if transposed else jax.vmap(call)
            return jax.vmap(convert(expanded,backend=backend,**({'warp_compile_profile':os.environ.get('JUGGLING_FACTOR_COMPILE_PROFILE','runtime')} if backend=='warp' else {})))
        compiled={name:compile_factor(f) for name,f in self.functions.items()}
        groups=self.streaming_layouts()
        small_evaluate=self.layout_evaluator(compiled)
        @jax.jit
        def assemble(z,p,projector):
            if p.shape[1]==40:p=jnp.concatenate((p,jnp.zeros((p.shape[0],6),p.dtype)),axis=1)
            b=z.shape[0];hv=jnp.zeros((b,self.P.nnz),z.dtype).at[:,self.diag].set(self.cfg['proximal']);av=jnp.zeros((b,self.A.nnz),z.dtype).at[:,self.identity].set(1.)
            g=jnp.zeros_like(z);con=jnp.zeros((b,self.m),z.dtype);cost=jnp.zeros(b,z.dtype)
            # Small validation/smoke batches benefit from combining independent
            # stages into each launch. The 60k collection never takes this path.
            if b<8192:
                values=small_evaluate(z,p,projector)
                for layout,mapping,outputs in zip(self.layouts,self.maps,values):
                    name,col,row,h=layout;hi,ai,nv,nc=mapping;H,G,A,C,V=outputs
                    hv=hv.at[:,hi].add(H.reshape(b,-1));av=av.at[:,ai].set(A.reshape(b,-1));g=g.at[:,col:col+nv].add(G.reshape(b,-1));con=con.at[:,row:row+nc].set(C.reshape(b,-1));cost+=V.reshape(b)
                return hv,g,av,con,cost
            for name,columns,rows,steps,his,ais,nv,nc in groups:
                columns=jnp.asarray(columns);rows=jnp.asarray(rows);steps=jnp.asarray(steps,z.dtype);his=jnp.asarray(his);ais=jnp.asarray(ais)
                def stage(i,carry):
                    hv,g,av,con,cost=carry;col=columns[i];row=rows[i]
                    local=jax.lax.dynamic_slice(z,(0,col),(b,nv))
                    aux=jnp.concatenate((jnp.full((b,1),steps[i],z.dtype),projector,jnp.full((b,1),col//34,z.dtype),jnp.full((b,1),jnp.asarray(self.meta['stage_times'],z.dtype)[col//34],z.dtype)),axis=1)
                    H,G,A,C,V=compiled[name](local,p,aux)
                    hv=hv.at[:,his[i]].add(H.reshape(b,-1));av=av.at[:,ais[i]].set(A.reshape(b,-1))
                    gradient=jax.lax.dynamic_slice(g,(0,col),(b,nv))+G.reshape(b,nv)
                    g=jax.lax.dynamic_update_slice(g,gradient,(0,col))
                    con=jax.lax.dynamic_update_slice(con,C.reshape(b,nc),(0,row));cost+=V.reshape(b)
                    return hv,g,av,con,cost
                # Keep just one stage's factor outputs live at a time. A Python
                # unroll/list would let XLA retain every horizon stage together.
                hv,g,av,con,cost=jax.lax.fori_loop(0,len(columns),stage,(hv,g,av,con,cost))
            return hv,g,av,con,cost
        return assemble

    def layout_evaluator(self,compiled):
        """Combine same-shape stages only for small validation batches."""
        import jax.numpy as jnp
        limit=int(os.environ.get('JUGGLING_FACTOR_STAGE_BATCH','20'))
        if limit<1:raise ValueError('Positive stage batch size required')
        by_name=[]
        for name in dict.fromkeys(row[0] for row in self.layouts):
            indices=[i for i,row in enumerate(self.layouts) if row[0]==name]
            by_name.append((name,indices))
        def evaluate(z,p,projector):
            b=z.shape[0];values=[None]*len(self.layouts)
            effective_limit=min(limit,max(1,8192//b))
            groups=[(name,indices[k:k+effective_limit]) for name,indices in by_name for k in range(0,len(indices),effective_limit)]
            for name,indices in groups:
                count=len(indices);nv=self.patterns[name][2]
                columns=np.array([self.layouts[i][1] for i in indices])[:,None]+np.arange(nv)[None,:]
                local=z[:,columns].reshape(b*count,nv)
                parameters=jnp.broadcast_to(p[:,None,:],(b,count,p.shape[1])).reshape(b*count,p.shape[1])
                h=jnp.broadcast_to(jnp.asarray([self.layouts[i][3] for i in indices],dtype=z.dtype)[None,:,None],(b,count,1))
                projectors=jnp.broadcast_to(projector[:,None,:],(b,count,49))
                stages=np.asarray([self.layouts[i][1]//34 for i in indices])
                location=jnp.broadcast_to(jnp.asarray(np.c_[stages,np.asarray(self.meta['stage_times'])[stages]],z.dtype)[None,:,:],(b,count,2))
                aux=jnp.concatenate((h,projectors,location),axis=2).reshape(b*count,52)
                outputs=compiled[name](local,parameters,aux)
                outputs=[v.reshape(b,count,int(np.prod(v.shape[1:]))) for v in outputs]
                for j,i in enumerate(indices):values[i]=tuple(v[:,j] for v in outputs)
            return values
        return evaluate

    def streaming_layouts(self):
        """Consecutive same-shape groups in the original assembly order.

        Only mapping metadata is grouped. Every evaluation still processes the
        complete world batch, with no horizon-sized list of factor outputs.
        """
        groups=[];start=0
        while start<len(self.layouts):
            name=self.layouts[start][0];stop=start+1
            while stop<len(self.layouts) and self.layouts[stop][0]==name:stop+=1
            layouts=self.layouts[start:stop];maps=self.maps[start:stop];_,_,nv,nc=maps[0]
            groups.append((name,np.asarray([v[1] for v in layouts],np.int32),np.asarray([v[2] for v in layouts],np.int32),
                np.asarray([v[3] for v in layouts]),np.stack([v[0] for v in maps]),np.stack([v[1] for v in maps]),nv,nc))
            start=stop
        return groups

    def compile_merit(self,backend='warp'):
        """Evaluate only nonlinear constraints and objective during backtracking."""
        import jax
        import jax.numpy as jnp
        from warpmpc.numpysadi import convert
        from chunked_factors import convert_chunked
        compiled={}
        for name,function in self.functions.items():
            inputs=[ca.MX.sym('arg'+str(i),function.sparsity_in(i)) for i in range(function.n_in())]
            outputs=function(*inputs)
            merit=ca.Function('merit_'+name,inputs,[outputs[3],outputs[4]]).expand()
            transposed=backend=='warp' and os.environ.get('JUGGLING_FACTOR_TRANSPOSE','1')=='1'
            call=convert_chunked(merit,batched=transposed) if backend=='warp' else convert(merit,backend=backend)
            compiled[name]=call if transposed else jax.vmap(call)
        groups=self.streaming_layouts()
        small_evaluate=self.layout_evaluator(compiled)
        @jax.jit
        def evaluate(z,p,projector):
            if p.shape[1]==40:p=jnp.concatenate((p,jnp.zeros((p.shape[0],6),p.dtype)),axis=1)
            b=z.shape[0];con=jnp.zeros((b,self.m),z.dtype);cost=jnp.zeros(b,z.dtype)
            if b<8192:
                values=small_evaluate(z,p,projector)
                for layout,mapping,outputs in zip(self.layouts,self.maps,values):
                    name,col,row,h=layout;_,_,nv,nc=mapping;C,V=outputs
                    con=con.at[:,row:row+nc].set(C.reshape(b,-1));cost+=V.reshape(b)
                return con,cost
            for name,columns,rows,steps,_,_,nv,nc in groups:
                columns=jnp.asarray(columns);rows=jnp.asarray(rows);steps=jnp.asarray(steps,z.dtype)
                def stage(i,carry):
                    con,cost=carry;col=columns[i];row=rows[i]
                    local=jax.lax.dynamic_slice(z,(0,col),(b,nv))
                    aux=jnp.concatenate((jnp.full((b,1),steps[i],z.dtype),projector,jnp.full((b,1),col//34,z.dtype),jnp.full((b,1),jnp.asarray(self.meta['stage_times'],z.dtype)[col//34],z.dtype)),axis=1)
                    C,V=compiled[name](local,p,aux)
                    return jax.lax.dynamic_update_slice(con,C.reshape(b,nc),(0,row)),cost+V.reshape(b)
                con,cost=jax.lax.fori_loop(0,len(columns),stage,(con,cost))
            return con,cost
        return evaluate
