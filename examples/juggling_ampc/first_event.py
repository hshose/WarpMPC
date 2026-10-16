"""Batched frozen first-impact objective lifecycle from native commit 71dd0a4.

Only the supplied observation and previously shifted plan identify an event.
Simulator contact counters never enter this objective.
"""
import jax
import jax.numpy as jnp


class FirstEvent:
    def __init__(self, factors, free, geometry):
        config=factors.meta['spec'].get('first_event_center',{})
        self.enabled=config.get('weight',0.)>0.
        times=jnp.asarray(factors.meta['stage_times'],jnp.float32)
        steps=jnp.asarray(factors.meta['stage_steps'],jnp.float32)
        scale=jnp.asarray(factors.arrays['scale'],jnp.float32)
        offset=jnp.asarray(factors.arrays['offset'],jnp.float32)
        N=factors.cfg['horizon'];acquire_lead=config.get('acquire_lead_s',.15)

        @jax.jit
        def update(z,x,state,elapsed):
            active,lead,previous,have=state
            lead=lead-elapsed
            predicted=free(previous,elapsed,jnp.zeros_like(x[:,6:13]))
            rebound=have&(elapsed>0.)&(x[:,5]>.15)&(x[:,5]-predicted[:,5]>.5)
            active=active&~(rebound|(lead<-.06))
            Y=z.reshape(-1,N+1,34)*scale+offset
            b=x.shape[0]
            def stage(k,carry):
                found,blocked,selected,closest=carry
                y=Y[:,k];yn=Y[:,k+1];j=y[:,30];fraction=y[:,31]
                candidate=times[k]+fraction*steps[k]
                eligible=jnp.isfinite(j)&(j>=jnp.where(active,2.,.25))&jnp.isfinite(fraction)&(fraction>=0)&(fraction<=1)&(candidate>=0)&(candidate<times[-1])
                eligible&=~blocked&jnp.where(active,(candidate>0)&(jnp.abs(candidate-lead)<=closest),~found)
                before=free(y[:,:23],fraction*steps[k],(yn[:,13:20]-y[:,13:20])/steps[k])
                eligible&=jnp.all(jnp.isfinite(before),axis=1)&(before[:,5]<-.05)
                observed=free(x,candidate,jnp.zeros_like(x[:,6:13]))
                geo=geometry(before).reshape(b,3)
                credible=(j>=2)&(active|(candidate>acquire_lead))&(observed[:,5]<-.05)&jnp.isfinite(geo[:,0])&jnp.isfinite(geo[:,2])&(jnp.abs(geo[:,0])<=.03)&(geo[:,2]<-.05)
                take=eligible&credible
                blocked|=eligible&~credible&~active
                selected=jnp.where(take,candidate,selected)
                closest=jnp.where(take&active,jnp.abs(candidate-lead),closest)
                return found|take,blocked,selected,closest
            found,_,selected,_=jax.lax.fori_loop(0,N,stage,(jnp.zeros(b,bool),jnp.zeros(b,bool),jnp.zeros(b,x.dtype),jnp.full(b,.060000000001,x.dtype)))
            refresh=jnp.minimum(.005,.25*elapsed)
            lead=jnp.where(found,jnp.where(active,lead+jnp.clip(selected-lead,-refresh,refresh),selected),lead)
            active|=found
            prediction=jnp.maximum(0.,lead)
            selected_stage=jnp.minimum(jnp.searchsorted(times,prediction+1e-12,side='right')-1,N-1)
            target=free(x,prediction,jnp.zeros_like(x[:,6:13]))[:,:3]
            parameters=jnp.concatenate((active[:,None],prediction[:,None],selected_stage[:,None],target),axis=1).astype(x.dtype)
            return (active,lead,x,jnp.ones(b,bool)),parameters
        self.update=update

    def empty(self,x):
        return jnp.zeros(x.shape[0],bool),jnp.zeros(x.shape[0],x.dtype),x,jnp.zeros(x.shape[0],bool)
