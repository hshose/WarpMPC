"""Ball-only supervisor guards; robot envelope flags do not censor trajectories."""
import numpy as np

def ball_lost(state, stale, reference, fixture, cfg, return_details=False):
    x=np.asarray(state);g=np.array([cfg['gravity_x'],cfg['gravity_y'],cfg['gravity_z']]);drag=cfg['drag']
    plane=cfg['radius']+reference[:,2]
    def position(t):
        a=-np.expm1(-drag*t)/drag;b=(t-a)/drag
        return x[:,:3]+a[:,None]*x[:,3:6]+b[:,None]*g
    lo=np.zeros(len(x));hi=np.ones(len(x))
    for _ in range(6):hi=np.where(position(hi)[:,2]>plane,hi*2,hi)
    for _ in range(40):
        mid=(lo+hi)*.5;above=position(mid)[:,2]>plane
        lo=np.where(above,mid,lo);hi=np.where(above,hi,mid)
    hit=position((lo+hi)*.5)[:,:2]
    def inside(p,front,side,rear):return (p[:,0]/np.where(p[:,0]>=0,front,rear))**2+(p[:,1]/side)**2<=1
    reach=inside(hit,cfg['landing_forward'],cfg['landing_side'],.35)
    if cfg['fixture_landing_forward']>0:
        reach &= inside(hit-fixture[:,:2],cfg['fixture_landing_forward'],cfg['fixture_landing_side'],.12)
    unreachable=(x[:,2]>plane)&~reach
    flags=dict(stale=np.asarray(stale,bool),nonfinite=~np.isfinite(x[:,np.r_[0:6,20:23]]).all(axis=1),height=(x[:,2]<-.15)|(x[:,2]>1.5),unreachable=unreachable)
    return flags if return_details else np.logical_or.reduce(list(flags.values()))
