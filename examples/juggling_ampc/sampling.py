"""Randomized drops within the unchanged controller/fixture workspace."""
import numpy as np

def drop_positions(rng,count,origin,config,recorded_ball_base,fixture_radius=.32,nominal_strike_z=None):
    if count<1 or not 0<fixture_radius<=.32:raise ValueError("Invalid sample count or native fixture radius")
    origin=np.asarray(origin,float);recorded=np.asarray(recorded_ball_base,float)
    if recorded.ndim!=2 or recorded.shape[1]!=3 or not np.isfinite(recorded).all():raise ValueError('Need recorded initial ball positions in robot-base metres')
    low=(origin[2] if nominal_strike_z is None else float(nominal_strike_z))+1.;high=float(np.max(recorded[:,2]))+.2
    if high<low:raise ValueError(f'Recorded drop height + 20 cm ({high}) is below strike plane + 1 m ({low})')
    # The live set_reference API additionally restricts fixture offsets to 32 cm.
    # Sample its intersection with the configured paddle workspace; IK must
    # still validate each initial fixture posture before the rollout starts.
    xy=[]
    while sum(len(v) for v in xy)<count:
        x=rng.uniform(-.4,config['workspace_forward'],count)
        y=rng.uniform(-config['workspace_side'],config['workspace_side'],count)
        keep=x*x+y*y<=fixture_radius**2;xy.append(np.c_[x[keep],y[keep]])
    positions=np.empty((count,3));positions[:,:2]=np.concatenate(xy)[:count]+origin[:2];positions[:,2]=rng.uniform(low,high,count)
    return positions,dict(z_min_base_m=low,z_max_base_m=high,fixture_radius_m=fixture_radius)
