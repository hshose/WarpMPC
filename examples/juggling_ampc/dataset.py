"""Trajectory-level splits and irreversible ball-loss censoring."""
import numpy as np

class BallLossMask:
    def __init__(self,batch):
        self.lost=np.zeros(batch,bool);self.first_loss=np.full(batch,np.inf)
    def update(self,time,ball_lost):
        event=np.asarray(ball_lost,bool)
        first=event&~self.lost;self.first_loss[first]=time;self.lost|=event
        # The event's input/label may already be unreliable; omit it as well.
        return ~self.lost.copy()

def split_trajectories(count,seed=2026,validation_fraction=.1,test_fraction=.1):
    if count<3 or validation_fraction<=0 or test_fraction<=0 or validation_fraction+test_fraction>=1:raise ValueError('Need three nonempty trajectory partitions')
    ids=np.random.default_rng(seed).permutation(count);nv=max(1,int(count*validation_fraction));nt=max(1,int(count*test_fraction))
    return dict(train=ids[nv+nt:],validation=ids[:nv],test=ids[nv:nv+nt])

def finite_samples(inputs,targets,alive):
    return np.asarray(alive,bool)&np.isfinite(inputs).all(axis=-1)&np.isfinite(targets).all(axis=-1)

def normalization(x,y):
    if len(x)==0:raise ValueError('No valid training samples')
    # Fit exclusively on training trajectories; retain failures up to loss.
    return dict(input_mean=np.mean(x,axis=0),input_scale=np.maximum(np.std(x,axis=0),1e-5),
                output_mean=np.mean(y,axis=0),output_scale=np.maximum(np.std(y,axis=0),1e-5))
