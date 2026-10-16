"""Impulse-based spinning-ball simulator used only to seed the MPC."""
import numpy as np
import pinocchio as pin
from .simulator import Simulator
from .spin_contact import impact


class SpinSimulator(Simulator):
    def __init__(self,robot,cfg,friction=.3,inertia_ratio=2/3):
        super().__init__(robot,cfg);self.friction=friction;self.inertia_ratio=inertia_ratio

    def step(self,x,ta,tb,h,wind=None,acc=None):
        wind=np.zeros(3) if wind is None else wind
        cfg=self.cfg;contacts=[];state=x.copy();elapsed=0.
        # Bounded search intervals also cover longer nominal warmstart steps.
        while elapsed<h-1e-14:
            dt=min(cfg.max_interval,h-elapsed)
            ua=ta+(tb-ta)*(elapsed/h);ub=ta+(tb-ta)*((elapsed+dt)/h)
            end=self.free(state,ua,ub,dt,wind,acc)
            ga=self.geometry(state);gb=self.geometry(end)
            if ga[0]>=-1e-10 and gb[0]<=0 and (ga[4]<0 or gb[4]<0):
                lo=0.;hi=dt
                while hi-lo>1e-10:
                    mid=.5*(lo+hi);um=ua+(ub-ua)*(mid/dt)
                    trial=self.free(state,ua,um,mid,wind,acc)
                    if self.geometry(trial)[0]>0:lo=mid
                    else:hi=mid
                hit=.5*(lo+hi);um=ua+(ub-ua)*(hit/dt)
                collision=self.free(state,ua,um,hit,wind,acc)
                gap,rad,n,J,speed=self.geometry(collision)
                if rad<=cfg.paddle_radius and speed<0:
                    p,n,Jp,W,R=self.robot.contact_fk(collision[6:13])
                    n=np.asarray(n).ravel();rel=collision[:3]-(np.asarray(p).ravel()-self.robot.origin)
                    tangent=rel-n*(n@rel);J=np.asarray(Jp)-pin.skew(tangent)@np.asarray(W)
                    relative=collision[3:6]-np.cross(collision[20:23],cfg.radius*n)-J@collision[13:20]
                    M=pin.crba(self.robot.pin,self.data,collision[6:13]).copy()
                    u,domega,dqjump,jn=impact(n,np.asarray(R)[:,:2],J,M,relative,cfg.ball_mass,cfg.radius,
                                             cfg.restitution,self.friction,self.inertia_ratio)
                    collision[3:6]+=u;collision[20:23]+=domega;collision[13:20]+=dqjump
                    collision[:3]+=(1e-10-gap)*n;elapsed+=hit
                    contacts.append(dict(time=elapsed,position=(collision[:3]-cfg.radius*n).tolist(),normal=n.tolist(),
                        incoming_normal_speed=float(speed),impulse_Ns=float(cfg.ball_mass*jn),radial_distance=float(rad),
                        ball_velocity_after=collision[3:6].tolist(),ball_spin_after=collision[20:23].tolist(),
                        joint_velocity_jump=dqjump.tolist(),impulse_vector_Ns=(cfg.ball_mass*u).tolist()))
                    state=collision
                    if len(contacts)>16:raise RuntimeError('Unsupported resting contact')
                    continue
            state=end;elapsed+=dt
        return state,contacts
